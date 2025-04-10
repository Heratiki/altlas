import requests
import json
import logging
import time
import threading
import asyncio
import aiohttp
from asyncio import Lock as AsyncLock, Queue as AsyncQueue
from typing import Union, Tuple, Dict, Any, Optional, List
from collections import deque # For recent latencies
import ast # For safely evaluating config strings like dictionaries

# Import the tokenizer
from tokenizer import Tokenizer
# --- Need to import torch for truncation logic ---
import torch


class LLMProvider:
    def __init__(self, config: dict = None):
        self.config = config or {}
        self.provider = self.config.get("provider", "vllm")
        self.vllm_base_url = self.config.get("vllm_base_url", "http://localhost:8000/v1")
        self.lmstudio_base_url = self.config.get("lmstudio_base_url", "http://localhost:1234/v1")
        self.model_preference = self.config.get("model_preference", ["wizardcoder", "codellama", "code-llama", "stable-code", "starcoder", "llama"])
        self.timeout = int(self.config.get("timeout", 120))
        self.temperature = float(self.config.get("temperature", 0.7))
        self.max_tokens = int(self.config.get("max_tokens", 500)) # Max *response* tokens

        # --- Prompt Templating and Token Management Config ---
        self.tokenizer = Tokenizer() # Instantiate the tokenizer
        self.prompt_templates = self._parse_config_dict(self.config.get("prompt_templates", {}))
        self.max_prompt_tokens = int(self.config.get("max_prompt_tokens", 4096)) # Max *prompt* tokens
        self.warn_prompt_tokens = int(self.config.get("warn_prompt_tokens", 3000)) # Warn threshold for prompt
        self.truncate_prompt = str(self.config.get("truncate_prompt", "false")).lower() == "true"
        # --- End Prompt Config ---

        # --- Async Config ---
        self._async_enabled = str(self.config.get("enable_async", "false")).lower() == "true"
        self._async_batch_window_ms = int(self.config.get("async_batch_window_ms", 20))
        self._aiohttp_session = None
        self._async_queue = None
        self._async_batch_loop_task = None
        self._async_stats_lock = AsyncLock()
        self._async_cache_lock = AsyncLock()
        # --- End Async Config ---

        # --- Cache Config ---
        self.cache = {}
        self.cache_order = []
        self.cache_size = int(self.config.get("cache_size", 100))
        self._cache_lock = threading.Lock() # Lock for sync cache updates
        self.cache_hits = 0
        self.cache_misses = 0

        # --- Performance stats ---
        self.total_requests = 0
        self.total_prompt_tokens = 0
        self.total_response_tokens = 0
        self.total_latency = 0.0
        self.failed_requests = 0
        self.recent_latencies = deque(maxlen=int(self.config.get("recent_latency_window", 50))) # Track last N latencies
        self.single_requests_processed = 0 # Requests processed outside batching loops
        self.batched_requests_processed = 0 # Individual requests processed *within* batches
        self.batches_processed = 0 # Number of times a batch was executed (sync or async)
        self.async_queue_max_observed_length = 0 # Max items seen in async queue during batch collection
        self._stats_lock = threading.Lock() # Lock for sync updates

        # --- Adaptive throttling ---
        self.throttle_threshold = float(self.config.get("throttle_threshold", 5.0))  # seconds
        self.throttle_delay = float(self.config.get("throttle_delay", 2.0))  # seconds

        # --- Sync Batching Config ---
        self._batch_enabled = str(self.config.get("enable_batching", "false")).lower() == "true"
        self._batch_window_ms = int(self.config.get("batch_window_ms", 20))
        self._batch_queue = [] # Stores dicts: {"prompt": str, "prompt_tokens": int, "kwargs": dict, "event": Event, "response_holder": list}
        self._batch_lock = threading.Lock()
        self._batch_thread = None

        # Start sync batch loop ONLY if sync batching is enabled AND async is disabled
        if self._batch_enabled and not self._async_enabled:
            self._batch_thread = threading.Thread(target=self._batch_loop, daemon=True)
            self._batch_thread.start()
            logging.info("LLMProvider initialized with sync batching enabled.")
        elif self._async_enabled:
            # Async batch loop started in async_init
            logging.info("LLMProvider initialized with async mode enabled. Call async_init() before use.")
        else:
            logging.info("LLMProvider initialized in synchronous, non-batching mode.")

    async def async_init(self):
        """Initialize aiohttp session and async batch loop."""
        if not self._async_enabled:
            logging.info("async_init() called but async mode is disabled.")
            return

        if self._aiohttp_session is None:
            # Use a timeout configuration for the session
            timeout = aiohttp.ClientTimeout(total=self.timeout + 10) # Add buffer to overall timeout
            self._aiohttp_session = aiohttp.ClientSession(timeout=timeout)
            logging.info("aiohttp ClientSession created.")

        # Start async batch loop only if batching is enabled
        if self._batch_enabled and self._async_queue is None:
            self._async_queue = AsyncQueue()
            self._async_batch_loop_task = asyncio.create_task(self._async_batch_loop())
            logging.info("Async batch loop started.")

    async def async_close(self):
        """Cleanup aiohttp session and stop async batch loop."""
        # Stop batch loop first
        if self._async_batch_loop_task:
            self._async_batch_loop_task.cancel()
            try:
                await self._async_batch_loop_task
            except asyncio.CancelledError:
                logging.info("Async batch loop task cancelled.")
            except Exception as e:
                # Log error if cancellation wasn't the cause
                logging.error(f"Error during async batch loop task cleanup: {e}")
            self._async_batch_loop_task = None
            self._async_queue = None # Clear queue reference

        # Close session
        if self._aiohttp_session:
            await self._aiohttp_session.close()
            self._aiohttp_session = None
            logging.info("aiohttp ClientSession closed.")

    def _parse_config_dict(self, config_value: Any) -> Dict:
        """Safely parse dict-like strings from config."""
        if isinstance(config_value, dict):
            return config_value
        if isinstance(config_value, str):
            try:
                parsed = ast.literal_eval(config_value)
                if isinstance(parsed, dict):
                    return parsed
                else:
                    logging.warning(f"Config value '{config_value}' parsed but is not a dict. Ignoring.")
                    return {}
            except (ValueError, SyntaxError, TypeError) as e:
                logging.error(f"Failed to parse config value as dict: '{config_value}'. Error: {e}. Using empty dict.")
                return {}
        logging.warning(f"Unexpected type for config value (expected str or dict): {type(config_value)}. Using empty dict.")
        return {}

    def _parse_config_list(self, config_value: Any) -> List:
        """Safely parse list-like strings from config."""
        if isinstance(config_value, list):
            return config_value
        if isinstance(config_value, str):
            try:
                parsed = ast.literal_eval(config_value)
                if isinstance(parsed, list):
                    return parsed
                else:
                    logging.warning(f"Config value '{config_value}' parsed but is not a list. Ignoring.")
                    return []
            except (ValueError, SyntaxError, TypeError) as e:
                logging.error(f"Failed to parse config value as list: '{config_value}'. Error: {e}. Using empty list.")
                return []
        logging.warning(f"Unexpected type for config value (expected str or list): {type(config_value)}. Using empty list.")
        return []

    def _count_tokens(self, text: str) -> int:
        """Counts tokens in the text using the project's tokenizer."""
        if not text:
            return 0
        try:
            # Encode and get the number of tokens. Subtract 2 for SOS/EOS added by encode.
            # Ensure the result is not negative if the text is too short.
            return max(0, self.tokenizer.encode(text).size(0) - 2)
        except Exception as e:
            logging.error(f"Tokenizer failed to encode text for counting: {e}. Falling back to space splitting.")
            # Fallback to simple word count if tokenizer fails
            return len(text.split())

    def _apply_template(self, template_name: str, context: Dict[str, Any]) -> str:
        """Applies a prompt template using the given context."""
        template = self.prompt_templates.get(template_name)
        if not template:
            logging.warning(f"Prompt template '{template_name}' not found in config. Returning raw context as string.")
            # Fallback: just return the context values concatenated (crude)
            return "\n".join(f"{k}: {v}" for k, v in context.items())

        try:
            # Basic placeholder filling using .format()
            # Ensure all required placeholders are present in the context
            # This will raise a KeyError if a placeholder is missing
            formatted_prompt = template.format(**context)
            return formatted_prompt
        except KeyError as e:
            logging.error(f"Missing key '{e}' in context for prompt template '{template_name}'. Context: {context}")
            # Fallback or raise error? For now, return template with missing key indicated
            return template + f"\n\n[ERROR: Missing context key: {e}]"
        except Exception as e:
            logging.error(f"Error formatting prompt template '{template_name}': {e}")
            # Fallback to raw template
            return template # Or maybe raise?

    def _handle_token_limits(self, prompt: str, template_name: Optional[str] = None) -> str:
        """Checks prompt token count, warns, and truncates if configured."""
        prompt_token_count = self._count_tokens(prompt)

        if prompt_token_count > self.max_prompt_tokens:
            logging.error(f"Prompt ({template_name or 'raw'}) exceeds max token limit ({prompt_token_count}/{self.max_prompt_tokens}).")
            if self.truncate_prompt:
                logging.warning(f"Truncating prompt to {self.max_prompt_tokens} tokens (simple truncation).")
                # Simple truncation: Keep the beginning. This is crude.
                # A better approach would involve smarter summarization or context-aware truncation.
                # We need to decode back from tokens to truncate accurately.
                try:
                    encoded_ids = self.tokenizer.encode(prompt)
                    # Keep SOS, truncate, add EOS back. Ensure we don't go below 2 tokens.
                    if encoded_ids.size(0) > 2:
                         # +1 because max_prompt_tokens doesn't include SOS/EOS, but slice needs endpoint
                        truncated_ids = encoded_ids[:self.max_prompt_tokens + 1]
                        # Ensure EOS is the last token if possible
                        if truncated_ids[-1] != self.tokenizer.eos_token_id:
                             truncated_ids = torch.cat((truncated_ids[:-1], torch.tensor([self.tokenizer.eos_token_id], dtype=torch.long)))

                        prompt = self.tokenizer.decode(truncated_ids)
                        prompt_token_count = self._count_tokens(prompt) # Recalculate after truncation
                        logging.info(f"Prompt truncated to {prompt_token_count} tokens.")
                    else:
                        logging.warning("Cannot truncate prompt further, already too short.")
                        # Return empty or original? Returning original for now.
                        # Consider returning an error indicator?

                except Exception as e:
                    logging.error(f"Failed to truncate prompt using tokenizer: {e}. Returning original prompt.")
            else:
                # If not truncating, maybe raise an error or return a specific error message?
                # For now, just log the error and proceed with the oversized prompt (API might reject it)
                 logging.error("Prompt truncation is disabled. Sending oversized prompt to LLM.")
                 # raise ValueError(f"Prompt exceeds max token limit ({prompt_token_count}/{self.max_prompt_tokens}) and truncation is disabled.")

        elif prompt_token_count > self.warn_prompt_tokens:
            logging.warning(f"Prompt ({template_name or 'raw'}) is approaching token limit ({prompt_token_count}/{self.max_prompt_tokens}).")

        # Store the final count for stats (even if truncated)
        self._current_prompt_token_count = prompt_token_count

        return prompt


    # --- Synchronous API ---

    def generate(self, prompt_or_template: Union[str, Tuple[str, Dict[str, Any]]], **kwargs) -> str:
        """
        Generates text using the LLM (Synchronous).

        If async mode is enabled, this will log a warning and perform a single, non-batched generation.
        If sync batching is enabled (and async is disabled), it uses the sync batch queue.
        Otherwise, performs a direct single generation.

        Args:
            prompt_or_template: Either a raw prompt string or a tuple (template_name, context).
            **kwargs: Additional arguments to override LLM defaults (e.g., temperature).

        Returns:
            The generated text response from the LLM.
        """
        if self._async_enabled:
            logging.warning("Calling synchronous generate() while async mode is enabled. Performing single, non-batched request.")
            # Note: We could potentially use asyncio.run() here to call the async version,
            # but that can be complex with running loops. Direct single call is simpler for now.
            return self._generate_single(prompt_or_template, **kwargs)

        # Use sync batching only if enabled and async is disabled
        if self._batch_enabled and not self._async_enabled:
            # Process template before queueing
            template_name = None
            if isinstance(prompt_or_template, tuple):
                template_name, context = prompt_or_template
                prompt_str = self._apply_template(template_name, context)
            else:
                prompt_str = prompt_or_template

            # Handle token limits before queueing
            processed_prompt = self._handle_token_limits(prompt_str, template_name)
            prompt_tokens = getattr(self, '_current_prompt_token_count', 0) # Get count set by _handle_token_limits

            # Queue the request for the sync batch loop
            event = threading.Event()
            response_holder = [None] # Use list to allow modification by the batch thread
            request_data = {
                "prompt": processed_prompt,
                "prompt_tokens": prompt_tokens, # Pass pre-calculated tokens
                "kwargs": kwargs,
                "event": event,
                "response_holder": response_holder
            }
            with self._batch_lock:
                self._batch_queue.append(request_data)

            # Wait for the batch thread to process, with timeout
            if not event.wait(timeout=self.timeout + 10): # Add buffer
                logging.error("Sync batch request timed out waiting for event.")
                # Attempt to remove the stale request (best effort)
                with self._batch_lock:
                    try:
                        # Find the specific request to remove
                        for i, req in enumerate(self._batch_queue):
                            if req["event"] == event:
                                del self._batch_queue[i]
                                logging.warning("Removed stale sync batch request from queue.")
                                break
                    except Exception as e:
                        logging.error(f"Error removing stale request from sync queue: {e}")
                return "Error: LLM request timed out during batch processing."
            else:
                # Event was set, check response holder
                if response_holder[0] is None:
                    # Should ideally not happen if event was set, but check anyway
                    logging.error("Sync batch event was set, but response holder is still None.")
                    return "Error: Unknown error during sync batch processing."
                else:
                    return response_holder[0]
        else:
            # No batching (sync or async)
            return self._generate_single(prompt_or_template, **kwargs)


    def _generate_single(self, prompt_or_template: Union[str, Tuple[str, Dict[str, Any]]], precomputed_prompt_tokens: Optional[int] = None, **kwargs) -> str:
        """
        Internal method to handle a single LLM generation request (Synchronous).
        Applies templates and handles token limits. Uses thread-safe locks for stats/cache.
        Accepts optional precomputed_prompt_tokens from batching logic.
        """
        template_name = None
        if isinstance(prompt_or_template, tuple):
            template_name, context = prompt_or_template
            prompt = self._apply_template(template_name, context)
        else:
            prompt = prompt_or_template # It's already a raw string

        # Handle token limits (warn, truncate) unless tokens were precomputed
        if precomputed_prompt_tokens is None:
            prompt = self._handle_token_limits(prompt, template_name)
            prompt_tokens = getattr(self, '_current_prompt_token_count', 0) # Get count set by _handle_token_limits
        else:
            prompt_tokens = precomputed_prompt_tokens
            # Assume prompt is already handled/truncated by the caller (batch loop)

        # --- Existing Logic ---
        # --- Throttling (Sync) ---
        # Access stats safely using sync lock
        with self._stats_lock:
            current_total_requests = self.total_requests
            current_total_latency = self.total_latency

        avg_latency = current_total_latency / current_total_requests if current_total_requests else 0
        if avg_latency > self.throttle_threshold:
            logging.warning(f"LLMProvider (Sync): High average latency ({avg_latency:.2f}s), throttling for {self.throttle_delay}s")
            time.sleep(self.throttle_delay)

        # --- Cache Check (Sync) ---
        cache_key = prompt # Use the potentially truncated prompt as the key
        with self._cache_lock: # Use sync lock
            cached_response = self.cache.get(cache_key)
            if cached_response is not None:
                self.cache_hits += 1 # Increment cache hits under lock

        if cached_response is not None:
            # Update stats even for cache hit (thread-safe)
            # Note: cache_hits incremented under _cache_lock above
            with self._stats_lock: # Use sync lock
                self.total_requests += 1
                self.total_prompt_tokens += prompt_tokens
                self.single_requests_processed += 1 # Count as single request if cache hit here
                # Don't add response tokens or latency for cache hits
            logging.info(f"Sync Cache hit. Prompt tokens: {prompt_tokens}")
            # Move key to end of order (thread-safe)
            with self._cache_lock: # Use sync lock
                try:
                    self.cache_order.remove(cache_key)
                    self.cache_order.append(cache_key)
                except ValueError: # Should not happen if key was in cache, but safety first
                    logging.warning(f"Cache key '{cache_key[:50]}...' found in cache but not in order list during cache hit.")
            return cached_response
        # --- Cache Miss ---
        with self._cache_lock:
            self.cache_misses += 1
        with self._stats_lock:
            self.single_requests_processed += 1 # Count as single request if cache miss here
        logging.info(f"Sync Cache miss. Prompt tokens: {prompt_tokens}")


        # Override defaults with kwargs
        temperature = kwargs.get("temperature", self.temperature)
        # max_tokens here refers to the *response* max tokens
        max_response_tokens = kwargs.get("max_tokens", self.max_tokens)

        start_time = time.time()
        response = None
        success = False
        response_tokens = 0 # Initialize response token count

        try:
            # Try vLLM first if configured
            if self.provider == "vllm":
                try:
                    response_content, response_tokens = self._call_llm_api(
                        base_url=self.vllm_base_url,
                        prompt=prompt, # Use the processed prompt
                        temperature=temperature,
                        max_tokens=max_response_tokens # Pass max *response* tokens
                    )
                    response = response_content
                    success = True
                except Exception as e:
                    logging.warning(f"vLLM call failed: {e}, falling back to LM Studio")

            if response is None:
                # Fallback to LM Studio
                try:
                    response_content, response_tokens = self._call_llm_api(
                        base_url=self.lmstudio_base_url,
                        prompt=prompt, # Use the processed prompt
                        temperature=temperature,
                        max_tokens=max_response_tokens # Pass max *response* tokens
                    )
                    response = response_content
                    success = True
                except Exception as e:
                    logging.error(f"Sync All LLM calls failed: {e}")
                    response = f"Error generating LLM response: {e}"
                    # Stats updated in _call_llm_api for failures during API call

        finally:
            # --- Update Stats and Cache (Sync) ---
            latency = time.time() - start_time
            with self._stats_lock: # Use sync lock
                self.total_requests += 1
                self.total_latency += latency
                self.recent_latencies.append(latency) # Track recent latency
                self.total_prompt_tokens += prompt_tokens
                if success:
                    self.total_response_tokens += response_tokens
                # failed_requests is incremented within _call_llm_api on exceptions

            logging.info(f"Sync LLM call latency: {latency:.2f}s, prompt tokens: {prompt_tokens}, response tokens: {response_tokens if success else 0}")

            # Save to cache (thread-safe)
            if success:
                with self._cache_lock: # Use sync lock
                    self.cache[cache_key] = response
                    self.cache_order.append(cache_key)
                    if len(self.cache_order) > self.cache_size:
                        oldest = self.cache_order.pop(0)
                        self.cache.pop(oldest, None)

        return response

    def _call_llm_api(self, base_url, prompt, temperature, max_tokens):
        """ Calls the LLM API synchronously and returns (response_content, response_tokens). """
        headers = {"Content-Type": "application/json"}

        # Select model (existing logic)
        chosen_model = None
        try:
            models_response = requests.get(f"{base_url}/models", headers=headers, timeout=5)
            if models_response.status_code == 200:
                models_data = models_response.json()
                available_models = [m["id"] for m in models_data.get("data", [])]
                logging.debug(f"Available models at {base_url}: {available_models}") # Debug level

                for preferred in self.model_preference:
                    for model in available_models:
                        if preferred.lower() in model.lower():
                            chosen_model = model
                            break
                    if chosen_model:
                        break

                if not chosen_model and available_models:
                    chosen_model = available_models[0]

                if chosen_model:
                    logging.info(f"Using model: {chosen_model}")
                else:
                    logging.warning("No models available at LLM endpoint")
                    raise RuntimeError("No models available at LLM endpoint")
            else:
                 # Log full error for debugging
                logging.error(f"Failed to get models from {base_url}: HTTP {models_response.status_code} - {models_response.text[:500]}") # Limit error text length
                raise RuntimeError(f"Failed to get models from {base_url}: HTTP {models_response.status_code}")
        except requests.exceptions.Timeout:
            logging.error(f"Timeout fetching models from {base_url}")
            raise RuntimeError(f"Timeout fetching models from {base_url}")
        except requests.exceptions.RequestException as e:
            logging.error(f"Network error fetching models from {base_url}: {e}")
            raise RuntimeError(f"Network error fetching models from {base_url}: {e}")
        except Exception as e:
            # Catch potential JSON parsing errors etc.
            logging.error(f"Error processing models response from {base_url}: {e}")
            raise RuntimeError(f"Error processing models response from {base_url}: {e}")

        # Prepare chat payload
        data = {
            "messages": [
                {"role": "system", "content": "You are an AI assistant specializing in code generation and reinforcement learning analysis. Provide insightful observations about training reports."},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": max_tokens, # This is max *response* tokens for the API
            "temperature": temperature,
            "model": chosen_model
            # Consider adding stop sequences if needed
        }

        # Use the pre-calculated prompt tokens if available (from _handle_token_limits or batch loop)
        prompt_token_count = getattr(self, '_current_prompt_token_count', self._count_tokens(prompt))
        logging.info(f"Sending sync prompt ({prompt_token_count} tokens) to {base_url} with model {chosen_model}")
        response_tokens = 0 # Initialize
        try:
            response = requests.post(f"{base_url}/chat/completions", headers=headers, data=json.dumps(data), timeout=self.timeout)
            response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)

            response_data = response.json()
            response_content = response_data['choices'][0]['message']['content'].strip()

            # --- Token Counting for Response ---
            # Use the actual response content to count tokens
            response_tokens = self._count_tokens(response_content)

            # Optionally, check API usage fields if available (more accurate)
            usage = response_data.get("usage")
            if usage:
                 api_prompt_tokens = usage.get("prompt_tokens")
                 api_completion_tokens = usage.get("completion_tokens")
                 # Log if our count differs significantly from API's count
                 # Note: Our prompt count might differ due to template processing/truncation
                 # if api_prompt_tokens is not None and abs(self._current_prompt_token_count - api_prompt_tokens) > 10:
                 #      logging.warning(f"Discrepancy in prompt token count: Local={self._current_prompt_token_count}, API={api_prompt_tokens}")
                 if api_completion_tokens is not None:
                     # Prefer API completion tokens if available and significantly different
                     if abs(response_tokens - api_completion_tokens) > 5:
                         logging.warning(f"Sync Discrepancy in response token count: Local={response_tokens}, API={api_completion_tokens}. Using API count.")
                         response_tokens = api_completion_tokens
                     else:
                         # If close, still prefer API count for consistency
                         response_tokens = api_completion_tokens
                 # total_tokens = usage.get("total_tokens") # Could also use this

            return response_content, response_tokens

        except requests.exceptions.Timeout:
            logging.error(f"Sync LLM API call to {base_url} timed out after {self.timeout} seconds.")
            # Increment failed count here for timeout
            with self._stats_lock: # Use sync lock
                self.failed_requests += 1
            raise RuntimeError(f"Sync LLM API call to {base_url} timed out")
        except requests.exceptions.RequestException as e:
            # Handle connection errors, HTTP errors, etc.
            error_message = f"Sync LLM API request error to {base_url}: {e}"
            if e.response is not None:
                error_message += f" - Status: {e.response.status_code} - Body: {e.response.text[:500]}"
            logging.error(error_message)
            with self._stats_lock: # Use sync lock
                self.failed_requests += 1
            raise RuntimeError(error_message)
        except (json.JSONDecodeError, KeyError, IndexError, TypeError, ValueError) as e: # Added TypeError/ValueError
            resp_text = "Error reading response text"
            try:
                resp_text = response.text[:500] if response else "No response object"
            except Exception:
                pass
            logging.error(f"Error parsing sync LLM API response from {base_url}: {e} - Response Text: {resp_text}")
            with self._stats_lock: # Use sync lock
                self.failed_requests += 1
            raise RuntimeError(f"Error parsing sync LLM API response from {base_url}: {e}")

    async def _call_llm_api_async(self, base_url, prompt, temperature, max_tokens):
        """ Calls the LLM API asynchronously using aiohttp and returns (response_content, response_tokens). """
        if not self._aiohttp_session or self._aiohttp_session.closed:
            logging.error("aiohttp session not initialized or closed. Call async_init() first.")
            # Optionally, try to re-initialize? For now, raise error.
            # await self.async_init() # Be careful with re-entrancy
            raise RuntimeError("aiohttp session not initialized or closed.")

        headers = {"Content-Type": "application/json"}
        chosen_model = None

        # --- Select Model (Async) ---
        # TODO: Consider caching model list for a short duration async
        try:
            # Use a shorter timeout specifically for fetching models
            async with self._aiohttp_session.get(f"{base_url}/models", headers=headers, timeout=aiohttp.ClientTimeout(total=10)) as models_response:
                if models_response.status == 200:
                    models_data = await models_response.json()
                    available_models = [m["id"] for m in models_data.get("data", [])]
                    logging.debug(f"Available models at {base_url}: {available_models}")

                    for preferred in self.model_preference:
                        for model in available_models:
                            if preferred.lower() in model.lower():
                                chosen_model = model
                                break
                        if chosen_model:
                            break

                    if not chosen_model and available_models:
                        chosen_model = available_models[0] # Fallback to first available

                    if chosen_model:
                        logging.info(f"Using model: {chosen_model}")
                    else:
                        logging.warning(f"No models available at LLM endpoint {base_url}")
                        raise RuntimeError(f"No models available at LLM endpoint {base_url}")
                else:
                    error_text = await models_response.text()
                    logging.error(f"Failed to get models from {base_url}: HTTP {models_response.status} - {error_text[:500]}")
                    raise RuntimeError(f"Failed to get models from {base_url}: HTTP {models_response.status}")
        except asyncio.TimeoutError:
            logging.error(f"Timeout fetching models from {base_url}")
            raise RuntimeError(f"Timeout fetching models from {base_url}")
        except aiohttp.ClientError as e:
            logging.error(f"Network error fetching models from {base_url}: {e}")
            raise RuntimeError(f"Network error fetching models from {base_url}: {e}")
        except Exception as e:
            # Catch potential JSON parsing errors etc.
            logging.error(f"Error processing models response from {base_url}: {e}")
            raise RuntimeError(f"Error processing models response from {base_url}: {e}")

        # --- Prepare and Send Request (Async) ---
        data = {
            "messages": [
                {"role": "system", "content": "You are an AI assistant specializing in code generation and reinforcement learning analysis. Provide insightful observations about training reports."},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "model": chosen_model
            # Consider adding stop sequences if needed
        }

        # Use the pre-calculated prompt tokens if available (from _handle_token_limits)
        prompt_token_count = getattr(self, '_current_prompt_token_count', self._count_tokens(prompt))
        logging.info(f"Sending async prompt ({prompt_token_count} tokens) to {base_url} with model {chosen_model}")
        response_tokens = 0
        response_content = None
        response = None # Define response in outer scope for error handling

        try:
            # Use the session's configured timeout for the main request
            async with self._aiohttp_session.post(f"{base_url}/chat/completions", headers=headers, json=data) as response:
                response.raise_for_status() # Raise ClientResponseError for bad responses (4xx or 5xx)

                response_data = await response.json()
                # Basic validation of response structure
                if not isinstance(response_data, dict) or 'choices' not in response_data or not response_data['choices']:
                    raise ValueError("Invalid response structure from LLM API")
                if 'message' not in response_data['choices'][0] or 'content' not in response_data['choices'][0]['message']:
                     raise ValueError("Missing 'content' in LLM API response choice")

                response_content = response_data['choices'][0]['message']['content'].strip()

                # --- Token Counting for Response ---
                response_tokens = self._count_tokens(response_content)
                usage = response_data.get("usage")
                if usage:
                    api_completion_tokens = usage.get("completion_tokens")
                    if api_completion_tokens is not None:
                        if abs(response_tokens - api_completion_tokens) > 5:
                            logging.warning(f"Async Discrepancy in response token count: Local={response_tokens}, API={api_completion_tokens}. Using API count.")
                        response_tokens = api_completion_tokens # Prefer API count

                return response_content, response_tokens

        except asyncio.TimeoutError:
            logging.error(f"Async LLM API call to {base_url} timed out after {self.timeout} seconds.")
            # Update stats using async lock
            async with self._async_stats_lock:
                self.failed_requests += 1
            raise RuntimeError(f"Async LLM API call to {base_url} timed out")
        except aiohttp.ClientResponseError as e: # Catch specific aiohttp HTTP errors
            error_message = f"Async LLM API request error to {base_url}: {e.status} {e.message}"
            try:
                error_body = await response.text() if response else "No response object"
                error_message += f" - Body: {error_body[:500]}"
            except Exception:
                pass # Ignore if reading body fails
            logging.error(error_message)
            async with self._async_stats_lock:
                self.failed_requests += 1
            raise RuntimeError(error_message) from e
        except aiohttp.ClientError as e: # Catch other client errors (connection, etc.)
            error_message = f"Async LLM API client error to {base_url}: {e}"
            logging.error(error_message)
            async with self._async_stats_lock:
                self.failed_requests += 1
            raise RuntimeError(error_message) from e
        except (json.JSONDecodeError, KeyError, IndexError, TypeError, ValueError) as e: # Added ValueError
            resp_text = "Error reading response text"
            try:
                resp_text = await response.text() if response else "No response object"
            except Exception:
                pass
            logging.error(f"Error parsing async LLM API response from {base_url}: {e} - Response Text: {resp_text[:500]}")
            async with self._async_stats_lock:
                self.failed_requests += 1
            raise RuntimeError(f"Error parsing async LLM API response from {base_url}: {e}")


    def generate_batch(self, prompts: List[Union[str, Tuple[str, Dict[str, Any]]]], **kwargs) -> List[str]:
        """
        Generate completions for a batch of prompts (Synchronous).

        If async mode is enabled, this logs a warning and calls _generate_single sequentially.
        Otherwise, it calls `generate` for each prompt, which handles sync batching or direct calls.
        """
        if self._async_enabled:
            logging.warning("Calling synchronous generate_batch() while async mode is enabled. Processing sequentially via _generate_single.")

        results = []
        for p_or_t in prompts:
            # Call the single generate method, which handles templates, tokens,
            # sync batching (if enabled/async disabled), async warning, or direct calls.
            results.append(self.generate(p_or_t, **kwargs))
        return results

    def _batch_loop(self):
        """Synchronous batch processing loop using threading."""
        if self._async_enabled:
            logging.warning("Sync batch loop started but async mode is enabled. Loop will exit.")
            return # Don't run sync loop if async is primary

        logging.info("Starting synchronous LLM batch processing thread.")
        while True:
            # Check if async got enabled after thread start (e.g., config reload)
            # This check is basic; a more robust system might involve signals/events.
            if self._async_enabled:
                logging.warning("Async mode detected enabled during sync batch loop execution. Exiting loop.")
                break

            batch_requests = []
            start_time = time.time()

            # Collect requests within the window or until queue is empty
            while (time.time() - start_time) * 1000 < self._batch_window_ms:
                with self._batch_lock:
                    if self._batch_queue:
                        batch_requests.append(self._batch_queue.pop(0))
                    else:
                        # If queue is empty, wait a very short time before checking again or finishing window
                        time.sleep(0.001) # Prevent busy-waiting

                # Exit inner loop if queue was empty or no more items expected in window
                if not self._batch_queue and not batch_requests:
                    break # Finish window early if queue empty
                if not self._batch_queue and batch_requests:
                    # Queue is empty now, but we have items, finish collecting for this window
                    break

            if not batch_requests:
                # No requests in this window, sleep briefly and continue
                # Sleep for roughly one window duration, but check periodically for async enable
                sleep_end_time = time.time() + (self._batch_window_ms / 1000.0)
                while time.time() < sleep_end_time:
                    if self._async_enabled: break # Exit sleep early if async enabled
                    time.sleep(0.01)
                if self._async_enabled: break # Exit outer loop
                continue
            # --- Update Batch Stats ---
            with self._stats_lock:
                self.batches_processed += 1
                self.batched_requests_processed += len(batch_requests)


            logging.info(f"Sync Batch Loop: Processing batch of {len(batch_requests)} requests.")

            # --- Process the batch (Sequentially using _generate_single for now) ---
            # TODO: Implement true sync batch API call here if available.
            for req_data in batch_requests:
                response = "Error: Default error in sync batch processing" # Default error
                try:
                    # Call _generate_single directly, passing precomputed tokens
                    response = self._generate_single(
                        req_data["prompt"],
                        precomputed_prompt_tokens=req_data["prompt_tokens"],
                        **req_data["kwargs"]
                    )
                except Exception as e:
                    logging.error(f"Error processing sync batch item: {e}")
                    response = f"Error: {e}" # Return error string
                finally:
                    # Set response and notify waiting thread
                    if req_data["response_holder"] is not None:
                         req_data["response_holder"][0] = response
                    if req_data["event"] is not None:
                         req_data["event"].set()

            # Brief sleep after processing a batch to yield control
            time.sleep(0.001)

    # --- Asynchronous API ---

    async def _generate_single_async(self, prompt_or_template: Union[str, Tuple[str, Dict[str, Any]]], precomputed_prompt_tokens: Optional[int] = None, **kwargs) -> str:
        """
        Internal method to handle a single LLM generation request asynchronously.
        Applies templates, handles token limits, uses async locks for stats/cache.
        Accepts optional precomputed_prompt_tokens from async batching logic.
        """
        if not self._async_enabled:
            # This shouldn't be called if async is disabled, but safety check
            logging.error("_generate_single_async called when async mode is disabled.")
            return "Error: Async mode is disabled."

        template_name = None
        if isinstance(prompt_or_template, tuple):
            template_name, context = prompt_or_template
            prompt = self._apply_template(template_name, context)
        else:
            prompt = prompt_or_template # Raw string

        # Handle token limits unless precomputed
        if precomputed_prompt_tokens is None:
            prompt = self._handle_token_limits(prompt, template_name)
            prompt_tokens = getattr(self, '_current_prompt_token_count', 0)
        else:
            prompt_tokens = precomputed_prompt_tokens
            # Assume prompt already handled/truncated by caller (async batch loop)

        # --- Async Throttling ---
        async with self._async_stats_lock:
            current_total_requests = self.total_requests
            current_total_latency = self.total_latency

        avg_latency = current_total_latency / current_total_requests if current_total_requests else 0
        if avg_latency > self.throttle_threshold:
            logging.warning(f"LLMProvider (Async): High average latency ({avg_latency:.2f}s), throttling for {self.throttle_delay}s")
            await asyncio.sleep(self.throttle_delay)

        # --- Async Cache Check ---
        cache_key = prompt
        async with self._async_cache_lock:
            cached_response = self.cache.get(cache_key)
            if cached_response is not None:
                self.cache_hits += 1 # Increment cache hits under async lock

        if cached_response is not None:
            # Update stats (async lock)
            # Note: cache_hits incremented under _async_cache_lock above
            async with self._async_stats_lock:
                self.total_requests += 1
                self.total_prompt_tokens += prompt_tokens
                self.single_requests_processed += 1 # Count as single request if cache hit here
            logging.info(f"Async Cache hit. Prompt tokens: {prompt_tokens}")
            # Move key to end (async lock)
            async with self._async_cache_lock:
                try:
                    self.cache_order.remove(cache_key)
                    self.cache_order.append(cache_key)
                except ValueError:
                    pass
            return cached_response
        # --- Cache Miss ---
        async with self._async_cache_lock:
            self.cache_misses += 1
        async with self._async_stats_lock:
            self.single_requests_processed += 1 # Count as single request if cache miss here
        logging.info(f"Async Cache miss. Prompt tokens: {prompt_tokens}")


        # --- Prepare for API Call ---
        temperature = kwargs.get("temperature", self.temperature)
        max_response_tokens = kwargs.get("max_tokens", self.max_tokens)

        start_time = time.time()
        response = None
        success = False
        response_tokens = 0

        try:
            # Try vLLM first if configured
            if self.provider == "vllm":
                try:
                    response_content, response_tokens = await self._call_llm_api_async(
                        base_url=self.vllm_base_url,
                        prompt=prompt,
                        temperature=temperature,
                        max_tokens=max_response_tokens
                    )
                    response = response_content
                    success = True
                except Exception as e:
                    logging.warning(f"Async vLLM call failed: {e}, falling back to LM Studio")

            # Fallback to LM Studio
            if response is None:
                try:
                    response_content, response_tokens = await self._call_llm_api_async(
                        base_url=self.lmstudio_base_url,
                        prompt=prompt,
                        temperature=temperature,
                        max_tokens=max_response_tokens
                    )
                    response = response_content
                    success = True
                except Exception as e:
                    logging.error(f"Async All LLM calls failed: {e}")
                    response = f"Error generating LLM response: {e}"
                    # Stats updated in _call_llm_api_async for failures during API call

        finally:
            # --- Update Stats and Cache (Async) ---
            latency = time.time() - start_time
            async with self._async_stats_lock:
                self.total_requests += 1
                self.total_latency += latency
                self.recent_latencies.append(latency) # Track recent latency
                self.total_prompt_tokens += prompt_tokens
                if success:
                    self.total_response_tokens += response_tokens
                # failed_requests is incremented within _call_llm_api_async on exceptions

            logging.info(f"Async LLM call latency: {latency:.2f}s, prompt tokens: {prompt_tokens}, response tokens: {response_tokens if success else 0}")

            # Save to cache (async lock)
            if success:
                async with self._async_cache_lock:
                    self.cache[cache_key] = response
                    self.cache_order.append(cache_key)
                    if len(self.cache_order) > self.cache_size:
                        oldest = self.cache_order.pop(0)
                        self.cache.pop(oldest, None)

        return response

    async def generate_async(self, prompt_or_template: Union[str, Tuple[str, Dict[str, Any]]], **kwargs) -> str:
        """
        Generates text using the LLM (Asynchronous).

        If async batching is enabled, queues the request. Otherwise, calls _generate_single_async directly.
        Raises RuntimeError if async mode is not enabled.
        """
        if not self._async_enabled:
            raise RuntimeError("Cannot call generate_async when async mode is disabled.")

        # Use async batching only if enabled
        if self._batch_enabled and self._async_queue is not None:
            # Process template before queueing
            template_name = None
            if isinstance(prompt_or_template, tuple):
                template_name, context = prompt_or_template
                prompt_str = self._apply_template(template_name, context)
            else:
                prompt_str = prompt_or_template

            # Handle token limits before queueing
            processed_prompt = self._handle_token_limits(prompt_str, template_name)
            prompt_tokens = getattr(self, '_current_prompt_token_count', 0)

            # Queue the request for the async batch loop
            future = asyncio.get_running_loop().create_future()
            request_data = {
                "prompt": processed_prompt,
                "prompt_tokens": prompt_tokens,
                "kwargs": kwargs,
                "future": future
            }
            await self._async_queue.put(request_data)

            # Wait for the batch loop to set the future's result
            try:
                # Add a timeout slightly longer than the API timeout
                return await asyncio.wait_for(future, timeout=self.timeout + 15)
            except asyncio.TimeoutError:
                logging.error("Async batch request timed out waiting for future result.")
                # Future might still be processed, but we return error
                return "Error: LLM request timed out during async batch processing."
            except Exception as e:
                logging.error(f"Error waiting for async batch future: {e}")
                return f"Error: {e}"
        else:
            # No batching or queue not ready, call single async directly
            return await self._generate_single_async(prompt_or_template, **kwargs)

    async def generate_batch_async(self, prompts: List[Union[str, Tuple[str, Dict[str, Any]]]], **kwargs) -> List[str]:
        """
        Generate completions for a batch of prompts asynchronously using asyncio.gather.

        Processes templates/tokens first, then calls generate_async for each.
        Raises RuntimeError if async mode is not enabled.
        """
        if not self._async_enabled:
            raise RuntimeError("Cannot call generate_batch_async when async mode is disabled.")

        tasks = []
        for p_or_t in prompts:
            # Create a task for each prompt using generate_async
            # generate_async handles batch queueing or direct call internally
            tasks.append(asyncio.create_task(self.generate_async(p_or_t, **kwargs)))

        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results, logging any exceptions
        final_results = []
        for i, res in enumerate(results):
            if isinstance(res, Exception):
                logging.error(f"Error in async batch generation for prompt {i}: {res}")
                final_results.append(f"Error: {res}") # Return error string in place
            else:
                final_results.append(res)

        return final_results

    async def _async_batch_loop(self):
        """Coroutine for processing the async request queue in batches."""
        if not self._async_queue:
            logging.error("Async batch loop started without a queue.")
            return

        logging.info("Starting async LLM batch processing loop.")
        while True:
            batch_requests = []
            start_time = asyncio.get_running_loop().time()
            try:
                # Collect requests within the window or until queue is empty
                while (asyncio.get_running_loop().time() - start_time) * 1000 < self._async_batch_window_ms:
                    try:
                        # --- Track Max Queue Size ---
                        # Check queue size before attempting to get an item
                        current_qsize = self._async_queue.qsize()
                        async with self._async_stats_lock: # Use async lock for stat update
                             if current_qsize > self.async_queue_max_observed_length:
                                 self.async_queue_max_observed_length = current_qsize

                        # Wait for an item with a short timeout to avoid blocking indefinitely
                        # if the queue becomes empty during the window
                        timeout = max(0.001, (self._async_batch_window_ms / 1000.0) - (asyncio.get_running_loop().time() - start_time))
                        req_data = await asyncio.wait_for(self._async_queue.get(), timeout=timeout)
                        batch_requests.append(req_data)
                        self._async_queue.task_done() # Mark task as done immediately after getting
                    except asyncio.TimeoutError:
                        # Timeout waiting for an item, window is likely ending or queue is empty
                        break
                    except Exception as e:
                        logging.error(f"Error getting item from async queue: {e}")
                        # Avoid busy-looping on error
                        await asyncio.sleep(0.01)
                        break # Exit inner loop on unexpected error

                if not batch_requests:
                    # No requests in this window, yield control briefly
                    await asyncio.sleep(0.001) # Small sleep to prevent tight loop when idle
                    continue
                # --- Update Batch Stats ---
                async with self._async_stats_lock:
                    self.batches_processed += 1
                    self.batched_requests_processed += len(batch_requests)

                logging.info(f"Async Batch Loop: Processing batch of {len(batch_requests)} requests.")

                # --- Process the batch (Sequentially using _generate_single_async for now) ---
                # TODO: Implement true async batch API call here if available.
                # This would involve preparing a single batch request and making one API call.
                # For now, we process them one by one using the single async method.
                results = []
                for req_data in batch_requests:
                    # Call _generate_single_async directly, passing precomputed tokens
                    task = asyncio.create_task(
                        self._generate_single_async(
                            req_data["prompt"],
                            precomputed_prompt_tokens=req_data["prompt_tokens"],
                            **req_data["kwargs"]
                        )
                    )
                    results.append((req_data["future"], task)) # Store future and task

                # Wait for all tasks in the current batch to complete
                for future, task in results:
                    if future.cancelled(): # Skip if the original caller timed out
                        if not task.done():
                            task.cancel() # Cancel the LLM task if future was cancelled
                        continue
                    try:
                        result = await task
                        if not future.cancelled(): # Check again before setting result
                            future.set_result(result)
                    except Exception as e:
                        logging.error(f"Error processing async batch item: {e}")
                        if not future.cancelled():
                            future.set_exception(e) # Propagate exception to the caller

            except asyncio.CancelledError:
                logging.info("Async batch loop cancelled.")
                # Set exception for any pending futures in the current batch
                for req_data in batch_requests:
                    if not req_data["future"].done():
                         req_data["future"].set_exception(asyncio.CancelledError("Batch loop cancelled during processing"))
                break # Exit the loop
            except Exception as e:
                logging.exception("Unexpected error in async batch loop.") # Log full traceback
                # Set exception for any pending futures in the current batch
                for req_data in batch_requests:
                     if not req_data["future"].done():
                          req_data["future"].set_exception(e)
                # Avoid crashing the loop, sleep briefly and continue
                await asyncio.sleep(1)


    def get_stats(self):
        """
        Return detailed LLM usage statistics (thread-safe for sync access).
        """
        # Use sync lock to ensure consistent read of stats
        with self._stats_lock:
            # Calculate recent average latency safely
            recent_latency_list = list(self.recent_latencies) # Copy deque under lock
            avg_recent_latency = sum(recent_latency_list) / len(recent_latency_list) if recent_latency_list else 0

            # Create a copy to avoid returning internal state directly
            stats = {
                "total_requests": self.total_requests,
                "failed_requests": self.failed_requests,
                "total_prompt_tokens": self.total_prompt_tokens,
                "total_response_tokens": self.total_response_tokens,
                "single_requests_processed": self.single_requests_processed,
                "batched_requests_processed": self.batched_requests_processed,
                "batches_processed": self.batches_processed,
                "async_queue_max_observed_length": self.async_queue_max_observed_length,
                "recent_latency_count": len(recent_latency_list),
                "total_tokens": self.total_prompt_tokens + self.total_response_tokens,
                "avg_latency_seconds": self.total_latency / self.total_requests if self.total_requests else 0,
                # Note: avg_recent_latency_seconds is added outside the lock after calculation
            }
        # Cache stats can be read outside lock, but length might be slightly off if accessed during update
        # For perfect consistency, cache lock could be used here too, but likely overkill.
        with self._cache_lock:
            stats["cache_size"] = len(self.cache)
            stats["cache_hits"] = self.cache_hits
            stats["cache_misses"] = self.cache_misses
            total_cache_lookups = self.cache_hits + self.cache_misses
            stats["cache_hit_rate"] = self.cache_hits / total_cache_lookups if total_cache_lookups > 0 else 0
            stats["cache_miss_rate"] = self.cache_misses / total_cache_lookups if total_cache_lookups > 0 else 0
        stats["avg_recent_latency_seconds"] = avg_recent_latency # Add calculated recent latency
        # Add config info (doesn't need locking)
        stats.update({
            "cache_capacity": self.cache_size,
            "provider": self.provider,
            "model_preference": self.model_preference,
            "max_prompt_tokens_limit": self.max_prompt_tokens,
            "truncate_enabled": self.truncate_prompt,
            "async_enabled": self._async_enabled,
            "batching_enabled": self._batch_enabled, # Reflects if *any* batching is on
            "sync_batch_window_ms": self._batch_window_ms,
            "async_batch_window_ms": self._async_batch_window_ms
        })
        return stats