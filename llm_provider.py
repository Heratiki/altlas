import requests
import json
import logging

import time

class LLMProvider:
    def __init__(self, config: dict = None):
        self.config = config or {}
        self.provider = self.config.get("provider", "vllm")
        self.vllm_base_url = self.config.get("vllm_base_url", "http://localhost:8000/v1")
        self.lmstudio_base_url = self.config.get("lmstudio_base_url", "http://localhost:1234/v1")
        self.model_preference = self.config.get("model_preference", ["wizardcoder", "codellama", "code-llama", "stable-code", "starcoder", "llama"])
        self.timeout = int(self.config.get("timeout", 120))
        self.temperature = float(self.config.get("temperature", 0.7))
        self.max_tokens = int(self.config.get("max_tokens", 500))
        # Simple LRU cache for prompt-response pairs
        self.cache = {}
        self.cache_order = []
        self.cache_size = int(self.config.get("cache_size", 100))
        # Performance stats
        self.total_requests = 0
        self.total_tokens = 0
        self.total_latency = 0.0
        self.failed_requests = 0
        # Adaptive throttling
        self.throttle_threshold = float(self.config.get("throttle_threshold", 5.0))  # seconds
        self.throttle_delay = float(self.config.get("throttle_delay", 2.0))  # seconds
        import threading
        self._batch_enabled = str(self.config.get("enable_batching", "false")).lower() == "true"
        self._batch_window_ms = int(self.config.get("batch_window_ms", 20))
        self._batch_queue = []
        self._batch_lock = threading.Lock()
        if self._batch_enabled:
            self._batch_thread = threading.Thread(target=self._batch_loop, daemon=True)
            self._batch_thread.start()
    def _batch_loop(self):
        pass

    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate text from the LLM provider (vLLM preferred, fallback to LM Studio).
        """
        # Adaptive throttling based on average latency
        avg_latency = self.total_latency / self.total_requests if self.total_requests else 0
        if avg_latency > self.throttle_threshold:
            logging.warning(f"LLMProvider: High average latency ({avg_latency:.2f}s), throttling for {self.throttle_delay}s before request")
            time.sleep(self.throttle_delay)

        # Check cache first
        if prompt in self.cache:
            return self.cache[prompt]

        # Override defaults with kwargs
        temperature = kwargs.get("temperature", self.temperature)
        max_tokens = kwargs.get("max_tokens", self.max_tokens)

        start_time = time.time()
        response = None
        success = False

        try:
            # Try vLLM first if configured
            if self.provider == "vllm":
                try:
                    response = self._call_llm_api(
                        base_url=self.vllm_base_url,
                        prompt=prompt,
                        temperature=temperature,
                        max_tokens=max_tokens
                    )
                    success = True
                except Exception as e:
                    logging.warning(f"vLLM call failed: {e}, falling back to LM Studio")

            if response is None:
                # Fallback to LM Studio
                try:
                    response = self._call_llm_api(
                        base_url=self.lmstudio_base_url,
                        prompt=prompt,
                        temperature=temperature,
                        max_tokens=max_tokens
                    )
                    success = True
                except Exception as e:
                    logging.error(f"All LLM calls failed: {e}")
                    response = f"Error generating LLM response: {e}"
                    self.failed_requests += 1

        finally:
            latency = time.time() - start_time
            self.total_requests += 1
            self.total_latency += latency
            prompt_tokens = len(prompt.split())
            response_tokens = len(response.split()) if isinstance(response, str) else 0
            self.total_tokens += prompt_tokens + response_tokens
            logging.info(f"LLM call latency: {latency:.2f}s, prompt tokens: {prompt_tokens}, response tokens: {response_tokens}")

        # Save to cache
        self.cache[prompt] = response
        self.cache_order.append(prompt)
        if len(self.cache_order) > self.cache_size:
            oldest = self.cache_order.pop(0)
            self.cache.pop(oldest, None)

        return response

    def _call_llm_api(self, base_url, prompt, temperature, max_tokens):
        headers = {"Content-Type": "application/json"}

        # Select model
        chosen_model = None
        try:
            models_response = requests.get(f"{base_url}/models", headers=headers, timeout=5)
            if models_response.status_code == 200:
                models_data = models_response.json()
                available_models = [m["id"] for m in models_data.get("data", [])]
                logging.info(f"Available models at {base_url}: {available_models}")

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
                raise RuntimeError(f"Failed to get models: HTTP {models_response.status_code}")
        except Exception as e:
            logging.warning(f"Error fetching models from {base_url}: {e}")
            raise

        # Prepare chat payload
        data = {
            "messages": [
                {"role": "system", "content": "You are an AI assistant specializing in code generation and reinforcement learning analysis. Provide insightful observations about training reports."},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "model": chosen_model
        }

        logging.info(f"Sending prompt to {base_url} with model {chosen_model}")
        response = requests.post(f"{base_url}/chat/completions", headers=headers, data=json.dumps(data), timeout=self.timeout)
        if response.status_code == 200:
            response_data = response.json()
            return response_data['choices'][0]['message']['content'].strip()
        else:
            raise RuntimeError(f"LLM API error: HTTP {response.status_code} - {response.text}")

    def generate_batch(self, prompts: list, **kwargs) -> list:
        """
        Generate completions for a batch of prompts.
        Currently sequential; future: optimize with real batch API calls.
        """
        return [self.generate(p, **kwargs) for p in prompts]

    def get_stats(self):
        """
        Return basic LLM usage statistics.
        """
        avg_latency = self.total_latency / self.total_requests if self.total_requests else 0
        return {
            "total_requests": self.total_requests,
            "failed_requests": self.failed_requests,
            "total_tokens": self.total_tokens,
            "avg_latency": avg_latency
        }