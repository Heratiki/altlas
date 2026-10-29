# T-003: llama.cpp provider for the LLM roles

Status: open
Opened: 2026-10-29 by the orchestrator, from maintainer ruling R4 after the project's 18-month idle.
Depends on: none

## What happens today

- `llm_provider.py:22-24` reads `provider` (default `vllm`), `vllm_base_url` (default `http://localhost:8000/v1`) and `lmstudio_base_url` (`http://localhost:1234/v1`). `generate` paths at `llm_provider.py:402-430` (sync) and `877-905` (async) try vLLM when `provider == "vllm"`, and on failure (or any other provider value) fall through to LM Studio. A value such as `llamacpp` would therefore be sent to port 1234.
- `_call_llm_api` (`llm_provider.py:457+`, async twin at `575+`) GETs `{base_url}/models`, picks the first id containing a `model_preference` entry (substring, case-insensitive), else the first listed id, then posts the completion. Errors raise RuntimeError; `generate` catches them and returns the string `Error generating LLM response: ...` (`llm_provider.py:428-431`), so a down server does not crash.
- `core/model_router.py` builds one `LLMProvider` per role from `default_config` plus role overrides; it does not branch on provider name (`describe_role` only reports it).
- `config.ini:66-72`: `Provider = lmstudio`, `vLLMBaseURL = http://localhost:8000/v1` (port 8000 belongs to another project on this host), `ModelPreference = wizardcoder,codellama,...`. Roles at `config.ini:77-100` prefer `gemma,deepseek-coder,...` (advisor, analyst) and `gpt,claude,gemini,gemma` (planner, critic, Tier = hosted).
- Config keys reach the provider only through `core/config_loader.py:116-180` (`get_llm_config`, `get_model_router_config`), which maps `vLLMBaseURL`/`LMStudioBaseURL` and hardcodes the 8000 default at lines 124, 138, 170. A new `LlamaCppBaseURL` key is dropped unless that file maps it.
- Host: a llama.cpp `llama-server` router (Vulkan build) serves an OpenAI-compatible API at `http://localhost:8080/v1`, one model loaded at a time (`--models-max 1`). Models listed: Qwen3-8B-Q4_K_M, Qwen3-4B-Instruct-2507-Q4_K_M, gemma-4-12B-it-qat-UD-Q4_K_XL, VibeThinker-3B-Agentic-Q5_K_M.

## Rulings (2026-10-29)

- R4: LLM roles use the llama.cpp `llama-server` through its OpenAI-compatible API; add it as a provider. Reason: it already runs on the host and uses the Vulkan GPU path. Rejected: ROCm torch for LLMs; hosted providers for planner/critic.
- Default provider becomes `llamacpp`; nothing defaults to port 8000.

## Settled design

- `llm_provider.py`: add `self.llamacpp_base_url = config.get("llamacpp_base_url", "http://localhost:8080/v1")`; change the `provider` default to `llamacpp`; change the `vllm_base_url` default away from port 8000 (the lane picks and states a replacement, e.g. require explicit config; vLLM stays usable when set). In both the sync and async `generate` paths, `provider == "llamacpp"` calls only `llamacpp_base_url`; on failure it logs an error and returns the same `Error generating LLM response: ...` string. No silent fall-through to LM Studio. `vllm` and `lmstudio` behave as today.
- Model selection keeps the substring preference but must accept the router's `/v1/models` ids as returned (e.g. `Qwen3-8B-Q4_K_M`); the `model` sent in the completion is that exact id. Because the router loads one model at a time, a role naming a different model triggers a load on the server; the lane documents this in a config.ini comment.
- `core/model_router.py`: no branching change expected; the lane verifies `describe_role` reports `llamacpp` and `_build_role_config` carries `llamacpp_base_url` to each role. Edit only if a defect is found.
- `core/config_loader.py` (added to Owns: required, see Confirm below): map `LlamaCppBaseURL` -> `llamacpp_base_url` in `get_llm_config` and `get_model_router_config`, both the success and the except-fallback dict; default provider `llamacpp`; replace the 8000 defaults.
- `config.ini` `[LLM]`: `Provider = llamacpp`, `LlamaCppBaseURL = http://localhost:8080/v1`, remove or repoint `vLLMBaseURL` off port 8000, `ModelPreference = Qwen3-8B-Q4_K_M,...` (a coder-capable default; stays a config value). Roles: set each `ModelPreference` to listed models (advisor/analyst e.g. `Qwen3-4B-Instruct-2507,Qwen3-8B`; planner/critic e.g. `Qwen3-8B,gemma-4-12B`), and `Tier = local` for all, since no hosted provider is configured. The lane records its choices in the commit message.

## Confirm before implementing

- `core/config_loader.py` is outside the Owns list in the brief but must change for `LlamaCppBaseURL` to take effect. This ticket owns it; the orchestrator confirms no other ticket touches it (T-001 and T-002 do not).
- The replacement for the `vllm_base_url` 8000 default (none vs an arbitrary port) is not ruled; the lane states its choice.
- Which role gets which model is not ruled beyond "a coder-capable default such as Qwen3-8B-Q4_K_M".

## Owns

- `llm_provider.py`
- `core/model_router.py`
- `core/config_loader.py`
- `config.ini`

## Out of scope

- Starting or configuring `llama-server`; installing models.
- Device selection (T-002), packaging (T-001).
- Changing prompt templates, caching, batching or token logic.

## Acceptance

- [ ] `grep -n "8000" llm_provider.py core/model_router.py core/config_loader.py config.ini` prints nothing.
- [ ] `grep -n "LlamaCppBaseURL\|llamacpp" config.ini core/config_loader.py llm_provider.py` shows the key, mapping and provider branch; `Provider = llamacpp` in `[LLM]`.
- [ ] With the server up (`curl -s localhost:8080/v1/models` lists the models): a script using `ConfigLoader().get_model_router_config()` and `ModelRouter` calls `generate("advisor", "Reply with the single word OK")`; the result does not start with `Error generating` and the log shows `Using model:` with an id from `/v1/models`. Also list the ids via `requests.get("http://localhost:8080/v1/models")` and show the chosen id is among them.
- [ ] `ModelRouter.describe_role("planner")["provider"] == "llamacpp"`.
- [ ] With the server down (point `LlamaCppBaseURL` at an unused port in a temp config copy, do not edit the real one): `import runner` and constructing `ModelRouter` + `LLMProvider` raise nothing, and `generate` returns a string starting `Error generating LLM response` (matches existing failure handling at `llm_provider.py:428-431`). No request reaches port 1234 or 8000.
- [ ] Existing providers: `Provider = lmstudio` and `vllm` code paths are unchanged (diff review).
- [ ] Ticket Status set to `landed (<branch>, YYYY-MM-DD)` in the lane's commit.
- [ ] Gates pass: `python -m py_compile runner.py altlas_ui.py benchmark_runner.py core/*.py agent_core/*.py evaluator/*.py guardian/*.py reinforcer/*.py task/*.py memory/*.py` and `node scripts/check-tickets.mjs`.

Do NOT run the test harness. Stage paths explicitly.
