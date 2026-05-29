# Design: DeepSeek Cache Optimisation & Default Model Update

## 1. Executive Summary

**Goal:** Maximise DeepSeek API cache-hit rate (to bring input cost from $0.14/Mtok → $0.0028/Mtok) and update the default model from the legacy `deepseek-chat` to `deepseek-v4-flash`.

### Key Findings

| Area | Current State | What Must Change |
|---|---|---|
| **Default model** | `"deepseek-chat"` (hardcoded in `_PROVIDER_DEFAULT_MODELS`) | `"deepseek-v4-flash"` |
| **Context limits** | DeepSeek models listed at 128K, but V4 Flash/Pro have 1M | Add V4 family entries |
| **`user_id` (cache key)** | **Not set anywhere** — `ChatOpenAI(...)` for DeepSeek has no `user` parameter | Add `user=<agent_name>` / `user=<agent_name>:tm` |
| **`thread_id`** | Correct: agent = `self.name`, TM = `f"{self.name}:tm"` | No change needed |

---

## 2. Background — Why This Matters

### 2.1 The Pricing Asymmetry

From the DeepSeek API pricing page (May 2026):

| Scenario | Price / 1M input tokens | 
|---|---|
| Cache **hit** | **$0.0028** (∼0.02 ₽) |
| Cache **miss** | $0.14 (∼1 ₽) |

A 50× cost difference. Since agents send large system prompts + tool schemas that seldom change between turns, a well-tuned cache strategy saves the dominant share of every invocation's input cost.

### 2.2 DeepSeek Cache Mechanics

DeepSeek uses **prefix-based KV-cache**: the first N tokens of the `messages` array are cached. The cache key is determined by **`user_id`** — different `user_id` values produce isolated caches.

- **Same `user_id` + same prefix → cache hit** (pay $0.0028/Mtok)
- **Different `user_id` → different cache** (even with identical prefixes)
- **Dynamic content at the start → cache miss** (even with matching `user_id`)

### 2.3 The Multi-Agent Problem

When multiple agents share one API key **without `user_id`**, their system prompts overlap in a single shared cache. This means:

- Agent A's messages evict Agent B's cached prefix
- Both agents constantly miss the cache
- No benefit from repeated identical system prompts within the same agent

**Solution:** Use **per-agent `user_id`** to give each agent its own isolated cache.

### 2.4 The `:tm` Sub-Thread Problem

Each agent may have a **TaskManager** (`-tm` flag). The TM runs as a separate LangGraph thread with `thread_id = f"{agent}:tm"`. If the TM uses the same `user_id` as the agent, their caches collide — a summarization in the main thread evicts the TM's cached prefix, and vice versa.

**Solution:** Give the TM its own `user_id = f"{agent}:tm"`.

---

## 3. Required Changes

### 3.1 Default Model: `"deepseek-chat"` → `"deepseek-v4-flash"`

**File:** `deep_worker_cli/config.py`

```python
# In _PROVIDER_DEFAULT_MODELS (line ~697):
_PROVIDER_DEFAULT_MODELS: dict[str, tuple[str | None, str | None]] = {
    "deepseek": ("DEEPSEEK_MODEL", "deepseek-v4-flash"),  # ← changed
    # ... rest unchanged
}
```

**Rationale:** `deepseek-chat` is a legacy alias that will stop working on 24 July 2026. The `deepseek-v4-flash` is the recommended replacement — faster, cheaper (cache hit: $0.0028 vs $0.07), same 1M context.

**Migration path for existing users:** Setting `DEEPSEEK_MODEL=deepseek-chat` in `.env` will override the default, so existing configurations continue working.

### 3.2 Context Limits for V4 Models

**File:** `deep_worker_cli/config.py` — `PROVIDER_CONTEXT_LIMITS` (line ~406)

Add entries:

```python
# DeepSeek V4 family
"deepseek-v4-flash": 1_000_000,
"deepseek-v4-pro": 1_000_000,
```

This enables optimal summarization config (trigger at 85%, keep 10%) for the 1M context window.

### 3.3 Add `user_id` to DeepSeek Model Creation

There are **two levels** at which we can inject the `user` parameter. The recommended approach is **Level A** because it keeps everything explicit.

#### Level A (Recommended): Add `user` parameter to `create_model()`

**File:** `deep_worker_cli/config.py` — `create_model()` function (line ~1409)

Add a `user: str | None = None` parameter:

```python
def create_model(
    model_name_override: str | None = None,
    *,
    provider: str | None = None,
    model_name: str | None = None,
    user: str | None = None,         # ← NEW: DeepSeek cache isolation key
) -> BaseChatModel:
```

Then, when constructing `ChatOpenAI` for `provider == "deepseek"` (line ~1589), pass `model_kwargs`:

```python
# Build an extra-body dict. Only include keys that are non-None.
extra: dict[str, Any] = {}
if user is not None:
    extra["user"] = user

model = ChatOpenAI(
    model=model_name,
    base_url="https://api.deepseek.com",
    api_key=settings.deepseek_api_key,
    model_kwargs=extra,
)
```

For `provider == "openai"`, we **don't** need to set `user` — OpenAI doesn't use `user` for cache isolation in the same way.

#### How Callers Pass `user`

**In `agent_process.py` (line ~626):**
```python
model = create_model(
    provider=self.config.provider,
    model_name=self.config.model,
    user=self.name,  # ← NEW
)
```

**In `task_manager.py` (line ~219):**
```python
model = create_model(
    provider=self.config.provider,
    model_name=self.config.model,
    user=f"{self.agent_name}:tm",  # ← NEW
)
```

**In `main.py` (line ~405) and `non_interactive.py` (line ~555):**
These use auto-detection and don't have an agent name. Passing `user=None` is fine — no cache isolation needed for one-off CLI invocations. Could also derive user from assistant_id if desired.

#### API Payload Effect

When `create_model()` is called with `user="my_agent"`, the resulting HTTP request to DeepSeek API will include:

```json
{
  "model": "deepseek-v4-flash",
  "messages": [...],
  "user": "my_agent"       // ← This is the cache isolation key
}
```

For TaskManager: `"user": "my_agent:tm"`.

#### Level B (Alternative): `bind()` After Construction

If we don't want to touch `create_model()`'s signature, we could use `.bind()`:

```python
model = create_model(provider=..., model_name=...)
model = model.bind(user="my_agent")
```

**Not recommended** because `.bind()` wraps the model and loses type information. It's also invisible in the code — harder to debug.

### 3.4 Summary of All Changes

| # | File | Change | Type |
|---|---|---|---|
| 1 | `config.py` line ~697 | `"deepseek-chat"` → `"deepseek-v4-flash"` | Model default |
| 2 | `config.py` line ~423 | Add `"deepseek-v4-flash"`, `"deepseek-v4-pro"` entries with 1M limit | Context limit |
| 3 | `config.py` line ~1409 | Add `user: str \| None = None` param to `create_model()` | Function signature |
| 4 | `config.py` line ~1589 | Add `model_kwargs={"user": user}` when constructing `ChatOpenAI` for deepseek | Model creation |
| 5 | `agent_process.py` line ~626 | Pass `user=self.name` to `create_model()` | Call site |
| 6 | `task_manager.py` line ~219 | Pass `user=f"{self.agent_name}:tm"` to `create_model()` | Call site |
| 7 | `main.py` line ~405 | Pass `user=None` (or derive from thread_id) | Call site |
| 8 | `non_interactive.py` line ~555 | Pass `user=None` (or derive from assistant_id) | Call site |

---

## 4. Verification

### 4.1 Cache Hit Rate Monitoring

After the change, every API response will include `usage.prompt_cache_hit_tokens` and `usage.prompt_cache_miss_tokens`. We can monitor this by:

1. **Logging at `ChatOpenAI` level** — add a callback that logs cache stats per-turn
2. **Logging in `dwcli` TUI** — display cache hit ratio in the token/pricing footer

Example expected payload from DeepSeek:

```json
{
  "usage": {
    "prompt_tokens": 1500,
    "prompt_cache_hit_tokens": 1200,   // 80% hit rate
    "prompt_cache_miss_tokens": 300,
    "completion_tokens": 500,
    "total_tokens": 2000
  }
}
```

### 4.2 Test Plan

| Test | Scenario | Expected |
|---|---|---|
| `test_default_model_is_v4_flash` | No env override | `get_default_model_name("deepseek")` → `"deepseek-v4-flash"` |
| `test_env_override_still_works` | `DEEPSEEK_MODEL=deepseek-chat` | `get_default_model_name("deepseek")` → `"deepseek-chat"` |
| `test_v4_context_limit` | `get_model_context_limit("deepseek-v4-flash", None)` | `1_000_000` |
| `test_create_model_with_user` | `create_model(provider="deepseek", ..., user="my_agent")` | `ChatOpenAI` created with `model_kwargs={"user": "my_agent"}` |
| `test_create_model_without_user` | `create_model(provider="deepseek", ...)` | `ChatOpenAI` created **without** `user` in model_kwargs |
| `test_agent_creates_model_with_user` | Agent `AgentProcess(name="foo")` calls `create_model()` | `user="foo"` |
| `test_tm_creates_model_with_user` | TaskManager `self.agent_name="foo"` calls `create_model()` | `user="foo:tm"` |

---

## 5. Appendix

### 5.1 Why `model_kwargs`?

`ChatOpenAI` (from `langchain-openai`) is a Pydantic model with a fixed field set. The `user` parameter is **not** a declared field. However, `_default_params` property (which builds the API payload) includes `**self.model_kwargs`, which are `dict[str, Any]`. This means:

```python
ChatOpenAI(..., model_kwargs={"user": "my_agent"})
```

produces `{"model": "deepseek-v4-flash", ..., "user": "my_agent"}` in the API call. Verified experimentally.

### 5.2 Why Not `user` as a Direct Parameter?

`ChatOpenAI` accepts `**kwargs` but only stores known fields. The `user` parameter would be silently dropped if passed as a top-level keyword. `model_kwargs` is the documented escape hatch for arbitrary API parameters.

### 5.3 No Changes Needed for `openai` Provider

OpenAI's API also accepts `user`, but OpenAI doesn't have the same 50× cache hit/miss price asymmetry. DeepSeek's pricing is unique in how aggressively it discounts cache hits. For the `openai` provider, `user` is irrelevant for cost optimisation.

### 5.4 Why `user` Not `extra_body`

DeepSeek API expects `user` as a top-level parameter in the request body, **not** inside `extra_body`. The `model_kwargs` approach injects it at the correct level.

```python
# ✅ Correct: user goes to top-level
model_kwargs={"user": "my_agent"}       → {"model": "...", "user": "my_agent", ...}

# ❌ Wrong: extra_body would add a nested object
extra_body={"user": "my_agent"}          → {"model": "...", "extra_body": {"user": "my_agent"}, ...}
```

DeepSeek API docs explicitly state `user` as a top-level field in the chat completions request, alongside `model`, `messages`, `temperature`, etc.
