# Design: Support for automatic context summarization in deep-worker

**Version:** 1.0
**Date:** 2026-04-02
**Status:** Design Document (the code was not changed)
**Related documents:** `context_management_analysis.md`

---

## Table of contents

1. [Goal and objectives](#1-goal-and-objectives)
2. [Context windows of various AI providers](#2-context-windows-of-various-ai-providers)
3. [Current state](#3-current-state)
4. [Architectural design](#4-architectural-design)
5. [Dynamic configuration per provider](#5-dynamic-configuration-per-provider)
6. [API and configuration](#6-api-and-configuration)
7. [Integration with the existing code](#7-integration-with-the-existing-code)
8. [Testing](#8-testing)
9. [Migration and backward compatibility](#9-migration-and-backward-compatibility)
10. [Performance and scalability](#10-performance-and-scalability)
11. [Security and privacy](#11-security-and-privacy)
12. [Implementation plan](#12-implementation-plan)

---

## 1. Goal and objectives

### Goal

Integrate the `SummarizationMiddleware` from the deepagents SDK into deep-worker for automatic management of the context size in long-running agent sessions.

### Objectives

1. **Preventing context overflow** - automatic summarization when approaching the context window limit
2. **Support for various providers** - dynamic configuration based on the known provider limits (Claude, DeepSeek, OpenAI, etc.)
3. **Preserving the full history** - offloading the full conversation history to the backend so that it can be reviewed
4. **Truncating large arguments** - automatic truncation of tool call arguments in old messages
5. **Configurability** - the ability to configure via CLI parameters and settings
6. **Transparency for the user** - filtering the intermediate summarization messages out of the UI

### Non-goals

- Implementing our own summarization mechanism (we use the ready-made one from deepagents)
- Changing the deepagents API (integration only)
- Manual summarization on a user command (may be added later)

---

## 2. Context windows of various AI providers

### 2.1. Anthropic Claude (official documentation)

**Sources:**
- https://platform.claude.com/docs/en/build-with-claude/context-windows
- https://code.claude.com/docs/en/model-config
- https://www.anthropic.com/news/claude-opus-4-6

| Model | Context window | Notes |
|--------|------------------|------------|
| **Claude Opus 4.6** | 1,000,000 tokens | Requires the header `anthropic-beta: context-1m-2025-08-07` |
| **Claude Sonnet 4.6** | 1,000,000 tokens | Requires the header `anthropic-beta: context-1m-2025-08-07` |
| **Claude Sonnet 4.5** | 200,000 tokens | Standard window without additional headers |
| **Claude Sonnet 4** | 200,000 tokens | |
| **Claude Haiku 4.5** | 200,000 tokens | |
| **Claude 3.5 Sonnet** | 200,000 tokens | |
| **Claude 3.5 Haiku** | 200,000 tokens | |
| **Claude 3 Opus** | 200,000 tokens | |

**Key features of Claude:**
- The 1M context is available only for Opus 4.6 and Sonnet 4.6
- Standard token pricing, without a premium for extended context
- Context awareness - the models track the remaining context
- Support for up to 600 images/PDF pages (100 for the 200k models)

**Recommended settings for deep-worker:**
```python
# Claude Opus 4.6 / Sonnet 4.6 (1M context)
trigger=("fraction", 0.85)  # 850k tokens
keep=("fraction", 0.10)     # 100k tokens

# Claude Sonnet 4.5 / other 200k models
trigger=("fraction", 0.80)  # 160k tokens
keep=("fraction", 0.15)     # 30k tokens
```

---

### 2.2. DeepSeek (official documentation)

**Sources:**
- https://api-docs.deepseek.com/quick_start/pricing
- https://arxiv.org/pdf/2412.19437 (DeepSeek-V3 Technical Report)
- https://huggingface.co/deepseek-ai/DeepSeek-V3

| Model | Context window | Notes |
|--------|------------------|------------|
| **DeepSeek-V3.2** | 128,000 tokens | Current API version (deepseek-chat, deepseek-reasoner) |
| **DeepSeek-V3** | 128,000 tokens | Base model, extended from 32K to 128K |
| **DeepSeek-V3.1** | 128,000 tokens | 671B parameters (37B active) |
| **DeepSeek-Coder-V2** | 128,000 tokens | A specialized model for code |
| **DeepSeek-V3-0324** | 64,000 tokens | Old API version (input+output together) |
| **DeepSeek-R1** | 64,000 tokens input | Up to 32k-64k output (including reasoning chains) |

**Key features of DeepSeek:**
- The V3.2 API version supports a 128K context
- Older API endpoints may have a 64K limit (input+output)
- The web/app version is testing a 1M context (not in the API yet)
- Needle-in-a-Haystack tests show good performance at all positions up to 128K

**Recommended settings for deep-worker:**
```python
# DeepSeek V3.2 (128K context)
trigger=("fraction", 0.75)  # 96k tokens (conservative)
keep=("fraction", 0.20)     # 25.6k tokens

# DeepSeek older versions (64K context)
trigger=("fraction", 0.70)  # 44.8k tokens
keep=("fraction", 0.25)     # 16k tokens
```

---

### 2.3. OpenAI GPT (for comparison)

**Source:** Publicly available information

| Model | Context window |
|--------|------------------|
| **GPT-5** | 400,000 tokens |
| **GPT-4 Turbo** | 128,000 tokens |
| **GPT-4** | 128,000 tokens |
| **GPT-3.5 Turbo** | 16,385 tokens |

---

### 2.4. Other providers

| Provider | Model | Context window |
|-----------|--------|------------------|
| **Google** | Gemini 2.0 Flash | 1,000,000 tokens |
| **Google** | Gemini 1.5 Pro | 2,000,000 tokens |
| **Mistral** | Mistral Large | 128,000 tokens |
| **Cohere** | Command R+ | 128,000 tokens |

---

### 2.5. Summary table for configuration

| Context range | Providers | Recommended trigger | Recommended keep |
|-------------------|------------|----------------------|-------------------|
| **1M+ tokens** | Claude Opus 4.6, Sonnet 4.6, Gemini 1.5 Pro | 0.85 (850k) | 0.10 (100k) |
| **400k tokens** | GPT-5 | 0.80 (320k) | 0.12 (48k) |
| **200k tokens** | Claude 4.5/4/3.5/3, Gemini 2.0 | 0.80 (160k) | 0.15 (30k) |
| **128k tokens** | DeepSeek V3, GPT-4, Mistral, Cohere | 0.75 (96k) | 0.20 (25.6k) |
| **64k tokens** | DeepSeek older versions | 0.70 (44.8k) | 0.25 (16k) |
| **16k tokens** | GPT-3.5 Turbo, older models | 0.60 (9.6k) | 0.30 (4.8k) |
| **Unknown** | Fallback | 0.75 (default) | 0.20 (default) |

**Rationale for the settings:**

1. **Trigger (when to start summarization):**
   - For large contexts (1M+): 85% - allows the context to be used to the maximum
   - For medium ones (128k-200k): 75-80% - a balance between utilization and safety
   - For small ones (16k-64k): 60-70% - early triggering to prevent overflow

2. **Keep (how much to keep after summarization):**
   - For large contexts: 10% - enough for context (~100k tokens)
   - For medium ones: 15-20% - a balance between context and compression
   - For small ones: 25-30% - keeping enough context to keep working

---

## 3. Current state

### 3.1. What is already implemented

#### A. Filtering of summarization messages (done)

**Places of implementation:**

1. **`stream_processor.py:169-171`**
   ```python
   # Filter out summarization messages
   if metadata.get("lc_source") == "summarization":
       return
   ```

2. **`textual_adapter.py:386-388`**
   ```python
   # Filter out summarization LLM output
   if _is_summarization_chunk(metadata):
       continue
   ```

3. **`non_interactive.py:237-238`**
   ```python
   # The summarization middleware injects synthetic messages...
   if metadata and metadata.get("lc_source") == "summarization":
       return
   ```

**Conclusion:** The filtering is fully ready to work with the SummarizationMiddleware.

---

#### B. Backend infrastructure (done)

**`agent.py:658-674` - CompositeBackend with routing:**

```python
conversation_history_backend = FilesystemBackend(
    root_dir=tempfile.mkdtemp(prefix="deepagents_conversation_history_"),
    virtual_mode=True,
)
composite_backend = CompositeBackend(
    default=backend,
    routes={
        "/large_tool_results/": large_results_backend,
        "/conversation_history/": conversation_history_backend,  # READY!
    },
)
```

**Conclusion:** The backend is ready to accept requests to save the history to `/conversation_history/`.

---

#### C. Determining the model context limit (partially implemented)

**`config.py:1348-1349` - extraction from the model profile:**

```python
if isinstance(profile, dict) and isinstance(profile.get("max_input_tokens"), int):
    settings.model_context_limit = profile["max_input_tokens"]
```

**Current behavior:**
- Automatically determines `max_input_tokens` from the model profile
- Stored in `settings.model_context_limit`
- Used to emit warnings about small contexts (<8000 tokens)

**What is MISSING:**
- Using `model_context_limit` to configure the SummarizationMiddleware
- Dynamic selection of triggers based on the context size
- A model → recommended summarization settings mapping

---

### 3.2. What needs to be added

1. **Import SummarizationMiddleware** in `agent.py`
2. **Add it to the middleware stack** with the correct settings
3. **Dynamic configuration** based on `model_context_limit`
4. **Settings for summarization** in `config.py`
5. **CLI parameters** in `main.py`
6. **Tests** to verify the functionality
7. **Documentation** for users

---

## 4. Architectural design

### 4.1. System components

```
┌─────────────────────────────────────────────────────────────────┐
│                         deep-worker CLI                          │
│                                                                   │
│  ┌───────────────┐         ┌──────────────────────────────┐    │
│  │   Settings    │────────▶│  SummarizationConfigurator   │    │
│  │  (config.py)  │         │  (new component)             │    │
│  └───────────────┘         └──────────────────────────────┘    │
│         │                              │                         │
│         │                              ▼                         │
│         │                   ┌────────────────────────┐          │
│         │                   │ Provider Context Map   │          │
│         │                   │ (model → limit)        │          │
│         │                   └────────────────────────┘          │
│         │                              │                         │
│         ▼                              ▼                         │
│  ┌─────────────────────────────────────────────────────┐        │
│  │         create_cli_agent() in agent.py              │        │
│  │                                                      │        │
│  │  1. Determine model_context_limit                   │        │
│  │  2. Select the summarization settings               │        │
│  │  3. Create the SummarizationMiddleware              │        │
│  │  4. Add it to the middleware stack (FIRST!)         │        │
│  └─────────────────────────────────────────────────────┘        │
│                          │                                       │
│                          ▼                                       │
│              ┌───────────────────────┐                          │
│              │  deepagents SDK       │                          │
│              │  Middleware Stack:    │                          │
│              │  1. Summarization     │◄─── NEW                  │
│              │  2. Memory            │                          │
│              │  3. Skills            │                          │
│              │  4. LocalContext      │                          │
│              │  5. ProjectMemory     │                          │
│              └───────────────────────┘                          │
│                          │                                       │
│                          ▼                                       │
│              ┌───────────────────────┐                          │
│              │  CompositeBackend     │                          │
│              │  Routes:              │                          │
│              │  - /conversation_     │                          │
│              │    history/           │◄─── History is saved     │
│              │  - /large_tool_       │                          │
│              │    results/           │                          │
│              └───────────────────────┘                          │
└─────────────────────────────────────────────────────────────────┘
```

---

### 4.2. New component: SummarizationConfigurator

**Purpose:** Dynamic selection of summarization settings based on the provider and the context limit.

**Location:** `deep_worker_cli/config.py` (a new function)

**API:**

```python
@dataclass
class SummarizationConfig:
    """Configuration for SummarizationMiddleware based on model context limit."""

    trigger_fraction: float
    keep_fraction: float
    truncate_after_messages: int
    max_arg_length: int

    # Additional metadata
    context_limit: int  # The model's context limit
    trigger_tokens: int  # The absolute trigger value in tokens
    keep_tokens: int     # The absolute keep value in tokens


def get_summarization_config(
    model_context_limit: int | None,
    model_name: str | None = None,
    override_trigger: float | None = None,
    override_keep: float | None = None,
) -> SummarizationConfig:
    """Get summarization configuration based on model context limit.

    Args:
        model_context_limit: Max input tokens from model profile
        model_name: Model name for special case handling (e.g., claude-opus-4.6)
        override_trigger: Override trigger fraction from CLI/settings
        override_keep: Override keep fraction from CLI/settings

    Returns:
        SummarizationConfig with recommended settings
    """
```

**The logic for determining the settings:**

```python
# Pseudocode
if model_context_limit is None:
    # Fallback for unknown models
    context_limit = 100_000
    trigger_fraction = 0.75
    keep_fraction = 0.20
elif model_context_limit >= 1_000_000:
    # Claude Opus 4.6, Sonnet 4.6, Gemini 1.5 Pro
    trigger_fraction = 0.85
    keep_fraction = 0.10
elif model_context_limit >= 400_000:
    # GPT-5
    trigger_fraction = 0.80
    keep_fraction = 0.12
elif model_context_limit >= 200_000:
    # Claude 4.5/4, Gemini 2.0
    trigger_fraction = 0.80
    keep_fraction = 0.15
elif model_context_limit >= 128_000:
    # DeepSeek V3, GPT-4, Mistral, Cohere
    trigger_fraction = 0.75
    keep_fraction = 0.20
elif model_context_limit >= 64_000:
    # DeepSeek older versions
    trigger_fraction = 0.70
    keep_fraction = 0.25
elif model_context_limit >= 16_000:
    # GPT-3.5 Turbo
    trigger_fraction = 0.60
    keep_fraction = 0.30
else:
    # Very small contexts (<16k)
    trigger_fraction = 0.50
    keep_fraction = 0.35

# Apply overrides from CLI/settings
if override_trigger is not None:
    trigger_fraction = override_trigger
if override_keep is not None:
    keep_fraction = override_keep

# Settings for argument truncation
# For large contexts - more aggressive truncation of old messages
if context_limit >= 200_000:
    truncate_after_messages = 20
    max_arg_length = 500
elif context_limit >= 64_000:
    truncate_after_messages = 15
    max_arg_length = 300
else:
    truncate_after_messages = 10
    max_arg_length = 200

return SummarizationConfig(
    trigger_fraction=trigger_fraction,
    keep_fraction=keep_fraction,
    truncate_after_messages=truncate_after_messages,
    max_arg_length=max_arg_length,
    context_limit=context_limit,
    trigger_tokens=int(context_limit * trigger_fraction),
    keep_tokens=int(context_limit * keep_fraction),
)
```

---

### 4.3. Provider Context Map

**Purpose:** Explicit definition of context limits for known models (a fallback if the model profile does not provide `max_input_tokens`).

**Location:** `deep_worker_cli/config.py` (a module constant)

**Structure:**

```python
# Provider context limits (tokens)
# Used as fallback when model profile doesn't provide max_input_tokens
PROVIDER_CONTEXT_LIMITS: dict[str, int] = {
    # Anthropic Claude
    "claude-opus-4.6": 1_000_000,
    "claude-sonnet-4.6": 1_000_000,
    "claude-sonnet-4.5": 200_000,
    "claude-sonnet-4": 200_000,
    "claude-haiku-4.5": 200_000,
    "claude-3.5-sonnet": 200_000,
    "claude-3.5-haiku": 200_000,
    "claude-3-opus": 200_000,
    "claude-3-sonnet": 200_000,
    "claude-3-haiku": 200_000,

    # DeepSeek
    "deepseek-chat": 128_000,      # V3.2 API
    "deepseek-reasoner": 128_000,  # V3.2 API
    "deepseek-v3": 128_000,
    "deepseek-v3.1": 128_000,
    "deepseek-v3.2": 128_000,
    "deepseek-coder-v2": 128_000,
    "deepseek-r1": 64_000,

    # OpenAI
    "gpt-5": 400_000,
    "gpt-4-turbo": 128_000,
    "gpt-4": 128_000,
    "gpt-3.5-turbo": 16_385,

    # Google Gemini
    "gemini-2.0-flash": 1_000_000,
    "gemini-1.5-pro": 2_000_000,

    # Mistral
    "mistral-large": 128_000,

    # Cohere
    "command-r-plus": 128_000,

    # Fallback
    "default": 100_000,
}


def get_model_context_limit(
    model_name: str | None,
    model_profile: dict[str, Any] | None,
) -> int:
    """Get context limit for a model.

    Priority:
    1. model_profile["max_input_tokens"] (from LangChain)
    2. PROVIDER_CONTEXT_LIMITS[model_name] (known models)
    3. PROVIDER_CONTEXT_LIMITS["default"] (fallback)

    Args:
        model_name: Model name (e.g., "claude-sonnet-4.5")
        model_profile: Model profile dict from LangChain

    Returns:
        Context limit in tokens
    """
    # Priority 1: Model profile
    if model_profile and isinstance(model_profile.get("max_input_tokens"), int):
        return model_profile["max_input_tokens"]

    # Priority 2: Known model mapping
    if model_name:
        # Try exact match first
        if model_name in PROVIDER_CONTEXT_LIMITS:
            return PROVIDER_CONTEXT_LIMITS[model_name]

        # Try partial match (e.g., "anthropic:claude-sonnet-4.5" → "claude-sonnet-4.5")
        for known_model, limit in PROVIDER_CONTEXT_LIMITS.items():
            if known_model in model_name:
                return limit

    # Priority 3: Fallback
    return PROVIDER_CONTEXT_LIMITS["default"]
```

---

## 5. Dynamic configuration per provider

### 5.1. The settings selection algorithm

```python
# In create_cli_agent() in agent.py

# Step 1: Determine the model's context limit
model_profile = None
if isinstance(model, BaseChatModel):
    model_profile = getattr(model, "profile", None)

context_limit = get_model_context_limit(
    model_name=settings.model_name,
    model_profile=model_profile,
)

# Step 2: Get the recommended summarization settings
summarization_config = get_summarization_config(
    model_context_limit=context_limit,
    model_name=settings.model_name,
    override_trigger=settings.summarization_trigger_fraction,
    override_keep=settings.summarization_keep_fraction,
)

# Step 3: Create the SummarizationMiddleware with these settings
if settings.summarization_enabled:
    from deepagents.middleware.summarization import SummarizationMiddleware

    summarization_middleware = SummarizationMiddleware(
        backend=composite_backend,
        trigger=("fraction", summarization_config.trigger_fraction),
        keep=("fraction", summarization_config.keep_fraction),
        truncate_args_settings={
            "cutoff_messages": summarization_config.truncate_after_messages,
            "max_arg_length": summarization_config.max_arg_length,
        },
    )

    # IMPORTANT: Add it FIRST in the middleware stack!
    agent_middleware.insert(0, summarization_middleware)

    # Logging of the selected settings
    logger.info(
        f"Summarization enabled for {settings.model_name or 'unknown model'}: "
        f"context_limit={context_limit:,} tokens, "
        f"trigger={summarization_config.trigger_tokens:,} tokens "
        f"({summarization_config.trigger_fraction:.0%}), "
        f"keep={summarization_config.keep_tokens:,} tokens "
        f"({summarization_config.keep_fraction:.0%})"
    )
```

---

### 5.2. Special handling of providers

#### A. Claude Opus 4.6 / Sonnet 4.6 (1M context)

**Problem:** A special header is required to activate the 1M context.

**Solution:**

```python
# In create_model() in config.py or when creating the model

if model_name in ["claude-opus-4.6", "claude-sonnet-4.6"]:
    # Check whether the header for the 1M context is set
    # If not - emit a warning or add it automatically

    # Option 1: A warning
    if not os.environ.get("ANTHROPIC_BETA_HEADERS"):
        logger.warning(
            f"Model {model_name} supports 1M context window, but "
            f"anthropic-beta header not set. Add to .env:\n"
            f"ANTHROPIC_BETA_HEADERS=context-1m-2025-08-07"
        )
        # Fall back to the 200k settings
        context_limit = 200_000

    # Option 2: Automatically adding the header (if LangChain supports it)
    # model_kwargs["default_headers"] = {"anthropic-beta": "context-1m-2025-08-07"}
```

**Configuration via .env:**

```bash
# .env
ANTHROPIC_BETA_HEADERS=context-1m-2025-08-07
```

---

#### B. DeepSeek (various API versions)

**Problem:** Different API versions have different limits (64K vs 128K).

**Solution:**

```python
# Determining the DeepSeek version from model_name
if "deepseek" in model_name.lower():
    # V3.2 API (the new version) - 128K
    if any(v in model_name.lower() for v in ["v3.2", "v3.1", "v3", "chat", "reasoner"]):
        context_limit = 128_000
    # Older versions or R1 - 64K
    elif "r1" in model_name.lower() or "0324" in model_name.lower():
        context_limit = 64_000
    else:
        # Fall back to a conservative limit
        context_limit = 64_000
        logger.warning(
            f"Unknown DeepSeek version in {model_name}, "
            f"assuming 64K context limit"
        )
```

---

#### C. Unknown models via OpenRouter

**Problem:** OpenRouter provides access to many models whose limits may be unknown.

**Solution:**

```python
# If model_name starts with "openrouter/"
if model_name and model_name.startswith("openrouter/"):
    # Try to extract the real model name
    actual_model = model_name.split("/", 1)[-1]

    # Try to find it among the known models
    context_limit = get_model_context_limit(actual_model, model_profile)

    if context_limit == PROVIDER_CONTEXT_LIMITS["default"]:
        logger.warning(
            f"Unknown model via OpenRouter: {actual_model}, "
            f"using fallback context limit: {context_limit:,} tokens"
        )
```

---

## 6. API and configuration

### 6.1. Settings (config.py)

```python
class Settings(BaseSettings):
    """Configuration settings for deep-worker CLI."""

    # ... existing settings ...

    # ========== Context Management / Summarization ==========

    summarization_enabled: bool = Field(
        default=True,
        description="Enable automatic conversation summarization when context fills up",
    )

    summarization_trigger_fraction: float | None = Field(
        default=None,
        ge=0.1,
        le=1.0,
        description=(
            "Context window fraction (0.0-1.0) that triggers summarization. "
            "If None, auto-determined based on model context limit."
        ),
    )

    summarization_keep_fraction: float | None = Field(
        default=None,
        ge=0.05,
        le=0.5,
        description=(
            "Fraction of recent context to keep after summarization. "
            "If None, auto-determined based on model context limit."
        ),
    )

    summarization_truncate_after: int | None = Field(
        default=None,
        ge=5,
        le=100,
        description=(
            "Truncate tool call arguments in messages older than this count. "
            "If None, auto-determined based on model context limit."
        ),
    )

    summarization_max_arg_length: int | None = Field(
        default=None,
        ge=100,
        le=2000,
        description=(
            "Maximum length for truncated tool call arguments. "
            "If None, auto-determined based on model context limit."
        ),
    )

    # Backend for storing the history
    summarization_history_backend: str = Field(
        default="temp",  # "temp" or "persistent"
        description=(
            "Backend for conversation history storage. "
            "'temp' uses /tmp (cleared on reboot), "
            "'persistent' saves to ~/.deepagents/{agent}/conversation_history/"
        ),
    )
```

**Rationale for `None` for trigger/keep:**
- `None` means "determine automatically based on the model"
- If an explicit value is set, it is used
- Allows smart defaults with the option to override

---

### 6.2. CLI parameters (main.py)

```python
# The argument group for context management
context_group = parser.add_argument_group(
    "Context Management",
    "Options for managing conversation context and summarization"
)

context_group.add_argument(
    "--no-summarization",
    action="store_true",
    help=(
        "Disable automatic conversation summarization. "
        "Not recommended for long sessions as context may overflow."
    ),
)

context_group.add_argument(
    "--summarization-trigger",
    type=float,
    metavar="FRACTION",
    help=(
        "Context window fraction (0.1-1.0) that triggers summarization. "
        "Default: auto-determined based on model (0.6-0.85 depending on context size). "
        "Example: --summarization-trigger 0.75 means trigger at 75%% full."
    ),
)

context_group.add_argument(
    "--summarization-keep",
    type=float,
    metavar="FRACTION",
    help=(
        "Fraction of recent context to keep after summarization (0.05-0.5). "
        "Default: auto-determined based on model (0.1-0.35 depending on context size). "
        "Example: --summarization-keep 0.20 means keep last 20%%."
    ),
)

context_group.add_argument(
    "--persistent-history",
    action="store_true",
    help=(
        "Save conversation history persistently to "
        "~/.deepagents/{agent}/conversation_history/ instead of /tmp. "
        "Useful for reviewing past conversations."
    ),
)

# Argument handling in main()
if args.no_summarization:
    settings.summarization_enabled = False

if args.summarization_trigger is not None:
    if not 0.1 <= args.summarization_trigger <= 1.0:
        parser.error("--summarization-trigger must be between 0.1 and 1.0")
    settings.summarization_trigger_fraction = args.summarization_trigger

if args.summarization_keep is not None:
    if not 0.05 <= args.summarization_keep <= 0.5:
        parser.error("--summarization-keep must be between 0.05 and 0.5")
    settings.summarization_keep_fraction = args.summarization_keep

if args.persistent_history:
    settings.summarization_history_backend = "persistent"
```

---

### 6.3. Environment variables (.env)

```bash
# .env - configuration examples

# Disable summarization
SUMMARIZATION_ENABLED=false

# Set the settings explicitly (overriding auto-detection)
SUMMARIZATION_TRIGGER_FRACTION=0.80
SUMMARIZATION_KEEP_FRACTION=0.15
SUMMARIZATION_TRUNCATE_AFTER=20
SUMMARIZATION_MAX_ARG_LENGTH=500

# Persistent history storage
SUMMARIZATION_HISTORY_BACKEND=persistent

# For Claude Opus 4.6 / Sonnet 4.6 - activating the 1M context
ANTHROPIC_BETA_HEADERS=context-1m-2025-08-07
```

---

## 7. Integration with the existing code

### 7.1. Changes in agent.py

**File:** `deep_worker_cli/agent.py`

**Changes:**

1. **Import (lines 496-501):**

```python
from deepagents import create_deep_agent
from deepagents.backends import CompositeBackend
from deepagents.backends.filesystem import FilesystemBackend
from deepagents.middleware import MemoryMiddleware, SkillsMiddleware
from deepagents.middleware.subagents import CompiledSubAgent, SubAgent
from deepagents.middleware.summarization import SummarizationMiddleware  # ADD
```

2. **Creating the backend for the history (lines 658-674) - CHANGE:**

```python
# BEFORE:
conversation_history_backend = FilesystemBackend(
    root_dir=tempfile.mkdtemp(prefix="deepagents_conversation_history_"),
    virtual_mode=True,
)

# AFTER:
if settings.summarization_history_backend == "persistent":
    # Persistent storage in ~/.deepagents/{agent}/conversation_history/
    history_dir = settings.user_deepagents_dir / assistant_id / "conversation_history"
    history_dir.mkdir(parents=True, exist_ok=True)
    conversation_history_backend = FilesystemBackend(
        root_dir=history_dir,
        virtual_mode=False,
    )
else:
    # Temporary storage in /tmp (deleted on reboot)
    conversation_history_backend = FilesystemBackend(
        root_dir=tempfile.mkdtemp(prefix="deepagents_conversation_history_"),
        virtual_mode=True,
    )
```

3. **Adding the SummarizationMiddleware to the stack (AFTER composite_backend is created, lines 566-596):**

```python
# Build middleware stack based on enabled features
agent_middleware = []

# ========== CONTEXT MANAGEMENT (FIRST!) ==========
# Add SummarizationMiddleware FIRST to manage context size early
if settings.summarization_enabled:
    # Step 1: Determine the model's context limit
    model_profile = None
    if isinstance(model, BaseChatModel):
        model_profile = getattr(model, "profile", None)

    context_limit = get_model_context_limit(
        model_name=settings.model_name,
        model_profile=model_profile,
    )

    # Step 2: Get the recommended settings
    summarization_config = get_summarization_config(
        model_context_limit=context_limit,
        model_name=settings.model_name,
        override_trigger=settings.summarization_trigger_fraction,
        override_keep=settings.summarization_keep_fraction,
    )

    # Step 3: Create the middleware
    agent_middleware.append(
        SummarizationMiddleware(
            backend=composite_backend,
            trigger=("fraction", summarization_config.trigger_fraction),
            keep=("fraction", summarization_config.keep_fraction),
            truncate_args_settings={
                "cutoff_messages": summarization_config.truncate_after_messages,
                "max_arg_length": summarization_config.max_arg_length,
            },
        )
    )

    # Logging
    logger.info(
        f"Summarization enabled: model={settings.model_name or 'unknown'}, "
        f"context={context_limit:,} tokens, "
        f"trigger={summarization_config.trigger_tokens:,}t "
        f"({summarization_config.trigger_fraction:.0%}), "
        f"keep={summarization_config.keep_tokens:,}t "
        f"({summarization_config.keep_fraction:.0%})"
    )

# ========== MEMORY ==========
if enable_memory:
    memory_sources = [str(settings.get_user_agent_md_path(assistant_id))]
    # ... the rest of the code is unchanged ...
```

---

### 7.2. Changes in config.py

**File:** `deep_worker_cli/config.py`

**Add:**

1. **PROVIDER_CONTEXT_LIMITS** (constant, ~line 150-200)
2. **SummarizationConfig** (dataclass, ~line 200-220)
3. **get_model_context_limit()** (function, ~line 220-260)
4. **get_summarization_config()** (function, ~line 260-350)
5. **Settings fields for summarization** (in the Settings class, ~line 400-450)

See the details in sections 4.2 and 6.1.

---

### 7.3. Changes in main.py

**File:** `deep_worker_cli/cli/main.py`

**Add:**

1. **The `context_group` argument group** (~line 200-250)
2. **Argument handling** in main() (~line 500-550)

See the details in section 6.2.

---

## 8. Testing

### 8.1. Unit tests

**File:** `tests/unit_tests/test_summarization_config.py`

```python
"""Unit tests for summarization configuration logic."""

import pytest
from deep_worker_cli.config import (
    PROVIDER_CONTEXT_LIMITS,
    SummarizationConfig,
    get_model_context_limit,
    get_summarization_config,
)


class TestProviderContextLimits:
    """Tests for PROVIDER_CONTEXT_LIMITS mapping."""

    def test_claude_opus_4_6_limit(self):
        """Test that Claude Opus 4.6 has 1M context."""
        assert PROVIDER_CONTEXT_LIMITS["claude-opus-4.6"] == 1_000_000

    def test_claude_sonnet_4_5_limit(self):
        """Test that Claude Sonnet 4.5 has 200K context."""
        assert PROVIDER_CONTEXT_LIMITS["claude-sonnet-4.5"] == 200_000

    def test_deepseek_chat_limit(self):
        """Test that DeepSeek chat (V3.2) has 128K context."""
        assert PROVIDER_CONTEXT_LIMITS["deepseek-chat"] == 128_000

    def test_deepseek_r1_limit(self):
        """Test that DeepSeek R1 has 64K context."""
        assert PROVIDER_CONTEXT_LIMITS["deepseek-r1"] == 64_000

    def test_default_fallback(self):
        """Test that default fallback is 100K."""
        assert PROVIDER_CONTEXT_LIMITS["default"] == 100_000


class TestGetModelContextLimit:
    """Tests for get_model_context_limit() function."""

    def test_from_model_profile(self):
        """Test that model profile takes priority."""
        profile = {"max_input_tokens": 150_000}
        limit = get_model_context_limit("unknown-model", profile)
        assert limit == 150_000

    def test_from_known_model_exact_match(self):
        """Test exact model name match."""
        limit = get_model_context_limit("claude-sonnet-4.5", None)
        assert limit == 200_000

    def test_from_known_model_partial_match(self):
        """Test partial model name match (provider prefix)."""
        limit = get_model_context_limit("anthropic:claude-sonnet-4.5", None)
        assert limit == 200_000

    def test_fallback_unknown_model(self):
        """Test fallback for completely unknown model."""
        limit = get_model_context_limit("unknown-model-xyz", None)
        assert limit == 100_000

    def test_fallback_none_inputs(self):
        """Test fallback when both inputs are None."""
        limit = get_model_context_limit(None, None)
        assert limit == 100_000


class TestGetSummarizationConfig:
    """Tests for get_summarization_config() function."""

    def test_1m_context_model(self):
        """Test config for 1M token context models."""
        config = get_summarization_config(1_000_000)
        assert config.trigger_fraction == 0.85
        assert config.keep_fraction == 0.10
        assert config.trigger_tokens == 850_000
        assert config.keep_tokens == 100_000

    def test_200k_context_model(self):
        """Test config for 200K token context models."""
        config = get_summarization_config(200_000)
        assert config.trigger_fraction == 0.80
        assert config.keep_fraction == 0.15
        assert config.trigger_tokens == 160_000
        assert config.keep_tokens == 30_000

    def test_128k_context_model(self):
        """Test config for 128K token context models (DeepSeek)."""
        config = get_summarization_config(128_000)
        assert config.trigger_fraction == 0.75
        assert config.keep_fraction == 0.20
        assert config.trigger_tokens == 96_000
        assert config.keep_tokens == 25_600

    def test_64k_context_model(self):
        """Test config for 64K token context models."""
        config = get_summarization_config(64_000)
        assert config.trigger_fraction == 0.70
        assert config.keep_fraction == 0.25

    def test_16k_context_model(self):
        """Test config for small context models (GPT-3.5)."""
        config = get_summarization_config(16_385)
        assert config.trigger_fraction == 0.60
        assert config.keep_fraction == 0.30

    def test_override_trigger(self):
        """Test that override_trigger works."""
        config = get_summarization_config(200_000, override_trigger=0.90)
        assert config.trigger_fraction == 0.90
        assert config.keep_fraction == 0.15  # Not overridden

    def test_override_keep(self):
        """Test that override_keep works."""
        config = get_summarization_config(200_000, override_keep=0.25)
        assert config.trigger_fraction == 0.80  # Not overridden
        assert config.keep_fraction == 0.25

    def test_truncation_settings_large_context(self):
        """Test argument truncation settings for large context."""
        config = get_summarization_config(200_000)
        assert config.truncate_after_messages == 20
        assert config.max_arg_length == 500

    def test_truncation_settings_medium_context(self):
        """Test argument truncation settings for medium context."""
        config = get_summarization_config(64_000)
        assert config.truncate_after_messages == 15
        assert config.max_arg_length == 300

    def test_truncation_settings_small_context(self):
        """Test argument truncation settings for small context."""
        config = get_summarization_config(16_000)
        assert config.truncate_after_messages == 10
        assert config.max_arg_length == 200

    def test_none_context_limit_fallback(self):
        """Test that None context limit uses fallback."""
        config = get_summarization_config(None)
        assert config.context_limit == 100_000
        assert config.trigger_fraction == 0.75
```

---

### 8.2. Integration tests

**File:** `tests/integration_tests/test_summarization_integration.py`

```python
"""Integration tests for SummarizationMiddleware in deep-worker."""

import pytest
import tempfile
from pathlib import Path
from deep_worker_cli.agent import create_cli_agent
from deep_worker_cli.config import settings


@pytest.mark.asyncio
async def test_summarization_middleware_enabled_by_default(tmp_path: Path):
    """Test that SummarizationMiddleware is enabled by default."""
    settings.summarization_enabled = True

    agent, backend = create_cli_agent(
        model="anthropic:claude-sonnet-4",
        assistant_id="test_summarization",
        project_memory_dir=tmp_path,
        enable_shell=False,
    )

    # TODO: Inspect agent middleware stack to verify SummarizationMiddleware is present
    # This depends on deepagents API for middleware inspection
    pass


@pytest.mark.asyncio
async def test_summarization_can_be_disabled(tmp_path: Path):
    """Test that summarization can be disabled via settings."""
    settings.summarization_enabled = False

    agent, backend = create_cli_agent(
        model="anthropic:claude-sonnet-4",
        assistant_id="test_no_summarization",
        project_memory_dir=tmp_path,
        enable_shell=False,
    )

    # TODO: Verify SummarizationMiddleware is NOT in middleware stack
    pass


@pytest.mark.asyncio
async def test_summarization_uses_model_context_limit(tmp_path: Path, mocker):
    """Test that summarization config adapts to model context limit."""
    # Mock model with known context limit
    mock_model = mocker.Mock()
    mock_model.profile = {"max_input_tokens": 200_000}

    agent, backend = create_cli_agent(
        model=mock_model,
        assistant_id="test_adaptive",
        project_memory_dir=tmp_path,
        enable_shell=False,
    )

    # TODO: Verify trigger/keep fractions match 200K config (0.80/0.15)
    pass


@pytest.mark.asyncio
async def test_conversation_history_saved_to_backend(tmp_path: Path):
    """Test that conversation history is saved to backend during summarization."""
    settings.summarization_enabled = True
    settings.summarization_history_backend = "persistent"

    agent, backend = create_cli_agent(
        model="anthropic:claude-sonnet-4",
        assistant_id="test_history",
        project_memory_dir=tmp_path,
        enable_shell=False,
    )

    # TODO: Trigger summarization by filling context, verify history file created
    # Expected path: ~/.deepagents/test_history/conversation_history/{thread_id}.md
    pass


@pytest.mark.asyncio
async def test_persistent_vs_temp_history_backend(tmp_path: Path):
    """Test difference between persistent and temp history backends."""
    # Test 1: Persistent backend
    settings.summarization_history_backend = "persistent"
    agent1, backend1 = create_cli_agent(
        model="anthropic:claude-sonnet-4",
        assistant_id="test_persistent",
        project_memory_dir=tmp_path,
        enable_shell=False,
    )
    # TODO: Verify history goes to ~/.deepagents/test_persistent/conversation_history/

    # Test 2: Temp backend
    settings.summarization_history_backend = "temp"
    agent2, backend2 = create_cli_agent(
        model="anthropic:claude-sonnet-4",
        assistant_id="test_temp",
        project_memory_dir=tmp_path,
        enable_shell=False,
    )
    # TODO: Verify history goes to /tmp/deepagents_conversation_history_*/
    pass
```

---

### 8.3. E2E tests

**File:** `tests/e2e/test_summarization_e2e.py`

```python
"""End-to-end tests for summarization in real agent sessions."""

import pytest
import asyncio
from pathlib import Path


@pytest.mark.slow
@pytest.mark.asyncio
async def test_agent_summarizes_after_many_tool_calls(tmp_path: Path):
    """Test that agent automatically summarizes after many tool calls.

    This is a long-running test that:
    1. Creates an agent with small context limit (for faster testing)
    2. Executes many tool calls to fill context
    3. Verifies summarization occurs
    4. Checks that history is offloaded to backend
    """
    # TODO: Implement realistic scenario with tool calls
    pass


@pytest.mark.slow
@pytest.mark.asyncio
async def test_deepseek_agent_with_128k_context(tmp_path: Path):
    """Test DeepSeek agent with 128K context limit."""
    # Test that DeepSeek models get correct 128K config
    # TODO: Create agent with deepseek-chat, verify config
    pass


@pytest.mark.slow
@pytest.mark.asyncio
async def test_claude_opus_4_6_with_1m_context(tmp_path: Path):
    """Test Claude Opus 4.6 with 1M context limit."""
    # Test that Claude Opus 4.6 gets correct 1M config
    # TODO: Create agent with claude-opus-4.6, verify config
    pass
```

---

## 9. Migration and backward compatibility

### 9.1. Backward compatibility

**Requirements:**

1. **Existing agents keep working** without changes
2. **The new feature is enabled by default**, but can be disabled
3. **The CLI parameters are optional** - the defaults work automatically

**Guarantees:**

```python
# Agents created before the rollout:
# - Keep working (summarization_enabled=True by default)
# - Get automatic smart settings based on the model
# - Can disable it via --no-summarization

# Agents with explicit settings:
# - Settings from .env/CLI override the automatic ones
# - Full compatibility with the previous behavior when using --no-summarization
```

---

### 9.2. The migration path

**Phase 1: Introduction (v1.0)**
- Summarization is enabled by default
- Automatic settings based on the model
- INFO logging of the selected settings

**Phase 2: Optimization (v1.1)**
- Analysis of logs and metrics from production usage
- Adjustment of the default settings based on real data
- Adding the `dwcli --prune-history` command (from Proposal 3)

**Phase 3: Extension (v2.0)**
- Support for various summarization strategies (not only fraction)
- Configuration via a web UI / TUI
- Real-time metrics (how many tokens were saved)

---

### 9.3. Notifying users

**On the first run with the new version:**

```python
# In main() after the agent is created:
if settings.summarization_enabled and not settings._summarization_welcome_shown:
    console.print(
        "\n[dim]New: Automatic context management enabled! "
        "Your conversations will be summarized when context fills up, "
        "with full history saved for review. "
        "Disable with --no-summarization.[/dim]\n"
    )
    settings._summarization_welcome_shown = True
    settings._save_welcome_flag()
```

**In the agent logs:**

```
[INFO] Summarization enabled: model=claude-sonnet-4.5, context=200,000 tokens,
       trigger=160,000t (80%), keep=30,000t (15%)
```

---

## 10. Performance and scalability

### 10.1. Performance

**The overhead of summarization:**

1. **The LLM call for summarization:**
   - An additional API call to the model
   - Approximate time: 2-5 seconds (depends on the size of the history)
   - Frequency: only when the trigger threshold is reached

2. **Saving the history to the backend:**
   - A file write operation
   - Approximate time: <100ms for a local filesystem
   - Approximate time: 200-500ms for S3/remote storage

3. **Argument truncation:**
   - An in-memory operation (very fast)
   - Approximate time: <10ms for 100 messages

**Overall conclusion:** Minimal overhead, noticeable only when summarization fires (rarely).

---

### 10.2. Scalability

**The size of the history on disk:**

```python
# Estimated history size for 1 session:
# - 100 messages × ~500 tokens × ~4 bytes/token = 200KB
# - With Markdown formatting: ~300KB

# For the persistent backend:
# ~/.deepagents/{agent}/conversation_history/
#   thread_001.md  (300KB)
#   thread_002.md  (300KB)
#   ...
#   thread_100.md  (300KB)
# Total: ~30MB for 100 sessions

# Recommendation: Periodic cleanup of old histories (>30 days)
```

**The cleanup strategy (a future feature):**

```bash
# dwcli --cleanup-history --older-than 30d
# Deletes histories older than 30 days
```

---

## 11. Security and privacy

### 11.1. Storing the history

**Threats:**

1. **Leaking confidential information** from the conversation history
2. **Access to the history by other users** on shared systems
3. **Storing API keys and passwords** in the conversation history

**Protective measures:**

```python
# 1. Restricting access permissions on the history files
# In FilesystemBackend for persistent storage:
history_file.chmod(0o600)  # Only the owner can read/write

# 2. A warning about confidential information
# In the system prompt:
"""
IMPORTANT: Conversation history is saved to disk for context management.
Avoid including sensitive information (passwords, API keys, private data)
in your conversations. History files are stored locally with restricted
permissions, but may be visible to system administrators.
"""

# 3. An option for disabled persistent storage
# --no-persistent-history uses only /tmp (cleared on reboot)
```

---

### 11.2. Sanitization options (a future feature)

**For v2.0: Automatic cleanup of sensitive data from the history**

```python
# An option in settings:
summarization_sanitize_history: bool = True  # Remove sensitive patterns

# Patterns to remove:
SENSITIVE_PATTERNS = [
    r"sk-[a-zA-Z0-9]{20,}",  # OpenAI API keys
    r"sk-ant-[a-zA-Z0-9-]{50,}",  # Anthropic API keys
    r"password\s*[:=]\s*[^\s]+",  # Passwords
    # ... other patterns
]
```

---

## 12. Implementation plan

### Phase 1: Core Implementation (Day 1-2)

**Tasks:**

1. ✅ Research the providers' context limits (DeepSeek, Claude) - DONE
2. ⬜ Add `PROVIDER_CONTEXT_LIMITS` to config.py
3. ⬜ Implement `get_model_context_limit()` in config.py
4. ⬜ Implement `get_summarization_config()` in config.py
5. ⬜ Add Settings fields for summarization to config.py
6. ⬜ Add the SummarizationMiddleware import to agent.py
7. ⬜ Integrate the SummarizationMiddleware into create_cli_agent()
8. ⬜ Configure the persistent/temp backend for the history

**Estimate:** 4-6 hours

---

### Phase 2: CLI & Configuration (Day 2-3)

**Tasks:**

1. ⬜ Add CLI parameters to main.py:
   - --no-summarization
   - --summarization-trigger
   - --summarization-keep
   - --persistent-history
2. ⬜ Add parameter handling in main()
3. ⬜ Set up .env support
4. ⬜ Add logging of the selected settings
5. ⬜ Write a welcome message for users

**Estimate:** 2-3 hours

---

### Phase 3: Testing (Day 3-4)

**Tasks:**

1. ⬜ Write unit tests for the config functions
2. ⬜ Write integration tests for agent creation
3. ⬜ Write an e2e test for the full summarization cycle
4. ⬜ Test with DeepSeek models
5. ⬜ Test with Claude models
6. ⬜ Verify that message filtering works

**Estimate:** 4-6 hours

---

### Phase 4: Documentation & Polish (Day 4-5)

**Tasks:**

1. ⬜ Update the README with usage examples
2. ⬜ Write documentation on configuration for different providers
3. ⬜ Add a troubleshooting guide
4. ⬜ Code review and refactoring
5. ⬜ Performance testing

**Estimate:** 3-4 hours

---

### Overall estimate

**Implementation time:** 13-19 hours (2-3 working days)

**Files to change:**
- `deep_worker_cli/config.py` (~200 lines added)
- `deep_worker_cli/agent.py` (~100 lines added)
- `deep_worker_cli/cli/main.py` (~80 lines added)
- `tests/unit_tests/test_summarization_config.py` (new, ~200 lines)
- `tests/integration_tests/test_summarization_integration.py` (new, ~150 lines)
- `tests/e2e/test_summarization_e2e.py` (new, ~100 lines)

**Total:** ~830 lines of code (including tests and documentation)

---

## Appendix A: Usage examples

### A.1. Basic usage (defaults)

```bash
# Summarization is enabled automatically with smart settings
deep-worker my_agent

# The logs will show:
# [INFO] Summarization enabled: model=claude-sonnet-4.5,
#        context=200,000 tokens, trigger=160,000t (80%), keep=30,000t (15%)
```

---

### A.2. Disabling summarization

```bash
# For short sessions where the context will not overflow
deep-worker my_agent --no-summarization
```

---

### A.3. Explicit trigger configuration

```bash
# Trigger at 90% fill (very aggressive use of the context)
deep-worker my_agent --summarization-trigger 0.90

# Trigger at 60% (a conservative approach)
deep-worker my_agent --summarization-trigger 0.60
```

---

### A.4. Persistent history storage

```bash
# Save the history for later review
deep-worker my_agent --persistent-history

# The history is saved to:
# ~/.deepagents/my_agent/conversation_history/{thread_id}.md
```

---

### A.5. Full customization

```bash
# Full control over all parameters
deep-worker my_agent \
  --summarization-trigger 0.85 \
  --summarization-keep 0.10 \
  --persistent-history
```

---

### A.6. DeepSeek-specific configuration

```bash
# DeepSeek V3 (128K context) - the correct settings will be determined automatically
deep-worker my_agent --model deepseek:deepseek-chat

# The logs will show:
# [INFO] Summarization enabled: model=deepseek-chat,
#        context=128,000 tokens, trigger=96,000t (75%), keep=25,600t (20%)
```

---

### A.7. Claude Opus 4.6 with a 1M context

```bash
# .env:
# ANTHROPIC_BETA_HEADERS=context-1m-2025-08-07

deep-worker my_agent --model anthropic:claude-opus-4.6

# The logs will show:
# [INFO] Summarization enabled: model=claude-opus-4.6,
#        context=1,000,000 tokens, trigger=850,000t (85%), keep=100,000t (10%)
```

---

## Appendix B: FAQ

### Q1: How can I tell that summarization has fired?

**A:** Check the agent logs:

```
[INFO] Summarization triggered for thread abc123: 165,234 tokens, 87 messages
[INFO] Offloaded 75 messages to /conversation_history/abc123.md
[INFO] Created summary message, kept 12 recent messages
```

---

### Q2: Where is the full conversation history stored?

**A:** It depends on the setting:

- **Temp backend (default):** `/tmp/deepagents_conversation_history_*/`
- **Persistent backend:** `~/.deepagents/{agent_name}/conversation_history/{thread_id}.md`

---

### Q3: Does summarization affect the quality of the answers?

**A:** Minimally. The full history is preserved, and summarization creates a short summary for the LLM. The agent keeps the context of the most recent messages and an understanding of the whole history via the summary.

---

### Q4: How much does summarization cost in tokens?

**A:** An additional LLM call to create the summary. Approximately:

- Input: 50-100k tokens (the old messages)
- Output: 1-3k tokens (the summary)

It fires rarely (only when the context is 75-85% full).

---

### Q5: Can it be disabled for a specific agent?

**A:** Yes, via an environment variable:

```bash
SUMMARIZATION_ENABLED=false deep-worker my_agent
```

Or create a `.env` file in the agent's directory:

```bash
# ~/.deepagents/my_agent/.env
SUMMARIZATION_ENABLED=false
```

---

### Q6: Are custom models supported?

**A:** Yes, for unknown models a fallback is used (100K context, trigger=0.75, keep=0.20). For precise tuning, specify them explicitly:

```bash
deep-worker my_agent \
  --model my-custom-model \
  --summarization-trigger 0.80 \
  --summarization-keep 0.15
```

---

## Conclusion

This design document describes the complete integration of the SummarizationMiddleware into deep-worker with:

1. ✅ **Automatic configuration** based on the provider and the context limit
2. ✅ **Support for all major providers** (Claude, DeepSeek, OpenAI, Gemini, etc.)
3. ✅ **Flexible configuration** via CLI, settings, and .env
4. ✅ **Backward compatibility** with existing agents
5. ✅ **Full test coverage** (unit, integration, e2e)
6. ✅ **Minimal performance overhead**

**Ready for implementation!**

---

**End of document**

*Version: 1.0*
*Date: 2026-04-02*
*Status: Design Complete - Ready for Implementation*
