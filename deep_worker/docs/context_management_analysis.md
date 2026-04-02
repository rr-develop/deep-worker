# Analysis: Solving the overly long context problem in deepagents and the agent

**Date:** 2026-04-02
**Status:** Analysis complete, the code was not changed

---

## Table of contents

1. [How deepagents solves the long context problem](#1-how-deepagents-solves-the-long-context-problem)
2. [Usage in the agent](#2-usage-in-the-agent)
3. [Proposed solutions](#3-proposed-solutions-for-the-long-context-problem-in-the-agent)
4. [Recommendations for choosing a solution](#4-recommendations-for-choosing-a-solution)
5. [Concrete integration steps](#5-concrete-integration-steps-proposal-1)
6. [Conclusion](#conclusion)

---

## 1. How deepagents solves the long context problem

deepagents uses the **SummarizationMiddleware** (`deepagents/middleware/summarization.py`) - a specialized middleware for automatic context management through summarization and history persistence.

### Key capabilities of the SummarizationMiddleware

#### A. Summarization triggers (configurable)

```python
# Trigger at 85% of the context window being filled
trigger=("fraction", 0.85)

# Trigger when 170k tokens are reached
trigger=("tokens", 170000)

# Trigger at 50 messages in the history
trigger=("messages", 50)
```

#### B. The context retention policy (the `keep` parameter)

```python
# Keep the last 10% of the context
keep=("fraction", 0.10)

# Keep the last 20 messages
keep=("messages", 20)

# Keep 10k tokens worth of context
keep=("tokens", 10000)
```

#### C. Two-phase processing

**Phase 1: Offloading (flushing to the backend)**
- The full history is saved to `/conversation_history/{thread_id}.md`
- The file contains all the information without any loss
- A backend is used for storage (FilesystemBackend, S3, etc.)

**Phase 2: Summarization**
- The LLM produces a short summary of the old messages
- The summary is inserted as a HumanMessage with a special field:
  ```python
  message.additional_kwargs["lc_source"] = "summarization"
  ```
- A reference to the full history is added to the summary
- The old messages are replaced with the summary message

#### D. Truncation (truncating arguments)

A separate feature for truncating large arguments in old tool calls:

```python
truncate_args_settings={
    "cutoff_messages": 20,  # Truncate arguments older than 20 messages
    "max_arg_length": 500,  # Max argument length: 500 characters
}
```

It applies to `write_file`, `edit_file` and other tools with large arguments.

### An example of usage in deepagents

```python
from deepagents.middleware.summarization import SummarizationMiddleware
from deepagents import create_deep_agent

# Creating the middleware with settings
summarization = SummarizationMiddleware(
    backend=backend,
    trigger=("fraction", 0.85),  # Trigger at 85% fill
    keep=("fraction", 0.10),     # Keep 10% of the context
    truncate_args_settings={
        "cutoff_messages": 20,
        "max_arg_length": 500,
    },
)

# Adding it to the middleware stack
agent = create_deep_agent(
    model=model,
    backend=backend,
    middleware=[summarization, ...],
    checkpointer=checkpointer,
)
```

### The mechanism for filtering summarization messages

Summarization messages are marked with `metadata["lc_source"] = "summarization"` and are filtered out in the stream handlers, so that the intermediate LLM calls that create the summary are not shown to the user.

**The filtering code in this project:**

**stream_processor.py:169-171:**
```python
# Filter out summarization messages
if metadata.get("lc_source") == "summarization":
    return
```

**textual_adapter.py:386-388:**
```python
# Filter out summarization LLM output
if _is_summarization_chunk(metadata):
    continue
```

**non_interactive.py:237-238:**
```python
# The summarization middleware injects synthetic messages...
if metadata and metadata.get("lc_source") == "summarization":
    return
```

---

## 2. Usage in the agent

### CONCLUSION: the agent does NOT use the SummarizationMiddleware YET!

The search showed:
- ✅ Filtering of summarization messages is **ALREADY implemented** (in 3 places)
- ❌ The SummarizationMiddleware is **NOT imported** in `agent.py`
- ❌ The SummarizationMiddleware is **NOT added** to the middleware stack

### The current middleware stack in `agent.py:566-596`

```python
agent_middleware = []

# 1. MemoryMiddleware (if enable_memory)
agent_middleware.append(MemoryMiddleware(...))

# 2. SkillsMiddleware (if enable_skills)
agent_middleware.append(SkillsMiddleware(...))

# 3. LocalContextMiddleware (only in local mode)
agent_middleware.append(LocalContextMiddleware())

# 4. ProjectMemoryMiddleware (semantic search)
agent_middleware.append(ProjectMemoryMiddleware(...))

# ❌ SummarizationMiddleware IS MISSING!
```

### Why is the filtering already there?

The code is prepared for using the SummarizationMiddleware - the filtering was added in advance so that connecting the middleware would not require changing the stream handling.

### The backend infrastructure is already in place!

In `agent.py:658-674` a `CompositeBackend` with routing is already configured:

```python
conversation_history_backend = FilesystemBackend(
    root_dir=tempfile.mkdtemp(prefix="deepagents_conversation_history_"),
    virtual_mode=True,
)
composite_backend = CompositeBackend(
    default=backend,
    routes={
        "/large_tool_results/": large_results_backend,
        "/conversation_history/": conversation_history_backend,  # ALREADY IN PLACE!
    },
)
```

**This means:**
- ✅ The infrastructure for storing the history is **ALREADY IN PLACE**
- ✅ The SummarizationMiddleware will be able to use `/conversation_history/` without changes
- ✅ The history will be saved to `/tmp/deepagents_conversation_history_*/`

---

## 3. Proposed solutions for the long context problem in deep-worker

### Proposal 1: Integrating the SummarizationMiddleware (RECOMMENDED)

#### Advantages
- ✅ A ready-made solution from the deepagents SDK
- ✅ The filtering is already implemented in deep-worker
- ✅ The full history is preserved without any loss
- ✅ Configurable triggers and policies
- ✅ Works with any backend (filesystem, S3, sandbox)
- ✅ Truncation of tool call arguments is already built in

#### Drawbacks
- ⚠️ Requires additional LLM calls for summarization
- ⚠️ A small delay when the trigger fires

#### Implementation

**Step 1: Add the import in `agent.py:496-501`**

```python
from deepagents import create_deep_agent
from deepagents.backends import CompositeBackend
from deepagents.backends.filesystem import FilesystemBackend
from deepagents.middleware import MemoryMiddleware, SkillsMiddleware
from deepagents.middleware.subagents import CompiledSubAgent, SubAgent
from deepagents.middleware.summarization import SummarizationMiddleware  # ADD
```

**Step 2: Add it to the middleware stack in `agent.py:566-596`**

```python
agent_middleware = []

# ADD FIRST for early context truncation
if True:  # settings.enable_summarization can be added
    # Settings for summarization
    # Trigger at 80% of the context being filled (conservative)
    # Keep the last 15% of the context (the last ~15 messages)
    agent_middleware.append(
        SummarizationMiddleware(
            backend=composite_backend,
            trigger=("fraction", 0.80),
            keep=("fraction", 0.15),
            truncate_args_settings={
                "cutoff_messages": 15,  # Truncate arguments older than 15 messages
                "max_arg_length": 300,  # Max 300 characters for old arguments
            },
        )
    )

# The remaining middleware...
if enable_memory:
    agent_middleware.append(MemoryMiddleware(...))
```

**Step 3: Add settings in `config.py`**

```python
class Settings(BaseSettings):
    # ... existing settings ...

    # Summarization settings
    summarization_enabled: bool = True
    summarization_trigger_fraction: float = 0.80
    summarization_keep_fraction: float = 0.15
    summarization_truncate_after: int = 15
    summarization_max_arg_length: int = 300
```

**Step 4: Add CLI parameters in `main.py`**

```python
parser.add_argument(
    "--no-summarization",
    action="store_true",
    help="Disable automatic conversation summarization",
)

parser.add_argument(
    "--summarization-trigger",
    type=float,
    default=0.80,
    help="Context fraction to trigger summarization (default: 0.80)",
)
```

---

### Proposal 2: Argument Truncation Only (a lightweight solution)

If full summarization is not needed, only the truncation of tool call arguments can be implemented.

#### Advantages
- ✅ Does not require LLM calls
- ✅ Fast execution
- ✅ Significantly reduces the context size
- ✅ Simple implementation

#### Drawbacks
- ⚠️ Does not compress textual content (arguments only)
- ⚠️ Does not create a summary for the LLM

#### Implementation via a custom middleware

**Create the file `deep_worker_cli/middleware/argument_truncation.py`:**

```python
"""Middleware for truncating large tool call arguments in old messages."""

from typing import Any
from langchain.agents.middleware.types import AgentMiddleware, AgentState, Runtime
from langchain_core.messages import AIMessage


class ArgumentTruncationMiddleware(AgentMiddleware):
    """Truncate large tool call arguments in old messages.

    This middleware reduces context size by truncating arguments in tool calls
    that are older than a specified cutoff, without requiring LLM calls for
    summarization.
    """

    def __init__(self, cutoff_messages: int = 15, max_arg_length: int = 300):
        """Initialize the middleware.

        Args:
            cutoff_messages: Number of recent messages to keep untruncated
            max_arg_length: Maximum length for truncated argument strings
        """
        self.cutoff_messages = cutoff_messages
        self.max_arg_length = max_arg_length
        self.tools = []  # No tools provided by this middleware

    def before_model(
        self, state: AgentState, runtime: Runtime
    ) -> dict[str, Any] | None:
        """Truncate arguments in old messages before passing to model.

        Args:
            state: Current agent state with messages
            runtime: Runtime context

        Returns:
            Dict with truncated messages, or None if no truncation needed
        """
        messages = state["messages"]

        if len(messages) <= self.cutoff_messages:
            return None  # No truncation needed

        # Truncate args in old messages
        truncated = []
        cutoff_index = len(messages) - self.cutoff_messages

        for i, msg in enumerate(messages):
            if i < cutoff_index and isinstance(msg, AIMessage):
                # Truncate tool_calls args
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    new_tool_calls = []
                    for tc in msg.tool_calls:
                        if isinstance(tc.get("args"), dict):
                            # Truncate each arg value
                            truncated_args = {}
                            for k, v in tc["args"].items():
                                if isinstance(v, str) and len(v) > self.max_arg_length:
                                    truncated_args[k] = (
                                        v[:self.max_arg_length]
                                        + f"... [truncated, was {len(v)} chars]"
                                    )
                                else:
                                    truncated_args[k] = v
                            new_tool_calls.append({**tc, "args": truncated_args})
                        else:
                            new_tool_calls.append(tc)

                    truncated.append(
                        AIMessage(
                            content=msg.content,
                            tool_calls=new_tool_calls,
                        )
                    )
                    continue

            truncated.append(msg)

        return {"messages": truncated}
```

**Add to `agent.py`:**

```python
from deep_worker_cli.middleware.argument_truncation import ArgumentTruncationMiddleware

# At the beginning of the middleware stack:
agent_middleware.append(
    ArgumentTruncationMiddleware(cutoff_messages=15, max_arg_length=300)
)
```

---

### Proposal 3: A combination of SummarizationMiddleware + Manual History Pruning

For aggressive context management: a combination of automatic summarization and manual cleanup.

#### Components

1. **SummarizationMiddleware** for automatic summarization
2. **Manual pruning** via the `dwcli --prune-history <agent_name>` command

#### Implementation of the prune command

**Add to `deep_worker_cli/cli/client_command.py`:**

```python
async def prune_history(agent_name: str, keep_messages: int = 20) -> int:
    """Prune conversation history for an agent, keeping only recent messages.

    This command connects to a running agent and removes old messages from
    its conversation history, keeping only the most recent ones.

    Args:
        agent_name: Name of agent to prune
        keep_messages: Number of recent messages to keep

    Returns:
        0 on success, 1 on error
    """
    try:
        registry = ProcessRegistry()
        agent_info = registry.lookup(agent_name)

        if not agent_info:
            print(f"Error: Agent '{agent_name}' not running", file=sys.stderr)
            return 1

        # Connect to agent via RPyC
        async with RpycAsyncSession("127.0.0.1", agent_info.rpyc_port) as session:
            # Call prune method
            result = await session.root.prune_history(keep_messages)

            if result["status"] == "success":
                print(f"✓ Pruned history for '{agent_name}'")
                print(f"  Kept: {result['kept_messages']} messages")
                print(f"  Removed: {result['removed_messages']} messages")
                return 0
            else:
                print(f"Error: {result['error']}", file=sys.stderr)
                return 1

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
```

**Add a CLI command in `main.py`:**

```python
parser.add_argument(
    "--prune-history",
    metavar="AGENT",
    help="Prune conversation history for an agent",
)

parser.add_argument(
    "--keep-messages",
    type=int,
    default=20,
    help="Number of recent messages to keep when pruning (default: 20)",
)

# In the command handling:
if args.prune_history:
    exit_code = await prune_history(args.prune_history, args.keep_messages)
    sys.exit(exit_code)
```

---

## 4. Recommendations for choosing a solution

### Scenario 1: Long sessions with a large number of tool calls

**Recommendation:** Proposal 1 (SummarizationMiddleware)

**Settings:**
```python
trigger=("fraction", 0.75)
keep=("messages", 20)
truncate_args_settings={
    "cutoff_messages": 15,
    "max_arg_length": 300,
}
```

**Why:** Automatic summarization prevents exceeding the context while preserving the full history for reference.

---

### Scenario 2: Quick tasks that rarely reach the limit

**Recommendation:** Proposal 2 (Argument Truncation Only)

**Settings:**
```python
cutoff_messages=20
max_arg_length=500
```

**Why:** Simple truncation without LLM calls. Sufficient for most short sessions.

---

### Scenario 3: Performance-critical, a minimum of LLM calls

**Recommendation:** Proposal 3 (Manual Pruning)

**Usage:**
```bash
# When the context becomes too large:
dwcli --prune-history my_agent --keep-messages 15
```

**Why:** The user decides when to clean up the history. There are no automatic LLM calls.

---

### Scenario 4: Maximum flexibility

**Recommendation:** All three approaches at once

**Components:**
1. SummarizationMiddleware (automatic summarization at 80% fill)
2. Argument Truncation (additional compression of old arguments)
3. Manual Pruning (emergency cleanup on command)

**Why:** Multi-layered protection against context overflow with the option of manual control.

---

## 5. Concrete integration steps (Proposal 1)

### Step 1: Add the import in `agent.py`

**File:** `deep_worker_cli/agent.py:496-501`

```python
from deepagents import create_deep_agent
from deepagents.backends import CompositeBackend
from deepagents.backends.filesystem import FilesystemBackend
from deepagents.middleware import MemoryMiddleware, SkillsMiddleware
from deepagents.middleware.subagents import CompiledSubAgent, SubAgent
from deepagents.middleware.summarization import SummarizationMiddleware  # ADD
```

---

### Step 2: Add the SummarizationMiddleware to the middleware stack

**File:** `deep_worker_cli/agent.py:566-596`

**IMPORTANT:** Add it FIRST in the middleware stack for early context truncation!

```python
agent_middleware = []

# ========== CONTEXT MANAGEMENT ==========
# Add SummarizationMiddleware FIRST to manage context size early
if settings.summarization_enabled:
    agent_middleware.append(
        SummarizationMiddleware(
            backend=composite_backend,
            trigger=("fraction", settings.summarization_trigger_fraction),
            keep=("fraction", settings.summarization_keep_fraction),
            truncate_args_settings={
                "cutoff_messages": settings.summarization_truncate_after,
                "max_arg_length": settings.summarization_max_arg_length,
            },
        )
    )

# ========== MEMORY ==========
if enable_memory:
    memory_sources = [str(settings.get_user_agent_md_path(assistant_id))]
    project_agent_md = settings.get_project_agent_md_path()
    if project_agent_md:
        memory_sources.append(str(project_agent_md))

    agent_middleware.append(
        MemoryMiddleware(
            backend=FilesystemBackend(root_dir=project_memory_dir),
            sources=memory_sources,
        )
    )

# ... the remaining middleware ...
```

---

### Step 3: Add settings in `config.py`

**File:** `deep_worker_cli/config.py`

```python
class Settings(BaseSettings):
    """Configuration settings for deep-worker CLI."""

    # ... existing settings ...

    # ========== Context Management / Summarization ==========
    summarization_enabled: bool = Field(
        default=True,
        description="Enable automatic conversation summarization when context fills up",
    )

    summarization_trigger_fraction: float = Field(
        default=0.80,
        ge=0.1,
        le=1.0,
        description="Context window fraction (0.0-1.0) that triggers summarization",
    )

    summarization_keep_fraction: float = Field(
        default=0.15,
        ge=0.05,
        le=0.5,
        description="Fraction of recent context to keep after summarization",
    )

    summarization_truncate_after: int = Field(
        default=15,
        ge=5,
        le=100,
        description="Truncate tool call arguments in messages older than this count",
    )

    summarization_max_arg_length: int = Field(
        default=300,
        ge=100,
        le=2000,
        description="Maximum length for truncated tool call arguments",
    )
```

---

### Step 4: Add CLI parameters in `main.py`

**File:** `deep_worker_cli/cli/main.py`

```python
# In the context management / performance arguments section:
context_group = parser.add_argument_group("Context Management")

context_group.add_argument(
    "--no-summarization",
    action="store_true",
    help="Disable automatic conversation summarization (not recommended for long sessions)",
)

context_group.add_argument(
    "--summarization-trigger",
    type=float,
    metavar="FRACTION",
    help=(
        "Context window fraction (0.0-1.0) that triggers summarization "
        "(default: 0.80 = 80%% full)"
    ),
)

context_group.add_argument(
    "--summarization-keep",
    type=float,
    metavar="FRACTION",
    help=(
        "Fraction of recent context to keep after summarization "
        "(default: 0.15 = keep last 15%%)"
    ),
)

# In the argument handling:
if args.no_summarization:
    settings.summarization_enabled = False

if args.summarization_trigger:
    if not 0.1 <= args.summarization_trigger <= 1.0:
        parser.error("--summarization-trigger must be between 0.1 and 1.0")
    settings.summarization_trigger_fraction = args.summarization_trigger

if args.summarization_keep:
    if not 0.05 <= args.summarization_keep <= 0.5:
        parser.error("--summarization-keep must be between 0.05 and 0.5")
    settings.summarization_keep_fraction = args.summarization_keep
```

---

### Step 5: Check the filtering (already implemented)

The filtering of summarization messages ALREADY works in 3 places:

1. **stream_processor.py:169-171** - filters during stream processing
2. **textual_adapter.py:386-388** - filters in the Textual UI
3. **non_interactive.py:237-238** - filters in non-interactive mode

**Action:** Nothing needs to be done - the filtering is ready!

---

### Step 6: Write tests

**Create the file:** `tests/unit_tests/test_summarization_integration.py`

```python
"""Tests for SummarizationMiddleware integration in deep-worker."""

import pytest
from pathlib import Path
from deep_worker_cli.agent import create_cli_agent
from deep_worker_cli.config import settings


@pytest.mark.asyncio
async def test_summarization_middleware_is_enabled(tmp_path: Path):
    """Test that SummarizationMiddleware is included in middleware stack."""
    from deepagents.middleware.summarization import SummarizationMiddleware

    # Enable summarization in settings
    settings.summarization_enabled = True

    # Create agent
    agent, backend = create_cli_agent(
        model="anthropic:claude-sonnet-4",
        assistant_id="test_summarization",
        project_memory_dir=tmp_path,
        enable_shell=False,
    )

    # Check that SummarizationMiddleware is in the middleware stack
    # This requires inspecting agent internals - implementation depends on deepagents API
    # TODO: Add proper middleware inspection once API is available
    pass


@pytest.mark.asyncio
async def test_summarization_triggers_at_threshold(tmp_path: Path):
    """Test that summarization triggers when context reaches threshold."""
    # TODO: Create agent, send many messages, verify summarization occurs
    pass


@pytest.mark.asyncio
async def test_history_offloaded_to_backend(tmp_path: Path):
    """Test that full history is offloaded to /conversation_history/."""
    # TODO: Trigger summarization, verify file exists in backend
    pass


@pytest.mark.asyncio
async def test_summarization_messages_filtered(tmp_path: Path):
    """Test that summarization messages are filtered from UI output."""
    # TODO: Verify metadata["lc_source"] == "summarization" messages are filtered
    pass


@pytest.mark.asyncio
async def test_argument_truncation_works(tmp_path: Path):
    """Test that tool call arguments are truncated in old messages."""
    # TODO: Create messages with large args, verify truncation after cutoff
    pass


@pytest.mark.asyncio
async def test_no_summarization_when_disabled(tmp_path: Path):
    """Test that summarization can be disabled via settings."""
    settings.summarization_enabled = False

    # Create agent
    agent, backend = create_cli_agent(
        model="anthropic:claude-sonnet-4",
        assistant_id="test_no_summarization",
        project_memory_dir=tmp_path,
        enable_shell=False,
    )

    # TODO: Verify SummarizationMiddleware is NOT in middleware stack
    pass
```

**The tests must verify:**
- ✅ The SummarizationMiddleware is included in the middleware stack
- ✅ The summarization trigger fires when the limit is reached
- ✅ The history is saved to `/conversation_history/{thread_id}.md`
- ✅ The summary message is created correctly
- ✅ Summarization messages are filtered out in the UI
- ✅ Truncation of tool call arguments works
- ✅ The ability to disable summarization via `--no-summarization`

---

## 6. Additional considerations

### An alternative for production: persistent history storage

Instead of `/tmp` (which is deleted on reboot), persistent storage can be used:

**File:** `deep_worker_cli/agent.py:664-666`

```python
# INSTEAD OF:
conversation_history_backend = FilesystemBackend(
    root_dir=tempfile.mkdtemp(prefix="deepagents_conversation_history_"),
    virtual_mode=True,
)

# USE:
history_dir = settings.user_deepagents_dir / assistant_id / "conversation_history"
history_dir.mkdir(parents=True, exist_ok=True)

conversation_history_backend = FilesystemBackend(
    root_dir=history_dir,
    virtual_mode=False,  # Persistent storage
)
```

**Advantages:**
- ✅ The history is preserved across reboots
- ✅ The user can review the full histories
- ✅ Useful for debugging and analysis

**Drawbacks:**
- ⚠️ Takes up disk space
- ⚠️ Requires periodic cleanup of old histories

---

### Monitoring the context size

Add logging when summarization fires:

```python
import logging

logger = logging.getLogger(__name__)

# In SummarizationMiddleware.before_model():
if should_summarize:
    logger.info(
        f"Summarization triggered for thread {thread_id}: "
        f"{total_tokens} tokens, {len(messages)} messages"
    )
```

The user will be able to see when summarization happens through the agent logs.

---

### Configuration for different models

Different models have different context windows:

```python
# In config.py or agent.py:
MODEL_CONTEXT_LIMITS = {
    "claude-sonnet-4": 200_000,
    "claude-opus-4": 200_000,
    "gpt-4": 128_000,
    "gpt-3.5-turbo": 16_000,
}

# Automatic trigger configuration based on the model:
model_name = settings.model_name
context_limit = MODEL_CONTEXT_LIMITS.get(model_name, 100_000)

# Trigger at 80% of the specific model's context being filled
trigger = ("tokens", int(context_limit * 0.80))
```

---

## Conclusion

### Current state

- ❌ deep-worker **does NOT use** the SummarizationMiddleware
- ✅ Filtering of summarization messages is **ALREADY implemented** (the preparation is complete)
- ✅ The backend infrastructure is **READY** for use
- ✅ The CompositeBackend with `/conversation_history/` routing is **configured**

### Recommendation

**Integrate the SummarizationMiddleware (Proposal 1)** with the settings:

```python
trigger=("fraction", 0.80)  # Trigger at 80% fill
keep=("fraction", 0.15)     # Keep 15% of the context
truncate_args_settings={
    "cutoff_messages": 15,
    "max_arg_length": 300,
}
```

### Benefits of the integration

- ✅ Automatic context management without user intervention
- ✅ Preservation of the full history without loss in the backend
- ✅ Truncation of large tool call arguments to save tokens
- ✅ Minimal code changes (4 files: agent.py, config.py, main.py, tests)
- ✅ Configurability via CLI parameters and settings
- ✅ The backend infrastructure is already in place

### Effort estimate

- **Implementation time:** 2-3 hours
- **Files to change:** 4 (agent.py, config.py, main.py, a new test file)
- **Lines of code:** ~150-200 (including tests and documentation)
- **Risks:** Minimal (using a ready-made SDK component)

### Next steps

1. Implement Proposal 1 (SummarizationMiddleware)
2. Write TDD tests to verify the functionality
3. Test on long sessions with a large number of tool calls
4. Document the new CLI parameters in the README
5. (Optional) Add Proposal 3 (Manual Pruning) for manual control

---

**End of document**

*Creation date: 2026-04-02*
*Status: Ready for implementation*