# Analysis: Agents have no internet access

## Problem

Deep-worker agents (via the claude-code provider and others) report that they have NO internet access:

```
> Do you have internet access?

No, I don't have internet access.

I work in an isolated sandbox environment and have access only to the local file system.

But I can't:
• Download files from the internet
• Access websites
• Make HTTP requests
```

## Investigation

### 1. Web tools exist in the codebase

**File:** `deep_worker/deep_worker_cli/deep_worker_cli/tools.py`

**Available functions:**
- `http_request()` - Make HTTP requests to APIs and web services (lines 24-88)
- `fetch_url()` - Fetch and convert web pages to markdown (lines 91-...)
- `web_search()` - Search the web using Tavily API (lines 91-...)

**Tavily Client** (lines 7-21):
```python
from tavily import TavilyClient

tavily_client = (
    TavilyClient(api_key=settings.tavily_api_key) if settings.has_tavily else None
)
```

### 2. Web tools are registered in main.py

**File:** `deep_worker/deep_worker_cli/deep_worker_cli/main.py`

**Lines 393-395:**
```python
tools = [http_request, fetch_url]
if settings.has_tavily:
    tools.append(web_search)
```

Web tools are added and passed to `create_cli_agent()` when dwcli is started interactively.

### 3. Web tools are NOT passed to AgentProcess

**File:** `deep_worker/deep_worker_cli/deep_worker_cli/process/agent_process.py`

**Lines 211-219:**
```python
self.agent, self.backend = create_cli_agent(
    model=model,
    assistant_id=self.name,
    auto_approve=self.skip_permissions,
    enable_memory=True,
    enable_skills=True,
    enable_shell=True,
    project_memory_dir=Path(self.config.working_dir),
)
```

**PROBLEM:** The `tools=...` parameter is **NOT passed**!

### 4. Tavily API Key configuration

**File:** `deep_worker/deep_worker_cli/deep_worker_cli/config.py`

**Line 475:**
```python
tavily_key = os.environ.get("TAVILY_API_KEY")
```

**Lines 522-524:**
```python
@property
def has_tavily(self) -> bool:
    """Check if Tavily API key is configured."""
    return self.tavily_api_key is not None
```

## Root cause

### In main.py (interactive dwcli):

```python
# main.py lines 393-395
tools = [http_request, fetch_url]
if settings.has_tavily:
    tools.append(web_search)

# main.py line 418+ (approximately)
agent, backend = create_cli_agent(
    model=model,
    assistant_id=assistant_id,
    tools=tools,  # ← Web tools ARE passed!
    ...
)
```

### In agent_process.py (RPyC agents):

```python
# agent_process.py lines 211-219
self.agent, self.backend = create_cli_agent(
    model=model,
    assistant_id=self.name,
    # tools=???  ← NO tools parameter!
    auto_approve=self.skip_permissions,
    ...
)
```

**Conclusion:** Web tools are NOT added for agents launched through AgentProcess (RPyC mode).

## Comparison: Interactive vs RPyC mode

| Function | Interactive dwcli (main.py) | RPyC agent (agent_process.py) |
|---------|-------------------------------|-------------------------------|
| `http_request` | ✅ Passed | ❌ NOT passed |
| `fetch_url` | ✅ Passed | ❌ NOT passed |
| `web_search` | ✅ Passed (if tavily_api_key) | ❌ NOT passed |
| `execute` | ✅ (via enable_shell) | ✅ (via enable_shell) |
| File tools | ✅ (via backend) | ✅ (via backend) |

## Solution

### Option 1: Add web tools to agent_process.py (simple)

**File:** `deep_worker/deep_worker_cli/deep_worker_cli/process/agent_process.py`

**Changes:**

```python
# ADD IMPORT (after line 199):
from deep_worker_cli.tools import http_request, fetch_url, web_search

# CHANGE the create_cli_agent call (lines 211-219):
# Prepare web tools
web_tools = [http_request, fetch_url]

# Load settings to check tavily_api_key
from deep_worker_cli.config import settings
if settings.has_tavily:
    web_tools.append(web_search)

self.agent, self.backend = create_cli_agent(
    model=model,
    assistant_id=self.name,
    tools=web_tools,  # ← ADD!
    auto_approve=self.skip_permissions,
    enable_memory=True,
    enable_skills=True,
    enable_shell=True,
    project_memory_dir=Path(self.config.working_dir),
)
```

### Option 2: Add a parameter to AgentConfig (more flexible)

**File:** the module that defines the `AgentConfig` class

Add:
```python
class AgentConfig:
    ...
    enable_web_tools: bool = True  # Enable http_request, fetch_url, web_search
```

Then check `self.config.enable_web_tools` in `agent_process.py`.

### Option 3: Always add web tools (recommended)

Web tools are safe (they only make GET/POST requests). There is no reason NOT to enable them.

**Recommendation:** Use Option 1 - simply add web tools to agent_process.py

## Tavily API Key requirements

**web_search** requires a Tavily API key:
- Get one at: https://tavily.com
- Set: `export TAVILY_API_KEY=tvly-...` in the local `.env` file

**http_request** and **fetch_url** work WITHOUT an API key (they only use `requests`).

## Current state

### .env file

**Location:** the local `.env` file

Check for its presence:
```bash
cat .env | grep TAVILY
```

If absent, add:
```bash
echo "TAVILY_API_KEY=tvly-YOUR_KEY_HERE" >> .env
```

## Testing after the fix

### 1. http_request (no API key)

```
> Make an HTTP request to https://api.github.com
```

Expected: Successful response with GitHub API data

### 2. fetch_url (no API key)

```
> Load the contents of the page https://example.com
```

Expected: Markdown content of the page

### 3. web_search (requires tavily_api_key)

```
> Find the latest news about Python 3.13
```

Expected:
- If tavily_api_key is configured → search results
- If NOT → error "Tavily API key not configured"

## Summary

**Cause:** Web tools exist but are NOT passed to `create_cli_agent()` in `agent_process.py`

**Solution:** Add the `tools=[http_request, fetch_url, web_search]` parameter when creating the agent

**Files to change:**
- `deep_worker/deep_worker_cli/deep_worker_cli/process/agent_process.py`

**API Keys:**
- `http_request`, `fetch_url` - work WITHOUT keys ✅
- `web_search` - requires `TAVILY_API_KEY` in the local `.env` file

**Additionally:** Check for the presence of the Tavily API key in the environment
