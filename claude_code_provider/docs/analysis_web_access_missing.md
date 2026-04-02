# Analysis: Agents Have No Internet Access

## Problem

Agents (via the claude-code provider and others) report that they have NO internet access:

```
> Do you have internet access?

No, I do not have internet access.

I work in an isolated sandbox environment and have access only to the local file system.

But I cannot:
• Download files from the internet
• Access web sites
• Make HTTP requests
```

## Investigation

### 1. Web tools exist in the code base

**File:** `tools.py`

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

**File:** `main.py`

**Lines 393-395:**
```python
tools = [http_request, fetch_url]
if settings.has_tavily:
    tools.append(web_search)
```

The web tools are added and passed into `create_cli_agent()` when the CLI client is started interactively.

### 3. Web tools are NOT passed into AgentProcess

**File:** `process/agent_process.py`

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

### 4. Tavily API key configuration

**File:** `config.py`

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

### In main.py (the interactive CLI client):

```python
# main.py lines 393-395
tools = [http_request, fetch_url]
if settings.has_tavily:
    tools.append(web_search)

# main.py line 418+ (approximately)
agent, backend = create_cli_agent(
    model=model,
    assistant_id=assistant_id,
    tools=tools,  # ← The web tools are passed!
    ...
)
```

### In agent_process.py (RPyC agents):

```python
# agent_process.py lines 211-219
self.agent, self.backend = create_cli_agent(
    model=model,
    assistant_id=self.name,
    # tools=???  ← There is NO tools parameter!
    auto_approve=self.skip_permissions,
    ...
)
```

**Conclusion:** The web tools are NOT added for agents started via AgentProcess (RPyC mode).

## Comparison: interactive vs RPyC mode

| Function | Interactive CLI client (main.py) | RPyC agent (agent_process.py) |
|---------|-------------------------------|-------------------------------|
| `http_request` | ✅ Passed | ❌ NOT passed |
| `fetch_url` | ✅ Passed | ❌ NOT passed |
| `web_search` | ✅ Passed (if tavily_api_key) | ❌ NOT passed |
| `execute` | ✅ (via enable_shell) | ✅ (via enable_shell) |
| File tools | ✅ (via backend) | ✅ (via backend) |

## Solution

### Option 1: Add the web tools in agent_process.py (simple)

**File:** `process/agent_process.py`

**Changes:**

```python
# ADD IMPORT (after line 199):
from deep_worker_cli.tools import http_request, fetch_url, web_search

# CHANGE the create_cli_agent call (lines 211-219):
# Prepare the web tools
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

**File:** `config/models.py`

Add:
```python
class AgentConfig:
    ...
    enable_web_tools: bool = True  # Enable http_request, fetch_url, web_search
```

Then check `self.config.enable_web_tools` in `agent_process.py`.

### Option 3: Always add the web tools (recommended)

The web tools are safe (they only make GET/POST requests). There is no reason NOT to enable them.

**Recommendation:** Use Option 1 - simply add the web tools in agent_process.py

## Tavily API key requirements

**web_search** requires a Tavily API key:
- Obtain it at: https://tavily.com
- Set it: `export TAVILY_API_KEY=tvly-...` in `~/.deepagents/.env`

**http_request** and **fetch_url** work WITHOUT an API key (they only use `requests`).

## Current state

### The .env file

**Location:** `~/.deepagents/.env`

Check whether it is present:
```bash
cat ~/.deepagents/.env | grep TAVILY
```

If not, add it:
```bash
echo "TAVILY_API_KEY=tvly-YOUR_KEY_HERE" >> ~/.deepagents/.env
```

## Testing after the fix

### 1. http_request (without an API key)

```
> Make an HTTP request to https://api.github.com
```

Expected: A successful response with GitHub API data

### 2. fetch_url (without an API key)

```
> Fetch the contents of the page https://example.com
```

Expected: The Markdown content of the page

### 3. web_search (requires tavily_api_key)

```
> Find the latest news about Python 3.13
```

Expected:
- If tavily_api_key is configured → search results
- If NOT → the error "Tavily API key not configured"

## Summary

**Cause:** The web tools exist but are NOT passed into `create_cli_agent()` in `agent_process.py`

**Solution:** Add the parameter `tools=[http_request, fetch_url, web_search]` when creating the agent

**Files to change:**
- `process/agent_process.py`

**API Keys:**
- `http_request`, `fetch_url` - work WITHOUT keys ✅
- `web_search` - requires `TAVILY_API_KEY` in `~/.deepagents/.env`

**Additionally:** Check for the presence of a Tavily API key in the environment
