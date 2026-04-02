# Summary: Agents have no internet access

## Problem

Agents started through RPyC (AgentProcess) report that they have NO internet access.

```
> Do you have internet access?
No, I do not have internet access.
```

## Cause

The web tools (`http_request`, `fetch_url`, `web_search`) are **NOT passed** to `create_cli_agent()` in `agent_process.py`.

### The main.py code (the interactive CLI client) - ✅ WORKS

```python
# main.py lines 393-395
tools = [http_request, fetch_url]
if settings.has_tavily:
    tools.append(web_search)

agent, backend = create_cli_agent(
    ...
    tools=tools,  # ← the web tools are passed!
)
```

### The agent_process.py code (RPyC) - ❌ DOES NOT WORK

```python
# agent_process.py lines 211-219
self.agent, self.backend = create_cli_agent(
    model=model,
    assistant_id=self.name,
    # tools=???  ← the parameter is MISSING!
    auto_approve=self.skip_permissions,
    ...
)
```

## Solution

### Change agent_process.py

**File:** `process/agent_process.py`

**1. Add the import (after line 199):**
```python
from deep_worker_cli.tools import http_request, fetch_url, web_search
from deep_worker_cli.config import settings
```

**2. Prepare the web tools (before line 211):**
```python
# Prepare web tools
web_tools = [http_request, fetch_url]
if settings.has_tavily:
    web_tools.append(web_search)
```

**3. Pass the tools to create_cli_agent (line 211):**
```python
self.agent, self.backend = create_cli_agent(
    model=model,
    assistant_id=self.name,
    tools=web_tools,  # ← ADD THIS!
    auto_approve=self.skip_permissions,
    enable_memory=True,
    enable_skills=True,
    enable_shell=True,
    project_memory_dir=Path(self.config.working_dir),
)
```

## API Keys

**http_request** and **fetch_url** - work WITHOUT any API keys ✅

**web_search** - requires a Tavily API key:
1. Obtain one at https://tavily.com
2. Add it to `~/.deepagents/.env`:
   ```
   TAVILY_API_KEY=tvly-YOUR_KEY_HERE
   ```

**Implementation status:**
- ✅ `http_request` and `fetch_url` have been added to agent_process.py
- ❌ `web_search` (Tavily) has not been added yet (requires an API key or an alternative)

## Testing

After the fix:

**1. http_request** (without an API key):
```
> Make an HTTP request to https://api.github.com
```

**2. fetch_url** (without an API key):
```
> Fetch https://example.com
```

**3. web_search** (requires tavily_api_key):
```
> Find the latest news about Python
```

## Documentation

**Detailed analysis:** `claude_code_provider/docs/analysis_web_access_missing.md`

It includes:
- A study of the code base
- A comparison of main.py vs agent_process.py
- Three solution options
- API key requirements
- The testing procedure
