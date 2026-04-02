# WebSearch Delegation Tool - Quick Reference

Universal web search tool for ANY AI provider via Claude Code SDK delegation.

## Quick Start

```python
from claude_code_provider import WebSearchDelegationTool

# Create tool
tool = WebSearchDelegationTool()

# Perform search
result = await tool.ainvoke({
    "query": "Python tutorials",
    "allowed_domains": ["python.org"]
})

# Results are JSON
import json
results = json.loads(result)
for item in results:
    print(f"{item['title']}: {item['url']}")
```

## With Any LLM Provider

### Gemini
```python
from langchain_google_genai import ChatGoogleGenerativeAI

model = ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp")
model_with_search = model.bind_tools([WebSearchDelegationTool()])
```

### OpenAI
```python
from langchain_openai import ChatOpenAI

model = ChatOpenAI(model="gpt-4o-mini")
model_with_search = model.bind_tools([WebSearchDelegationTool()])
```

### DeepSeek
```python
from langchain_openai import ChatOpenAI

model = ChatOpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com/v1",
    model="deepseek-chat"
)
model_with_search = model.bind_tools([WebSearchDelegationTool()])
```

## Parameters

### Tool Configuration
```python
tool = WebSearchDelegationTool(
    timeout=30.0,   # Timeout in seconds
    verbose=True,   # Debug logging
    model="claude-sonnet-4-5-20250929"  # Claude model
)
```

### Search Parameters
```python
await tool.ainvoke({
    "query": "search query",              # Required
    "allowed_domains": ["domain.com"],    # Optional whitelist
    "blocked_domains": ["spam.com"]       # Optional blacklist
})
```

## Result Format

```json
[
  {
    "title": "Page Title",
    "url": "https://example.com/page",
    "snippet": "Brief description of the page content..."
  },
  ...
]
```

## Error Handling

Errors are returned as JSON (never raises exceptions):

```json
{
  "error": "Error message",
  "query": "original query"
}
```

## Examples

Run the example script:
```bash
python3 examples/websearch_delegation_example.py
```

## Requirements

- Claude CLI installed and authenticated
- Internet connection
- Claude Code SDK (automatically imported from source)

## Documentation

- [Design Document](../docs/WEBSEARCH_DELEGATION_DESIGN.md)
- [Implementation Summary](../docs/WEBSEARCH_IMPLEMENTATION_SUMMARY.md)
- [Tests](tests/test_web_search_delegation.py)

## Architecture

```
AI Provider → WebSearchDelegationTool → ClaudeSDKClient
              ↓ PreToolUse Hook enforces params
              ↓ Claude WebSearch executes
              ↓ Results extracted
              → Formatted JSON returned
```

## Test Status

✅ **23/23 tests passing (100% coverage)**

Run tests:
```bash
python3 -m pytest claude_code_provider/tests/test_web_search_delegation.py -v
```

## License

Same as parent project (see root LICENSE file)
