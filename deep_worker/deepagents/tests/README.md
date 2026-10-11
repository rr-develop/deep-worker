# Deep Agents SDK Tests

## Origin note

Most of the files in `tests/integration_tests/` were imported wholesale
from the upstream `deepagents` reference implementation (commit
`56d41c8`, "Add deep_worker: Copy of deepagents-cli renamed to
deep-worker"). The upstream tests drive
`langchain_anthropic.ChatAnthropic` directly (via direct construction,
`init_chat_model("anthropic:...")`, `SAMPLE_MODEL`, or the implicit
default in `deepagents.graph.get_default_model()`), which talks to the
public Anthropic HTTPS API and is the wrong transport on this host.

This host uses the `claude-code-sdk` provider
(`deep_worker_cli.config.create_model(provider="claude-code-sdk", ...)`),
which authenticates through the user's Claude Code SDK login session
and does **not** require any Anthropic env key. `tests/integration_tests/conftest.py`
therefore identifies the upstream tests that actually drive the
Anthropic SDK and skips them with a clear, documented reason; tests
written on this host (`test_project_memory.py`) run normally.

## API Keys

### Required

None for the `claude-code-sdk` provider (auth comes from the Claude Code
SDK login session).

### Optional

- **`LANGSMITH_API_KEY`** or **`LANGCHAIN_API_KEY`** - Enables LangSmith tracing for test runs

## Test Utilities

Shared test utilities are in `tests/utils.py`:

- Mock tools (`get_weather`, `get_soccer_scores`, etc.)
- Middleware classes (`ResearchMiddleware`, `WeatherToolMiddleware`, etc.)
- Assertion helpers (`assert_all_deepagent_qualities`)
