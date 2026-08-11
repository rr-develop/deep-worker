"""Regression: the wait_for_job / kill_job tools must reach the claude-code
model the same way `execute` does — i.e. bind_tools must turn them into valid
ToolMetadata (name + input schema with the right params).

When `execute` times out, the command keeps running in the background and the
agent gets a job_id. The agent can only act on it if wait_for_job/kill_job are
actually exposed to the model. The LangChain middleware framework collects
``middleware.tools`` (factory.py:845), and ClaudeCodeChatModel.bind_tools turns
each BaseTool into ToolMetadata for the proxy MCP. This test locks in that the
two new tools survive that conversion with correct schemas.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from deep_worker.providers.claude_code_provider import ClaudeCodeChatModel
from deep_worker_cli.backends import CLIFilesystemMiddleware, CLIShellBackend


def _job_tools():
    backend = CLIShellBackend(root_dir=Path("/tmp"), inherit_env=True)
    mw = CLIFilesystemMiddleware(backend=backend)
    return [t for t in mw.tools if t.name in ("execute", "wait_for_job", "kill_job")]


def test_job_tools_become_tool_metadata() -> None:
    tools = _job_tools()
    model = ClaudeCodeChatModel(model="opus")
    bound = model.bind_tools(tools)

    md = {m["name"]: m for m in bound.tools_metadata}
    assert "wait_for_job" in md, (
        "wait_for_job must reach the model as ToolMetadata so the agent can "
        "act on a backgrounded command"
    )
    assert "kill_job" in md
    assert "execute" in md

    # wait_for_job exposes job_id + timeout; kill_job exposes job_id.
    wait_props = md["wait_for_job"]["input_schema"].get("properties", {})
    assert "job_id" in wait_props
    assert "timeout" in wait_props
    # runtime is an injected arg and must NOT leak into the model-facing schema.
    assert "runtime" not in wait_props

    kill_props = md["kill_job"]["input_schema"].get("properties", {})
    assert "job_id" in kill_props
    assert "runtime" not in kill_props


def test_execute_timeout_param_is_advertised() -> None:
    """The execute tool's timeout param must remain exposed (so the agent can
    set its own timeout, as requested)."""
    tools = _job_tools()
    model = ClaudeCodeChatModel(model="opus")
    bound = model.bind_tools(tools)
    md = {m["name"]: m for m in bound.tools_metadata}
    exec_props = md["execute"]["input_schema"].get("properties", {})
    assert "command" in exec_props
    assert "timeout" in exec_props


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
