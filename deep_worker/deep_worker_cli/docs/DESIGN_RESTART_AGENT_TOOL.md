# Design: `restart_agent` Tool

Status: implemented

## Goal

Expose an agent-usable tool that can restart a named deep-worker agent through the same lifecycle path as `deep-worker --restart <agent_name>`.

The tool is not a general-purpose convenience command. It is intended for cases where the human user or an explicit procedure requires a specific agent to be running, and the current agent needs to bring that dependency back online.

## Tool Contract

Name: `restart_agent`

Arguments:

| Field | Type | Required | Meaning |
|---|---:|---:|---|
| `agent_name` | string | yes | Registry/session name of the agent to restart. |

Return value is a non-raising structured dict:

```python
{
    "success": bool,
    "agent_name": "<agent>",
    "already_running": bool,
    "message": "human-readable status",
    "error": "optional_error_code",
    "stdout": "optional captured stdout",
    "stderr": "optional captured stderr",
}
```

Error codes:

| Code | Meaning |
|---|---|
| `invalid_argument` | `agent_name` was empty or not a string. |
| `self_restart_forbidden` | The caller tried to restart itself. |
| `already_running` | Target is registered and its process is alive, so restart is unnecessary. |
| `no_launch_config` | `deep-worker --restart` reported no saved launch configuration. |
| `actively_running` | CLI refused because the target status is `running`. |
| `restart_failed` | CLI returned non-zero for another reason. |
| `restart_timeout` | Restart command or post-start wait timed out. |
| `command_unavailable` | `deep-worker` was not found on `PATH`. |

`already_running` is returned with `success=True` and `already_running=True`: the requested invariant, "agent is running", is satisfied without killing a healthy process.

## Usage Policy

The tool description must explicitly instruct the model not to use it by default. Allowed use cases:

1. The human user explicitly asks to restart an agent.
2. The human user explicitly asks to ensure an agent is running.
3. A documented or user-provided procedure explicitly requires a named agent to be in a running state.

The tool must not be used for speculative cleanup, optimization, or autonomous process management.

## Architecture

Implement the tool in `deep_worker_cli.tools` as `make_restart_agent_tool(sender_name: str) -> BaseTool`, mirroring `make_send_message_to_agent_tool`.

Reasons:

- The sender identity is known in `AgentProcess`, not in static module-level tools.
- The tool can reject self-restart deterministically before touching process state.
- Registration stays close to the existing agent-only tool factory pattern.

Register it in `AgentProcess.start()` alongside `make_send_message_to_agent_tool(self.name)`.

Add HITL gating in `deep_worker_cli.agent.get_interrupt_on_config()` so `restart_agent` requires approval whenever global `-sp / --skip-permissions` is not active. It should not be auto-approved by `-asm`; that flag is only for inter-agent messaging.

## Lifecycle Behavior

The tool should not duplicate the restart implementation. It shells out to the installed `deep-worker` command:

```bash
deep-worker --restart <agent_name>
```

This deliberately reuses the existing CLI behavior:

- load saved `launch_config` from `~/.deep_worker/<agent>/session_metadata.json`
- refuse if no launch config exists
- refuse if target status is `running`
- stop idle/waiting/error/shutdown agents via existing `stop_agent`
- daemonize and launch with the saved effective config

Before invoking the CLI, the tool checks `ProcessRegistry.lookup(agent_name)` and `/proc/<pid>`. If the target is already registered and alive, return `success=True` without restarting. This matches the primary goal of satisfying an explicit "agent must be running" requirement and avoids unnecessary disruption.

After invoking the CLI, the tool waits briefly for `ProcessRegistry.lookup(agent_name)` to show a live PID. Suggested timeout: 30 seconds, polling every 0.5 seconds. The CLI daemonizes quickly, so this wait validates practical availability rather than only command acceptance.

## Path and Environment

Use `shutil.which("deep-worker")` to resolve the command. If unavailable, return `command_unavailable`.

The subprocess call should capture stdout/stderr and run with a bounded timeout. No shell string is needed; pass argv as a list.

## Tests

TDD order:

1. Add unit tests for `make_restart_agent_tool` before implementation:
   - factory returns a tool named `restart_agent`
   - schema exposes only `agent_name`
   - empty names fail with `invalid_argument`
   - self-restart fails with `self_restart_forbidden`
   - already-running target returns success without subprocess
   - missing command returns `command_unavailable`
   - non-zero CLI output maps `no_launch_config`, `actively_running`, and generic `restart_failed`
   - successful CLI call waits until registry reports a live PID
2. Add registration tests:
   - `AgentProcess` includes `restart_agent` in its web tools
   - `get_interrupt_on_config()` gates `restart_agent` by default
   - `-asm` does not ungate `restart_agent`
   - `-sp` still ungates all tools
3. Add a no-mock integration test that creates a temporary fake `deep-worker` executable on `PATH`, writes a realistic registry entry after invocation, and verifies the real tool subprocess path plus registry wait behavior. This avoids launching an LLM-backed agent while still exercising subprocess execution, PATH lookup, and registry observation without monkeypatching.

## Non-Goals

- Do not add restart overrides such as model/provider/dir to the tool in this feature. The requested behavior is equivalent to `deep-worker --restart <agent_name>`.
- Do not let an agent restart itself.
- Do not add autonomous retry loops around failed restarts.
- Do not bypass `deep-worker --restart` by importing CLI internals and reconstructing argparse namespaces.
