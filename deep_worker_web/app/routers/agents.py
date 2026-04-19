"""API router for agent operations.

This module provides REST API endpoints for:
- Listing agents
- Getting agent info
- Sending messages to agents
- Retrieving conversation history
- Getting last assistant response
- Stopping TaskManager
- Getting detailed status

All endpoints follow REST conventions and return JSON responses.
"""

import asyncio
import logging
import os
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import PlainTextResponse

from app.schemas.agent_schemas import (
    AgentInfo,
    AgentStatusInfo,
    BatchRequest,
    MessageInfo,
    MessageRequest,
    RestartRequest,
    StopTaskManagerRequest,
)
from app.services.agent_service import AgentService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/agents", tags=["agents"])


def get_agent_service(request: Request) -> AgentService:
    """Dependency to get AgentService from app state.
    
    Args:
        request: FastAPI request object.
        
    Returns:
        AgentService instance.
        
    Raises:
        HTTPException: If service is not initialized.
    """
    if not hasattr(request.app.state, "agent_service"):
        raise HTTPException(
            status_code=500,
            detail="AgentService not initialized",
        )
    return request.app.state.agent_service


@router.get("", response_model=list[AgentInfo])
async def list_agents(
    service: AgentService = Depends(get_agent_service),
) -> list[AgentInfo]:
    """List all discovered deep-worker agents.
    
    Returns:
        List of AgentInfo for all agents.
    """
    logger.debug("Listing all agents")
    agents = await service.list_agents()
    logger.debug(f"Found {len(agents)} agents")
    return agents


@router.get("/{agent_name}", response_model=AgentInfo)
async def get_agent(
    agent_name: str,
    service: AgentService = Depends(get_agent_service),
) -> AgentInfo:
    """Get information about a specific agent.
    
    Args:
        agent_name: Name of the agent.
        
    Returns:
        AgentInfo for the requested agent.
        
    Raises:
        HTTPException: If agent is not found.
    """
    logger.debug(f"Getting agent: {agent_name}")
    agent = await service.get_agent(agent_name)
    
    if agent is None:
        logger.warning(f"Agent not found: {agent_name}")
        raise HTTPException(
            status_code=404,
            detail=f"Agent '{agent_name}' not found",
        )
    
    return agent


@router.post("/{agent_name}/message")
async def send_message(
    agent_name: str,
    request: MessageRequest,
    service: AgentService = Depends(get_agent_service),
) -> dict[str, Any]:
    """Send a message to an agent.
    
    The message will be delivered to the agent via RPyC. The agent will
    process it and broadcast responses via WebSocket.
    
    Args:
        agent_name: Name of the agent.
        request: Message request with message content.
        
    Returns:
        Success response.
        
    Raises:
        HTTPException: If agent is not found or not connected.
    """
    logger.info(f"Sending message to agent: {agent_name}")
    
    # Check if agent exists
    agent = await service.get_agent(agent_name)
    if agent is None:
        raise HTTPException(
            status_code=404,
            detail=f"Agent '{agent_name}' not found",
        )
    
    # Send message via RPyC
    response = await service.send_message(agent_name, request.message)
    
    if not response.get("success"):
        raise HTTPException(
            status_code=500,
            detail=f"Failed to send message: {response.get('error', 'Unknown error')}",
        )
    
    return {
        "success": True,
        "message": "Message sent successfully",
        "agent_name": agent_name,
    }


@router.get("/{agent_name}/history")
async def get_history(
    agent_name: str,
    limit: int = 50,
    offset: int = 0,
    request_id: str | None = None,
    service: AgentService = Depends(get_agent_service),
) -> dict[str, Any]:
    """Get conversation history for an agent (fire-and-forget).

    Returns immediately with request_id. Actual history arrives via WebSocket.

    Args:
        agent_name: Name of the agent.
        limit: Maximum number of messages to retrieve.
        offset: Number of recent messages to skip.
        request_id: Optional request ID for correlation.

    Returns:
        Immediate response with request_id. Actual history delivered via WebSocket
        with type="history_response" or "history_error".

    Raises:
        HTTPException: If agent is not found or not connected.
    """
    logger.debug(f"Getting history for agent: {agent_name}")

    # Check if agent exists
    agent = await service.get_agent(agent_name)
    if agent is None:
        raise HTTPException(
            status_code=404,
            detail=f"Agent '{agent_name}' not found",
        )

    # Get history via fire-and-forget
    response = await service.get_agent_history(agent_name, limit, offset, request_id)

    if not response.get("success"):
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get history: {response.get('error', 'Unknown error')}",
        )

    # Return immediate response with request_id
    return response


@router.get("/{agent_name}/last-response")
async def get_last_response(
    agent_name: str,
    request_id: str | None = None,
    service: AgentService = Depends(get_agent_service),
) -> dict[str, Any]:
    """Get the last assistant response for an agent (fire-and-forget).

    Returns immediately with request_id. Actual response arrives via WebSocket
    with type="last_response_result" or "last_response_error".

    Args:
        agent_name: Name of the agent.
        request_id: Optional request ID for correlation.

    Returns:
        Immediate response with request_id.

    Raises:
        HTTPException: If agent is not found or not connected.
    """
    logger.debug(f"Getting last response for agent: {agent_name}")

    agent = await service.get_agent(agent_name)
    if agent is None:
        raise HTTPException(
            status_code=404,
            detail=f"Agent '{agent_name}' not found",
        )

    response = await service.get_agent_last_response(agent_name, request_id)

    if not response.get("success"):
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get last response: {response.get('error', 'Unknown error')}",
        )

    return response


@router.post("/{agent_name}/approve")
async def approve_action(
    agent_name: str,
    request: Request,
    service: AgentService = Depends(get_agent_service),
) -> dict[str, Any]:
    """Send approval/rejection decision for a tool interrupt.

    Args:
        agent_name: Name of the agent.

    Request Body:
        approval_id: Unique ID of the approval request.
        decision: "approve", "reject", or "auto_approve_all".

    Raises:
        HTTPException: If agent not found or approval fails.
    """
    body = await request.json()
    approval_id = body.get("approval_id")
    decision = body.get("decision", "approve")

    if not approval_id:
        raise HTTPException(status_code=400, detail="approval_id is required")

    agent = await service.get_agent(agent_name)
    if agent is None:
        raise HTTPException(status_code=404, detail=f"Agent '{agent_name}' not found")

    response = await service.send_approval(agent_name, approval_id, decision)

    if not response.get("success"):
        raise HTTPException(
            status_code=500,
            detail=response.get("error", "Failed to send approval"),
        )

    return response


@router.post("/{agent_name}/task-manager/toggle")
async def toggle_task_manager(
    agent_name: str,
    request: Request,
    service: AgentService = Depends(get_agent_service),
) -> dict[str, Any]:
    """Enable or disable TaskManager for an agent.

    Request Body: {"enabled": true|false}
    """
    body = await request.json()
    enabled = body.get("enabled", True)

    agent = await service.get_agent(agent_name)
    if agent is None:
        raise HTTPException(status_code=404, detail=f"Agent '{agent_name}' not found")

    response = await service.set_task_manager_enabled(agent_name, enabled)
    if not response.get("success"):
        raise HTTPException(status_code=500, detail=response.get("error"))
    return response


@router.post("/{agent_name}/task-manager/reset")
async def reset_task_manager_endpoint(
    agent_name: str,
    service: AgentService = Depends(get_agent_service),
) -> dict[str, Any]:
    """Reset TaskManager to initial state — clears evaluation history."""
    agent = await service.get_agent(agent_name)
    if agent is None:
        raise HTTPException(status_code=404, detail=f"Agent '{agent_name}' not found")

    response = await service.reset_task_manager(agent_name)
    if not response.get("success"):
        raise HTTPException(status_code=500, detail=response.get("error"))
    return response


@router.post("/{agent_name}/stop_task_manager")
async def stop_task_manager(
    agent_name: str,
    request: StopTaskManagerRequest,
    service: AgentService = Depends(get_agent_service),
) -> dict[str, Any]:
    """Stop TaskManager using fire-and-forget pattern.

    This stops the internal TaskManager running within the agent process.

    NOTE: This feature requires the agent to expose stop_task_manager method.
    If not available, returns a stub response.

    Args:
        agent_name: Name of the agent.
        request: Stop request with force flag.

    Returns:
        Immediate response. Result will arrive via WebSocket with
        type="task_manager_stopped" or "task_manager_stop_error".

    Raises:
        HTTPException: If agent is not found or not connected.
    """
    logger.info(f"Stopping TaskManager for agent: {agent_name} (force={request.force})")

    # Check if agent exists
    agent = await service.get_agent(agent_name)
    if agent is None:
        raise HTTPException(
            status_code=404,
            detail=f"Agent '{agent_name}' not found",
        )

    # Try to stop TaskManager via fire-and-forget
    from rpyc.utils.helpers import fire_and_forget_async

    if await service.is_connected(agent_name):
        root = service.get_root(agent_name)
        if root and hasattr(root, 'stop_task_manager'):

            async def on_success(response: dict[str, Any]) -> None:
                """Handle successful TaskManager stop."""
                logger.info(f"TaskManager stopped for {agent_name}")
                # Broadcast to WebSocket
                await service._broadcast_message({
                    "type": "task_manager_stopped",
                    "agent_name": agent_name,
                    "success": True,
                    "force": request.force,
                })

            async def on_error(exc: BaseException) -> None:
                """Handle TaskManager stop errors."""
                logger.error(f"Failed to stop TaskManager for {agent_name}: {exc}")
                await service._broadcast_message({
                    "type": "task_manager_stop_error",
                    "agent_name": agent_name,
                    "success": False,
                    "error": str(exc),
                })

            # Fire and forget with 5s timeout
            fire_and_forget_async(
                root.stop_task_manager(force=request.force),
                timeout=5.0,
                success_callback=on_success,
                error_callback=on_error,
                name=f"stop_task_manager_{agent_name}",
            )

            return {
                "success": True,
                "message": "TaskManager stop requested (result will arrive via WebSocket)",
                "agent_name": agent_name,
                "force": request.force,
            }

    # Fallback if method not available
    return {
        "success": True,
        "message": "TaskManager stop requested (feature may not be available)",
        "agent_name": agent_name,
        "force": request.force,
    }


@router.post("/{agent_name}/interrupt")
async def interrupt_agent_endpoint(
    agent_name: str,
    service: AgentService = Depends(get_agent_service),
) -> dict[str, Any]:
    """Interrupt the agent's currently running execution (fire-and-forget).

    Cancels the agent's current worker task via RPyC. The agent broadcasts
    ``agent_interrupted`` to all WebSocket subscribers, and we additionally
    broadcast ``agent_interrupt_result`` for the originating client.

    Unlike ``stop`` (which terminates the process) and ``stop_task_manager``
    (which shuts down TaskManager), ``interrupt`` only cancels the current
    in-flight invocation. The agent process keeps running and is then ready
    to accept the next user message.

    Returns:
        Immediate JSON ``{"success": true, "message": ...}`` on dispatch.
        Actual interrupt result arrives via WebSocket later.

    Raises:
        HTTPException: 404 if agent is not found.
    """
    logger.info(f"Interrupting agent: {agent_name}")

    agent = await service.get_agent(agent_name)
    if agent is None:
        raise HTTPException(
            status_code=404,
            detail=f"Agent '{agent_name}' not found",
        )

    response = await service.interrupt_agent(agent_name)
    if not response.get("success"):
        # Connection-level failure (couldn't reach RPyC). Surface as 500.
        raise HTTPException(
            status_code=500,
            detail=response.get("error", "Failed to interrupt agent"),
        )
    return response


@router.get("/{agent_name}/status")
async def get_status(
    agent_name: str,
    request_id: str | None = None,
    service: AgentService = Depends(get_agent_service),
) -> dict[str, Any]:
    """Get detailed status information for an agent (fire-and-forget).

    Returns immediately with request_id. Actual status arrives via WebSocket.

    Args:
        agent_name: Name of the agent.
        request_id: Optional request ID for correlation.

    Returns:
        Immediate response with request_id. Actual status delivered via WebSocket
        with type="detailed_status_response" or "detailed_status_error".

    Raises:
        HTTPException: If agent is not found or not connected.
    """
    logger.debug(f"Getting detailed status for agent: {agent_name}")

    # Check if agent exists
    agent = await service.get_agent(agent_name)
    if agent is None:
        raise HTTPException(
            status_code=404,
            detail=f"Agent '{agent_name}' not found",
        )

    # Get detailed status via fire-and-forget
    response = await service.get_agent_detailed_status(agent_name, request_id)

    if not response.get("success"):
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get status: {response.get('error', 'Unknown error')}",
        )

    # Return immediate response with request_id
    return response


@router.get("/{agent_name}/dialog-text", response_class=PlainTextResponse)
async def get_dialog_text(
    agent_name: str,
    verbosity: Literal["final", "text", "full"] = "final",
    service: AgentService = Depends(get_agent_service),
) -> str:
    """Get full dialog text for an agent as plain text.

    Returns ALL messages (no artificial limit). Verbosity controls how
    each turn renders:

    - ``final`` (default): one assistant block per turn, the LAST text-bearing
      AIMessage of that turn.
    - ``text``: every text-bearing AIMessage of the turn.
    - ``full``: like ``text`` plus compact tool-call summaries.

    Three sources of user-side messages are all rendered as first-class turns
    at every verbosity:

    - Direct user input → ``--- User ---``
    - TaskManager prompts → ``--- TaskManager ---``
    - Inter-agent messages → ``--- From agent:<name> ---``

    Only summarization-time synthetic injections are filtered.

    Args:
        agent_name: Name of the agent.
        verbosity: Rendering verbosity (validated by FastAPI; unknown
            values cause HTTP 422).

    Raises:
        HTTPException: 404 if agent unknown, 500 if connection fails.
    """
    logger.debug(f"Getting dialog text for agent: {agent_name} (verbosity={verbosity})")

    # Check if agent exists
    agent = await service.get_agent(agent_name)
    if agent is None:
        raise HTTPException(
            status_code=404,
            detail=f"Agent '{agent_name}' not found",
        )

    # Connect if not connected
    if not await service.is_connected(agent_name):
        if not await service.connect_to_agent(agent_name):
            raise HTTPException(
                status_code=500,
                detail=f"Failed to connect to agent '{agent_name}'",
            )

    # Get formatted dialog text from agent
    try:
        dialog_text = await service.get_agent_dialog_formatted(
            agent_name, verbosity=verbosity
        )
        return dialog_text

    except Exception as e:
        logger.error(f"Error getting dialog text for {agent_name}: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error getting dialog: {str(e)}",
        )


# ===========================================================================
# Agent Stop / Restart / Batch
# ===========================================================================


async def stop_agent_process(agent_name: str, service: AgentService) -> dict[str, Any]:
    """Stop an agent process gracefully.

    Uses deep_worker_cli.process.agent_stop.stop_agent under the hood.

    Returns:
        Dict with success/message or raises RuntimeError/TimeoutError.
    """
    import asyncio
    import json
    import os
    import signal
    from pathlib import Path

    from deep_worker_cli.process.registry import ProcessRegistry

    registry = ProcessRegistry()
    agent_info = registry.lookup(agent_name)
    if agent_info is None:
        return {"success": True, "message": f"Agent '{agent_name}' is not running"}

    # Check status
    data_dir = Path("~/.deep_worker").expanduser() / agent_name
    metadata_file = data_dir / "session_metadata.json"
    status = "idle"
    if metadata_file.exists():
        try:
            data = json.loads(metadata_file.read_text())
            status = data.get("status", "idle")
        except (json.JSONDecodeError, OSError):
            pass

    if status == "running":
        raise RuntimeError(
            f"Agent '{agent_name}' is actively running (status: running). "
            f"Wait for it to finish or interrupt it first."
        )

    pid = agent_info.pid
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        return {"success": True, "message": f"Agent '{agent_name}' already stopped"}

    # Wait for exit
    for _ in range(20):  # 10 seconds with 0.5s intervals
        if not Path(f"/proc/{pid}").exists():
            # Broadcast termination
            await service._broadcast_message({
                "type": "agent_terminated",
                "agent_name": agent_name,
            })
            return {"success": True, "message": f"Agent '{agent_name}' stopped"}
        await asyncio.sleep(0.5)

    raise TimeoutError(f"Agent '{agent_name}' (PID {pid}) did not stop within 10 seconds")


def load_launch_config(agent_name: str) -> dict[str, Any] | None:
    """Load launch_config from session_metadata.json."""
    import json
    from pathlib import Path

    data_dir = Path("~/.deep_worker").expanduser() / agent_name
    metadata_file = data_dir / "session_metadata.json"

    if not metadata_file.exists():
        return None
    try:
        data = json.loads(metadata_file.read_text())
        return data.get("launch_config")
    except (json.JSONDecodeError, OSError):
        return None


async def _wait_for_agent_in_registry(agent_name: str, timeout: float = 15.0) -> bool:
    """Wait for agent to appear in ProcessRegistry.

    Returns True if agent appeared, False if timeout.
    """
    from deep_worker_cli.process.registry import ProcessRegistry

    elapsed = 0.0
    poll = 0.5
    registry = ProcessRegistry()
    while elapsed < timeout:
        info = registry.lookup(agent_name)
        if info is not None:
            return True
        await asyncio.sleep(poll)
        elapsed += poll
    return False


async def spawn_agent_process(
    agent_name: str,
    config: dict[str, Any],
    *,
    model: str | None = None,
    provider: str | None = None,
    clean: bool = False,
) -> dict[str, Any]:
    """Spawn a new agent process via subprocess.

    Builds the CLI command from config and runs it as a daemon.
    Sets PYTHONPATH explicitly so the subprocess can find deep_worker_cli
    and deepagents regardless of the web server's environment.
    Waits for agent to appear in registry before returning success.
    """
    cmd = [
        "python3", "-m", "deep_worker_cli.cli.agent_command",
        agent_name,
        "--provider", provider or config.get("provider", "deepseek"),
        "--model", model or config.get("model", "unknown"),
        "--dir", config.get("working_dir", "."),
    ]
    if config.get("skip_permissions"):
        cmd.append("-sp")
    if config.get("task_manager"):
        cmd.append("-tm")
    if config.get("memory_only"):
        cmd.append("--memory-only")
    if config.get("use_sqlite"):
        cmd.append("--use-sqlite")
    if clean:
        cmd.append("--clean")

    # Build environment with correct PYTHONPATH
    # The web server may not have deep_worker_cli/deepagents on its path,
    # so we set it explicitly for the subprocess.
    env = os.environ.copy()
    # Locate sibling packages relative to this checkout so the subprocess
    # can find them without host-specific absolute paths.
    _repo_root = Path(__file__).resolve().parents[3]
    env["PYTHONPATH"] = ":".join([
        str(_repo_root / "deep_worker" / "deep_worker_cli"),
        str(_repo_root / "deep_worker" / "deepagents"),
    ])
    # Preserve DEEP_WORKER_ORIGINAL_CWD for the working directory
    env["DEEP_WORKER_ORIGINAL_CWD"] = config.get("working_dir", ".")

    logger.info(f"Spawning agent: {' '.join(cmd)}")

    # Redirect stderr to agent's log file for debugging (not /dev/null)
    log_dir = Path("~/.deep_worker").expanduser() / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    err_log = log_dir / f"{agent_name}.spawn.log"

    with open(err_log, "w") as err_file:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=err_file,
            env=env,
            start_new_session=True,  # Detach from parent
        )

    # The CLI daemonizes itself (double-fork), so proc exits quickly.
    # Wait for the actual daemon to appear in the registry.
    appeared = await _wait_for_agent_in_registry(agent_name, timeout=15.0)

    if appeared:
        return {
            "success": True,
            "pid": proc.pid,
            "message": f"Agent '{agent_name}' restarted successfully",
        }
    else:
        # Read spawn log for error details
        error_details = ""
        if err_log.exists():
            error_details = err_log.read_text()[-500:]  # Last 500 chars
        return {
            "success": False,
            "error": f"Agent '{agent_name}' did not appear in registry within 15s. "
                     f"Check {err_log} for details. {error_details}",
        }


@router.post("/{agent_name}/stop")
async def stop_agent(
    agent_name: str,
    service: AgentService = Depends(get_agent_service),
) -> dict[str, Any]:
    """Stop an agent process gracefully.

    Sends SIGTERM and waits for the process to exit.
    Returns 409 if agent is actively running (status=running).
    """
    agent = await service.get_agent(agent_name)
    if agent is None:
        raise HTTPException(status_code=404, detail=f"Agent '{agent_name}' not found")

    try:
        result = await stop_agent_process(agent_name, service)
        return result
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except TimeoutError as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{agent_name}/restart")
async def restart_agent(
    agent_name: str,
    request: RestartRequest | None = None,
    service: AgentService = Depends(get_agent_service),
) -> dict[str, Any]:
    """Restart an agent using saved launch_config.

    Stops the running agent (if idle/waiting), then re-launches with saved
    parameters. Optional overrides can be passed in the request body.
    Returns 404 if no launch_config exists, 409 if actively running.
    """
    # Load saved config
    config = load_launch_config(agent_name)
    if config is None:
        raise HTTPException(
            status_code=404,
            detail=f"No previous launch configuration for '{agent_name}'. Start it normally first.",
        )

    # Stop if running
    try:
        await stop_agent_process(agent_name, service)
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except TimeoutError as e:
        raise HTTPException(status_code=500, detail=str(e))

    # Apply overrides
    model = request.model if request and request.model else None
    provider = request.provider if request and request.provider else None
    clean = request.clean if request else False

    # Spawn new process
    result = await spawn_agent_process(
        agent_name, config,
        model=model, provider=provider, clean=clean,
    )
    if not result.get("success"):
        raise HTTPException(status_code=500, detail=result.get("error", "Spawn failed"))
    return result


@router.post("/batch")
async def batch_operation(
    request: BatchRequest,
    service: AgentService = Depends(get_agent_service),
) -> dict[str, Any]:
    """Execute batch operations on multiple agents.

    Supported actions: 'stop', 'restart'.
    """
    if request.action not in ("stop", "restart"):
        raise HTTPException(status_code=400, detail=f"Invalid action: '{request.action}'. Use 'stop' or 'restart'.")

    if not request.agent_names:
        raise HTTPException(status_code=400, detail="agent_names list cannot be empty")

    results: dict[str, dict[str, Any]] = {}

    for name in request.agent_names:
        try:
            if request.action == "stop":
                result = await stop_agent_process(name, service)
                results[name] = result
            elif request.action == "restart":
                config = load_launch_config(name)
                if config is None:
                    results[name] = {"success": False, "error": f"No launch config for '{name}'"}
                    continue
                await stop_agent_process(name, service)
                result = await spawn_agent_process(name, config)
                results[name] = result
        except RuntimeError as e:
            results[name] = {"success": False, "error": str(e)}
        except TimeoutError as e:
            results[name] = {"success": False, "error": str(e)}
        except Exception as e:
            results[name] = {"success": False, "error": str(e)}

    return {"results": results}
