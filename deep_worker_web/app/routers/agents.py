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
import re
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import PlainTextResponse

from app.schemas.agent_schemas import (
    AgentInfo,
    BatchAcceptedResponse,
    BatchRequest,
    JobAcceptedResponse,
    MessageRequest,
    RestartRequest,
    StopRequest,
    StopTaskManagerRequest,
)
from app.services.agent_service import AgentRefusedError, AgentService
from deep_worker_cli.storage_paths import deep_worker_home, read_agent_file

# Provider↔model compatibility. This used to be a hand-copied duplicate of
# the CLI predicate, kept local because importing ``deep_worker_cli.config``
# drags in langchain and breaks the web unit-test environment. The two copies
# then drifted: both carried a stale Claude Code whitelist, so a saved
# ``fable`` model failed the check here and the web spawned the agent with no
# ``--model`` at all, silently falling back to sonnet (observed in a past
# incident).
#
# ``deep_worker_cli.model_predicate`` is a dependency-free leaf module created
# for exactly this: one implementation, importable from both sides. It has no
# heavy or order-sensitive imports, so it belongs in the normal import block.
# Do not reintroduce a local copy.
from deep_worker_cli.model_predicate import (
    model_can_belong_to_provider as _model_can_belong_to_provider,
)

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
        error = response.get("error", "Unknown error")
        # "Failed to connect" is the canonical marker the service
        # uses when the RPyC connection is unavailable. Translate
        # that into 503 Service Unavailable so the client can
        # distinguish "agent down, retry later" from a real 500.
        if "Failed to connect" in str(error) or "agent_disconnected" in str(error).lower():
            raise HTTPException(
                status_code=503,
                detail={
                    "error": "agent_disconnected",
                    "agent_name": agent_name,
                    "message": str(error),
                },
            )
        raise HTTPException(
            status_code=500,
            detail=f"Failed to send message: {error}",
        )

    return {
        "success": True,
        "message": "Message sent successfully",
        "agent_name": agent_name,
    }


@router.post("/{agent_name}/summarize", status_code=202)
async def summarize_agent(
    agent_name: str,
    service: AgentService = Depends(get_agent_service),
) -> dict[str, Any]:
    agent = await service.get_agent(agent_name)
    if agent is None:
        raise HTTPException(
            status_code=404,
            detail=f"Agent '{agent_name}' not found",
        )
    response = await service.summarize_agent(agent_name)
    if not response.get("success"):
        raise HTTPException(status_code=503, detail=response.get("error"))
    return response


@router.get("/{agent_name}/stats")
async def get_agent_stats(
    agent_name: str,
    service: AgentService = Depends(get_agent_service),
) -> dict[str, Any]:
    """Get token/cache usage statistics for an agent (§6.1).

    Contract (DESIGN_AGENT_TOKEN_STATS.md §6.1):

    - The "known agent" criterion is the same as ``list_agents`` (the
      web registry). Unknown agent → 404.
    - Known agent with no stats file → 200 with zero-filled aggregates;
      the GET never creates a file/directory.
    - Invalid agent name → 400 (ValueError from ``validate_agent_name``
      is a client error, not a 500).

    The reader/aggregator is the single shared implementation in
    ``deep_worker_cli.stats.token_stats`` (MAJOR-1 v2) — no duplicated
    formulas. The path is resolved via ``storage_paths.agent_root``
    (no raw path join from the URL).
    """
    # Client error first: a malformed name must never reach path
    # resolution.
    from deep_worker_cli.storage_paths import (  # noqa: PLC0415
        agent_root,
        validate_agent_name,
    )

    try:
        validate_agent_name(agent_name)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid agent name: {agent_name!r}",
        ) from None

    # Known-agent criterion: the web registry (same source as
    # list_agents). Unknown → 404.
    agent = await service.get_agent(agent_name)
    if agent is None:
        raise HTTPException(
            status_code=404,
            detail=f"Agent '{agent_name}' not found",
        )

    from deep_worker_cli.stats.token_stats import stats_for_agent  # noqa: PLC0415

    data_dir = agent_root(agent_name, create=False)
    return stats_for_agent(agent_name, data_dir).to_dict()


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
    from rpyc_async.core.netref import async_call
    from rpyc_async.utils.helpers import fire_and_forget_call

    # Distinguish two failure modes that used to look identical:
    #
    # * **Disconnected**: ``is_connected`` is False AND we cannot
    #   re-establish — this is a transient agent-unreachable state.
    #   Respond 503 so the UI shows a retry banner.
    # * **No TaskManager feature**: we ARE connected, but the agent
    #   was started without ``--task-manager``, so ``root`` has no
    #   ``stop_task_manager`` method. This is a normal "nothing to
    #   do" state — respond 200 (pre-existing contract).
    # NON-BLOCKING: ensure_connecting returns current connectivity and
    # kicks a single-flight background connect if needed — it never waits
    # on a slow/stuck peer (see a related internal incident analysis, not
    # included here).
    is_connected = await service.ensure_connecting(agent_name)

    root = service.get_root(agent_name) if is_connected else None

    if root is None:
        # Connection genuinely unavailable — this is the disconnect
        # case.
        raise HTTPException(
            status_code=503,
            detail={
                "error": "agent_disconnected",
                "agent_name": agent_name,
                "message": (
                    "TaskManager stop cannot be delivered: agent is "
                    "not connected (no live RPyC session)"
                ),
            },
        )

    if not hasattr(root, "stop_task_manager"):
        # Connected, but the feature isn't installed on this agent.
        # Keep the pre-existing 200 contract — clients (and the
        # integration test at
        # tests/integration/test_api_agents_integration.py::
        # test_stop_task_manager) expect a success-shaped "nothing
        # to do" response here, not a 503.
        return {
            "success": True,
            "message": (
                "TaskManager stop requested (feature may not be "
                "available on this agent build)"
            ),
            "agent_name": agent_name,
            "force": request.force,
        }

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
    fire_and_forget_call(
        lambda: async_call(root, "stop_task_manager", force=request.force),
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


# ---------------------------------------------------------------------------
# Full Dialog = static, click-time snapshot (DESIGN_FULL_DIALOG_STATIC_SNAPSHOT.md)
# ---------------------------------------------------------------------------
# The Full Dialog tab is ONE static ``text/plain`` document produced at the
# moment the operator clicks the button. It carries no script, is never
# streamed, never served from cache, and never updates itself. The two-line
# banner below makes that contract visible in the tab (agent, verbosity, the
# click-time stamp, and the words "never updates").
#
# Bounded wait: the service waits at most 30 s for the agent's history
# (agent_service.get_agent_dialog_formatted). If the agent is busy/wedged the
# tab must show WHY (504 + reason) instead of a generic 500, and a fetch that
# is slow but succeeds leaves a bounded breadcrumb (elapsed time only, never
# content) so slow-agent episodes are diagnosable from the web log.

#: Fetch+render longer than this (seconds) logs one WARNING breadcrumb.
SLOW_DIALOG_WARN_SECONDS: float = 5.0

#: Bound enforced inside ``AgentService.get_agent_dialog_formatted``.
DIALOG_HISTORY_TIMEOUT_SECONDS: int = 30


def _snapshot_now() -> datetime:
    """Click-time stamp: server local time with UTC offset (monkeypatchable)."""
    return datetime.now().astimezone()


def _snapshot_banner(agent_name: str, verbosity: str, taken_at: datetime | None = None) -> str:
    """Two banner lines + blank separator that prefix every Full Dialog body."""
    taken = (taken_at or _snapshot_now()).isoformat(timespec="seconds")
    return (
        f"=== Full Dialog snapshot: agent={agent_name}  verbosity={verbosity}  "
        f"taken={taken} ===\n"
        "=== Static snapshot \u2014 this page never updates. "
        "Click \"Full Dialog\" again for a fresh one. ===\n"
        "\n"
    )


_SNAPSHOT_HEADERS: dict[str, str] = {
    # Every click is a fresh snapshot; a browser must never reuse an older
    # cached copy (would show something other than the click-time state).
    "Cache-Control": "no-store",
    # Pin text/plain rendering: no MIME sniffing → no HTML/script path, ever.
    "X-Content-Type-Options": "nosniff",
}

# Characters allowed in the download filename. Anything else becomes "_" so
# the value can never break out of the quoted Content-Disposition parameter
# (no quotes, no CR/LF, no ";"). See DESIGN_FULL_DIALOG_ACTION_MODE.md §3.3.
_FILENAME_SAFE = re.compile(r"[^A-Za-z0-9_.-]")


def _download_filename(agent_name: str, verbosity: str, taken: datetime) -> str:
    """``dialog_<agent>_<verbosity>_<YYYYMMDD-HHMMSS>.txt`` with a sanitised agent."""
    safe_agent = _FILENAME_SAFE.sub("_", agent_name) or "agent"
    stamp = taken.strftime("%Y%m%d-%H%M%S")
    return f"dialog_{safe_agent}_{verbosity}_{stamp}.txt"


@router.get("/{agent_name}/dialog-text", response_class=PlainTextResponse)
async def get_dialog_text(
    agent_name: str,
    verbosity: Literal["final", "text", "full", "system_prompt", "tools"] = "final",
    download: bool = False,
    timestamps: Literal["local", "utc"] = "local",
    service: AgentService = Depends(get_agent_service),
) -> PlainTextResponse:
    """Get full dialog text for an agent as plain text.

    ``download=1`` returns the very same bytes with an additional
    ``Content-Disposition: attachment`` header so the browser saves a
    ``.txt`` instead of rendering a tab. This exists for screen-reader
    users: rendering a huge plain-text document forces the browser to
    build a full accessibility tree and serve the reader synchronously on
    its UI thread (measured 57 s / 152 812 nodes for 150 KB), freezing the
    whole browser. See ``docs/DESIGN_FULL_DIALOG_ACTION_MODE.md``.

    The response is a STATIC SNAPSHOT taken at request (button-click) time:
    a single ``text/plain`` document with ``Cache-Control: no-store``,
    ``X-Content-Type-Options: nosniff`` and an explicit ``Content-Length``,
    prefixed by a two-line banner naming the agent, the verbosity and the
    click-time stamp. It never streams and never updates itself. See
    ``docs/DESIGN_FULL_DIALOG_STATIC_SNAPSHOT.md``.

    Returns ALL messages (no artificial limit). Verbosity controls how
    each turn renders:

    - ``final`` (default): one assistant block per turn, the LAST text-bearing
      AIMessage of that turn.
    - ``text``: every text-bearing AIMessage of the turn.
    - ``full``: like ``text`` plus compact tool-call summaries.

    Four sources of user-side messages are all rendered as first-class turns
    at every verbosity:

    - Direct user input → ``--- User ---``
    - TaskManager prompts → ``--- TaskManager ---``
    - Inter-agent messages → ``--- From agent:<name> ---``
    - Context-compaction summarization injections → ``--- Summarization ---``

    No message is filtered out — the summarization node renders with its
    own explicit header so readers can tell synthetic context-management
    events apart from real user input without losing the "why did the
    agent pivot?" signal after a compaction. See
    ``docs/DESIGN_SUMMARIZATION_VISIBLE.md``.

    Every block header carries the message's own ``created_at``, e.g.
    ``--- Assistant 2026-08-29 14:29:22 +02:00 ---``. ``timestamps=local``
    (default) renders in the host timezone, ``timestamps=utc`` in UTC; the
    zone is always printed, so a rendering cannot be misread as another
    clock. Messages written before timestamps existed simply render without
    one rather than showing a made-up time.

    Args:
        agent_name: Name of the agent.
        verbosity: Rendering verbosity (validated by FastAPI; unknown
            values cause HTTP 422).
        timestamps: ``local`` (default) | ``utc`` — zone of the per-message
            stamp in block headers.

    Raises:
        HTTPException: 404 if agent unknown, 503 if not connected yet,
            504 if the agent did not answer within the 30 s bound,
            500 on any other failure.
    """
    logger.debug(f"Getting dialog text for agent: {agent_name} (verbosity={verbosity})")

    # Check if agent exists
    agent = await service.get_agent(agent_name)
    if agent is None:
        raise HTTPException(
            status_code=404,
            detail=f"Agent '{agent_name}' not found",
        )

    # NON-BLOCKING connect gate: never wait on a slow/stuck peer. If the
    # agent is not connected yet, a single-flight background connect is
    # kicked off and we return 503 immediately (UI shows a retry banner)
    # instead of hanging this handler (see a related internal incident
    # analysis, not included here).
    if not await service.ensure_connecting(agent_name):
        raise HTTPException(
            status_code=503,
            detail=f"Agent '{agent_name}' not connected (connecting…)",
        )

    # Get formatted dialog text from agent (bounded wait inside the service).
    started = time.perf_counter()
    try:
        dialog_text = await service.get_agent_dialog_formatted(
            agent_name, verbosity=verbosity, timestamps=timestamps
        )
    except asyncio.TimeoutError:
        # Agent busy/wedged: the bounded wait expired. Tell the operator WHY
        # and that the main UI is unaffected — the tab must not look like a
        # silent hang. Must precede the generic ``except Exception`` below.
        elapsed = time.perf_counter() - started
        logger.warning(
            "Full Dialog for %s: agent did not answer get_history within %ss "
            "(elapsed %.1fs, verbosity=%s)",
            agent_name, DIALOG_HISTORY_TIMEOUT_SECONDS, elapsed, verbosity,
        )
        raise HTTPException(
            status_code=504,
            detail=(
                f"Agent '{agent_name}' did not answer get_history within "
                f"{DIALOG_HISTORY_TIMEOUT_SECONDS}s (agent busy or wedged). "
                "The main UI is unaffected; retry later."
            ),
        )
    except AgentRefusedError as e:
        # The agent answered, and the answer was a refusal. That is an
        # upstream failure, not a fault of this server: 500 sent operators
        # hunting a web bug that was not there. 502 says "the peer refused",
        # and the detail carries the agent's own words verbatim.
        #
        # Deliberately a dedicated type, not bare ValueError: any other
        # ValueError from this path is a genuine server-side bug and must
        # keep returning 500 (see test_generic_error_still_500).
        logger.warning("Dialog text for %s refused by agent: %s", agent_name, e)
        raise HTTPException(
            status_code=502,
            detail=f"Agent '{agent_name}' could not return its dialog: {e}",
        )
    except Exception as e:
        logger.error(f"Error getting dialog text for {agent_name}: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error getting dialog: {str(e)}",
        )

    elapsed = time.perf_counter() - started
    if elapsed > SLOW_DIALOG_WARN_SECONDS:
        # Bounded breadcrumb: sizes/times only, never content.
        logger.warning(
            "Full Dialog for %s was slow: %.1fs (verbosity=%s, %d chars)",
            agent_name, elapsed, verbosity, len(dialog_text),
        )

    taken = _snapshot_now()
    headers = dict(_SNAPSHOT_HEADERS)
    if download:
        headers["Content-Disposition"] = (
            f'attachment; filename="{_download_filename(agent_name, verbosity, taken)}"'
        )
    return PlainTextResponse(
        _snapshot_banner(agent_name, verbosity, taken) + dialog_text,
        headers=headers,
    )


# ===========================================================================
# Agent Stop / Restart / Batch (fire-and-forget)
# ===========================================================================
#
# These endpoints are the fire-and-forget rewrite described in
# ``docs/DESIGN_FIRE_AND_FORGET_LIFECYCLE.md``.  The HTTP handler does only:
#
#   1. validate input
#   2. ask AgentLifecycleService to schedule the op (returns a request_id)
#   3. return 202 Accepted with {"request_id"} — immediately
#
# All progress (started / escalated / terminated / completed / failed) is
# broadcast via WebSocket with that request_id as the correlation key.
#
# Event-driven internals live in ``app/services/lifecycle.py``:
#   * ``os.pidfd_open`` + ``loop.add_reader`` for process exit (no polling)
#   * ``inotify`` + ``loop.add_reader`` for spawn registry wait (no polling)
#   * one ``loop.call_later`` per grace / hard-timeout
#
# NOTE: the old helpers ``stop_agent_process`` and ``_wait_for_agent_in_registry``
# have been removed — they violated the no-polling / no-blocking contract and
# had no other callers.  ``load_launch_config`` and ``_spawn_subprocess`` are
# kept here as pure-function helpers because ``lifecycle.py`` depends on them
# (see ``AgentLifecycleService._drive_spawn``).


def load_launch_config(agent_name: str) -> dict[str, Any] | None:
    """Load launch_config from session_metadata.json."""
    import json

    raw = read_agent_file(agent_name, "session_metadata.json")
    if raw is None:
        return None
    try:
        data = json.loads(raw)
        cfg = data.get("launch_config")
        return cfg if isinstance(cfg, dict) else None
    except (json.JSONDecodeError, OSError):
        return None


def _spawn_subprocess(
    agent_name: str,
    config: dict[str, Any],
    *,
    model: str | None = None,
    provider: str | None = None,
    clean: bool = False,
) -> subprocess.Popen[bytes]:
    """Launch the agent daemon via subprocess. Returns immediately.

    This is a pure *fire* helper — it does NOT wait for the registry.
    ``AgentLifecycleService._drive_spawn`` handles the event-driven wait
    via inotify.
    """
    effective_provider = provider or config.get("provider", "deepseek")
    cmd = [
        "python3", "-m", "deep_worker_cli.cli.agent_command",
        agent_name,
        "--provider", effective_provider,
        "--dir", config.get("working_dir", "."),
    ]
    # Resolve the model: explicit arg wins, else the saved config value.
    # ``"unknown"`` is the legacy poison sentinel — NEVER a real model —
    # so treat it (and an empty/None value) as "no model specified" and
    # omit ``--model`` entirely. The agent then picks its provider
    # default (codex CLI chooses its own current default). Passing
    # ``--model unknown`` is exactly what sent an agent to ERROR with a
    # Codex HTTP 400 on every web restart in a past incident.
    saved_provider = config.get("provider")
    resolved_model = model
    if resolved_model is None and saved_provider == effective_provider:
        resolved_model = config.get("model")
    if _model_can_belong_to_provider(effective_provider, resolved_model):
        cmd += ["--model", resolved_model]
    elif resolved_model:
        # A non-empty model was discarded. Say so: silently substituting the
        # provider default is the defect from that incident. This is a SERVER-side
        # warning — the HTTP response and the browser UI are deliberately
        # unchanged; surfacing it to the operator's browser would mean
        # extending the endpoint contract, which is out of scope here.
        logger.warning(
            "Dropping model %r for agent %r: not compatible with provider %r. "
            "The agent will use that provider's default instead.",
            resolved_model,
            agent_name,
            effective_provider,
        )
    # Preserve every persisted boolean option across restart.
    #
    # We iterate the single source of truth (``_BOOLEAN_FLAGS`` in
    # agent_command) rather than maintain a parallel hand-coded sequence
    # of ``if config.get(...)`` lines. The opt-in pattern is what dropped
    # ``-m`` (history_memory) on the 2026-06-05 web restart and would have
    # silently dropped every future boolean too. The coverage test
    # ``tests/unit_tests/cli/test_boolean_flags_registry.py`` enforces that
    # adding a new bool to ``_PARSER_DEFAULTS`` requires classifying it,
    # so this loop automatically picks it up.
    #
    # ``clean`` is intentionally separate: it lives in
    # ``_NON_PRESERVED_BOOLEAN_OPTIONS`` and is governed exclusively by the
    # explicit ``clean=`` argument to this function, never by saved config.
    # See ``deep_worker_web/docs/DESIGN_RESTART_OPTION_PRESERVATION.md``.
    from deep_worker_cli.cli.agent_command import _BOOLEAN_FLAGS
    for _dest, _true_flag, _false_flag in _BOOLEAN_FLAGS:
        if config.get(_dest):
            cmd.append(_true_flag)

    # External MCP ad-hoc layers are persistent launch options just like
    # provider/model. Preserve each occurrence across a web-driven restart.
    for _path in config.get("mcp_config_paths", []) or []:
        cmd += ["--mcp-config", str(_path)]
    for _server in config.get("mcp_server_inline", []) or []:
        cmd += ["--mcp-server", str(_server)]

    _context_limit = config.get("context_limit")
    if isinstance(_context_limit, int):
        cmd += ["--context_limit", str(_context_limit)]

    if clean:
        cmd.append("--clean")

    # Which codebase to relaunch the agent on?
    #
    # The agent's ORIGINAL tree is authoritative, NOT the web server's. An
    # agent launched with ``--dev`` (dev codebase) must be RESTARTED on the
    # dev codebase even when the restart is driven by the PROD web UI (:7887),
    # and vice-versa. We persist that choice as ``launch_config['dev_tree']``
    # (build_launch_config), so honor it here. Previously this hard-coded the
    # tree to the web server's OWN checkout (``__file__``), so restarting a
    # ``--dev`` agent from the prod web UI silently dropped ``--dev`` and
    # respawned it on prod — the bug this fixes (2026-05-23). It also meant the
    # dev web server stamped DEEP_WORKER_DEV_TREE=1 onto EVERY agent it spawned,
    # even non-dev ones. The CLI ``deep-worker --restart`` launcher already
    # auto-detects dev_tree; this brings the web path in line. See
    # docs/DESIGN_DEV_FLAG.md.
    #
    # __file__ here = <WEB_TREE>/deep_worker_web/app/routers/agents.py;
    # parent x4 = <WEB_TREE> — the fallback tree for non-dev agents (prod when
    # the web server runs from the production tree).
    _web_tree_root = Path(__file__).resolve().parent.parent.parent.parent
    if config.get("dev_tree"):
        # DEEP_WORKER_DEV_TREE_ROOT_OVERRIDE mirrors the bash launcher's
        # testing hook so tests don't need a real dev tree.
        _tree_root = Path(
            os.environ.get(
                "DEEP_WORKER_DEV_TREE_ROOT_OVERRIDE", "/nonexistent/dev-tree"
            )
        )
    else:
        _tree_root = _web_tree_root
    env = os.environ.copy()
    env["PYTHONPATH"] = ":".join([
        str(_tree_root / "deep_worker" / "deep_worker_cli"),
        str(_tree_root / "deep_worker" / "deepagents"),
        "/path/to/rpyc_async",
    ])
    env["DEEP_WORKER_ORIGINAL_CWD"] = config.get("working_dir", ".")
    # Mark the spawned agent as a dev-tree agent iff it actually IS one (its
    # persisted dev_tree), NOT merely because this web server runs from the
    # dev checkout. build_launch_config reads DEEP_WORKER_DEV_TREE=1 to
    # re-persist dev_tree across the restart. Setting it for a dev agent
    # restarted from prod (and clearing any inherited marker for a non-dev
    # agent) is what keeps dev_tree stable across restarts.
    if config.get("dev_tree"):
        env["DEEP_WORKER_DEV_TREE"] = "1"
    else:
        env.pop("DEEP_WORKER_DEV_TREE", None)
    env["DEEP_WORKER_TREE_ROOT"] = str(_tree_root)

    _logged_cmd: list[str] = []
    _redact_next = False
    for _arg in cmd:
        if _redact_next:
            _logged_cmd.append("<redacted-inline-mcp-server>")
            _redact_next = False
            continue
        _logged_cmd.append(_arg)
        _redact_next = _arg == "--mcp-server"
    logger.info("Spawning agent: %s", " ".join(_logged_cmd))

    log_dir = deep_worker_home() / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    err_log = log_dir / f"{agent_name}.spawn.log"

    err_file = open(err_log, "w")  # kept open for subprocess stderr redirection
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=err_file,
        env=env,
        start_new_session=True,
    )
    return proc


@router.post("/{agent_name}/stop", response_model=JobAcceptedResponse, status_code=202)
async def stop_agent(
    agent_name: str,
    request: StopRequest | None = None,
    service: AgentService = Depends(get_agent_service),
) -> JobAcceptedResponse:
    """Fire-and-forget agent stop.

    Accepts instantly; the kill pipeline runs as a background task.
    Watch WS events (``agent_stop_started``, ``agent_stop_escalated``,
    ``agent_terminated``, ``agent_stop_completed`` / ``agent_stop_failed``)
    correlated by ``request_id``.

    ``force=True`` — SIGKILL immediately (like ``dwcli --stop`` panic path).
    Otherwise SIGTERM → wait ``grace_seconds`` → SIGKILL.

    Note: we intentionally do NOT 404 when the agent isn't in the in-memory
    cache — the lifecycle service does its own lookup against the live
    registry and broadcasts ``agent_stop_failed`` if the agent is truly
    unknown. That keeps the contract uniform (every button click gets a
    request_id it can track).
    """
    params = request if request is not None else StopRequest(grace_seconds=5.0, force=False)
    lifecycle = service.get_lifecycle()
    request_id = await lifecycle.request_stop(
        agent_name,
        grace_seconds=params.grace_seconds,
        force=params.force,
    )
    return JobAcceptedResponse(
        success=True,
        request_id=request_id,
        message="agent_stop scheduled; watch WebSocket for progress",
        agent_name=agent_name,
    )


@router.post("/{agent_name}/restart", response_model=JobAcceptedResponse, status_code=202)
async def restart_agent(
    agent_name: str,
    request: RestartRequest | None = None,
    service: AgentService = Depends(get_agent_service),
) -> JobAcceptedResponse:
    """Fire-and-forget restart: stop → spawn.

    Returns immediately with a ``request_id``; WS events per phase
    (``agent_stop_*``, ``agent_spawn_*``, ``agent_restart_completed`` /
    ``agent_restart_failed``).
    """
    params = request if request is not None else RestartRequest(
        model=None, provider=None, clean=False, grace_seconds=5.0, force=False,
    )
    lifecycle = service.get_lifecycle()
    request_id = await lifecycle.request_restart(
        agent_name,
        grace_seconds=params.grace_seconds,
        force=params.force,
        model=params.model,
        provider=params.provider,
        clean=params.clean,
    )
    return JobAcceptedResponse(
        success=True,
        request_id=request_id,
        message="agent_restart scheduled; watch WebSocket for progress",
        agent_name=agent_name,
    )


@router.post("/batch", response_model=BatchAcceptedResponse, status_code=202)
async def batch_operation(
    request: BatchRequest,
    service: AgentService = Depends(get_agent_service),
) -> BatchAcceptedResponse:
    """Fire-and-forget batch stop / restart.

    Fans out across agents in parallel — HTTP returns once all child
    jobs are *scheduled*, not completed. Each child has its own
    ``request_id``; follow them individually via WS, plus aggregate
    ``batch_started`` / ``batch_progress`` / ``batch_completed``.
    """
    if request.action not in ("stop", "restart"):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid action: '{request.action}'. Use 'stop' or 'restart'.",
        )
    if not request.agent_names:
        raise HTTPException(
            status_code=400,
            detail="agent_names list cannot be empty",
        )

    params: dict[str, Any] = {
        "grace_seconds": request.grace_seconds,
        "force": request.force,
        "model": request.model,
        "provider": request.provider,
        "clean": request.clean,
    }
    lifecycle = service.get_lifecycle()
    handle = await lifecycle.request_batch(
        action=request.action,
        agent_names=list(request.agent_names),
        params=params,
    )
    jobs = [
        JobAcceptedResponse(
            success=True,
            request_id=j["request_id"],
            message=f"{request.action} scheduled",
            agent_name=j["agent_name"],
        )
        for j in handle["jobs"]
    ]
    return BatchAcceptedResponse(
        success=True,
        request_id=handle["request_id"],
        action=handle["action"],
        jobs=jobs,
    )
