"""Two-real-agents A2A end-to-end fixture (no mocks).

Spawns TWO real OS subprocesses (via multiprocessing 'spawn'), each running a
real A2A server (Starlette + uvicorn + a2a-sdk + DeepWorkerAgentExecutor) over
real HTTP. Each subprocess registers itself in the real ProcessRegistry with
its real ``a2a_url`` (RC-6), exactly the same way a production AgentProcess
does. The parent (pytest) talks to nobody directly — agentA finds agentB by
NAME through the registry and sends the actual A2A JSON-RPC over the wire.

What's REAL here (covered by these e2e tests):
* Two OS processes.
* Real ProcessRegistry on disk (with file lock).
* Real A2A HTTP server on each side (real port, real /well-known/agent-card).
* Real outbound A2A client + ``a2a_*`` LLM tools.
* Real registry-based name-to-URL discovery between processes.

What's stubbed (out of scope for A2A integration):
* The LLM turn engine. Each subprocess uses a deterministic ``_handle_invoke``
  that interprets the inbound text as a tiny DSL (``upper:foo`` → ``FOO``;
  ``echo:foo`` → ``foo``; arbitrary text → ``ack: <text>``). This isolates
  what we want to verify (the A2A integration) from LLM cost / flakiness,
  which is independently exercised by ``tests/integration_ai/``.

Cleanup: SIGTERM to each subprocess on teardown; safety-belt unregister from
the registry and rmtree of any per-agent data dirs that may have been
created. Names enforced to start with ``test_`` so a misconfigured fixture
can never touch a production agent.
"""

from __future__ import annotations

import contextlib
import logging
import multiprocessing
import os
import shutil
import signal
import socket
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Generator

logger = logging.getLogger(__name__)


_READY_TIMEOUT_SEC = 30.0
_SHUTDOWN_GRACE_SEC = 5.0


@dataclass
class A2AEndpoint:
    """What the parent test needs to talk to a spawned agent."""

    name: str
    pid: int
    a2a_url: str
    rpyc_port: int  # dummy bound socket; no RPyC service runs there


def _ensure_sys_path() -> None:
    """Subprocess-side sys.path bootstrap (mirrors integration_ai fixture)."""
    # tests/deepworker_a2a/ -> tests/ -> deep_worker_cli/ -> deep_worker/
    here = Path(__file__).resolve()
    deep_worker_cli = str(here.parents[2])
    deep_worker_dir = str(here.parents[3])
    deepagents = str(here.parents[3] / "deepagents")
    for p in (deep_worker_cli, deep_worker_dir, deepagents):
        if p not in sys.path:
            sys.path.insert(0, p)


def _stub_handle_invoke_factory(agent_name: str):
    """Return a stub `_handle_invoke` coroutine that imitates a turn engine.

    Supported micro-DSL (lets tests pin exact behaviour without an LLM):
    * ``upper:<text>``  → returns ``<TEXT UPPERCASED>``
    * ``echo:<text>``   → returns ``<text>`` verbatim
    * ``slow:<n>:<t>``  → emits ``<n>`` streamed chunks then a final result
    * anything else     → returns ``"ack:" + text + " (from " + agent_name + ")"``
    """

    async def _handle_invoke(  # noqa: D401 — matches AgentProcess signature
        command,
        connection=None,
        notify_callback=None,
        approval_callback=None,
        thread_id_override=None,
    ):
        text = (command or {}).get("message", "") or ""

        if text.startswith("upper:"):
            answer = text[len("upper:") :].upper()
        elif text.startswith("echo:"):
            answer = text[len("echo:") :]
        elif text.startswith("slow:"):
            try:
                _, n_str, body = text.split(":", 2)
                n = max(0, int(n_str))
            except (ValueError, IndexError):
                n = 0
                body = text
            for i in range(n):
                if notify_callback is not None:
                    await notify_callback(
                        {"type": "assistant", "role": "assistant",
                         "content": f"chunk-{i+1}/{n} "}
                    )
            answer = body
        else:
            answer = f"ack: {text} (from {agent_name})"

        # Always emit ONE final streamed chunk so streaming clients see the
        # full answer as artifact text too (the non-streaming path reads the
        # final task.status.message via get_message_text — see client.py).
        if notify_callback is not None:
            await notify_callback(
                {"type": "assistant", "role": "assistant", "content": answer}
            )
        return {"success": True,
                "response": {"role": "assistant", "content": answer}}

    return _handle_invoke


class _StubAgentProcess:
    """Minimal object satisfying the AgentExecutor contract.

    Mirrors only what ``DeepWorkerAgentExecutor`` touches:
    ``name``, ``_handle_invoke(...)``, ``cancel_current_worker_task()``.
    """

    def __init__(self, name: str):
        self.name = name
        self._cancelled = False
        self._handle_invoke = _stub_handle_invoke_factory(name)

    def cancel_current_worker_task(self) -> bool:
        self._cancelled = True
        return True


def _bind_dummy_rpyc_socket() -> tuple[socket.socket, int]:
    """Bind 127.0.0.1:0 so we have a real port to advertise in the registry.

    The registry contract REQUIRES ``rpyc_port`` and verifies the PID is
    alive, but A2A discovery only reads ``a2a_url`` / ``a2a_port`` — nothing
    in these e2e tests connects to the RPyC port. We just hold the socket
    open for the agent's lifetime so the port stays reserved.
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("127.0.0.1", 0))
    sock.listen(1)
    return sock, sock.getsockname()[1]


def _agent_subprocess_entry(
    agent_name: str, ready_pipe, data_dir: str
) -> None:
    """Subprocess body. Boots a real A2A server + registers; blocks on SIGTERM."""
    _ensure_sys_path()

    import asyncio
    from datetime import datetime

    from deep_worker_cli.process.registry import ProcessRegistry
    from deep_worker_cli.types import AgentProcessInfo
    from deep_worker.providers.deepworker_a2a.config import A2AConfig
    from deep_worker.providers.deepworker_a2a.server import start_a2a_server

    # ProcessRegistry insists the pid be alive AND the cmdline contain one of
    # {deep-worker, deep_worker, pytest, python}. multiprocessing spawn uses
    # the current python interpreter, so the child's argv[0] is /usr/bin/python
    # — the registry liveness check accepts that.

    rpyc_sock, rpyc_port = _bind_dummy_rpyc_socket()

    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True, mode=0o755)
    pid_file = data_path / f"{agent_name}.pid"
    pid_file.write_text(str(os.getpid()))

    stub_ap = _StubAgentProcess(agent_name)

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    cfg = A2AConfig(enabled=True, host="127.0.0.1", port=0)
    handle = loop.run_until_complete(
        start_a2a_server(stub_ap, cfg, agent_name=agent_name, version="1.0.0")
    )
    # uvicorn flips ``started`` once the loop is serving requests.
    deadline = time.monotonic() + 10
    while not getattr(handle.server, "started", False):
        if time.monotonic() > deadline:
            break
        loop.run_until_complete(asyncio.sleep(0.02))

    registry = ProcessRegistry()
    info = AgentProcessInfo(
        name=agent_name,
        pid=os.getpid(),
        pid_file=pid_file,
        working_dir=str(data_path),
        started_at=datetime.now(),
        rpyc_port=rpyc_port,
        a2a_port=handle.port,
        a2a_url=handle.url,
    )
    registry.register(info)

    # Signal the parent that we're up and reachable.
    try:
        ready_pipe.send({"pid": os.getpid(), "a2a_url": handle.url,
                         "rpyc_port": rpyc_port})
        ready_pipe.close()
    except Exception:  # noqa: BLE001
        pass

    # Block until SIGTERM. We use a simple signal-driven wait — no polling.
    stop_event = asyncio.Event()

    def _on_sig(*_args):
        loop.call_soon_threadsafe(stop_event.set)

    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, _on_sig)

    async def _wait_and_shutdown() -> None:
        await stop_event.wait()
        with contextlib.suppress(Exception):
            await handle.shutdown()
        with contextlib.suppress(Exception):
            registry.unregister(agent_name)

    try:
        loop.run_until_complete(_wait_and_shutdown())
    finally:
        with contextlib.suppress(Exception):
            rpyc_sock.close()
        with contextlib.suppress(Exception):
            loop.close()


def _spawn_agent(agent_name: str, data_root: Path):
    if not agent_name.startswith("test_"):
        raise ValueError(
            f"Agent name must start with 'test_' for safety; got {agent_name!r}"
        )
    ctx = multiprocessing.get_context("spawn")
    parent_pipe, child_pipe = ctx.Pipe(duplex=False)
    data_dir = str(data_root / agent_name)
    proc = ctx.Process(
        target=_agent_subprocess_entry,
        args=(agent_name, child_pipe, data_dir),
        name=f"a2a-e2e-{agent_name}",
        daemon=True,
    )
    proc.start()
    child_pipe.close()  # parent only reads
    return proc, parent_pipe


def _wait_ready(parent_pipe, proc, agent_name: str) -> dict:
    deadline = time.monotonic() + _READY_TIMEOUT_SEC
    while time.monotonic() < deadline:
        if parent_pipe.poll(0.1):
            return parent_pipe.recv()
        if not proc.is_alive():
            raise RuntimeError(
                f"Agent {agent_name!r} subprocess died before signalling ready "
                f"(exitcode={proc.exitcode})"
            )
    raise TimeoutError(
        f"Agent {agent_name!r} did not signal ready within {_READY_TIMEOUT_SEC}s"
    )


def _terminate(proc) -> None:
    if not proc.is_alive():
        return
    try:
        proc.terminate()  # SIGTERM — subprocess handles it and exits cleanly
    except Exception:  # noqa: BLE001
        return
    proc.join(timeout=_SHUTDOWN_GRACE_SEC)
    if proc.is_alive():
        try:
            proc.kill()
        except Exception:  # noqa: BLE001
            pass
        proc.join(timeout=2.0)


def _safety_cleanup(agent_name: str, data_root: Path) -> None:
    """Best-effort: drop stale registry entry + remove the per-agent data dir."""
    try:
        from deep_worker_cli.process.registry import ProcessRegistry

        ProcessRegistry().unregister(agent_name)
    except Exception:  # noqa: BLE001
        pass
    d = data_root / agent_name
    if d.exists():
        with contextlib.suppress(OSError):
            shutil.rmtree(d)


@contextlib.contextmanager
def two_a2a_agents(
    name_a: str = "test_a2a_e2e_a",
    name_b: str = "test_a2a_e2e_b",
    *,
    data_root: Path | None = None,
) -> Generator[tuple[A2AEndpoint, A2AEndpoint], None, None]:
    """Context manager: spawn two real A2A-enabled agent subprocesses.

    Yields ``(endpoint_a, endpoint_b)``. On exit (whether the body raised or
    returned cleanly) both subprocesses get SIGTERM'd, registry entries are
    cleaned up, and per-agent data dirs are removed.
    """
    if data_root is None:
        data_root = Path(f"/tmp/deepworker_a2a_e2e_{os.getpid()}")
    data_root.mkdir(parents=True, exist_ok=True, mode=0o755)

    # Clean any leftover state from a previous run with the same names.
    for n in (name_a, name_b):
        _safety_cleanup(n, data_root)

    proc_a = pipe_a = proc_b = pipe_b = None
    try:
        proc_a, pipe_a = _spawn_agent(name_a, data_root)
        proc_b, pipe_b = _spawn_agent(name_b, data_root)

        ready_a = _wait_ready(pipe_a, proc_a, name_a)
        ready_b = _wait_ready(pipe_b, proc_b, name_b)

        ep_a = A2AEndpoint(name=name_a, pid=ready_a["pid"],
                           a2a_url=ready_a["a2a_url"],
                           rpyc_port=ready_a["rpyc_port"])
        ep_b = A2AEndpoint(name=name_b, pid=ready_b["pid"],
                           a2a_url=ready_b["a2a_url"],
                           rpyc_port=ready_b["rpyc_port"])

        yield ep_a, ep_b
    finally:
        for p in (proc_a, proc_b):
            if p is not None:
                _terminate(p)
        for n in (name_a, name_b):
            _safety_cleanup(n, data_root)
        with contextlib.suppress(OSError):
            if data_root.exists() and not any(data_root.iterdir()):
                data_root.rmdir()
