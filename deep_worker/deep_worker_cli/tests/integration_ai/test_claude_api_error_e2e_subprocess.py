"""E2E integration: Claude API error → real recoverable path in a REAL agent.

Contract source: docs/DESIGN_FIX_API_ERROR_AS_FINAL_ANSWER.md §5.2 (integration
plan) — full subprocess variant with pytest-owned ``sitecustomize`` shim on
the child's ``PYTHONPATH``, double gate, inherited-pipe canary, real agent
lifecycle over RPyC, real notify subscription, real history/checkpointer.

What is REAL end-to-end here:

  * Real ``deep-worker`` console-script spawned as a fresh subprocess.
  * Real ``AgentProcess``, real ``ClaudeCodeChatModel``, real ``ainvoke``,
    real capture of ``AssistantMessage`` / ``ResultMessage`` fields, real
    ``returning_tool_calls`` guard, real ``detect_api_error``, real raise of
    ``ClaudeApiResponseError``, real ``_handle_invoke`` catch-all, real
    ``is_provider_api_error`` registry lookup, real recoverable branch, real
    ``_append_recoverable_error_to_history`` (writes to real checkpointer),
    real ``notify_callback`` fan-out to subscribers via real RPyC.

The ONLY substitution — permitted by POLICIES §10 (external precondition
that cannot be reproduced) — is at the transport boundary of the SDK
itself: we cannot force Anthropic to return an HTTP 400 on demand. The
substitution is installed by CPython's standard ``sitecustomize`` hook via
a pytest-owned directory on the child's ``PYTHONPATH`` (double gate) —
production code has zero imports from that directory (enforced by
``test_no_prod_import_of_test_support``).

Bootstrap correctness is proven by a one-shot canary written to an
inherited pipe file descriptor: if ``sitecustomize`` never ran, no canary
byte arrives and the test fails with a clean "seam bootstrap did not fire"
rather than an ambiguous timeout.

Coverage:

  * ``test_e2e_api_error_recoverable_path`` — verbatim real API-error payload
    shape → agent stays IDLE, ``invoke`` returns ``success=False`` with the
    real API text (no ``AIMessage`` carrying the error), a persistent
    ``type='error'`` notification is delivered, and the recoverable-error
    SystemMessage is written to the real checkpointer history.

  * ``test_e2e_bypass_detector_reddens`` — mutation test: the SAME scenario
    with the detector short-circuited (``DEEP_WORKER_TEST_FAKE_CLAUDE_SDK_
    BYPASS=1``) must FAIL the "no error-text AIMessage" assertion — proving
    the assertions above really catch the bug rather than pass on nothing.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pytest


pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# VERBATIM recorded API-error payload (docs/DESIGN §1.2). Reproducing this
# byte-for-byte is a POLICIES §10 requirement — earlier grok 402 fixes
# passed on a flattened shape the real API never sends and shipped broken.
# ---------------------------------------------------------------------------
_REAL_400_TEXT = (
    "API Error: 400 messages.18.content.0.tool_result.content.1: "
    "Input tag 'image_url' found using 'type' does not match any of the "
    "expected tags: 'document', 'image', 'search_result', 'text', "
    "'tool_reference'"
)
_ZERO_USAGE = {
    "input_tokens": 0,
    "output_tokens": 0,
    "cache_creation_input_tokens": 0,
    "cache_read_input_tokens": 0,
}
_FAKE_SDK_SCRIPT = [
    {
        "type": "AssistantMessage",
        "content_text": _REAL_400_TEXT,
        "usage": _ZERO_USAGE,
        "model": "claude-fake-e2e",
    },
    {
        "type": "ResultMessage",
        "subtype": "success",       # NOT is_error — recorded shape
        "is_error": False,
        "num_turns": 1,
        "session_id": "e2e-fake",
        "stop_reason": "end_turn",  # verbatim — looks like a normal answer
        "usage": _ZERO_USAGE,
    },
]
_RECOVERY_SCRIPT = [
    _FAKE_SDK_SCRIPT,
    [
        {
            "type": "AssistantMessage",
            "content_text": "4",
            "usage": {"input_tokens": 1, "output_tokens": 1},
            "model": "claude-fake-e2e",
        },
        {
            "type": "ResultMessage",
            "subtype": "success",
            "is_error": False,
            "num_turns": 1,
            "session_id": "clean-session-id",
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 1, "output_tokens": 1},
        },
    ],
]

# Hard budget per the bounded integration-feedback session.
_TEST_BUDGET_S = 60.0
_AGENT_READY_TIMEOUT_S = 20.0
_CANARY_TIMEOUT_S = 15.0
_INVOKE_TIMEOUT_S = 20.0


def _repo_root() -> Path:
    # tests/integration_ai/<file>.py → parents[4] is the tree root.
    return Path(__file__).resolve().parents[4]


def _tree_paths() -> tuple[Path, Path, Path]:
    """PYTHONPATH entries the child needs to import our first-party trees.

    Return the *parent* of ``claude_code_provider/`` (i.e. the repo root),
    NOT the package directory itself — CPython resolves ``import
    claude_code_provider`` against parents of the package.
    """
    root = _repo_root()
    return (
        root,  # parent of claude_code_provider/
        root / "deep_worker" / "deep_worker_cli",
        root / "deep_worker" / "deepagents",
    )


def _support_dir() -> Path:
    return _repo_root() / "claude_code_provider" / "tests" / "_support"


def _make_child_env(
    *,
    canary_fd: int,
    script_path: Path,
    evidence_path: Path | None = None,
    extra: dict[str, str] | None = None,
) -> dict[str, str]:
    """Environment for the ``deep-worker`` child. Prepends the _support
    directory to ``PYTHONPATH`` — this is what activates ``sitecustomize``.
    """
    env = os.environ.copy()
    support = str(_support_dir())
    ccp, dwcli, dagents = _tree_paths()
    prepend = os.pathsep.join([support, str(ccp), str(dwcli), str(dagents)])
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = prepend + (os.pathsep + existing if existing else "")

    # Double gate — both must be set for the shim to activate.
    env["DEEP_WORKER_TEST_FAKE_CLAUDE_SDK_PYTEST"] = "1"
    env["DEEP_WORKER_TEST_FAKE_CLAUDE_SDK_CANARY_FD"] = str(canary_fd)
    env["DEEP_WORKER_TEST_FAKE_CLAUDE_SDK_SCRIPT"] = str(script_path)
    if evidence_path is not None:
        env["DEEP_WORKER_TEST_FAKE_CLAUDE_SDK_EVIDENCE"] = str(evidence_path)
    if extra:
        env.update(extra)
    return env


async def _await_canary(read_fd: int, timeout: float) -> bytes:
    """Event-driven wait for the canary byte-string. No polling."""
    loop = asyncio.get_running_loop()
    future: asyncio.Future[bytes] = loop.create_future()
    buf = bytearray()

    def _on_readable() -> None:
        try:
            chunk = os.read(read_fd, 4096)
        except BlockingIOError:
            return
        if not chunk:
            # EOF on write-end.
            if not future.done():
                future.set_result(bytes(buf))
            loop.remove_reader(read_fd)
            return
        buf.extend(chunk)
        if b"\n" in bytes(buf) and not future.done():
            future.set_result(bytes(buf))
            loop.remove_reader(read_fd)

    os.set_blocking(read_fd, False)
    loop.add_reader(read_fd, _on_readable)
    try:
        return await asyncio.wait_for(future, timeout=timeout)
    finally:
        try:
            loop.remove_reader(read_fd)
        except Exception:
            pass


def _drain_child_output(proc: subprocess.Popen) -> tuple[str, str]:
    """Best-effort capture of whatever the child has written so far.

    The child is spawned with stdout/stderr=PIPE. If we never read them the
    pipe buffer can fill and *block the child*, and — worse for a test — the
    real failure reason stays invisible. Called on every failure path.
    """
    out, err = "", ""
    try:
        # The launcher daemonizes and normally has already exited; only
        # signal it if it is somehow still alive, and never block long.
        if proc.poll() is None:
            proc.send_signal(signal.SIGTERM)
        out_b, err_b = proc.communicate(timeout=10.0)
        out = (out_b or b"").decode("utf-8", "replace")
        err = (err_b or b"").decode("utf-8", "replace")
    except Exception as e:  # noqa: BLE001
        err = f"<failed to drain child output: {e!r}>"
    return out, err


def _child_diagnostics(proc: subprocess.Popen, agent_name: str) -> str:
    """Full diagnostic block for any readiness/early-exit failure."""
    out, err = _drain_child_output(proc)
    log_tail = ""
    log_dir = Path("~/.deep_worker").expanduser() / "logs"
    for log_file in sorted(log_dir.glob(f"{agent_name}.*")):
        try:
            text = log_file.read_text(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            continue
        log_tail += f"\n--- {log_file.name} (tail) ---\n" + "\n".join(
            text.splitlines()[-60:]
        )
    return (
        f"child returncode={proc.returncode!r}\n"
        f"--- child stdout (tail) ---\n" + "\n".join(out.splitlines()[-60:]) + "\n"
        f"--- child stderr (tail) ---\n" + "\n".join(err.splitlines()[-60:]) + "\n"
        f"{log_tail}"
    )


async def _await_agent_ready(
    agent_name: str, timeout: float, proc: subprocess.Popen
) -> None:
    """Block until the agent registers, using the production inotify barrier.

    Uses ``AgentRegistry.wait_for_registration`` — the same inotify-based
    primitive production code uses. NO retry loop and NO sleep-backoff, so
    the no-polling contract of the signed design holds.

    Registration is the *last* step of agent bring-up (agent_process.py:
    ``_register()`` runs only after the RPyC server is up), so it is a sound
    readiness barrier.

    If the child dies early we fail immediately with its returncode plus the
    captured stdout/stderr/log tail instead of burning the full timeout on a
    process that is already gone.
    """
    from deep_worker_cli.process.registry import ProcessRegistry  # noqa: PLC0415

    registry = ProcessRegistry()
    loop = asyncio.get_running_loop()

    # Race the readiness barrier against *abnormal* child death. Both are
    # event-driven (inotify / waitpid in a thread) — no retry loop, no sleep.
    #
    # NOTE: ``deep-worker`` DAEMONIZES: the process we spawned forks the real
    # agent and exits 0 almost immediately. So returncode==0 is the normal
    # path and must NOT be treated as failure; only a NON-ZERO exit means the
    # launcher itself failed (bad args, import error, missing tree).
    ready_task = asyncio.ensure_future(
        registry.wait_for_registration(agent_name, timeout=timeout)
    )
    exit_task = asyncio.ensure_future(loop.run_in_executor(None, proc.wait))
    try:
        done, _ = await asyncio.wait(
            {ready_task, exit_task},
            timeout=timeout,
            return_when=asyncio.FIRST_COMPLETED,
        )
        if exit_task in done and ready_task not in done:
            rc = exit_task.result()
            if rc != 0:
                raise AssertionError(
                    f"deep-worker launcher exited rc={rc} before the agent "
                    f"registered.\n" + _child_diagnostics(proc, agent_name)
                )
            # Daemonized cleanly — keep waiting on the inotify barrier alone.
            info = await ready_task
            if info is None:
                raise AssertionError(
                    f"agent {agent_name!r} daemonized (rc=0) but never "
                    f"registered within {timeout}s.\n"
                    + _child_diagnostics(proc, agent_name)
                )
            return
        if ready_task in done:
            info = ready_task.result()
            if info is None:
                raise AssertionError(
                    f"agent {agent_name!r} did not register within {timeout}s "
                    f"(inotify barrier returned no info).\n"
                    + _child_diagnostics(proc, agent_name)
                )
            return
        raise AssertionError(
            f"agent {agent_name!r} did not become ready within {timeout}s.\n"
            + _child_diagnostics(proc, agent_name)
        )
    finally:
        for task in (ready_task, exit_task):
            if not task.done():
                task.cancel()


def _make_notify_collector() -> Any:
    """Build a real ``rpyc.Service`` subscriber for the notify channel.

    Mirrors ``cli/interactive.py`` (``MessageCallbackService``) exactly: the
    agent's RPyC AsyncioServer invokes ``exposed_on_message`` as a COROUTINE
    in its own loop, so the method must be ``async def``. A plain sync class
    silently fails to deliver, which is what previously made the notify
    assertion vacuous.

    Payloads arrive as RPyC netrefs; we deep-copy each one into a local dict
    immediately so nothing depends on the connection still being alive.
    """
    # rpyc ALWAYS comes from the rpyc_async source tree, where the
    # package is named ``rpyc_async`` (same alias production uses — see
    # cli/interactive.py). Never ``import rpyc``.
    import rpyc_async as rpyc  # noqa: PLC0415

    messages: list[dict] = []
    message_event = asyncio.Event()

    class _NotifyCollector(rpyc.Service):
        async def exposed_on_message(self_inner, msg: Any) -> None:  # noqa: N805
            try:
                if isinstance(msg, str):
                    messages.append(json.loads(msg))
                else:
                    # netref -> plain dict (obtain copies it across the link)
                    try:
                        from rpyc_async.utils.classic import obtain  # noqa: PLC0415

                        msg = obtain(msg)
                    except Exception:  # noqa: BLE001
                        pass
                    messages.append({str(k): msg[k] for k in msg.keys()})
            except Exception as e:  # noqa: BLE001
                messages.append({"_collector_error": repr(e), "raw": str(msg)})
            finally:
                message_event.set()

    collector = _NotifyCollector()
    collector.messages = messages  # type: ignore[attr-defined]
    collector.message_event = message_event  # type: ignore[attr-defined]
    return collector


def _sync_wait_process(proc: subprocess.Popen, timeout: float) -> None:
    """SIGTERM → wait ≤timeout → SIGKILL fallback (§5.2 teardown)."""
    if proc.poll() is not None:
        return
    try:
        proc.send_signal(signal.SIGTERM)
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5.0)


def _kill_daemon(agent_name: str, timeout: float) -> None:
    """SIGTERM → wait ≤timeout → SIGKILL the DAEMONIZED agent process.

    ``deep-worker`` forks the real agent and the launcher exits immediately,
    so reaping the launcher leaves the agent running. We resolve its pid via
    the pid file the launcher advertises and shut it down deterministically.
    Also drops the registry entry so a crashed run cannot poison later ones.
    """
    pid_file = Path(f"/tmp/deep_worker_{agent_name}.pid")
    pid: int | None = None
    try:
        pid = int(pid_file.read_text(encoding="utf-8").strip())
    except Exception:  # noqa: BLE001
        pid = None

    if pid is not None:
        deadline = time.monotonic() + timeout
        try:
            os.kill(pid, signal.SIGTERM)
            while time.monotonic() < deadline:
                try:
                    os.kill(pid, 0)  # liveness probe, not a poll of state
                except OSError:
                    break
                time.sleep(0.05)
            else:
                os.kill(pid, signal.SIGKILL)
        except OSError:
            pass  # already gone

    pid_file.unlink(missing_ok=True)

    # Remove any registry entry so a leaked name cannot collide later.
    try:
        from deep_worker_cli.process.registry import ProcessRegistry  # noqa: PLC0415

        ProcessRegistry().unregister(agent_name)
    except Exception:  # noqa: BLE001
        pass


def _clean_agent_dir(agent_name: str) -> None:
    for d in (
        Path("~/.deep_worker").expanduser() / agent_name,
    ):
        if d.exists():
            shutil.rmtree(d, ignore_errors=True)
    log_dir = Path("~/.deep_worker").expanduser() / "logs"
    if log_dir.exists():
        for log_file in log_dir.glob(f"{agent_name}.*"):
            log_file.unlink(missing_ok=True)


async def _run_e2e(
    agent_name: str, bypass_detector: bool, recovery_probe: bool = False
) -> dict:
    """Common driver for the two E2E tests. Returns the collected evidence."""
    _clean_agent_dir(agent_name)

    read_fd, write_fd = os.pipe()
    tmpdir = Path(tempfile.mkdtemp(prefix="dwtest_e2e_api_error_"))
    script_path = tmpdir / "fake_sdk_script.json"
    script_path.write_text(
        json.dumps(_RECOVERY_SCRIPT if recovery_probe else _FAKE_SDK_SCRIPT),
        encoding="utf-8",
    )
    evidence_path = tmpdir / "fake_sdk_evidence.jsonl"

    extra = {"DEEP_WORKER_TEST_FAKE_CLAUDE_SDK_BYPASS": "1"} if bypass_detector else None
    env = _make_child_env(
        canary_fd=write_fd,
        script_path=script_path,
        evidence_path=evidence_path,
        extra=extra,
    )

    # Spawn the REAL deep-worker console script. skip-permissions + claude
    # provider so no HITL or real network is required.
    # NOTE: ``--dev`` is REQUIRED and is not cosmetic. The installed
    # console-script launcher dispatches to the PRODUCTION tree by
    # default and *overwrites* PYTHONPATH with its own tree (launcher line
    # "export PYTHONPATH=$SCRIPT_DIR:..."), which is prepended ahead of ours.
    # Without --dev the child would import deep_worker_cli /
    # claude_code_provider from master — i.e. the code WITHOUT this fix — so
    # the test would exercise the wrong tree entirely.
    proc = subprocess.Popen(
        [
            "deep-worker",
            "--dev",
            agent_name,
            "--dir", str(tmpdir),
            "--skip-permissions",
            "--provider", "claude-code",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        pass_fds=(write_fd,),
        cwd=str(tmpdir),
    )
    # Parent closes its copy of the write-end so EOF is observable.
    os.close(write_fd)

    result: dict = {"proc": proc, "read_fd": read_fd, "tmpdir": tmpdir}
    try:
        # 1. Await the canary — proves sitecustomize actually ran BEFORE the
        # agent entry point started using the SDK. If it did not, fail with
        # a clean diagnostic — do NOT let a broken bootstrap silently
        # degrade to a real-SDK call and an ambiguous invoke timeout.
        canary = await _await_canary(read_fd, timeout=_CANARY_TIMEOUT_S)
        result["canary"] = canary
        assert canary.startswith(b"CANARY_OK"), (
            f"seam bootstrap did not fire (canary missing / mismatched). "
            f"Got: {canary!r}. Check that ``sitecustomize`` is on the "
            f"child PYTHONPATH and both gate env vars are set.\n"
            + _child_diagnostics(proc, agent_name)
        )
        # 1b. Prove the child imported THIS tree. Without --dev the launcher
        # would silently hand us the production checkout (no fix) and every
        # downstream assertion would be meaningless.
        expected_tree = str(_repo_root())
        canary_text = canary.decode("utf-8", "replace").strip()
        assert f"TREE={expected_tree}" in canary_text, (
            f"child imported claude_code_provider from the WRONG tree.\n"
            f"expected TREE={expected_tree}\ngot canary: {canary_text!r}\n"
            f"This means the deep-worker launcher dispatched elsewhere "
            f"(is ``--dev`` still present in the argv?)."
        )

        # 2. Await agent readiness via the real RPyC registry handshake.
        await _await_agent_ready(
            agent_name, timeout=_AGENT_READY_TIMEOUT_S, proc=proc
        )

        # 3. Subscribe to notifications and drive the real invoke.
        from deep_worker_cli.client.agent_client import AgentClient  # noqa: PLC0415
        from deep_worker_cli.exceptions import IPCConnectionError  # noqa: PLC0415

        client = AgentClient(agent_name)
        await client.connect()
        try:
            collector = _make_notify_collector()
            # subscribe_messages is the real event-driven notify surface —
            # no polling. This is a HARD requirement, not best-effort: G2 of
            # the signed design is "a persistent type=error notification
            # reaches the UI channel", and history does NOT substitute for
            # it. A swallowed subscription would make that assertion vacuous.
            subscribed = await asyncio.wait_for(
                client._conn.root.subscribe_messages(collector), timeout=10.0
            )
            assert subscribed, (
                "subscribe_messages returned falsy — the notify channel was "
                "never established, so the G2 assertion below could not be "
                "meaningfully evaluated."
            )

            invoke_error: str | None = None
            invoke_response: dict | None = None
            try:
                invoke_response = await asyncio.wait_for(
                    client.invoke_agent("please analyse this input", stream=False),
                    timeout=_INVOKE_TIMEOUT_S,
                )
            except IPCConnectionError as e:
                # Recoverable API error surfaces here as success=False,
                # which invoke_agent raises as IPCConnectionError.
                invoke_error = str(e)
            result["invoke_error"] = invoke_error
            result["invoke_response"] = invoke_response

            if recovery_probe:
                result["second_response"] = await asyncio.wait_for(
                    client.invoke_agent("2+2", stream=False),
                    timeout=_INVOKE_TIMEOUT_S,
                )
                result["sdk_evidence"] = [
                    json.loads(line)
                    for line in evidence_path.read_text(encoding="utf-8").splitlines()
                ]

            # 4. Await the event raised by the real RPyC notify callback.
            await asyncio.wait_for(collector.message_event.wait(), timeout=5.0)
            result["notifications"] = list(collector.messages)

            # 5. Read the real history (real checkpointer path).
            try:
                history = await asyncio.wait_for(
                    client.get_history(limit=100), timeout=5.0
                )
            except Exception as e:  # noqa: BLE001
                history = {"error": repr(e)}
            result["history"] = history

            # 6. Read status via metadata.
            meta_path = Path("~/.deep_worker").expanduser() / agent_name / "session_metadata.json"
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception as e:  # noqa: BLE001
                meta = {"error": repr(e)}
            result["session_metadata"] = meta
        finally:
            try:
                await client.disconnect()
            except Exception:  # noqa: BLE001
                pass
        return result
    finally:
        # The launcher we spawned daemonizes, so reaping it is NOT enough —
        # the real agent lives on as an orphan and would leak a process plus
        # a stale registry entry into every subsequent run. Kill the daemon
        # by its PID file, then reap the launcher.
        _kill_daemon(agent_name, timeout=5.0)
        _sync_wait_process(proc, timeout=5.0)
        try:
            os.close(read_fd)
        except OSError:
            pass
        shutil.rmtree(tmpdir, ignore_errors=True)
        _clean_agent_dir(agent_name)


@pytest.mark.asyncio
async def test_e2e_api_error_recoverable_path() -> None:
    """Full §5.2 E2E: verbatim recorded payload → recoverable path in a real
    ``deep-worker`` subprocess. Bounded by ``_TEST_BUDGET_S``."""
    agent_name = f"test_ccp_api_err_e2e_{os.getpid()}"
    result = await asyncio.wait_for(
        _run_e2e(agent_name, bypass_detector=False),
        timeout=_TEST_BUDGET_S,
    )

    # G1: no AIMessage with the API text is emitted from the model boundary.
    # invoke_agent must have surfaced success=False (IPCConnectionError) with
    # the real API text in the error — NOT return a healthy "assistant"
    # response carrying the error text as content.
    assert result["invoke_response"] is None, (
        f"invoke should have surfaced the recoverable API error as "
        f"success=False, not return a healthy response. "
        f"Got: {result['invoke_response']!r}"
    )
    assert result["invoke_error"] is not None
    assert _REAL_400_TEXT in result["invoke_error"], (
        f"invoke error must carry the REAL upstream API text; "
        f"got: {result['invoke_error']!r}"
    )
    assert "ClaudeApiResponseError" in result["invoke_error"], (
        f"invoke error must name the ClaudeApiResponseError class; "
        f"got: {result['invoke_error']!r}"
    )

    # G3: agent stays IDLE (recoverable — NOT ERROR).
    meta = result["session_metadata"]
    assert meta.get("status") == "idle", (
        f"ClaudeApiResponseError is recoverable — agent must return to "
        f"IDLE status. session_metadata={meta!r}"
    )

    # G2/G4: verify both persistent surfaces independently: the real
    # checkpointer history and the subscribed RPyC UI notification channel.
    hist = result.get("history", {})
    hist_msgs = hist.get("history", []) if isinstance(hist, dict) else []
    error_in_history = [
        m for m in hist_msgs
        if isinstance(m, dict)
        and _REAL_400_TEXT in (m.get("content") or "")
    ]
    assert error_in_history, (
        f"POLICIES §9 + DESIGN §3.4 G4: recoverable-error SystemMessage "
        f"must be appended to history carrying the REAL upstream API text "
        f"so it is included in the next request. "
        f"history_len={len(hist_msgs)}, response={hist!r}"
    )
    # The exception class name is baked into the SystemMessage content per
    # agent_process._append_recoverable_error_to_history — so downstream
    # tools / users grepping the history can identify the fault.
    assert any(
        "ClaudeApiResponseError" in (m.get("content") or "")
        for m in error_in_history
    ), (
        f"history entry must name the exception class; got: {error_in_history!r}"
    )

    # error_message in session_metadata must NOT be set for a recoverable
    # error (that field is reserved for AgentStatus.ERROR); the recoverable
    # path leaves it untouched.
    assert not meta.get("error_message"), (
        f"recoverable API error must NOT populate session_metadata."
        f"error_message (reserved for fatal ERROR); got: "
        f"{meta.get('error_message')!r}"
    )

    # G2 (STRICT): the persistent UI notification channel must have carried
    # a type="error" payload with the real API text. History does NOT
    # substitute for this — they are two distinct goals in the signed design
    # (§3.4 G2 = UI surface, G4 = next-request context). Asserted strictly,
    # never "opportunistically".
    notifs = result.get("notifications", [])
    error_notifs = [
        n for n in notifs
        if isinstance(n, dict) and n.get("type") == "error"
    ]
    assert error_notifs, (
        f"DESIGN §3.4 G2: at least one type='error' notification must reach "
        f"the subscriber so the error is persistently visible in the UI. "
        f"Got {len(notifs)} notification(s), types="
        f"{[n.get('type') for n in notifs if isinstance(n, dict)]!r}"
    )
    assert any(
        _REAL_400_TEXT in (n.get("content") or "") for n in error_notifs
    ), (
        f"the type='error' notification must carry the REAL upstream API "
        f"text; got: {[n.get('content') for n in error_notifs]!r}"
    )
    assert any(
        "ClaudeApiResponseError" in (n.get("content") or "")
        for n in error_notifs
    ), (
        f"the type='error' notification must name ClaudeApiResponseError; "
        f"got: {[n.get('content') for n in error_notifs]!r}"
    )


@pytest.mark.asyncio
async def test_e2e_api_error_rotates_session_before_next_request() -> None:
    agent_name = f"test_ccp_api_err_recovery_{os.getpid()}"
    result = await asyncio.wait_for(
        _run_e2e(agent_name, bypass_detector=False, recovery_probe=True),
        timeout=_TEST_BUDGET_S,
    )

    assert result["invoke_error"] is not None
    second = result["second_response"]
    assert second["success"] is True
    assert second["response"] == {"content": "4", "role": "assistant"}
    evidence = result["sdk_evidence"]
    creates = [row for row in evidence if row.get("event") == "create"]
    disconnects = [row for row in evidence if row.get("event") == "disconnect"]
    assert [row["generation"] for row in creates] == [1, 2]
    assert [row["resume_session_id"] for row in creates] == [None, None]
    assert disconnects == [{"event": "disconnect", "generation": 1}]


@pytest.mark.asyncio
async def test_e2e_bypass_detector_reddens() -> None:
    """Mutation test: with the detector short-circuited, the SAME scenario
    MUST fail the G1 assertion — proving the assertions actually catch the
    bug, not GREEN-pass on nothing."""
    agent_name = f"test_ccp_api_err_mut_{os.getpid()}"
    result = await asyncio.wait_for(
        _run_e2e(agent_name, bypass_detector=True),
        timeout=_TEST_BUDGET_S,
    )
    # First: prove the BYPASS hook was actually active in the child. Without
    # this the whole mutation test could "pass" simply because the hook
    # never installed and the run silently behaved like the positive case.
    canary_text = result["canary"].decode("utf-8", "replace")
    assert "BYPASS=1" in canary_text, (
        f"mutation hook was NOT active in the child — the bypass canary "
        f"marker is missing, so this test proves nothing. canary="
        f"{canary_text!r}"
    )

    # With the detector bypassed we require the EXACT production regression
    # (the recorded API-error payload): a healthy AIMessage whose content
    # carries the API error text. A permissive OR (e.g. "history is missing
    # the entry") is deliberately NOT accepted — that could pass because of
    # an unrelated history-read failure and would make this test vacuous.
    assert result["invoke_response"] is not None, (
        f"MUTATION FAILURE: with the detector bypassed the invoke must "
        f"SUCCEED and return the healthy-but-wrong response (the recorded "
        f"regression). Instead it errored: {result.get('invoke_error')!r}"
    )
    content = (
        (result["invoke_response"].get("response") or {}).get("content") or ""
    )
    assert _REAL_400_TEXT in content, (
        f"MUTATION FAILURE: expected the exact recorded regression — the "
        f"API error text landing in AIMessage content. Got content="
        f"{content[:300]!r}"
    )

    # And the positive test's G2 surface must be ABSENT here; if a
    # type='error' notification still appeared, the positive G2 assertion
    # would pass regardless of the detector and would be meaningless.
    notifs = result.get("notifications", [])
    error_notifs = [
        n for n in notifs
        if isinstance(n, dict)
        and n.get("type") == "error"
        and _REAL_400_TEXT in (n.get("content") or "")
    ]
    assert not error_notifs, (
        f"MUTATION FAILURE: a type='error' notification carrying the API "
        f"text was delivered even with the detector bypassed, so the "
        f"positive G2 assertion does not actually discriminate. "
        f"got={error_notifs!r}"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
