"""Fake ``ClaudeSDKClient`` transport installer — pytest-only.

Contract source: docs/DESIGN_FIX_API_ERROR_AS_FINAL_ANSWER.md §5.2.

Called only from the sibling ``sitecustomize.py`` shim under a double gate.
Never imported by production. Enforced by
``tests/test_no_prod_import_of_test_support``.

``install()`` performs two side effects:

  1. Monkey-patches ``claude_code_provider.client_manager.ClientManager
     .get_or_create_client`` so that every ``ClaudeCodeChatModel.ainvoke``
     call in the child interpreter receives a fake ``(client, proxy)`` tuple
     yielding a pre-baked SDK stream. The stream is loaded from the JSON
     script pointed at by ``DEEP_WORKER_TEST_FAKE_CLAUDE_SDK_SCRIPT``.

  2. Emits the CANARY (a single ``b"CANARY_OK\\n"`` byte-string) to the
     inherited pipe file descriptor whose write-end number is passed via
     ``DEEP_WORKER_TEST_FAKE_CLAUDE_SDK_CANARY_FD``. This is how the test
     proves the shim actually executed inside the child before the invoke —
     otherwise a broken bootstrap would time out ambiguously.

  3. (Mutation-test hook) When ``DEEP_WORKER_TEST_FAKE_CLAUDE_SDK_BYPASS=1``
     is additionally set, the fake ``ClaudeCodeChatModel.ainvoke`` is
     monkey-patched to bypass the API-error detector entirely (returning a
     raw ``AIMessage`` with the error text). This lets
     ``test_bypass_detector_reddens`` prove the E2E assertions actually
     RED-fail when the detector is removed.
"""

from __future__ import annotations

import json
import os
from typing import Any


_CANARY_BYTES = b"CANARY_OK\n"


def install() -> None:
    """Install the fake SDK monkeypatches. Never raises to the caller."""
    try:
        _install_fake_client_manager()
        if os.environ.get("DEEP_WORKER_TEST_FAKE_CLAUDE_SDK_BYPASS") == "1":
            _install_detector_bypass()
        _fire_canary()
    except Exception as exc:  # noqa: BLE001 - shim must never crash the host
        # Write a diagnostic byte so the test's canary reader sees SOMETHING
        # non-empty (the assertion "canary saw CANARY_OK" will still fail,
        # but with a diagnostic tail rather than silence).
        try:
            _write_canary_bytes(f"BOOT_FAILED:{type(exc).__name__}:{exc}\n".encode())
        except Exception:
            pass


def _fire_canary() -> None:
    """Emit CANARY_OK plus the tree the child actually imported from.

    The tree path is NOT decoration. The `deep-worker` launcher
    overwrites PYTHONPATH with its own tree and defaults to the PRODUCTION
    checkout, so a child spawned without ``--dev`` silently imports the code
    WITHOUT the fix under test. Reporting the resolved module file lets the
    parent assert it is exercising the tree it thinks it is.
    """
    try:
        import deep_worker.providers.claude_code_provider.chat_model as _cm  # noqa: PLC0415

        tree = os.path.dirname(os.path.dirname(os.path.abspath(_cm.__file__)))
    except Exception:  # noqa: BLE001
        tree = "<unresolved>"
    # Report whether the mutation (detector-bypass) hook is VERIFIED active
    # (patch re-resolved through production's own import path), not merely
    # whether the env var was set. This is what lets the mutation test prove
    # the hook was genuinely in effect.
    bypass = "1" if _bypass_installed else "0"
    _write_canary_bytes(
        b"CANARY_OK TREE=" + tree.encode() + b" BYPASS=" + bypass.encode() + b"\n"
    )


def _write_canary_bytes(data: bytes) -> None:
    fd_env = os.environ.get("DEEP_WORKER_TEST_FAKE_CLAUDE_SDK_CANARY_FD")
    if not fd_env:
        return
    fd = int(fd_env)
    # Best-effort write + close of our copy of the write-end. The read-end
    # in the parent stays open until the parent closes it.
    try:
        os.write(fd, data)
    finally:
        try:
            os.close(fd)
        except OSError:
            pass


def _load_script() -> list[Any]:
    """Load the fake SDK stream script. Format:

        [
          {"type": "AssistantMessage", "content_text": "...", "usage": {...},
           "error": null},
          {"type": "ResultMessage", "is_error": false, "subtype": "success",
           "stop_reason": "end_turn", ...},
        ]
    """
    path = os.environ.get("DEEP_WORKER_TEST_FAKE_CLAUDE_SDK_SCRIPT")
    if not path:
        return []
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _install_fake_client_manager() -> None:
    """Replace ``ClientManager.get_or_create_client`` with a fake."""
    # Delayed import — the shim runs at interpreter startup, and importing
    # claude_code_provider that early would drag in half the world before
    # sys.path is fully configured. This works because monkeypatch happens
    # BEFORE any test code (or agent process) tries to call ainvoke —
    # AgentProcess.start() → RPyC ready → invoke arrives strictly after
    # ``site`` has finished importing sitecustomize.
    from deep_worker.providers.claude_code_provider import client_manager as cm_mod  # noqa: PLC0415
    from claude_agent_sdk import (  # noqa: PLC0415
        AssistantMessage as ClaudeAssistantMessage,
        ResultMessage,
        TextBlock,
    )

    script = _load_script()
    streams = script if script and isinstance(script[0], list) else [script]
    receive_count = 0

    async def _fake_receive_response():
        nonlocal receive_count
        index = min(receive_count, len(streams) - 1)
        receive_count += 1
        for entry in streams[index]:
            kind = entry.get("type")
            if kind == "AssistantMessage":
                text = entry.get("content_text", "")
                usage = entry.get("usage")
                error = entry.get("error")
                kwargs: dict[str, Any] = {
                    "content": [TextBlock(text=text)],
                    "model": entry.get("model", "claude-fake-e2e"),
                }
                if usage is not None:
                    kwargs["usage"] = usage
                if error is not None:
                    kwargs["error"] = error
                yield ClaudeAssistantMessage(**kwargs)
            elif kind == "ResultMessage":
                rm_kwargs: dict[str, Any] = {
                    "subtype": entry.get("subtype", "success"),
                    "duration_ms": int(entry.get("duration_ms", 1)),
                    "duration_api_ms": int(entry.get("duration_api_ms", 1)),
                    "is_error": bool(entry.get("is_error", False)),
                    "num_turns": int(entry.get("num_turns", 1)),
                    "session_id": entry.get("session_id", "fake-session"),
                    "stop_reason": entry.get("stop_reason"),
                }
                if "usage" in entry:
                    rm_kwargs["usage"] = entry["usage"]
                if "result" in entry:
                    rm_kwargs["result"] = entry["result"]
                if "errors" in entry:
                    rm_kwargs["errors"] = entry["errors"]
                yield ResultMessage(**rm_kwargs)

    evidence_path = os.environ.get("DEEP_WORKER_TEST_FAKE_CLAUDE_SDK_EVIDENCE")

    def _record_evidence(payload: dict[str, Any]) -> None:
        if evidence_path:
            with open(evidence_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(payload) + "\n")

    class _FakeClient:
        def __init__(self, generation: int) -> None:
            self.generation = generation
            self.session_id = f"fake-session-{generation}"

        async def connect(self, *a: Any, **kw: Any) -> None:
            return None

        async def query(self, *a: Any, **kw: Any) -> None:
            return None

        def receive_response(self) -> Any:
            return _fake_receive_response()

        async def disconnect(self) -> None:
            _record_evidence(
                {"event": "disconnect", "generation": self.generation}
            )

        async def interrupt(self) -> None:
            return None

    class _FakeProxy:
        def has_pending_calls(self) -> bool:
            return False

        def get_pending_calls(self) -> list[Any]:
            return []

        def resolve_call(self, *a: Any, **kw: Any) -> None:
            return None

        async def wait_for_pending_call_by_id(self, *a: Any, **kw: Any) -> None:
            return None

        def cancel_pending_waiter(self, *a: Any, **kw: Any) -> None:
            return None

    generation = 0

    async def _fake_get_or_create_client(
        self: Any, thread_id: str, *a: Any, **kw: Any
    ) -> tuple[Any, Any]:
        nonlocal generation
        generation += 1
        client = _FakeClient(generation)
        proxy = _FakeProxy()
        self.clients[thread_id] = (client, proxy)
        _record_evidence(
            {
                "event": "create",
                "generation": generation,
                "thread_id": thread_id,
                "resume_session_id": kw.get("resume_session_id"),
            }
        )
        return client, proxy

    cm_mod.ClientManager.get_or_create_client = _fake_get_or_create_client  # type: ignore[method-assign]


_bypass_installed = False


def _install_detector_bypass() -> None:
    """Mutation-test hook: short-circuit the detector so the E2E assertions
    RED-fail — proving they actually catch the bug.

    ``chat_model.ainvoke`` does a FUNCTION-LOCAL import
    (``from deep_worker.providers.claude_code_provider.api_error_detect import detect_api_error``)
    which re-resolves the attribute from the module object on every call, so
    patching the module attribute is sufficient and takes effect even though
    the import statement lives inside the function body.

    We nonetheless verify the patch actually took, and record that fact in
    ``_bypass_installed`` so the canary can report a *verified* hook rather
    than merely echoing the env var back at the test.
    """
    global _bypass_installed
    from deep_worker.providers.claude_code_provider import api_error_detect as det_mod  # noqa: PLC0415

    def _no_op_detect(**_kw: Any) -> None:
        return None

    det_mod.detect_api_error = _no_op_detect  # type: ignore[assignment]

    # Re-resolve through the exact same import path production uses and
    # confirm we get the no-op back. If this ever stops holding (e.g. the
    # production import is hoisted to module level and binds early), the
    # canary will report BYPASS=0 and the mutation test fails loudly instead
    # of silently proving nothing.
    from deep_worker.providers.claude_code_provider.api_error_detect import (  # noqa: PLC0415
        detect_api_error as _resolved,
    )

    _bypass_installed = _resolved is _no_op_detect
