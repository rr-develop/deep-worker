"""Native built-in passthrough must be wired by CAPABILITY, not provider name.

Regression cover for the agent_a defect (2026-09-28). The wiring used to be
keyed on ``provider == "grok"`` / ``provider == "codex"``, so claude-code — whose
profile declares ``{Bash, Read, Write, Edit, ...}`` and whose chat model
faithfully surfaces those names as ``AIMessage.tool_calls`` — had NO matching
``BaseTool`` bound. LangGraph's ToolNode then took its unknown-name branch and
checkpointed ``ToolMessage(status="error", "Error: Bash is not a valid tool, try
one of [...]")`` WITHOUT executing anything, while the CLI had already run the
command. 3439 such fabricated refusals landed in that one agent's checkpoint
(2396 of them ``Bash``) and the real output was persisted nowhere.

The tests below assert the OUTCOME (ToolNode no longer fabricates a refusal),
not merely that a helper returns a tuple, and they pin the two axes a
name-keyed regression would break: which built-ins get bound, and what the
generic recorder is named.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from deep_worker.providers.acp.builtin_tool_passthrough import (
    PLAN_TOOL_NAME,
    build_passthrough_tools_for_profile,
)

# The production modules whose wiring must stay capability-keyed.
_AGENT_PROCESS = (
    Path(__file__).resolve().parents[2].parent
    / "deep_worker_cli"
    / "process"
    / "agent_process.py"
)
_TASK_MANAGER = (
    Path(__file__).resolve().parents[2].parent
    / "deep_worker_cli"
    / "task_manager"
    / "task_manager.py"
)
# Provenance reads the SAME shadowed attribute to report the ACP adapter.
_AGENT_MODULE = (
    Path(__file__).resolve().parents[2].parent / "deep_worker_cli" / "agent.py"
)

CLAUDE_CODE_BUILTINS = frozenset(
    {
        "Bash",
        "Read",
        "Write",
        "Edit",
        "Glob",
        "Grep",
        "Agent",
        "Task",
        "WebFetch",
        "WebSearch",
    }
)


@dataclass(frozen=True)
class _StubProfile:
    """Minimal stand-in for ``AcpProfile`` (duck-typed by the helper)."""

    provider: str
    known_builtins: frozenset[str] = frozenset()
    generic_builtin_name: str = ""
    colliding_builtins: frozenset[str] = field(default_factory=frozenset)
    # Defaults mirror AcpProfile's, for the tests that drive _apply_update's
    # tool_call path (which reads these).
    mcp_dispatch_style: str = "mcp_name_prefix"
    mcp_dispatch_tool: str = "CallMcpTool"
    tool_name_from_update: str = "from_claude_code_meta"
    vendor_prefix: str = ""


def _names(tools) -> set[str]:
    return {t.name for t in tools}


# --------------------------------------------------------------------------
# Axis 1: which built-ins get bound
# --------------------------------------------------------------------------


def test_claude_code_profile_binds_its_native_builtins() -> None:
    """The names the CLI actually runs must each get a BaseTool."""
    tools = build_passthrough_tools_for_profile(
        _StubProfile(provider="claude-code", known_builtins=CLAUDE_CODE_BUILTINS)
    )
    bound = _names(tools)

    # The three that dominated the agent_a corruption.
    for critical in ("Bash", "Edit", "Read"):
        assert critical in bound, f"{critical} must be bound or ToolNode fabricates a refusal"
    assert CLAUDE_CODE_BUILTINS <= bound


def test_profile_without_builtins_binds_nothing() -> None:
    """A provider that runs no local built-ins must not gain phantom tools."""
    assert build_passthrough_tools_for_profile(
        _StubProfile(provider="some-acp-cli", known_builtins=frozenset())
    ) == ()


def test_non_acp_profile_shapes_bind_nothing() -> None:
    """``profile`` is a field on langchain's BaseChatModel, so EVERY model has
    the attribute; only the value discriminates. An upstream ``ModelProfile``
    is a plain dict with no ``known_builtins`` and must yield no tools rather
    than raising — otherwise a provider with no built-ins fails at startup."""

    class _Bare:
        pass

    assert build_passthrough_tools_for_profile({}) == ()
    assert build_passthrough_tools_for_profile({"max_input_tokens": 1000}) == ()
    assert build_passthrough_tools_for_profile(_Bare()) == ()


def test_non_string_builtin_names_are_rejected_not_bound() -> None:
    """A duck-typed profile must never produce a non-string tool name: it would
    reach ``BaseTool.name`` and raise a pydantic ValidationError, converting
    "this provider has no built-ins" into a hard startup crash."""
    tools = build_passthrough_tools_for_profile(
        _StubProfile(
            provider="weird",
            known_builtins=frozenset({"Bash", 42, None, ""}),  # type: ignore[arg-type]
        )
    )
    assert "Bash" in _names(tools)
    assert all(isinstance(t.name, str) and t.name for t in tools)


# --------------------------------------------------------------------------
# Axis 2: the generic recorder's name
# --------------------------------------------------------------------------


def test_generic_recorder_name_matches_chat_model_derivation() -> None:
    """The name the chat model surfaces for an UNKNOWN built-in and the name of
    the tool bound for it must agree. A mismatch reintroduces the exact
    fabricated-refusal defect for every built-in not in ``known_builtins``."""
    from deep_worker.providers.acp.chat_model import AcpChatModel

    profile = _StubProfile(
        provider="claude-code", known_builtins=CLAUDE_CODE_BUILTINS
    )

    # Derivation used by AcpChatModel._generic_builtin_name, driven through the
    # real function object so a change to it breaks this test.
    model_side = AcpChatModel._generic_builtin_name(
        type("_M", (), {"_profile": profile})()
    )

    bound = _names(build_passthrough_tools_for_profile(profile))
    assert model_side == "claudecodeBuiltin"
    assert model_side in bound, (
        "generic recorder name drifted between chat_model and the passthrough "
        "builder; unknown built-ins would be fabricated as errors again"
    )


def test_explicit_generic_name_is_honoured() -> None:
    """grok pins ``GrokBuiltin`` as a public contract (bounded_tool_call keys
    its 600s imagine budget off it, and it is already in recorded history)."""
    bound = _names(
        build_passthrough_tools_for_profile(
            _StubProfile(
                provider="grok",
                known_builtins=frozenset({"Shell"}),
                generic_builtin_name="GrokBuiltin",
            )
        )
    )
    assert "GrokBuiltin" in bound


# --------------------------------------------------------------------------
# The outcome that actually matters: ToolNode stops fabricating refusals
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_toolnode_no_longer_fabricates_invalid_tool_error() -> None:
    """End-to-end on the real ToolNode: with the passthrough tools bound, a
    ``Bash`` tool_call must NOT produce the synthetic
    ``"Bash is not a valid tool"`` ToolMessage that corrupted agent_a."""
    from langchain_core.messages import AIMessage
    from langgraph.prebuilt import ToolNode

    tools = build_passthrough_tools_for_profile(
        _StubProfile(provider="claude-code", known_builtins=CLAUDE_CODE_BUILTINS)
    )
    node = ToolNode(list(tools))

    call = {
        "id": "toolu_fabricate_probe",
        "name": "Bash",
        "args": {"command": "pytest -q"},
        "type": "tool_call",
    }

    # Validation happens before the body runs. The body parks on a Future that
    # only the ACP reader resolves, so we assert on the VALIDATION verdict
    # rather than letting the (correctly) blocking body run to completion.
    assert node._validate_tool_call(call) is None, (
        "ToolNode still rejects Bash as an unknown name — the passthrough tool "
        "is not bound and history would record a refusal for work that ran"
    )

    # And the converse, proving the assertion above has teeth: an unbound name
    # IS still rejected with exactly the message seen in the corrupted thread.
    rejected = node._validate_tool_call(
        {**call, "name": "DefinitelyNotBound", "id": "toolu_probe2"}
    )
    assert rejected is not None
    assert "is not a valid tool" in rejected.content
    assert rejected.status == "error"

    _ = AIMessage  # imported to document the surfaced-call shape


# --------------------------------------------------------------------------
# The wiring itself: both call sites must select by capability, not by name
# --------------------------------------------------------------------------


@pytest.mark.parametrize("source", [_AGENT_PROCESS, _TASK_MANAGER])
def test_call_sites_wire_the_capability_helper(source: Path) -> None:
    """Guard the PROVING axis: the helper is reachable from the real startup
    path. A test that only calls the helper directly would still pass if the
    call site were reverted to a name-keyed gate."""
    text = source.read_text(encoding="utf-8")
    assert "build_passthrough_tools_for_profile" in text, (
        f"{source.name} no longer wires the capability-keyed passthrough helper"
    )

    tree = ast.parse(text)
    calls = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Name)
        and n.func.id == "build_passthrough_tools_for_profile"
    ]
    assert calls, f"{source.name} imports the helper but never calls it"


# --------------------------------------------------------------------------
# update_plan: a name the SHARED layer mints, so every ACP provider needs it
# --------------------------------------------------------------------------


def test_plan_recorder_bound_for_every_acp_provider() -> None:
    """``update_plan`` is emitted by acp/chat_model.py for ANY provider whose
    CLI sends ``kind == "plan"`` — there is no provider check at the emission
    site. So the recorder must be bound regardless of what the profile
    declares; otherwise the next provider fabricates "update_plan is not a
    valid tool" exactly as claude-code did."""
    for profile in (
        _StubProfile(provider="claude-code", known_builtins=CLAUDE_CODE_BUILTINS),
        _StubProfile(provider="some-new-acp-cli", known_builtins=frozenset({"Shell"})),
        _StubProfile(
            provider="grok",
            known_builtins=frozenset({"Shell"}),
            generic_builtin_name="GrokBuiltin",
        ),
    ):
        bound = _names(build_passthrough_tools_for_profile(profile))
        assert PLAN_TOOL_NAME in bound, (
            f"{profile.provider} would fabricate a refusal for {PLAN_TOOL_NAME}"
        )


def test_plan_recorder_not_declared_as_a_cli_builtin() -> None:
    """``update_plan`` must NOT be smuggled into a profile's ``known_builtins``:
    no CLI ships a tool by that name — our own layer synthesises it. Declaring
    it there would be a lie about the CLI's surface AND would leave the next
    provider broken until someone remembered to add it.

    Reads the real KNOWN_BUILTINS constant rather than calling
    ``build_profile()``: that constructor requires the npm adapter installed on
    disk, which CI does not have (CI_CD_TESTS.md portability rule)."""
    from deep_worker.providers.claude_code_provider.profile import KNOWN_BUILTINS

    assert PLAN_TOOL_NAME not in KNOWN_BUILTINS


def test_real_claude_code_builtins_accept_plan_and_keep_builtins() -> None:
    """The reported defect, on claude-code's REAL built-in set (not a
    hand-written stub list) and the real ToolNode: update_plan must be
    accepted, the built-ins must still be accepted, and an unbound name must
    still be rejected."""
    from deep_worker.providers.claude_code_provider.profile import KNOWN_BUILTINS
    from langgraph.prebuilt import ToolNode

    node = ToolNode(
        list(
            build_passthrough_tools_for_profile(
                _StubProfile(provider="claude-code", known_builtins=KNOWN_BUILTINS)
            )
        )
    )

    for name in (PLAN_TOOL_NAME, "Bash", "Edit", "Read"):
        verdict = node._validate_tool_call(
            {"id": f"tc_{name}", "name": name, "args": {}, "type": "tool_call"}
        )
        assert verdict is None, f"{name} still fabricates: {verdict and verdict.content!r}"

    rejected = node._validate_tool_call(
        {"id": "tc_x", "name": "NotABoundTool", "args": {}, "type": "tool_call"}
    )
    assert rejected is not None and "is not a valid tool" in rejected.content


@pytest.mark.asyncio
async def test_plan_recorder_returns_immediately_and_records_the_plan() -> None:
    """A plan is a COMPLETED FACT: the CLI never sends ``tool_call_update
    status=completed`` for it. So the body must NOT park on an unresolved
    Future — that would hang the ToolNode forever, which is worse than the
    fabricated error it replaces. chat_model pre-resolves the Future at
    emission time; this asserts the body returns without waiting and that the
    plan text actually reaches history."""
    import asyncio

    from deep_worker.providers.acp import chat_model as cm
    from deep_worker.providers.acp.client_state import get_global_client_state
    from deep_worker.providers.claude_code_provider.profile import KNOWN_BUILTINS

    tools = {
        t.name: t
        for t in build_passthrough_tools_for_profile(
            _StubProfile(provider="claude-code", known_builtins=KNOWN_BUILTINS)
        )
    }
    thread_id = "test-plan-thread"
    state = get_global_client_state().get_or_create(thread_id)

    # Drive the REAL update path, not the helper: _apply_update is what
    # production calls, so this also proves the pre-resolve is WIRED there.
    # Calling _preresolve_plan_result directly would pass even if the call site
    # were deleted, leaving the body to hang forever in production.
    notif = {
        "params": {
            "update": {
                "sessionUpdate": "plan",
                "toolCallId": "plan_tc_1",
                "entries": [
                    {"content": "Fix the bug", "status": "in_progress"},
                    {"content": "Run tests", "status": "pending"},
                ],
            }
        }
    }
    model = object.__new__(cm.AcpChatModel)
    tool_calls_acc: list[dict] = []
    cm.AcpChatModel._apply_update(
        model, state, thread_id, notif, [], tool_calls_acc
    )

    assert [tc["name"] for tc in tool_calls_acc] == [PLAN_TOOL_NAME], (
        f"plan update did not surface an {PLAN_TOOL_NAME} tool_call"
    )
    assert "plan_tc_1" in state.builtin_passthrough_results, (
        "no pre-resolved result — the recorder body would hang forever"
    )

    # A real timeout, so a regression to parking FAILS instead of hanging CI.
    msg = await asyncio.wait_for(
        tools[PLAN_TOOL_NAME].ainvoke(
            {"id": "plan_tc_1", "name": PLAN_TOOL_NAME, "args": {}, "type": "tool_call"},
            config={"configurable": {"thread_id": thread_id}},
        ),
        timeout=5,
    )
    assert msg.status == "success"
    assert "Fix the bug" in msg.content and "Run tests" in msg.content


@pytest.mark.asyncio
async def test_native_builtin_records_real_result_not_a_refusal() -> None:
    """Axis (a) of the defect, proven at the RECORD level, not the verdict level.

    The sibling test above asserts only that ToolNode's validator accepts
    ``Bash``. That is necessary but not sufficient: the point of the fix is that
    history gets the CLI's REAL result instead of a fabricated refusal. So this
    drives the whole production path — a ``tool_call`` update through the real
    ``_apply_update``, then the reader resolving it through the real
    ``resolve_builtin_future`` — and asserts on the resulting ToolMessage.
    """
    import asyncio

    from deep_worker.providers.acp import chat_model as cm
    from deep_worker.providers.acp.builtin_tool_passthrough import BUILTIN_MARKER
    from deep_worker.providers.acp.client_state import get_global_client_state
    from deep_worker.providers.claude_code_provider.profile import KNOWN_BUILTINS

    profile = _StubProfile(provider="claude-code", known_builtins=KNOWN_BUILTINS)
    tools = {t.name: t for t in build_passthrough_tools_for_profile(profile)}
    thread_id = "test-bash-record-thread"
    state = get_global_client_state().get_or_create(thread_id)

    # Built WITHOUT the constructor (no subprocess, no singleton mutation),
    # setting only what the mapping path reads — mirrors the _model() helper in
    # tests/acp_provider/test_chat_model_refine.py.
    model = object.__new__(cm.AcpChatModel)
    model._profile = profile  # type: ignore[attr-defined]
    model._proxy_mcp = None  # type: ignore[attr-defined]
    model._tools_metadata = []  # type: ignore[attr-defined]
    model._client_state = get_global_client_state()  # type: ignore[attr-defined]

    # 1. The CLI announces it is running Bash. Real update path.
    tool_calls_acc: list[dict] = []
    cm.AcpChatModel._apply_update(
        model,
        state,
        thread_id,
        {
            "params": {
                "update": {
                    "sessionUpdate": "tool_call",
                    "toolCallId": "bash_tc_1",
                    "title": "Bash",
                    "rawInput": {"command": "pytest -q"},
                    "_meta": {"claudeCode": {"toolName": "Bash"}},
                }
            }
        },
        [],
        tool_calls_acc,
    )
    assert [tc["name"] for tc in tool_calls_acc] == ["Bash"], (
        "Bash was not surfaced as a real tool_call — LangGraph would never "
        "route to the recorder at all"
    )

    # 2. The CLI reports the command finished, carrying its real output. Again
    #    the REAL update path, so the reader's resolve wiring is covered too.
    cm.AcpChatModel._apply_update(
        model,
        state,
        thread_id,
        {
            "params": {
                "update": {
                    "sessionUpdate": "tool_call_update",
                    "toolCallId": "bash_tc_1",
                    "title": "Bash",
                    "status": "completed",
                    "_meta": {"claudeCode": {"toolName": "Bash"}},
                    "content": [
                        {
                            "type": "content",
                            "content": {"type": "text", "text": "1220 passed"},
                        }
                    ],
                }
            }
        },
        [],
        [],
    )

    # 3. What actually lands in history.
    msg = await asyncio.wait_for(
        tools["Bash"].ainvoke(
            {"id": "bash_tc_1", "name": "Bash", "args": {}, "type": "tool_call"},
            config={"configurable": {"thread_id": thread_id}},
        ),
        timeout=5,
    )
    assert msg.status == "success", f"recorded as a failure: {msg.content!r}"
    assert "is not a valid tool" not in msg.content, (
        "history still records the fabricated refusal for a built-in that ran"
    )
    assert "1220 passed" in msg.content, (
        f"the CLI's real output did not reach history: {msg.content!r}"
    )
    assert msg.additional_kwargs.get(BUILTIN_MARKER) is True


def _acp_like_model():
    """A model shaped like ``AcpChatModel``: pydantic + a ``profile`` property.

    Built here rather than via ``ClaudeCodeAcpChatModel`` because that
    constructor calls ``build_profile``, which requires the npm adapter on
    disk (``AcpAdapterError`` on CI). What matters for this regression is the
    SHAPE — a pydantic model that declares ``profile`` both as an inherited
    field and as a ``@property`` — and that is reproducible without node.
    """
    from pydantic import BaseModel

    class _Profile:
        known_builtins = frozenset({"Bash", "Read", "Edit"})
        generic_builtin_name = None

    class _Model(BaseModel):
        profile: object | None = None

        def __init__(self, **kwargs: object) -> None:
            super().__init__(**kwargs)
            object.__setattr__(self, "_profile_obj", _Profile())

        @property  # type: ignore[misc]
        def profile(self) -> object:  # noqa: D401
            return self._profile_obj

    model = _Model()
    # Guard the premise: this stub deliberately reproduces the UNFIXED shape,
    # so that the call-site fallback below is exercised against the hazard it
    # was written for. Production no longer has this shape — the base class
    # clears the shadow (see the AcpChatModel tests further down); this stub
    # keeps the belt honest for any path that reintroduces the collision.
    assert isinstance(
        model.__dict__.get("profile"), property
    ), "premise broken: pydantic no longer shadows the property"
    return model


def test_profile_is_read_past_the_pydantic_property_shadow() -> None:
    """``getattr(model, "profile")`` yields the ``property`` OBJECT, not the
    profile — so the wiring must read ``_profile`` first.

    agent_a, 2026-09-29. The capability wiring landed and still bound ZERO
    recorders: ``AcpChatModel`` declares ``profile`` as a ``@property`` while
    inheriting it as a pydantic field, pydantic strips the property off the
    class, and the instance ``__dict__`` shadows it. ``getattr`` therefore
    returned a ``property`` object — not None, so the ``is not None`` guard
    passed — and the helper found no ``known_builtins`` on it and returned
    ``()``. ToolNode went straight back to fabricating "Error: Bash is not a
    valid tool" for a POST-fix process: 122 Bash / 61 Edit / 43 Read refusals.
    """
    model = _acp_like_model()

    # The naive read — what the wiring used to do — recovers nothing.
    naive = getattr(model, "profile", None)
    assert isinstance(naive, property)
    assert build_passthrough_tools_for_profile(naive) == (), (
        "premise broken: the property object must yield no recorders"
    )

    # The production read must see through the shadow.
    resolved = getattr(model, "_profile_obj", None) or getattr(
        model, "profile", None
    )
    names = {t.name for t in build_passthrough_tools_for_profile(resolved)}
    assert {"Bash", "Read", "Edit"} <= names, (
        f"native built-ins not bound through the shadowed property: {names}"
    )


@pytest.mark.parametrize(
    "source", [_AGENT_PROCESS, _TASK_MANAGER, _AGENT_MODULE]
)
def test_wiring_reads_private_profile_before_the_public_attribute(
    source: Path,
) -> None:
    """Both call sites must read ``_profile`` BEFORE ``profile``.

    An AST assertion rather than a string match: it pins the ORDER of the
    ``or`` fallback, which is the whole fix. Reading the public attribute
    first would resolve to the ``property`` object and bind nothing.
    """
    tree = ast.parse(source.read_text(encoding="utf-8"))

    found = False
    for node in ast.walk(tree):
        if not isinstance(node, ast.BoolOp) or not isinstance(node.op, ast.Or):
            continue
        attrs = []
        for value in node.values:
            if (
                isinstance(value, ast.Call)
                and isinstance(value.func, ast.Name)
                and value.func.id == "getattr"
                and len(value.args) >= 2
                and isinstance(value.args[1], ast.Constant)
            ):
                attrs.append(value.args[1].value)
        if attrs[:2] == ["_profile", "profile"]:
            found = True
            break

    assert found, (
        f"{source.name} must read the profile as "
        f"getattr(model, '_profile', None) or getattr(model, 'profile', None) "
        f"— the public attribute alone is the shadowed property object"
    )


def test_acp_chat_model_profile_is_not_a_pydantic_field() -> None:
    """The base class must not carry ``profile`` as a pydantic field.

    This is the root fix. While ``profile`` was a field, pydantic stored our
    ``@property`` as its default, so every instance carried a raw ``property``
    under ``__dict__["profile"]`` and ``model.profile`` returned it. Pinning
    the ABSENCE of the field is what makes the shadow unreproducible, rather
    than merely worked around at the call sites.
    """
    from deep_worker.providers.acp.chat_model import AcpChatModel

    assert "profile" not in AcpChatModel.model_fields, (
        "profile is a pydantic field again — pydantic will capture the "
        "@property as its default and model.profile will yield a property"
    )
    assert isinstance(AcpChatModel.__dict__.get("profile"), property), (
        "profile must be a real class-level descriptor"
    )


@pytest.mark.parametrize(
    "module_path, class_name",
    [
        ("deep_worker.providers.acp.chat_model", "AcpChatModel"),
        ("deep_worker.providers.grok_provider.chat_model", "GrokChatModel"),
        (
            "deep_worker.providers.claude_code_provider.chat_model",
            "ClaudeCodeAcpChatModel",
        ),
    ],
)
def test_every_acp_subclass_clears_the_profile_shadow(
    module_path: str, class_name: str
) -> None:
    """Subclasses must clear the shadow too.

    pydantic rebuilds ``model_fields`` per class, so each subclass re-captures
    the inherited property as a field default unless it is cleared again. A
    fix applied only to ``AcpChatModel`` would leave every concrete provider —
    the ones actually used — still broken.
    """
    import importlib

    cls = getattr(importlib.import_module(module_path), class_name)
    assert "profile" not in cls.model_fields, (
        f"{class_name} re-captured the profile property as a field default"
    )
    assert isinstance(getattr(cls, "profile", None), property), (
        f"{class_name} lost the profile descriptor"
    )


def test_constructed_acp_model_keeps_the_profile_out_of_its_instance_dict() -> None:
    """A real constructed instance must not carry ``profile`` in ``__dict__``.

    ``GrokChatModel`` rather than the claude-code model because its
    constructor needs no npm adapter, so this runs unconditionally on CI —
    the end-to-end test below skips without one.

    This pins the premise the class-level tests cannot: a property descriptor
    on the class MASKS a stale ``__dict__`` entry, so ``.profile`` could read
    correctly while the raw property still leaked into ``__dict__`` and
    ``model_dump()``. Without this assert the fix could "go green" with the
    shadow quietly restored.
    """
    from deep_worker.providers.grok_provider.chat_model import GrokChatModel

    model = GrokChatModel()

    assert "profile" not in model.__dict__, (
        f"raw {type(model.__dict__['profile']).__name__} in the instance dict"
    )
    assert not isinstance(getattr(model, "profile", None), property)
    assert "profile" not in model.model_dump()


@pytest.mark.parametrize(
    "module_path, class_name",
    [
        ("deep_worker.providers.codex_provider.chat_model", "CodexChatModel"),
    ],
)
def test_non_acp_providers_keep_the_inherited_profile_field(
    module_path: str, class_name: str
) -> None:
    """The fix must not reach providers that never had the collision.

    ``profile`` is declared on ``BaseChatModel``, so every provider inherits
    it — but only classes that also define a ``profile`` property collide.
    The repair is guarded on ``isinstance(field.default, property)``; these
    providers must therefore keep the ordinary field with a ``None`` default,
    exactly as LangChain intends.
    """
    import importlib

    import deep_worker.providers.acp.chat_model  # noqa: F401  (applies the fix)

    cls = getattr(importlib.import_module(module_path), class_name)
    field = cls.model_fields.get("profile")
    assert field is not None, f"{class_name} lost the inherited profile field"
    assert field.default is None, (
        f"{class_name} profile default is {type(field.default).__name__}"
    )


def test_unshadow_is_a_no_op_on_a_class_that_never_collided() -> None:
    """The repair must refuse to touch an uncollided ``profile`` field.

    Production only ever calls the helper on ACP classes, so the class-level
    assertions above cannot exercise this guard — they would pass even if it
    were deleted. Call the helper DIRECTLY on a non-ACP model to pin it.

    Without the ``isinstance(field.default, property)`` guard this deletes a
    perfectly healthy inherited field, leaving no descriptor behind: every
    subsequent ``model.profile`` on that provider raises ``AttributeError``.
    """
    from deep_worker.providers.acp.chat_model import _unshadow_profile_property
    from deep_worker.providers.codex_provider.chat_model import CodexChatModel

    before = CodexChatModel.model_fields.get("profile")
    assert before is not None and before.default is None, (
        "premise broken: CodexChatModel should carry a plain profile field"
    )

    _unshadow_profile_property(CodexChatModel)

    after = CodexChatModel.model_fields.get("profile")
    assert after is not None, (
        "the repair deleted an uncollided field — model.profile now raises "
        "AttributeError on every non-ACP provider"
    )
    assert after.default is None


def test_real_claude_code_model_binds_native_builtins_via_public_profile() -> None:
    """End-to-end on the real constructor: ``model.profile`` yields recorders.

    The prior tests all fed the helper a profile directly, which is exactly
    the gap that let the defect ship: the helper was correct, but what reached
    it was a ``property``. This builds the model the production way and reads
    the PUBLIC attribute only.
    """
    pytest.importorskip("deep_worker.providers.claude_code_provider.chat_model")
    from deep_worker.providers.claude_code_provider.chat_model import (
        ClaudeCodeAcpChatModel,
    )

    try:
        model = ClaudeCodeAcpChatModel()
    except Exception as exc:  # pragma: no cover - needs the npm adapter on disk
        pytest.skip(f"claude-code adapter unavailable: {type(exc).__name__}: {exc}")

    profile = getattr(model, "profile", None)
    assert not isinstance(profile, property), (
        "model.profile is a property object — the shadow is back"
    )
    assert profile is getattr(model, "_profile", None)
    # The shadow lived in the INSTANCE dict, and a property descriptor on the
    # class would mask a stale entry there — so ``model.profile`` could read
    # correctly while the raw property still leaked into ``__dict__`` and
    # ``model_dump()``. Pin the instance dict, not just the attribute read.
    assert "profile" not in model.__dict__, (
        f"raw {type(model.__dict__['profile']).__name__} left in the instance "
        f"dict — it leaks through __dict__/model_dump even if .profile reads ok"
    )

    names = {t.name for t in build_passthrough_tools_for_profile(profile)}
    assert {"Bash", "Read", "Edit", "Write"} <= names, (
        f"native built-ins not bound from the public profile: {sorted(names)}"
    )


def test_plan_render_never_yields_an_empty_message() -> None:
    """The plan IS the result, so a malformed/empty payload must still produce
    a readable line rather than a blank ToolMessage that reads as a no-op."""
    from deep_worker.providers.acp.chat_model import _render_plan_entries

    for payload in (None, [], "junk", [{}], [{"content": "   "}]):
        assert _render_plan_entries(payload).strip()
    rendered = _render_plan_entries([{"content": "Ship it", "status": "completed"}])
    assert "Ship it" in rendered and "completed" in rendered
