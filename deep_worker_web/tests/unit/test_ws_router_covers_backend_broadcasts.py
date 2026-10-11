"""AST-guard: every WS ``type`` value the backend broadcasts MUST have a
``case`` in ``handleWebSocketMessage`` (main.js).

This closes the regression class where a backend broadcast slips through
because the frontend switch was silently missing a case (2026-08-22:
``agent_context_changed`` from commit 3bdf351). The router falls through
to ``default: 'Unknown message type'`` and the UI stops auto-updating
that surface.

How the check works:

1. Walk ``deep_worker_web/app/**/*.py`` and collect every literal
   ``"type": "<name>"`` inside a ``dict`` that is passed to a broadcaster
   (``_broadcast_message``, ``self._broadcast``,
   ``send_json_to_agent_subscribers``, ``websocket.send_json``) — via
   ``ast.parse`` walk, not regex, so multi-line dicts and reflowed
   formatting keep working.
2. Expand the lifecycle shape ``f"{op}_started|completed|failed"`` by
   collecting every literal ``op`` value passed to
   ``LifecycleJobRegistry.start_job(op, ...)`` and pairing each with the
   three suffixes.
3. Parse ``static/js/main.js`` for the ``switch (data.type) { ... }``
   block inside ``handleWebSocketMessage`` and extract every ``case
   '...':`` literal — supports both single-quoted and double-quoted case
   labels.
4. Assert every backend-broadcast type has a ``case`` in the switch,
   modulo a small whitelist of control-only frames (``broadcast_failed``,
   ``error``, ``subscription_*``, ``ping``, ``raw``, ``auto_approve_all``)
   that are never surfaced to the router. ``pong`` is NOT in the
   whitelist — it is a real switch case (main.js keeps it as a no-op
   arm) and appears in the broadcast inventory (WS loop ``send_json``).

The inventory is generated at test time so a NEW broadcast the operator
adds to the backend automatically requires a matching switch case — no
one has to remember to update a hand-maintained list.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
WEB_ROOT = REPO_ROOT / "deep_worker_web"
MAIN_JS = WEB_ROOT / "static" / "js" / "main.js"
BACKEND_ROOTS = [WEB_ROOT / "app"]

# Frames that never reach ``handleWebSocketMessage`` as a UI event.
# ``broadcast_failed`` / ``error`` are server-internal error acks written
# on a specific client socket in the WS loop. ``subscription_*`` are
# per-client responses to subscribe requests. ``ping``/``pong`` is
# keep-alive. ``raw`` is a stringification fallback the payload
# normaliser emits internally and never as a top-level frame type.
CONTROL_WHITELIST = frozenset({
    # Server-internal error acks written directly on the failing socket
    # from inside the WS loop — never surfaces to the router.
    "broadcast_failed",
    # Handled by an early ``if message.type === 'error'`` guard in
    # handleAgentMessage (main.js:1145), not by the switch.  We assert
    # separately that ``error`` is NOT a switch case (see
    # test_control_whitelist_types_are_not_in_the_switch).
    "error",
    # ping is client → server; the client never receives it.
    "ping",
    # Per-client subscribe responses; not broadcast to all subscribers.
    "subscription_error",
    "subscription_failed",
    "subscription_success",
    # ``auto_approve_all`` is a decision_dict INSIDE ``approval_result``
    # payloads, not a top-level WS type — the recipient side reads it via
    # ``data.type`` on the inner approval, not the outer frame.
    "auto_approve_all",
    # ``raw`` is a stringification fallback the payload normaliser emits
    # for un-serialisable message bodies; never a top-level frame type.
    "raw",
})

# Callables that put a frame on the wire. We check whether the first
# positional arg (a dict literal) carries a ``"type"`` key.
BROADCAST_CALLEES = frozenset({
    "_broadcast_message",
    "_broadcast",
    "send_json_to_agent_subscribers",
    "send_json",
})


def _first_arg_dict(call: ast.Call) -> ast.Dict | None:
    """Return the first positional argument if it is a dict literal."""
    if not call.args:
        return None
    arg = call.args[0]
    return arg if isinstance(arg, ast.Dict) else None


def _dict_type_literal(node: ast.Dict) -> str | None:
    """Extract the ``"type"`` key's literal string value, if any."""
    for k, v in zip(node.keys, node.values):
        if isinstance(k, ast.Constant) and k.value == "type":
            if isinstance(v, ast.Constant) and isinstance(v.value, str):
                return v.value
            # f-string case handled by the caller (lifecycle expansion).
    return None


def _dict_type_fstring_op(node: ast.Dict) -> str | None:
    """Detect the lifecycle f-string shape ``f"{op}_<suffix>"`` on the
    ``type`` key and return the suffix (``started`` / ``completed`` /
    ``failed``). Returns None for anything else."""
    for k, v in zip(node.keys, node.values):
        if not (isinstance(k, ast.Constant) and k.value == "type"):
            continue
        if not isinstance(v, ast.JoinedStr):
            return None
        # Expect ``[FormattedValue(op), Constant('_<suffix>')]``.
        if len(v.values) != 2:
            return None
        fv, tail = v.values
        if not (
            isinstance(fv, ast.FormattedValue)
            and isinstance(fv.value, ast.Name)
            and fv.value.id == "op"
            and isinstance(tail, ast.Constant)
            and isinstance(tail.value, str)
            and tail.value.startswith("_")
        ):
            return None
        return tail.value[1:]  # strip the leading underscore
    return None


def _walk_python(paths: list[Path]) -> list[ast.AST]:
    trees: list[ast.AST] = []
    for root in paths:
        for p in root.rglob("*.py"):
            trees.append((p, ast.parse(p.read_text(), filename=str(p))))
    return trees


def _collect_broadcast_types() -> tuple[set[str], set[str]]:
    """Return ``(literal_types, lifecycle_ops)``.

    ``literal_types`` — every literal ``"type": "..."`` value passed to a
    broadcast callable.

    ``lifecycle_ops`` — every ``op`` string passed as the first arg of
    ``.start_job(...)``, used to expand ``f"{op}_started|completed|failed"``.
    """
    literals: set[str] = set()
    ops: set[str] = set()
    for _path, tree in _walk_python(BACKEND_ROOTS):
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            # Callee name — ``func.id`` for bare, ``func.attr`` for method.
            name: str | None
            if isinstance(node.func, ast.Attribute):
                name = node.func.attr
            elif isinstance(node.func, ast.Name):
                name = node.func.id
            else:
                name = None
            # Lifecycle op capture (regardless of broadcast).
            if name == "start_job" and node.args:
                first = node.args[0]
                if isinstance(first, ast.Constant) and isinstance(first.value, str):
                    ops.add(first.value)
            if name not in BROADCAST_CALLEES:
                continue
            d = _first_arg_dict(node)
            if d is None:
                continue
            lit = _dict_type_literal(d)
            if lit is not None:
                literals.add(lit)
            # f-string ``{op}_<suffix>`` — expanded later per collected op.
    return literals, ops


def _switch_case_labels() -> set[str]:
    """Extract every ``case '<name>':`` label inside the
    ``handleWebSocketMessage`` function body from main.js."""
    src = MAIN_JS.read_text()
    # Slice the function body — from ``function handleWebSocketMessage``
    # to the matching closing brace of its outermost body.
    start = src.index("function handleWebSocketMessage")
    # Advance to the opening brace of the function body.
    brace_open = src.index("{", start)
    depth = 0
    i = brace_open
    while i < len(src):
        c = src[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                body_end = i
                break
        i += 1
    else:  # pragma: no cover — defensive; unbalanced braces should not happen
        raise AssertionError("handleWebSocketMessage body is not balanced")
    body = src[brace_open:body_end]
    # Match ``case 'name':`` or ``case "name":``.
    return set(re.findall(r"case\s+['\"]([a-z_]+)['\"]\s*:", body))


class TestWsRouterCoversAllBroadcasts:
    def test_every_backend_broadcast_type_has_a_switch_case(self) -> None:
        literal_types, ops = _collect_broadcast_types()
        # Expand lifecycle shapes.
        for op in ops:
            for suf in ("_started", "_completed", "_failed"):
                literal_types.add(op + suf)
        # Whitelist the control frames we intentionally don't route.
        broadcasts = literal_types - CONTROL_WHITELIST
        assert broadcasts, (
            "backend broadcast inventory came back empty — the "
            "collector is broken (regex/AST paths); this test would "
            "silently pass and let real regressions through"
        )
        cases = _switch_case_labels()
        missing = sorted(broadcasts - cases)
        assert not missing, (
            "WS broadcast types the backend emits but the frontend "
            "switch does NOT route (auto-refresh will be silently "
            "broken for these): "
            f"{missing}. Add a ``case '<name>': handle...(data); break;`` "
            "arm to handleWebSocketMessage in static/js/main.js. If the "
            "type is genuinely server-internal (never routed to the UI), "
            "add it to CONTROL_WHITELIST in this test with a comment "
            "explaining why."
        )

    def test_inventory_covers_the_known_regression(self) -> None:
        """The 2026-08-22 case (``agent_context_changed``) must be in the
        collected inventory — this is the fixture that pins the collector
        itself. If the collector silently misses it, the primary test
        would go green even without the switch case restored."""
        literals, _ = _collect_broadcast_types()
        assert "agent_context_changed" in literals, (
            "collector regression: agent_context_changed is broadcast "
            "from agent_service._maybe_broadcast_context_changed but the "
            "backend inventory walker did not pick it up"
        )

    def test_control_whitelist_types_are_not_in_the_switch(self) -> None:
        """Control frames must NOT appear as switch cases — if one does,
        either the whitelist is wrong or a genuine broadcast was
        accidentally named after a control frame. Either way, flag it."""
        cases = _switch_case_labels()
        # ``error`` is intentionally handled OUTSIDE the switch (early
        # ``if message.type === 'error'`` guard in handleAgentMessage —
        # main.js:1145). It must still not be a switch case.
        overlap = CONTROL_WHITELIST & cases
        assert not overlap, (
            "control-frame types must not be handled in the router "
            f"switch: {sorted(overlap)}. If a type moved from control "
            "to UI event, remove it from CONTROL_WHITELIST."
        )
