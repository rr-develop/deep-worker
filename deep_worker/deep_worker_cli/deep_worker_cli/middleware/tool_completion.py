"""Generalized turn-ending tools.

See design document:
    deep_worker_cli/docs/DESIGN_TOOL_COMPLETION_MIDDLEWARE.md

Some tools *are* the agent's answer for a turn: once they succeed there
is nothing useful left for the model to say, and invoking it again only
produces filler. ``send_message_to_agent`` is the original example, but
it is not special — it is simply the first member of a class.

This module implements that class. A launcher declares which tools end
the turn by passing :class:`TurnEndingTool` declarations; nothing else
makes a tool turn-ending. In particular a tool's *name* never implies
semantics: names are matched with ``==`` against an explicit registry,
so an MCP server cannot change an agent's control flow by choosing a
suggestive tool name.

On success the middleware either injects an echo ``AIMessage`` (when the
declaration names an ``echo_field``) or ends the turn with no message at
all (``echo_field=None``). It never manufactures an empty ``AIMessage``:
downstream consumers skip empty AI content anyway, so an empty echo
would be invisible while still costing history and tokens.
"""

from __future__ import annotations

import argparse
import ast
import json
import logging
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
    hook_config,
)
from langchain_core.messages import AIMessage, ToolMessage

from deepagents.middleware.message_timestamps import ensure_message_created_at_after

if TYPE_CHECKING:
    from langgraph.runtime import Runtime

logger = logging.getLogger(__name__)


def _default_is_success(data: dict[str, Any]) -> bool:
    """Default success predicate: the ``success`` key is truthy."""
    return bool(data.get("success"))


@dataclass(frozen=True, slots=True)
class TurnEndingTool:
    """Declaration that one specific tool ends the agent's turn when it
    succeeds.

    Attributes:
        tool_name: exact tool name as it appears on ``ToolMessage.name``.
            Compared with ``==``. No prefix, suffix, or fuzzy matching:
            a tool is turn-ending only because it was registered here.
        is_success: predicate over the parsed tool-result dict. Returns
            True when this call should end the turn. Defaults to
            "the ``success`` key is truthy". If the payload cannot be
            parsed into a dict the turn never ends, regardless.
        echo_field: name of the field whose non-empty string value is
            emitted as the injected ``AIMessage.content``. ``None`` means
            "end the turn without emitting any message" (jump-only).
    """

    tool_name: str
    is_success: Callable[[dict[str, Any]], bool] = field(default=_default_is_success)
    echo_field: str | None = "delivered_content"

    def __post_init__(self) -> None:
        if not isinstance(self.tool_name, str) or not self.tool_name.strip():
            raise ValueError(
                "TurnEndingTool.tool_name must be a non-empty, non-whitespace string"
            )
        if not callable(self.is_success):
            raise TypeError("TurnEndingTool.is_success must be callable")
        if self.echo_field is not None:
            if not isinstance(self.echo_field, str) or not self.echo_field.strip():
                raise ValueError(
                    "TurnEndingTool.echo_field must be None or a non-empty, "
                    "non-whitespace string"
                )


class DuplicateTurnEndingToolError(ValueError):
    """Raised when the same tool name is declared turn-ending twice.

    ``turn_ending_tools`` is public configuration, so a duplicate is
    reported rather than silently dropped: silently ignoring it would
    hide typos and config layering, and would let a launcher believe it
    had redefined a built-in tool's semantics when it had not.
    """


class TurnEndingToolFlagError(argparse.ArgumentTypeError):
    """Raised when a ``--turn-ending-tool`` spec string is malformed.

    Subclasses ``argparse.ArgumentTypeError`` so that a bad value passed
    on the command line is reported by argparse as a clean usage error
    instead of a traceback, while direct callers (``--restart`` re-parse,
    ``AgentProcess.__init__``) can still catch it by its own name.
    """


def _validate_turn_ending_spec_str(raw: str) -> tuple[str, str | None]:
    """Validate one ``--turn-ending-tool`` value; return ``(name, field)``.

    The single source of truth for the grammar (design §3.2). Every entry
    point — the argparse ``type=`` handler, the ``--restart`` re-parse and
    ``AgentProcess.__init__`` — goes through this function, so a spec that
    is accepted on first launch is accepted on every later launch too.

    Accepted forms are exactly ``<name>`` and ``<name>:<field>``.

    Raises:
        TurnEndingToolFlagError: on any violation.
    """
    if not isinstance(raw, str):
        raise TurnEndingToolFlagError(
            "--turn-ending-tool: expected a string spec of the form "
            f"'<name>' or '<name>:<field>', got {type(raw).__name__}"
        )
    if any(ch.isspace() for ch in raw):
        # Whitespace-tolerant grammars silently paper over shell quoting
        # bugs and would let two visually different specs collapse onto
        # the same middleware key. Reject, do not strip.
        raise TurnEndingToolFlagError(
            f"--turn-ending-tool: {raw!r} must not contain whitespace; "
            "use '<name>' or '<name>:<field>' with no spaces"
        )
    if not raw:
        raise TurnEndingToolFlagError(
            "--turn-ending-tool: spec must not be empty; use '<name>' or "
            "'<name>:<field>'"
        )
    if raw.count(":") > 1:
        raise TurnEndingToolFlagError(
            f"--turn-ending-tool: {raw!r} has more than one ':'; the only "
            "accepted forms are '<name>' and '<name>:<field>'"
        )
    if ":" not in raw:
        return raw, None

    name, field_name = raw.split(":", 1)
    if not name:
        raise TurnEndingToolFlagError(
            f"--turn-ending-tool: {raw!r} has an empty tool name; "
            "use '<name>:<field>'"
        )
    if not field_name:
        raise TurnEndingToolFlagError(
            f"--turn-ending-tool: {raw!r} has an empty field name; use "
            f"'{name}' for the default success predicate, or "
            f"'{name}:<field>' to test a specific field"
        )
    if "." in field_name:
        raise TurnEndingToolFlagError(
            f"--turn-ending-tool: {raw!r} uses a dotted field name; nested "
            "field paths are out of scope, the field must be a single "
            "top-level key"
        )
    return name, field_name


def _canonical_turn_ending_spec_str(raw: str) -> str:
    """``argparse`` ``type=`` handler for ``--turn-ending-tool``.

    Validates via :func:`_validate_turn_ending_spec_str` and returns the
    CANONICAL raw string. Today canonical == input (whitespace is
    rejected outright, so there is nothing to normalise), but routing
    through this function guarantees the contract that matters: the
    argparse namespace attribute ``turn_ending_tools`` is always a
    ``list[str]``, hence JSON-serialisable straight into
    ``session_metadata.json`` by ``build_launch_config()``.
    """
    _validate_turn_ending_spec_str(raw)
    return raw


def _validate_turn_ending_specs_full_list(specs: list[TurnEndingTool]) -> None:
    """Reject cross-entry collisions BEFORE any resource allocation.

    Complements :func:`_validate_turn_ending_spec_str`, which sees one
    value at a time and therefore cannot detect:

    1. two entries declaring the same ``tool_name`` (note that ``foo``
       and ``foo:done`` are the same ``tool_name`` — a duplicate);
    2. an entry declaring the built-in ``send_message_to_agent``, which
       is always registered and cannot be redeclared from the CLI.

    Both used to surface only inside ``SendMessageCompletionMiddleware``
    — i.e. inside ``create_cli_agent``, inside the swallowing init
    ``try`` of ``AgentProcess.start()`` — which turned a config typo
    into an ERROR-mode ghost process instead of a clean startup failure.
    """
    # Local import: ``send_message_completion`` imports THIS module at
    # module scope, so a top-level import here would be circular. By the
    # time this function runs, ``tool_completion`` is fully loaded.
    from deep_worker_cli.middleware.send_message_completion import (
        TOOL_NAME as _BUILT_IN,
    )

    seen: set[str] = set()
    for spec in specs:
        if spec.tool_name == _BUILT_IN:
            raise DuplicateTurnEndingToolError(
                f"--turn-ending-tool: {_BUILT_IN!r} is a built-in "
                "turn-ending tool and cannot be redeclared from the CLI "
                "(it is always registered)"
            )
        if spec.tool_name in seen:
            raise DuplicateTurnEndingToolError(
                f"--turn-ending-tool: {spec.tool_name!r} is declared more "
                "than once; a bare form and a ':<field>' form of the same "
                "name are still duplicates — pick one"
            )
        seen.add(spec.tool_name)


def _parse_turn_ending_tool_specs(
    raw_specs: Iterable[str],
) -> list[TurnEndingTool]:
    """Turn raw ``--turn-ending-tool`` strings into declarations.

    Runs BOTH validation levels: per-string grammar
    (:func:`_validate_turn_ending_spec_str`) and cross-entry constraints
    (:func:`_validate_turn_ending_specs_full_list`), so every entry point
    that parses a full list gets identical enforcement.

    CLI-registered tools are always jump-only (``echo_field=None``):
    echoing a third-party tool's payload into the transcript is a
    separate feature nobody has asked for.

    Raises:
        TurnEndingToolFlagError: malformed spec string.
        DuplicateTurnEndingToolError: duplicate name, or the built-in.
    """
    parsed: list[TurnEndingTool] = []
    for raw in raw_specs:
        name, field_name = _validate_turn_ending_spec_str(raw)
        if field_name is None:
            parsed.append(TurnEndingTool(tool_name=name, echo_field=None))
        else:
            parsed.append(
                TurnEndingTool(
                    tool_name=name,
                    # Explicit default arg: without it every lambda would
                    # close over the loop variable and read the LAST field.
                    is_success=lambda d, _f=field_name: d.get(_f) is True,
                    echo_field=None,
                )
            )
    _validate_turn_ending_specs_full_list(parsed)
    return parsed


def _parse_tool_result(raw: Any) -> dict[str, Any] | None:
    """Best-effort parse of a ``ToolMessage.content`` payload.

    LangChain serializes dict tool results in at least two ways
    depending on the integration path:

    * JSON (most common): ``'{"success": true, "delivered_content": "..."}'``
    * Python-repr (``str(dict)``): ``"{'success': True, 'delivered_content': '...'}"``

    Returns the parsed dict on success, ``None`` on anything else.
    """
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str):
        return None

    # Try JSON first (preferred, produced by json.dumps()).
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        parsed = None

    # Fall back to Python literal eval for str(dict) style.
    if parsed is None:
        try:
            parsed = ast.literal_eval(raw)
        except (ValueError, SyntaxError, MemoryError, RecursionError):
            return None

    if isinstance(parsed, dict):
        return parsed
    return None


def _trailing_tool_message_run_length(messages: Sequence[Any]) -> int:
    """Number of contiguous ``ToolMessage``s at the end of ``messages``.

    The run boundary is the first preceding message that is NOT a
    ``ToolMessage`` — in practice the ``AIMessage`` carrying the tool
    call(s) that produced this run. Returns 0 when the tail is not a
    ``ToolMessage``.
    """
    count = 0
    for msg in reversed(messages):
        if not isinstance(msg, ToolMessage):
            break
        count += 1
    return count


class ToolCompletionMiddleware(AgentMiddleware):
    """Ends the agent turn when an explicitly registered tool succeeds.

    Implementation: ``before_model`` hook with
    ``@hook_config(can_jump_to=["end"])``, i.e. it runs *after* the tool
    executed and *before* the model would be called again.
    """

    # Explicit empty list — this middleware registers no tools of its own.
    # Declared at class level so ``middleware.tools`` is always iterable
    # even before ``__init__`` runs (some langgraph code introspects this).
    tools: list = []

    def __init__(self, specs: Iterable[TurnEndingTool] = ()) -> None:
        super().__init__()
        by_name: dict[str, TurnEndingTool] = {}
        for spec in specs:
            if not isinstance(spec, TurnEndingTool):
                raise TypeError(
                    "turn-ending tool declarations must be TurnEndingTool, "
                    f"got {type(spec).__name__}"
                )
            if spec.tool_name in by_name:
                raise DuplicateTurnEndingToolError(
                    f"tool {spec.tool_name!r} is declared turn-ending more than "
                    "once; remove the duplicate declaration"
                )
            by_name[spec.tool_name] = spec
        self._by_name = by_name

    @hook_config(can_jump_to=["end"])
    def before_model(
        self,
        state: AgentState,
        runtime: "Runtime[Any] | None" = None,  # noqa: ARG002
    ) -> dict[str, Any] | None:
        # Defensive: state might be an unusual shape in tests; use .get().
        messages = state.get("messages", []) if isinstance(state, dict) else []
        if not messages:
            return None

        # Gate 1: the tail must be a tool result.
        if not isinstance(messages[-1], ToolMessage):
            return None

        # Gate 2: only a SOLE trailing tool result may end the turn.
        #
        # When the model emits parallel tool calls, LangGraph appends one
        # ToolMessage per call and this hook runs once, after all of them.
        # Ending the turn on a matching last result could hide a failed
        # sibling from the model, so any parallel batch falls through and
        # lets the model see every result. Checked BEFORE the registry
        # lookup so the rule holds wherever the registered tool sits in
        # the batch.
        if _trailing_tool_message_run_length(messages) != 1:
            return None

        last = messages[-1]

        # Gate 3: the tool must be explicitly registered. (Anthropic
        # sometimes omits .name; getattr with default handles that.)
        spec = self._by_name.get(getattr(last, "name", None))
        if spec is None:
            return None

        # Gate 4: parse the tool's structured result.
        data = _parse_tool_result(last.content)
        if data is None:
            logger.debug(
                "ToolCompletionMiddleware: could not parse ToolMessage.content "
                "for %r; letting the normal flow continue",
                spec.tool_name,
            )
            return None

        # Gate 5: success only — failures should surface to the model.
        try:
            if not spec.is_success(data):
                return None
        except Exception:
            logger.warning(
                "ToolCompletionMiddleware: turn-ending predicate for %r raised; "
                "not ending the turn",
                spec.tool_name,
                exc_info=True,
            )
            return None

        # Jump-only: the tool's result is not conversational output, so no
        # message is emitted. Never inject an empty AIMessage.
        if spec.echo_field is None:
            logger.info(
                "ToolCompletionMiddleware: ending turn after %r (no echo)",
                spec.tool_name,
            )
            return {"jump_to": "end"}

        # Gate 6: the echo field must be a non-empty string.
        value = data.get(spec.echo_field, "")
        if not isinstance(value, str) or not value.strip():
            return None

        logger.info(
            "ToolCompletionMiddleware: ending turn after %r with echo "
            "(preview=%r)",
            spec.tool_name,
            value[:120],
        )
        echo = AIMessage(content=value)
        ensure_message_created_at_after(messages, echo)
        return {"messages": [echo], "jump_to": "end"}
