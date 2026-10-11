"""Restricting the tool set an agent may call.

Measuring whether a tool earns its place requires an agent that has only
that tool. The selection is expressed as a small spec — see
``docs/DESIGN_TOOL_SELECTION.md`` — and enforced at the one boundary the
model actually reads tools from.

The spec is deliberately tiny:

``[]``
    Every tool. This is the default and must stay indistinguishable from
    having no selection at all, because every existing agent relies on it.
``["-*"]``
    No tools.
``["-*", "read_file"]``
    No tools except the ones named after the star.
``["read_file", "grep"]``
    The same thing without the star: a bare list keeps only what it names.
``["-execute"]``
    Every tool except the ones named with a leading minus.
``["*", "-execute"]``
    The same thing said out loud: ``*`` is every tool, which is already the
    baseline, so the star is redundant but permitted.

``*`` means EVERY TOOL, matching glob, ``.gitignore`` and every ACL syntax a
reader arrives with. The negation is ``-*``, built from the same ``-name``
rule that removes a single tool. An earlier revision spelled deny-all as a
bare ``*``, which read as "all tools are available" and meant the opposite;
see ``_reject_legacy_deny_all`` for why that spelling is now an error rather
than a silent re-interpretation.

``none`` is an accepted alias for ``-*``: ``argparse`` swallows a value that
starts with ``-``, so ``--tools -*`` cannot reach this module at all and only
``--tools=-*`` survives. Deny-everything is the most common spec of the
feature, so it must not require remembering the ``=``.

Keep-these and remove-those do not mix — ``["read_file", "-execute"]``
states two different selections — and ``-*`` beside ``-name`` is redundant
at best, so both are rejected rather than resolved by picking a winner.
"""

from __future__ import annotations

import fnmatch
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

ALLOW_ALL = "*"
DENY_ALL = "-*"

#: ``argparse`` treats any value starting with ``-`` as the next option, so
#: ``--tools -*`` is rejected before this module ever sees it and only
#: ``--tools=-*`` gets through. Requiring the ``=`` form for the single most
#: common spec — deny everything — is a trap, so ``none`` is accepted as an
#: alias. It needs no ``=``, no quoting, and no shell-glob care.
DENY_ALL_ALIAS = "none"


class ToolSelectionError(ValueError):
    """A tool selection that cannot be honoured as written."""


def _tool_name(tool: Any) -> str:
    """Return a tool's name whether it is an object or a plain dict."""
    name = getattr(tool, "name", None)
    if name is None and isinstance(tool, dict):
        name = tool.get("name")
    return str(name) if name is not None else ""


def tool_names(tools: Iterable[Any]) -> set[str]:
    """Names of *tools*, skipping anything nameless."""
    return {n for n in (_tool_name(t) for t in tools) if n}


def middleware_tool_names(middleware: Iterable[Any]) -> set[str]:
    """Names of tools the *middleware* contribute.

    Tools reach the model from two directions, and a selection is
    validated before the middleware ones exist in any list. Asking the
    middleware themselves keeps the known-names set honest: a hand-written
    constant would drift the moment a middleware gains or renames a tool,
    and the failure mode is the worst kind — a correct spec rejected as a
    typo, or a typo accepted as correct.
    """
    names: set[str] = set()
    for mw in middleware:
        for tool in getattr(mw, "tools", ()) or ():
            name = _tool_name(tool)
            if name:
                names.add(name)
    return names


def sdk_tool_names() -> set[str]:
    """Names of tools the deepagents SDK builds after we validate.

    ``create_deep_agent`` constructs ``FilesystemMiddleware`` and the
    subagent middleware itself, so their tools exist only once the graph
    is built — strictly after the selection has to be checked. Validating
    against the middleware list alone therefore rejects ``read_file`` and
    ``task`` as typos even though they are exactly the tools a
    tool-effectiveness experiment is most likely to name (caught by a live
    ``create_cli_agent`` build; a unit test on the list could not see it).

    The names are read off the real middleware classes rather than
    hard-coded, so renaming a tool upstream cannot leave a stale constant
    behind.

    Failures here are deliberately NOT swallowed. An earlier revision
    wrapped this in ``except Exception: pass``, which turned any breakage
    into an empty set — and an empty set does not fail loudly, it silently
    reclassifies ``read_file`` as a typo and aborts the launch with a
    misleading message. A broken lookup must surface as itself.

    Note the result depends on whether ``patch_filesystem_middleware()``
    has run: after patching, the subclass also contributes ``wait_for_job``
    and ``kill_job``. That only ever ADDS names, so validation cannot
    reject a real tool because of ordering; callers must not assume a
    fixed-size set.
    """
    from deepagents.middleware.filesystem import FilesystemMiddleware
    from deepagents.middleware.subagents import SubAgentMiddleware  # noqa: F401

    names = tool_names(getattr(FilesystemMiddleware(), "tools", ()) or ())
    if not names:
        raise RuntimeError(
            "FilesystemMiddleware exposed no tools; the tool-selection "
            "known-name set would silently reject read_file/execute as "
            "typos. Refusing to validate against an empty set."
        )
    # SubAgentMiddleware needs a live backend to instantiate, so take the
    # one name it contributes from the module's own tool factory.
    names.add("task")
    return names


def is_pattern(name: str) -> bool:
    """True when *name* is a wildcard pattern rather than a literal name."""
    return "*" in name or "?" in name or "[" in name


def specificity(pattern: str) -> int:
    """How specific *pattern* is, for resolving overlapping rules.

    The rule that names a tool most precisely wins, so a literal name
    always beats any wildcard that also matches it. Ranking by the length
    of the literal (non-wildcard) part expresses that with one number:
    ``serena__find_symbol`` (19) outranks ``serena__f*`` (8), which
    outranks ``serena__*`` (8 → 8 is the prefix ``serena__``) and ``*``
    (0). A literal gets a bonus so it can never tie with a pattern whose
    literal part happens to be as long.
    """
    literal = len(pattern.replace("*", "").replace("?", ""))
    return literal * 2 + (1 if not is_pattern(pattern) else 0)


@dataclass(frozen=True, order=True)
class _Rule:
    """One ``add``/``remove`` decision plus how specifically it was stated."""

    pattern: str
    allow: bool

    @property
    def specificity(self) -> int:
        return specificity(self.pattern)

    def matches(self, name: str) -> bool:
        if not is_pattern(self.pattern):
            return name == self.pattern
        return fnmatch.fnmatchcase(name, self.pattern)


@dataclass(frozen=True)
class ToolSelection:
    """A parsed tool selection.

    ``allow`` and ``deny`` are never both non-empty: the spec grammar
    admits either an allow-list (after ``*``) or a deny-list, not both.

    ``rules`` carries the incremental ``--tools-add``/``--tools-remove``
    layer, which CAN mix the two directions because each rule states how
    specifically it was written and the most specific one wins. It is
    empty for every spec that predates the flags, so the original
    behaviour is untouched.
    """

    deny_all: bool = False
    allow: frozenset[str] = frozenset()
    deny: frozenset[str] = frozenset()
    rules: tuple[_Rule, ...] = field(default=())

    @property
    def restricts(self) -> bool:
        """True when this selection removes anything at all.

        The empty selection is the default, and callers use this to skip
        the filtering machinery entirely so that the untouched path stays
        exactly as it was.
        """
        return self.deny_all or bool(self.deny) or bool(self.rules)

    def _base_permits(self, name: str) -> bool:
        if self.deny_all:
            return name in self.allow
        return name not in self.deny

    def permits(self, name: str) -> bool:
        """Whether *name* survives the base spec and the incremental rules.

        The most specifically-stated rule wins, which is what makes
        ``--tools-rm 'serena__*' --tools-add serena__find_symbol`` mean
        "drop the family, keep that one".

        Two DIFFERENT patterns can be equally specific and still both
        match one name — ``ab*`` and ``*ab`` both match ``abab`` at
        specificity 4 — so a tie is reachable here even though
        ``validate_rules`` rejects the ones it can see. It cannot see
        them all: it needs the tool inventory to know two patterns
        overlap, and a spec re-read from ``session_metadata.json`` is
        parsed before any inventory exists.

        A tie is therefore resolved, not assumed away, and it resolves to
        DENY. Taking the position-first rule would make the outcome depend
        on flag order, which is the hidden precedence this feature must
        not have; and of the two directions only deny is safe to pick
        blind, because guessing wrong hands an agent a tool it was
        confined away from rather than withholding one.
        """
        winner: _Rule | None = None
        for rule in self.rules:
            if not rule.matches(name):
                continue
            if winner is None or rule.specificity > winner.specificity:
                winner = rule
            elif rule.specificity == winner.specificity and not rule.allow:
                winner = rule
        if winner is not None:
            return winner.allow
        return self._base_permits(name)

    def apply(self, tools: Sequence[Any]) -> list[Any]:
        """Return the subset of *tools* this selection permits."""
        if not self.restricts:
            return list(tools)
        return [t for t in tools if self.permits(_tool_name(t))]

    def unknown_names(self, available: Iterable[str]) -> list[str]:
        """Names in the spec that no available tool answers to.

        A typo would otherwise yield an agent with quietly fewer tools
        than the experiment intended — which corrupts the measurement
        this feature exists to support, so callers turn this into an
        error rather than a warning.

        Patterns are held to the same standard: one matching nothing is
        reported too. ``--tools-rm 'lsp_*'`` when the tools are really
        called ``serena__*`` would otherwise be a no-op that looks like a
        restriction, which is the same silent-miss failure a typo'd
        literal causes.
        """
        known = set(available)
        missing = set((self.allow | self.deny) - known)
        for rule in self.rules:
            if is_pattern(rule.pattern):
                if not any(rule.matches(n) for n in known):
                    missing.add(rule.pattern)
            elif rule.pattern not in known:
                missing.add(rule.pattern)
        return sorted(missing)

    def ambiguous_rules(self, available: Iterable[str]) -> list[tuple[str, str]]:
        """Equally-specific rule pairs that disagree about a real tool.

        ``_reject_contradictory_rules`` groups by the pattern STRING, so
        it only ever catches ``+lsp_*`` against ``~lsp_*``. Two different
        patterns can be equally specific and still both match one name —
        ``ab*`` and ``*ab`` both match ``abab`` at specificity 4 — and
        that pair has no non-arbitrary winner either.

        Deciding it needs the inventory: whether two patterns overlap is
        a question about the tools that exist, not about the strings.
        That is why this is a separate check taking *available* rather
        than part of parsing, which runs with no inventory in hand.

        Returned pairs are ordered within themselves and overall so the
        error message is stable regardless of how the flags were written.
        """
        known = set(available)
        conflicts: set[tuple[str, str]] = set()
        for i, first in enumerate(self.rules):
            for second in self.rules[i + 1:]:
                if first.allow == second.allow:
                    continue
                if first.specificity != second.specificity:
                    continue
                if any(
                    first.matches(n) and second.matches(n) for n in known
                ):
                    conflicts.add(
                        tuple(sorted((first.pattern, second.pattern)))  # type: ignore[arg-type]
                    )
        return sorted(conflicts)


EMPTY_SELECTION = ToolSelection()


def validate_against_inventory(
    selection: ToolSelection, available: Iterable[str]
) -> None:
    """Raise on anything only the tool inventory can reveal.

    Two checks need the real tool names, and both have the same failure
    mode if skipped — an agent that launches with quietly different tools
    than asked for. Pairing them behind one call is deliberate: a caller
    cannot run the typo check and forget the ambiguity one.

    Raises:
        ToolSelectionError: a name or pattern matches no tool, or two
            equally specific rules disagree about a tool that exists.
    """
    known = set(available)
    unknown = selection.unknown_names(known)
    if unknown:
        raise ToolSelectionError(
            f"tool selection names unknown tool(s): {', '.join(unknown)}. "
            f"A typo here would silently produce an agent with fewer "
            f"tools than intended. Known tools: {', '.join(sorted(known))}."
        )
    ambiguous = selection.ambiguous_rules(known)
    if ambiguous:
        pairs = "; ".join(f"'{a}' vs '{b}'" for a, b in ambiguous)
        overlap = sorted(
            n
            for n in known
            for a, b in ambiguous
            if _Rule(a, True).matches(n) and _Rule(b, True).matches(n)
        )
        raise ToolSelectionError(
            f"{pairs} are equally specific and disagree about "
            f"{', '.join(overlap)}, so neither side can win. Order will "
            f"not decide it: the spec is re-parsed from "
            f"session_metadata.json on every restart, where order is an "
            f"implementation detail. Name the tool literally on the side "
            f"you mean, or make one pattern more specific."
        )


def _reject_legacy_deny_all(
    *, allow_all: bool, allow: set[str], deny: set[str], deny_all: bool
) -> None:
    """Refuse the old spelling where a bare ``*`` meant "no tools".

    Two legacy specs mean the opposite of what they used to, and both would
    WIDEN an agent that was deliberately confined — silently, at the next
    restart, since specs live in ``session_metadata.json`` and survive one:

    ``["*"]``
        Meant "no tools at all"; would now mean every tool. The whole
        registry handed back to an agent that had none. This is the larger
        of the two, and it is the form the shipped documentation taught.
    ``["*", "project_memory"]``
        Meant "only project_memory"; would now mean "every tool, plus
        project_memory".

    Quietly re-interpreting a spec whose whole purpose is to take tools away
    is the one outcome worth refusing to start over. So both are errors
    carrying the rewrite, and the agent does not launch until a human has
    said which of the two things they meant.

    Refusing a bare ``*`` costs the new grammar nothing: "every tool" is
    already what an empty spec says, so the star is only ever redundant
    there. What stays legal is ``*`` beside ``-name`` — ``["*", "-execute"]``
    — where the star is the baseline being subtracted from and the meaning
    is identical with or without it.
    """
    if not allow_all or deny_all or deny:
        return
    if allow:
        rewritten = ", ".join([DENY_ALL, *sorted(allow)])
        kept = ", ".join(sorted(allow))
        raise ToolSelectionError(
            f"'{ALLOW_ALL}' now means EVERY tool, so '{ALLOW_ALL}, {kept}' "
            f"would widen this agent instead of restricting it. It previously "
            f"meant 'no tools except {kept}', which is now written "
            f"'{rewritten}'. Refusing to guess: write '{rewritten}' to keep "
            f"only {kept}, or drop '{ALLOW_ALL}' to say the same thing more "
            f"briefly."
        )
    raise ToolSelectionError(
        f"'{ALLOW_ALL}' now means EVERY tool, so a bare '{ALLOW_ALL}' would "
        f"hand this agent the whole registry. It previously meant 'no tools "
        f"at all', which is now written '{DENY_ALL}' (or "
        f"'{DENY_ALL_ALIAS}'). Refusing to guess: write '{DENY_ALL_ALIAS}' "
        f"for no tools, or drop the selection entirely — every tool is "
        f"already the default."
    )


def parse_tool_selection(spec: Sequence[str] | None) -> ToolSelection:
    """Parse a tool-selection spec.

    Raises:
        ToolSelectionError: the spec is self-contradictory or malformed.
    """
    if not spec:
        return EMPTY_SELECTION

    deny_all = False
    allow_all = False
    allow: set[str] = set()
    deny: set[str] = set()
    rules: list[_Rule] = []

    for raw in spec:
        entry = str(raw).strip()
        if not entry:
            continue
        if entry[:1] in (INCREMENT_ADD, INCREMENT_REMOVE):
            name = entry[1:].strip()
            if not name:
                raise ToolSelectionError(
                    f"'{entry[:1]}' must be followed by a tool name or "
                    f"pattern, e.g. '{entry[:1]}serena__find_symbol'."
                )
            rules.append(_Rule(pattern=name, allow=entry[0] == INCREMENT_ADD))
            continue
        if entry == DENY_ALL or entry == DENY_ALL_ALIAS:
            deny_all = True
        elif entry == ALLOW_ALL:
            allow_all = True
        elif entry.startswith("-"):
            name = entry[1:].strip()
            if not name:
                raise ToolSelectionError(
                    "'-' must be followed by a tool name, e.g. '-execute'."
                )
            deny.add(name)
        else:
            allow.add(entry)

    _reject_legacy_deny_all(
        allow_all=allow_all, allow=allow, deny=deny, deny_all=deny_all
    )

    if deny_all and deny:
        raise ToolSelectionError(
            f"'{DENY_ALL}' already denies every tool, so combining it with "
            f"{sorted('-' + n for n in deny)} is a contradiction. Either list "
            f"the tools to keep after '{DENY_ALL}', or drop '{DENY_ALL}' and "
            f"list only the tools to remove."
        )
    if allow and deny:
        raise ToolSelectionError(
            f"{sorted(allow)} names tools to keep while "
            f"{sorted('-' + n for n in deny)} names tools to remove. Say one "
            f"or the other: a bare list keeps only what it names, and '-name' "
            f"removes only what it names."
        )

    # A bare list of names is an allow-list. Writing them out and getting
    # *more* than you listed would be the surprising reading, and this is
    # also the shape ``tools:`` had in config files before the field did
    # anything — those configs now mean what they always appeared to mean.
    #
    # ``allow`` is empty whenever ``*`` was written, because ``*`` with bare
    # names is rejected above — so a spec like ``["*", "-execute"]`` reaches
    # here as a pure deny-list and keeps its "every tool except" meaning.
    if allow and not deny_all:
        deny_all = True

    _reject_contradictory_rules(rules)
    return ToolSelection(
        deny_all=deny_all,
        allow=frozenset(allow),
        deny=frozenset(deny),
        rules=tuple(rules),
    )


def _reject_contradictory_rules(rules: Sequence[_Rule]) -> None:
    """Refuse a pattern that is both added and removed.

    Equal specificity with opposite directions has no non-arbitrary
    winner. Picking one silently would make the outcome depend on flag
    order — exactly the hidden precedence this feature must not have,
    since the spec is re-parsed from ``session_metadata.json`` on every
    restart and order there is an implementation detail.

    This catches only the case visible in the strings themselves: the
    SAME pattern in both directions. Two different patterns can tie too
    (``ab*`` against ``*ab`` over ``abab``), but seeing that requires
    knowing which tools exist — see ``validate_against_inventory``, with
    ``ToolSelection.permits`` resolving to deny for any tie that reaches
    it anyway.
    """
    by_pattern: dict[str, set[bool]] = {}
    for rule in rules:
        by_pattern.setdefault(rule.pattern, set()).add(rule.allow)
    contradictory = sorted(p for p, dirs in by_pattern.items() if len(dirs) > 1)
    if contradictory:
        raise ToolSelectionError(
            f"{contradictory} is both added and removed, so neither side "
            f"can win. Name it in one of them, or make one side more "
            f"specific — a literal name beats a pattern, and a longer "
            f"pattern beats a shorter one."
        )


def _split_names(value: str | None) -> list[str]:
    """Split one comma-separated ``--tools-add``/``--tools-remove`` value."""
    if not value:
        return []
    return [part.strip() for part in value.split(",") if part.strip()]


def parse_increment_spec(
    add: Sequence[str] | None, remove: Sequence[str] | None
) -> tuple[_Rule, ...]:
    """Build the incremental rule layer from the add/remove flags.

    Both flags are repeatable and each value may be comma-separated, so
    the two argument lists arrive already flattened by the caller.

    Raises:
        ToolSelectionError: a name is empty, or the same name is both
            added and removed at equal specificity.
    """
    rules: list[_Rule] = []
    for value in add or ():
        for name in _split_names(value):
            rules.append(_Rule(pattern=name, allow=True))
    for value in remove or ():
        for name in _split_names(value):
            rules.append(_Rule(pattern=name, allow=False))

    _reject_contradictory_rules(rules)
    return tuple(rules)


#: Marks an incremental entry inside the flat spec list. The add/remove
#: flags have to survive in ``session_metadata.json`` alongside the base
#: spec, and the storage schema is a list of strings — so the layer is
#: encoded in that same list rather than by widening the schema. ``+name``
#: adds, ``~name`` removes; both carry a pattern and a specificity, which
#: is what lets the more specific of two overlapping rules win.
INCREMENT_ADD = "+"
INCREMENT_REMOVE = "~"


def encode_increments(rules: Sequence[_Rule]) -> list[str]:
    """Render the rule layer as spec entries that round-trip a restart."""
    return [
        (INCREMENT_ADD if r.allow else INCREMENT_REMOVE) + r.pattern
        for r in rules
    ]


def merge_increments(spec: Sequence[str], rules: Sequence[_Rule]) -> list[str]:
    """Layer new add/remove *rules* onto an existing *spec*.

    A later rule for the same pattern replaces the earlier one, so
    restarting with ``--tools-add grep`` after ``--tools-rm grep`` leaves
    one rule rather than two contradictory ones. Order is otherwise
    preserved, and it carries no meaning: overlapping rules are decided
    by specificity, and a tie between two different patterns resolves to
    deny in ``ToolSelection.permits`` rather than to whichever was
    written first. Appending here therefore cannot make a new rule lose
    to an old one by position alone.
    """
    if not rules:
        return list(spec)
    superseded = {r.pattern for r in rules}
    kept = [
        entry
        for entry in spec
        if not (
            entry[:1] in (INCREMENT_ADD, INCREMENT_REMOVE)
            and entry[1:] in superseded
        )
    ]
    return [*kept, *encode_increments(rules)]


def parse_tool_selection_spec(value: str | None) -> list[str]:
    """Split the comma-separated CLI form into the config's list form.

    Validation happens by parsing the result, so a malformed ``--tools``
    is rejected at launch rather than at agent construction.
    """
    if value is None:
        return []
    spec = [part.strip() for part in value.split(",") if part.strip()]
    parse_tool_selection(spec)
    return spec


def parse_tool_selection_flag(value: str | None) -> ToolSelection:
    """Parse the comma-separated CLI form, e.g. ``--tools '*,read_file'``."""
    if value is None:
        return EMPTY_SELECTION
    return parse_tool_selection([part for part in value.split(",")])


def describe_selection(selection: ToolSelection) -> str:
    """One line stating what the agent is left with, for logs and notices."""
    if not selection.restricts:
        return "all tools"
    if selection.deny_all:
        base = "no tools" if not selection.allow else (
            "only " + ", ".join(sorted(selection.allow))
        )
    elif selection.deny:
        base = "all tools except " + ", ".join(sorted(selection.deny))
    else:
        base = "all tools"
    if not selection.rules:
        return base
    # The increments decide the outcome for anything they match, so a
    # description that omitted them would understate the restriction.
    added = sorted(r.pattern for r in selection.rules if r.allow)
    removed = sorted(r.pattern for r in selection.rules if not r.allow)
    parts = [base]
    if removed:
        parts.append("minus " + ", ".join(removed))
    if added:
        parts.append("plus " + ", ".join(added))
    return " ".join(parts)


# Providers whose CLI runs its own built-in tools that no channel of ours
# can switch off. Verified per provider; see docs/DESIGN_TOOL_SELECTION.md
# §4 for the evidence and ``acp/builtin_tool_passthrough.py`` for the
# protocol-level statement ("they cannot be told not to").
_UNSUPPRESSABLE_NATIVES: dict[str, str] = {
    "claude-code": (
        "the ACP adapter exposes no tool-configuration channel — session/new "
        "carries only {cwd, mcpServers} — so its built-ins (Bash, Read, Write, "
        "Edit, Glob, Grep, WebFetch, WebSearch, Task, …) stay callable"
    ),
    "grok": (
        "the Grok CLI runs its built-ins locally in 'agent stdio' mode and "
        "cannot be told not to, so they stay callable"
    ),
    "codex": (
        "Codex hard-codes apply_patch, update_plan, view_image, "
        "request_user_input and the three list/read_mcp_resource tools; no "
        "--disable flag removes them (the rest are already disabled)"
    ),
}


# Sections of ``default_agent_prompt.md`` that exist to teach ONE tool,
# keyed by the exact heading line and mapped to the tools they describe. A
# section is dropped when none of its tools survive the selection.
#
# The base prompt is the largest single source of tool advertising, and it
# is static text — it never passes through ``wrap_model_call``, so the
# middleware's block-level pruning cannot reach it. Leaving it intact is
# how a "only project_memory" agent still reads three screens explaining
# ``read_file``, ``execute`` and ``send_message_to_agent``: an invitation
# to call them, and (before the dispatch barrier) an effective one.
#
# Prose OUTSIDE these sections is left alone. Rewriting the agent's
# general instructions to avoid naming a tool would change behaviour far
# beyond tool availability, so the rule is deliberately narrow: whole
# sections whose entire subject is a removed tool, nothing else.
_PROMPT_SECTIONS: tuple[tuple[str, frozenset[str]], ...] = (
    ("## Working with Subagents (task tool)", frozenset({"task"})),
    ("### shell", frozenset({"execute"})),
    (
        "### File Tools",
        frozenset({"ls", "read_file", "write_file", "edit_file", "glob", "grep"}),
    ),
    ("## File Reading Best Practices", frozenset({"read_file"})),
    ("### web_search", frozenset({"web_search"})),
    ("### http_request", frozenset({"http_request"})),
    ("### fetch_url", frozenset({"fetch_url"})),
    ("### read_image", frozenset({"read_image"})),
    (
        "### send_message_to_agent — Inter-Agent Messaging",
        frozenset({"send_message_to_agent"}),
    ),
    ("### restart_agent — Agent Restart Tool", frozenset({"restart_agent"})),
    ("### restart_self — Self-Restart Tool", frozenset({"restart_self"})),
)

# ``## Task Management`` is two lines whose whole subject is write_todos,
# but it carries no tool-specific heading marker of its own; it is listed
# here so the same rule covers it.
_PROMPT_SECTIONS_EXTRA: tuple[tuple[str, frozenset[str]], ...] = (
    ("## Task Management", frozenset({"write_todos"})),
)


def _section_bounds(lines: list[str], heading: str) -> tuple[int, int] | None:
    """Half-open line range of the section introduced by *heading*.

    A section runs to the next heading of the SAME OR SHALLOWER depth, so
    ``## Tools`` is not swallowed by its own ``### shell`` child.
    """
    depth = len(heading) - len(heading.lstrip("#"))
    for i, line in enumerate(lines):
        if line.strip() != heading:
            continue
        for j in range(i + 1, len(lines)):
            stripped = lines[j].lstrip()
            if not stripped.startswith("#"):
                continue
            other = len(stripped) - len(stripped.lstrip("#"))
            if other <= depth:
                return i, j
        return i, len(lines)
    return None


def strip_disabled_prompt_sections(
    prompt: str, selection: ToolSelection
) -> str:
    """Remove base-prompt sections describing tools the selection removed.

    Returns *prompt* unchanged when the selection restricts nothing, so
    the default path is byte-identical to having no selection at all.
    """
    if not selection.restricts:
        return prompt

    lines = prompt.splitlines(keepends=True)
    doomed: list[tuple[int, int]] = []
    for heading, owners in (*_PROMPT_SECTIONS, *_PROMPT_SECTIONS_EXTRA):
        if any(selection.permits(name) for name in owners):
            continue
        bounds = _section_bounds(lines, heading)
        if bounds is not None:
            doomed.append(bounds)

    if not doomed:
        return prompt
    cut = {i for start, end in doomed for i in range(start, end)}
    kept = [line for i, line in enumerate(lines) if i not in cut]
    return "".join(_drop_empty_parents(kept))


# Container headings that exist only to introduce the per-tool sections
# above them. Once every child is gone, the bare heading is left pointing
# at nothing — "## Tools" followed immediately by "## Code References"
# reads as a tool list the model cannot see rather than as an absence.
_CONTAINER_HEADINGS: frozenset[str] = frozenset({"## Tools"})


def _drop_empty_parents(lines: list[str]) -> list[str]:
    """Drop container headings whose body was emptied by the strip."""
    out: list[str] = []
    for i, line in enumerate(lines):
        if line.strip() not in _CONTAINER_HEADINGS:
            out.append(line)
            continue
        depth = len(line.strip()) - len(line.strip().lstrip("#"))
        has_body = False
        for nxt in lines[i + 1:]:
            stripped = nxt.lstrip()
            if stripped.startswith("#"):
                other = len(stripped) - len(stripped.lstrip("#"))
                if other <= depth:
                    break
                has_body = True
                break
            if nxt.strip():
                has_body = True
                break
        if has_body:
            out.append(line)
    return out


def native_tool_caveat(provider: str | None, selection: ToolSelection) -> str | None:
    """Warn when a restriction cannot reach a provider's native tools.

    A restricted run is a measurement, and a measurement that silently
    leaks the very tools it claims to have removed is worse than no
    measurement. Returns ``None`` when there is nothing to disclose —
    either the selection restricts nothing, or the provider's natives are
    fully suppressed (``claude-code-sdk`` blocks its built-ins outright).
    """
    if not selection.restricts or not provider:
        return None
    detail = _UNSUPPRESSABLE_NATIVES.get(provider)
    if detail is None:
        return None
    return (
        f"Tool selection ({describe_selection(selection)}) applies to "
        f"deep-worker tools only: on provider '{provider}' {detail}. "
        f"For a hermetic run use provider 'claude-code-sdk'."
    )
