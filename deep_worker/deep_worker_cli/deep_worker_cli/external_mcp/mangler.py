"""MCPToolNameMangler — a bijection between canonical MCP names (with
dots/slashes, as allowed by the MCP spec) and provider-safe names
(only ``^[a-zA-Z0-9_-]+$``, so OpenAI-class API and Claude-Code CLI
validation passes without silent mangling).

See docs/DESIGN_EXTERNAL_MCP_TOOL_NAME_MANGLING.md.

Strategy:

* Mangling is a pure function ``mangle()``: every ``.`` and ``/`` → ``__``
  (double underscore). No ``__DOT__``/``__SLASH__`` markers — only
  something a human can read without decoding.
* ``forward``/``reverse`` store ONLY actually-mangled names.
  Lookups return the argument as-is when it is not in the map:
  ``mangled = forward.get(canonical, canonical)``.
* Collisions (two canonicals → one mangled, or a mangled name clashing
  with a literal unmangled name) fail loud with
  ``MCPToolNameCollisionError``. A single check through the
  ``_all_mangled`` set — see §3.2 of the design doc.
"""
from __future__ import annotations


class MCPToolNameCollisionError(ValueError):
    """Two different canonical names produced the same provider-safe name, OR
    a new mangled name collided with an already-registered literal name."""

    def __init__(self, canonical: str, mangled: str) -> None:
        super().__init__(
            f"MCP tool name collision: adding {canonical!r} would produce "
            f"the mangled name {mangled!r}, which is already in use by "
            f"another tool registered in this mangler. Rename one of the "
            f"MCP tools (e.g. use distinct namespacing) — see "
            f"docs/DESIGN_EXTERNAL_MCP_TOOL_NAME_MANGLING.md §3.2."
        )
        self.canonical = canonical
        self.mangled = mangled


def mangle(canonical: str) -> str:
    """Pure function: every ``.`` and ``/`` → ``__``. Everything else is
    left unchanged (the name is already valid for all known providers).

    Idempotent: ``mangle(mangle(x)) == mangle(x)``.
    """
    # A single replacement pass — simple and fast. Order does not matter
    # (two independent replacements), but we fix it for determinism.
    return canonical.replace(".", "__").replace("/", "__")


class MCPToolNameMangler:
    """Per-agent, pool-level bijection of canonical ↔ provider-safe names.

    Contract:

    * ``add(canonical) -> mangled`` — registers a tool. Idempotent for a
      repeated ``add`` of the same ``canonical``. A collision on the final
      name → ``MCPToolNameCollisionError``.
    * ``to_mangled(canonical)`` / ``to_canonical(mangled)`` — lookup,
      returns the argument by default (for names without mangling).
    * ``forward`` / ``reverse`` — public for introspection (UI/tests);
      contain ONLY mangled pairs.
    """

    __slots__ = ("forward", "reverse", "_all_mangled")

    def __init__(self) -> None:
        self.forward: dict[str, str] = {}
        self.reverse: dict[str, str] = {}
        # All final names ever issued (mangled AND literal).
        # Collisions are caught against exactly this set — see §3.2 of the design doc.
        self._all_mangled: set[str] = set()

    def add(self, canonical: str) -> str:
        """Register a canonical name, return the provider-safe one.

        Idempotent: a repeated ``add`` of the same ``canonical`` is a no-op
        and returns the same mangled name. This matters for repeated init/retry.
        A collision with a DIFFERENT already-registered name —
        ``MCPToolNameCollisionError``.
        """
        m = mangle(canonical)

        # Idempotency for the same canonical (mangled or not):
        # if canonical→m is already recorded, a repeated add is a no-op.
        if canonical in self.forward and self.forward[canonical] == m:
            return m
        # For unmangled names: adding the same literal name again is also a
        # no-op (m in _all_mangled with m == canonical and no forward entry
        # means "this literal was already registered").
        if m == canonical and m in self._all_mangled and canonical not in self.reverse:
            return m

        # Single collision check (see §3.2 of the design doc): any final
        # name already issued blocks re-issuing the same mangled name to a
        # different canonical.
        if m in self._all_mangled:
            raise MCPToolNameCollisionError(canonical, m)

        if m != canonical:
            # Only actually-mangled names go into forward/reverse.
            self.forward[canonical] = m
            self.reverse[m] = canonical
        # _all_mangled is ALWAYS updated — it is the source of truth for
        # final-name uniqueness.
        self._all_mangled.add(m)
        return m

    def to_mangled(self, canonical: str) -> str:
        """canonical → provider-safe. For unregistered and unmangled
        names, returns the argument as-is."""
        return self.forward.get(canonical, canonical)

    def to_canonical(self, mangled: str) -> str:
        """provider-safe → canonical. For unregistered and unmangled
        names, returns the argument as-is."""
        return self.reverse.get(mangled, mangled)

    def all_mangled_names(self) -> frozenset[str]:
        """Snapshot of all provider-safe names already issued (mangled AND
        literal). Used for two-phase commit in the pool
        (see ExternalMCPClientPool._start_all)."""
        return frozenset(self._all_mangled)

    def all_canonical_names(self) -> frozenset[str]:
        """Snapshot of all canonical names known to the mangler.

        Includes both mangled ones (forward.keys()) and literal ones
        (names from ``_all_mangled`` not present in ``reverse`` — they
        were not mangled, so canonical == mangled)."""
        literal_names = self._all_mangled - set(self.reverse)
        return frozenset(self.forward) | literal_names
