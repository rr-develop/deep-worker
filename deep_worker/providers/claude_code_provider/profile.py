"""claude-code provider profile — pins the claude-agent-acp adapter.

Preflight (DESIGN §5.2): the adapter is installed OUTSIDE the wheel at
``~/.deep_worker/acp/claude-agent-acp`` (npm install --save-exact with the
owner-approved pinned version). The provider refuses to run on a missing or
version-mismatched adapter — a broken ACP handshake is far worse than a
clear install error at startup.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import subprocess
from pathlib import Path
from typing import Any

from ..acp.profile import AcpProfile, PermissionPolicy

logger = logging.getLogger(__name__)

# Owner-gate Q3: exact pinned adapter version. Compatible ranges are ONLY
# allowed with explicit owner approval — a bump must go through design review.
PINNED_ADAPTER_VERSION = "0.68.0"
ADAPTER_PACKAGE = "@agentclientprotocol/claude-agent-acp"
ADAPTER_ROOT = os.path.expanduser("~/.deep_worker/acp/claude-agent-acp")
ADAPTER_ENTRY = (
    Path(ADAPTER_ROOT)
    / "node_modules"
    / ADAPTER_PACKAGE
    / "dist"
    / "index.js"
)

# Live-verified 0.68.0. ``ToolSearch`` is Claude Code's tool-discovery
# built-in (surfaces as a regular tool_call whose rawInput carries
# ``select:mcp__<server>__<tool>``); ``Skill``/``AskUserQuestion``/
# ``ReportFindings``/``ExitPlanMode`` are CLI built-ins that can surface as
# tool_calls. ``TodoWrite`` and ``Task*`` are suppressed by the adapter
# (plan lane) and never surface as tool_calls — deliberately not listed.
KNOWN_BUILTINS: frozenset[str] = frozenset(
    {
        "Bash",
        "Read",
        "Write",
        "Edit",
        "Glob",
        "Grep",
        "WebFetch",
        "WebSearch",
        "Agent",
        "Task",
        "Skill",
        "AskUserQuestion",
        "ReportFindings",
        "ExitPlanMode",
        "ToolSearch",
    }
)

# claude-code-sdk usage-limit markers (live-verified wording). The generic
# markers in limit_errors already cover "hit your .* limit"; the claude-code-sdk
# banner is ``You've hit your session limit · resets <TIME>``.
EXTRA_LIMIT_MARKERS: tuple[re.Pattern, ...] = (
    re.compile(r"You['\u2019]ve hit your session limit"),
    re.compile(r"session limit \u00b7 resets", re.IGNORECASE),
)


# Model selection (owner-reported defect: ``--model`` was a dead setting —
# the value reached the Python layer, was echoed into token stats, and was
# never sent to the CLI, so the agent silently ran Claude Code's default).
#
# The pinned adapter takes the model from the ENVIRONMENT. Its documented
# priority (dist/acp-agent.js, live-verified 0.68.0) is:
#   1. ANTHROPIC_MODEL   2. settings.model   3. resumed session's live model
#   4. models[0] default
# ANTHROPIC_MODEL is therefore the highest-priority channel, and the adapter
# re-asserts the override via ``setModel`` on resumed sessions too, so the pin
# survives ``session/load``.
MODEL_ENV_VAR = "ANTHROPIC_MODEL"


class AcpAdapterError(RuntimeError):
    """Adapter missing, unusable, or version-mismatched (preflight failure)."""


class AcpModelError(ValueError):
    """The pinned ``ANTHROPIC_MODEL`` is not offered by the live adapter
    picker for this account/version.

    Raised at ``session/new`` (not at construction) because the whitelist
    lives on the adapter side — see :func:`validate_session_new_reply`.
    The message includes the ACTUAL picker options from the live reply
    and the adapter's current resolved value.
    """


# The adapter is the SOURCE OF TRUTH for accepted model values, and it tells
# us its verdict in every ``session/new`` reply. We MUST NOT hardcode a
# Python-side whitelist. Owner rule (2026-08-15, repeated): "HARDCODED MODEL
# NAMES ARE STRICTLY FORBIDDEN, because they will go stale within a month. The
# provider must use the current version of the model of the specified class
# unless the user supplied an exact model name. The provider must accept the
# short name."
#
# Contract:
#   * ``--model`` empty/None → no ANTHROPIC_MODEL pin; the adapter selects
#     its ``default`` option, which is always the current default class.
#   * ``--model <name>`` → passed VERBATIM to ANTHROPIC_MODEL. The adapter
#     resolves aliases, case, and versioned ids on its own.
#   * At session/new we read the adapter's verdict out of the model picker.
#
# HOW THE VERDICT IS ENCODED (live-verified 0.68.0, this is NOT the naive
# "is the pin one of the listed values?" check — that check always passes):
# the adapter INJECTS the pinned value into the picker it returns, so the pin
# is present in ``options`` no matter what. What differs is the metadata it
# attaches to that injected entry:
#
#   pin                        currentValue          injected entry
#   -------------------------  --------------------  ----------------------------
#   fable                      'fable'               name='Fable',  real description
#                                                    (REPLACES claude-fable-5[1m])
#   opusplan                   'opusplan'            name='Opus Plan Mode', real desc
#   claude-sonnet-4-20250514   same                  name='Sonnet 4', real desc
#   Opus                       'opus[1m]'            (normalized, nothing injected)
#   claude-opus-4-20250514     'default'             (pin DROPPED, nothing injected)
#   bogus-9999 / opud          same as pin           name==value, description='Custom model'
#
# So an UNRECOGNIZED model is exactly the entry the adapter labels
# ``Custom model``: it echoes the string back with no real display name and
# no real description. Those are the values that later blow up at
# ``session/prompt`` with ``model_not_found`` — which, without this gate,
# surfaces to the operator as a mute, resultless turn.
#
# We deliberately key on the description marker rather than on membership:
# membership is useless (always true), and the marker is what the adapter
# itself uses to say "I do not know this one, I am passing it through".

MODEL_CONFIG_OPTION_ID = "model"

# The adapter's own label for a model string it does not recognise and is
# merely passing through to the API (dist/acp-agent.js, live-verified).
UNKNOWN_MODEL_DESCRIPTION = "custom model"

# The picker entry meaning "whatever the current default class resolves to".
# A nonempty pin that the adapter drops comes back resolved to THIS value.
DEFAULT_MODEL_OPTION_VALUE = "default"


def _picker_entry(reply: dict) -> tuple[list[dict], str | None]:
    """Extract (options_list, currentValue) from a session/new reply.

    Returns ``([], None)`` when the adapter did not include a model picker
    (older versions or degenerate replies) — the caller must treat that as
    "no verdict available" and skip validation rather than fabricate one.
    Never raises.
    """
    result = reply.get("result") if isinstance(reply, dict) else None
    if not isinstance(result, dict):
        return [], None
    config_options = result.get("configOptions")
    if not isinstance(config_options, list):
        return [], None
    for entry in config_options:
        if not isinstance(entry, dict):
            continue
        if entry.get("id") != MODEL_CONFIG_OPTION_ID:
            continue
        options = entry.get("options")
        current = entry.get("currentValue")
        current = current if isinstance(current, str) else None
        return (options if isinstance(options, list) else []), current
    return [], None


def _known_option_values(options: list) -> list[str]:
    """Picker values the adapter actually KNOWS — i.e. everything except the
    passthrough entries it labels ``Custom model``. This is what we show the
    operator when their pin is rejected."""
    known: list[str] = []
    for opt in options:
        if not isinstance(opt, dict):
            continue
        value = opt.get("value")
        if not isinstance(value, str):
            continue
        description = opt.get("description")
        if (
            isinstance(description, str)
            and description.strip().lower() == UNKNOWN_MODEL_DESCRIPTION
        ):
            continue
        known.append(value)
    return known


def validate_session_new_reply(
    reply: dict, env: dict[str, str] | None
) -> None:
    """Post-session/new gate for claude-code.

    Reads the adapter's verdict on the pinned ``ANTHROPIC_MODEL`` out of the
    live model picker and raises :class:`AcpModelError` in the two cases
    where the requested model is NOT what the session will actually run:

    * **Unrecognized passthrough** — the pin appears in the picker labelled
      ``description == "Custom model"``. The adapter does not know it and is
      forwarding the raw string; the first prompt fails with
      ``model_not_found``, which without this gate surfaces as a mute,
      resultless turn.
    * **Dropped pin** — the pin is nonempty and absent from the picker while
      the adapter fell back to its ``default`` entry. Live-verified with
      ``claude-opus-4-20250514``: ``currentValue`` comes back ``'default'``
      and the pin is nowhere in ``options``. Inference would silently
      proceed on a DIFFERENT model than the operator requested — exactly the
      requested-vs-executed divergence this whole change exists to remove
      (flagged as blocking in review). We fail loudly instead of warning,
      because a warning still lets the wrong model run.

      We deliberately do NOT try to map a dated id onto its class here: the
      only class signal the adapter exposes is prose in the picker
      (``name='Opus (1M context)'``), and matching on that is a model-name
      hardcode in a different costume — forbidden by the owner rule. The
      honest move is to fail and name the short class values the adapter
      offers right now. ``--model default`` (or no ``--model``) remains the
      supported way to ask for the current default class.

    Fail-open cases that must NOT raise:
      * no pin (operator wants the CLI's default);
      * adapter did not report a model picker (older or non-standard reply —
        we cannot invent a verdict, and prompt-time failure is the residual
        guard);
      * the pin is present with real metadata (adapter knows it), or absent
        while ``currentValue`` is something OTHER than ``default`` — that is
        a genuine normalization, live-verified with ``Opus`` → ``opus[1m]``.

    Comparison is case-insensitive (the adapter itself is, live-verified).
    """
    pinned = (env or {}).get(MODEL_ENV_VAR)
    if not pinned:
        return
    options, current = _picker_entry(reply)
    if not options:
        return
    lowered = pinned.strip().lower()
    listed: dict | None = None
    for opt in options:
        if not isinstance(opt, dict):
            continue
        value = opt.get("value")
        if isinstance(value, str) and value.strip().lower() == lowered:
            listed = opt
            break

    if listed is not None:
        description = listed.get("description")
        if not (
            isinstance(description, str)
            and description.strip().lower() == UNKNOWN_MODEL_DESCRIPTION
        ):
            return  # adapter knows this one
        reason = (
            f"the adapter does not recognise model {pinned!r} (it listed it "
            f'as a passthrough "Custom model"), so the first prompt would '
            f"fail with model_not_found"
        )
    else:
        resolved_default = (
            current is not None
            and current.strip().lower() == DEFAULT_MODEL_OPTION_VALUE
            and lowered != DEFAULT_MODEL_OPTION_VALUE
        )
        if not resolved_default:
            return  # normalized to a concrete option (e.g. Opus -> opus[1m])
        reason = (
            f"the adapter DROPPED model {pinned!r} and fell back to its "
            f"{DEFAULT_MODEL_OPTION_VALUE!r} entry, so this session would "
            f"silently run a different model than requested"
        )

    known = _known_option_values(options)
    pretty = ", ".join(known) if known else "(none reported)"
    resolved = current if current else "(unknown)"
    raise AcpModelError(
        f"claude-code: {reason}. Models it offers right now: [{pretty}]. "
        f"Session value: {resolved!r}. Pass --model with one of those (the "
        f"short class names work), or drop --model entirely to run the "
        f"adapter's own default, which always tracks the current version of "
        f"the default class."
    )


def _read_package_version(package_dir: Path) -> str | None:
    pkg_json = package_dir / "package.json"
    try:
        data = json.loads(pkg_json.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    version = data.get("version")
    return str(version) if isinstance(version, str) and version else None


# Process-wide cache of the parsed `node --version` major (0 = node broken).
# Shared by the sync probe (`_node_major`, CLI diagnostics outside a running
# loop) and the async probe (`_node_major_async`, startup preflight) — one
# spawn per process, whichever path runs first.
_NODE_MAJOR_CACHE: tuple[int, ...] | None = None


def _node_major() -> int:
    """Return the installed node major version, 0 when node is unusable.

    SYNC probe — only call from a non-async context (CLI diagnostics, tests).
    Cached process-wide: one quick ``node --version`` subprocess per process
    (short 5s timeout), never per model construction. Inside a running event
    loop use :func:`_node_major_async` instead — a sync subprocess here would
    freeze every asyncio task (review item 7).
    """
    global _NODE_MAJOR_CACHE
    if _NODE_MAJOR_CACHE is not None:
        return _NODE_MAJOR_CACHE[0]
    try:
        proc = subprocess.run(
            ["node", "--version"],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        _NODE_MAJOR_CACHE = (0,)
        return 0
    version = proc.stdout.strip() or proc.stderr.strip()
    match = re.match(r"v?(\d+)", version)
    major = int(match.group(1)) if match else 0
    _NODE_MAJOR_CACHE = (major,)
    return major


def _node_version_ok() -> bool:
    """True when node is present AND its major version is >= 22 (the
    claude-agent-acp adapter's engines requirement, review item 7).

    SYNC — same non-async caveat as :func:`_node_major`; prefer
    :func:`_node_version_ok_async` inside a running event loop."""
    return _node_major() >= 22


async def _node_major_async() -> int:
    """Async ``node --version`` probe (create_subprocess_exec).

    Never blocks the event loop (review item 7). Shares the process-wide
    cache with the sync probe — whichever path runs first wins the single
    spawn. 0 = node missing/unusable.
    """
    global _NODE_MAJOR_CACHE
    if _NODE_MAJOR_CACHE is not None:
        return _NODE_MAJOR_CACHE[0]
    try:
        proc = await asyncio.create_subprocess_exec(
            "node", "--version",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=5)
    except (OSError, asyncio.TimeoutError):
        _NODE_MAJOR_CACHE = (0,)
        return 0
    version = (
        stdout.decode(errors="replace") or stderr.decode(errors="replace")
    ).strip()
    match = re.match(r"v?(\d+)", version)
    major = int(match.group(1)) if match else 0
    _NODE_MAJOR_CACHE = (major,)
    return major


async def _node_version_ok_async() -> bool:
    """Async node >= 22 gate — non-blocking (review item 7)."""
    return (await _node_major_async()) >= 22


def _adapter_version_from_package() -> str | None:
    """Trusted metadata: the pinned adapter's installed version from its
    package.json — NO node spawn, so preflight never blocks the event loop
    during async agent init (review item 7)."""
    return _read_package_version(Path(ADAPTER_ROOT) / "node_modules" / ADAPTER_PACKAGE)


async def _smoke_check_version_async() -> str:
    """LIVE async smoke: run ``node <entry> --version`` via asyncio.

    Used by diagnostics when a caller has a running loop and wants the
    actually-reported version (not just trusted package.json metadata).
    Never blocks the event loop (no subprocess.run inside async code).
    """
    if not ADAPTER_ENTRY.is_file():
        raise AcpAdapterError(
            f"claude-agent-acp adapter not installed at {ADAPTER_ENTRY}"
        )
    proc = await asyncio.create_subprocess_exec(
        "node", str(ADAPTER_ENTRY), "--version",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=30)
    if proc.returncode != 0:
        raise AcpAdapterError(
            f"claude-agent-acp --version probe failed "
            f"(rc={proc.returncode}): {stderr.decode(errors='replace').strip()[:300]}"
        )
    return (stdout.decode(errors="replace") or stderr.decode(errors="replace")).strip()


# Where the CLI keeps its own per-session transcript. Live-verified: every
# ACP ``sessionId`` we have ever created has a file named after it here,
# and the directory is a mangling of the session's cwd. We do NOT rebuild
# that mangling — it has corners (``.deep_worker`` becomes
# ``--deep-worker``) and it is the CLI's private business. The session id
# is unique, so a glob across project directories finds the file without
# encoding anyone else's convention.
SESSION_TRANSCRIPT_ROOT = Path.home() / ".claude" / "projects"


def read_session_model(
    session_id: str, *, root: Path | None = None
) -> str | None:
    """Return the model the CLI reports for ``session_id``, or ``None``.

    The ACP adapter never says which model it is running: the whole
    ``initialize`` → ``session/new`` → ``session/prompt`` exchange never
    mentions one, and ``session/new``'s picker reports ``currentValue:
    "default"`` — a selector, not a name. The name appears only in the
    picker's prose (``description: "Opus (1M context)"``), and reading it
    from there would be a model-name hardcode in a different costume
    (owner rule, see :func:`validate_session_new_reply`).

    But the CLI writes its own transcript, and every assistant record in
    it carries the ``model`` the API actually answered with, next to that
    request's real ``usage``. That is a report of what happened, not a
    declaration of intent — the strongest source available, and it costs
    nothing: the file is already on disk, written by the session we are
    already running. We do not spawn anything to ask.

    Returns the model from the LAST assistant record (a resumed session's
    earlier turns could name an older model), or ``None`` when the
    transcript does not exist yet or names no model — which is the honest
    answer before the session's first reply.
    """
    base = root if root is not None else SESSION_TRANSCRIPT_ROOT
    if not session_id:
        return None
    model: str | None = None
    try:
        for path in base.glob(f"*/{session_id}.jsonl"):
            with path.open("r", errors="replace") as fh:
                for line in fh:
                    try:
                        record = json.loads(line)
                    except ValueError:
                        continue
                    message = record.get("message")
                    if not isinstance(message, dict):
                        continue
                    found = message.get("model")
                    # '<synthetic>' marks a locally generated message (an
                    # error stub, a cancellation) — the API never answered,
                    # so it names no model.
                    if (
                        isinstance(found, str)
                        and found
                        and not found.startswith("<")
                    ):
                        model = found
    except OSError as exc:
        logger.warning(
            "[claude-code] could not read the session transcript for %s "
            "(%s: %s); the reported model stays unset",
            session_id, type(exc).__name__, exc,
        )
        return None
    return model


# Process-wide async preflight future (review item 7). A failed preflight
# resets it so the next start retries; a successful one is remembered for
# the life of the process.
_ASYNC_PREFLIGHT: asyncio.Future | None = None


def reset_async_preflight() -> None:
    """Test hook: forget a completed/failed async preflight."""
    global _ASYNC_PREFLIGHT
    _ASYNC_PREFLIGHT = None


async def preflight_async() -> None:
    """Async §5.2 preflight: node >= 22 + live adapter ``--version`` smoke.

    Runs INSIDE the event loop using ``create_subprocess_exec`` — a sync
    ``subprocess.run`` here would freeze every asyncio task during async
    agent/TaskManager init (review item 7). Cached process-wide; concurrent
    first-callers share one probe. Called from
    ``ClaudeCodeAcpChatModel._ensure_acp_started`` before the handshake.
    """
    global _ASYNC_PREFLIGHT
    fut = _ASYNC_PREFLIGHT
    if fut is not None:
        await asyncio.shield(fut)
        return
    loop = asyncio.get_running_loop()
    fut = loop.create_future()
    _ASYNC_PREFLIGHT = fut
    try:
        if not ADAPTER_ENTRY.is_file():
            raise AcpAdapterError(
                f"claude-agent-acp adapter not installed at {ADAPTER_ENTRY} — "
                f"run: mkdir -p ~/.deep_worker/acp && npm install --save-exact "
                f"--prefix ~/.deep_worker/acp/claude-agent-acp "
                f"{ADAPTER_PACKAGE}@{PINNED_ADAPTER_VERSION}"
            )
        major = await _node_major_async()
        if major < 22:
            raise AcpAdapterError(
                "claude-code requires node >= 22 (adapter engines) — "
                f"found node major {major or 'missing'}"
            )
        # Live smoke: the adapter binary actually starts and reports a
        # version (raises AcpAdapterError on any nonzero exit).
        await _smoke_check_version_async()
        fut.set_result(None)
    except BaseException as exc:  # noqa: BLE001 — cache must reset on failure
        _ASYNC_PREFLIGHT = None  # transient failure — allow a retry
        fut.set_exception(exc)
        raise


def build_profile(
    *,
    model: str | None = None,
    auto_approve: bool = False,
    verify: bool = True,
) -> AcpProfile:
    """Build the claude-code :class:`AcpProfile`.

    ``verify=True`` (default) runs the §5.2 preflight WITHOUT spawning node
    (review item 7): adapter entry present, and the adapter version from its
    installed package.json matches ``PINNED_ADAPTER_VERSION`` (trusted
    metadata — a file read, never a subprocess). The node >= 22 engine gate
    is NOT enforced here with a sync spawn (a model is frequently
    constructed while the agent's event loop is running, and a 5s
    ``subprocess.run`` would freeze every asyncio task); it runs ASYNC at
    the first ``ainvoke`` via :func:`preflight_async`. If a sync context
    already probed node (cache warm), we still fail fast here on a bad
    version. Preflight failure raises :class:`AcpAdapterError` with install
    diagnostics.

    ``auto_approve`` mirrors deep-worker's ``-sp`` operator flag: with it
    EVERY permission request is approved; with ``auto_approve=False``
    NOTHING is approved (fail-closed). There is no per-tool allowlist —
    see :class:`PermissionPolicy` for why one must never come back.

    ``model`` pins the CLI's model via ``ANTHROPIC_MODEL``. The value is
    passed through to the adapter verbatim (whitespace-stripped) — the
    adapter itself owns the whitelist and reports it in the ``session/new``
    picker; the shared client validates the pin against that live picker at
    session-creation time (see :func:`validate_session_new_reply`). ``None``
    or empty leaves the CLI's own default in place, which is how "the
    current version of the default class" is honoured without any code
    knowing today's model id.
    """
    if verify:
        # Warm-cache fail-fast only: the authoritative node >= 22 check is
        # async (preflight_async) so construction never blocks the loop.
        if _NODE_MAJOR_CACHE is not None and not _node_version_ok():
            raise AcpAdapterError(
                "claude-code requires node >= 22 (adapter engines) — "
                f"found node major {_NODE_MAJOR_CACHE[0]}"
            )
        if not ADAPTER_ENTRY.is_file():
            raise AcpAdapterError(
                f"claude-agent-acp adapter not installed at {ADAPTER_ENTRY} — "
                f"run: mkdir -p ~/.deep_worker/acp && npm install --save-exact "
                f"--prefix ~/.deep_worker/acp/claude-agent-acp "
                f"{ADAPTER_PACKAGE}@{PINNED_ADAPTER_VERSION}"
            )
        reported = _adapter_version_from_package()
        if reported != PINNED_ADAPTER_VERSION:
            raise AcpAdapterError(
                f"claude-agent-acp version mismatch: pinned "
                f"{PINNED_ADAPTER_VERSION}, found {reported!r}. Compatible "
                f"ranges require explicit owner approval — bump the pin in "
                f"claude_code_provider/profile.py through design review."
            )
    else:
        # Still fail on a hard-missing entry; only the checks that need node
        # are skipped (unit tests exercise the profile shape without node).
        if not ADAPTER_ENTRY.is_file():
            raise AcpAdapterError(
                f"claude-agent-acp adapter not installed at {ADAPTER_ENTRY}"
            )

    # No allowlist. It used to default to KNOWN_BUILTINS — the built-ins
    # known when that line was written — so the first tool Anthropic added
    # would have been denied under ``-sp`` with no warning anywhere. grok hit
    # exactly that failure mode on 2026-08-24 and lost every shell tool for
    # four days; the lesson is not "keep the list fresh" but "do not keep a
    # list". ``-sp`` approves whatever the CLI asks about.
    policy = PermissionPolicy(
        auto_approve=auto_approve,
        # claude-agent-acp permission options (live-verified 0.68.0):
        #   deny:  reject_once   (optionId "reject")
        #   allow: allow_once    (optionId "allow")
        #          allow_always  (optionId "allow_always")
        allow_kind_prefix="allow",
        deny_kind_prefix="reject",
    )
    # The pin travels VERBATIM into ANTHROPIC_MODEL — the adapter owns the
    # whitelist (per-account, per-adapter-version, ROTATING). The shared
    # client validates against the live picker at session/new time via
    # ``session_new_validator`` below. NO hardcoded model list here.
    pinned = (model or "").strip() or None
    profile_env: tuple[tuple[str, str], ...] = (
        ((MODEL_ENV_VAR, pinned),) if pinned else ()
    )

    return AcpProfile(
        provider="claude-code",
        cmd=("node", str(ADAPTER_ENTRY)),
        env=profile_env,
        session_new_validator=validate_session_new_reply,
        vendor_prefix="_claude/",
        # MCP tools surface as plain tool_call named mcp__<server>__<tool> —
        # there is NO CallMcpTool dispatch built-in in claude-agent-acp.
        mcp_dispatch_tool="",  # unused for mcp_name_prefix style
        mcp_dispatch_style="mcp_name_prefix",
        known_builtins=KNOWN_BUILTINS,
        session_load_supported=True,  # initialize advertises loadSession: true
        # authMethods: [] — claude-agent-acp authenticates from Claude Code's
        # own credential store; an authenticate call is not part of the flow.
        requires_authenticate=False,
        # Real tool name lives in _meta.claudeCode.toolName; title is
        # human-readable prose ("Read /path (1-30)").
        tool_name_from_update="from_claude_code_meta",
        permission_policy=policy,
        extra_limit_markers=EXTRA_LIMIT_MARKERS,
        adapter=ADAPTER_PACKAGE,
        adapter_version=PINNED_ADAPTER_VERSION,
        session_capability="load-v1",
    )


def describe_preflight() -> dict[str, Any]:
    """Return the §5.2 preflight state for diagnostics (no exceptions, NEVER
    spawns node — review item 7: trusted package.json metadata + the CACHED
    node probe). When node was never probed in this process, ``node_major``
    is None and the node gate is reported unchecked rather than failing."""
    entry_ok = ADAPTER_ENTRY.is_file()
    version = _adapter_version_from_package() if entry_ok else None
    major = _NODE_MAJOR_CACHE[0] if _NODE_MAJOR_CACHE is not None else None
    node_ok = major is not None and major >= 22
    return {
        "node_present": node_ok,
        "node_major": major,
        "adapter_entry": str(ADAPTER_ENTRY),
        "adapter_entry_present": entry_ok,
        "pinned_version": PINNED_ADAPTER_VERSION,
        "reported_version": version,
        "ok": (major is None or node_ok)
        and entry_ok
        and version == PINNED_ADAPTER_VERSION,
    }
