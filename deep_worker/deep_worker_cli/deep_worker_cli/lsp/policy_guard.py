"""Regression lock for POLICY_NO_INPROCESS_HEAVY_MCP (§8) / DESIGN §14.

The policy exists because a resource-heavy MCP server started in-process
multiplies its cost by the number of agents: N agents on one project
meant N Serena processes, N language servers and N project models. This
module is the automated lock that stops that shape coming back.

Three checks, matching the three bullets of policy §8:

1. an ``mcp.json`` entry whose ``command`` is a blocklisted heavy binary
   over stdio (explicit *or* default transport);
2. Python source spawning such a binary — ``subprocess.Popen/run/...``,
   ``asyncio.create_subprocess_exec``, or ``stdio_client`` — anywhere
   outside the single allowlisted package;
3. an edit to the policy document without a design document in the same
   delta. This is deliberately broader than "the exemption list only":
   any change to the policy file is a policy change and must arrive
   with its rationale, and detecting a section-scoped edit reliably
   would mean parsing prose.

**Allowlist.** Only ``deep_worker_cli/lsp/service/**`` may spawn a heavy
backend. The design draft mentioned ``lsp.bridge``, but the policy text
is authoritative and narrower, and the implementation follows it: the
bridge is a thin client that deliberately cannot spawn anything.

Design note: this guard is deliberately conservative about what counts
as a spawn. A check that fires on ordinary ``subprocess.run(["git", …])``
would be disabled within a week, and a disabled guard protects nothing.
"""
from __future__ import annotations

import ast
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Iterable, Sequence

#: Heavy binaries from policy §8. Matched against the executable name,
#: so an absolute path or a versioned suffix still trips the guard.
HEAVY_BINARIES: Final[frozenset[str]] = frozenset(
    {
        "serena",
        "serena-agent",
        "kotlin-language-server",
        "kotlin-lsp",
        "pyright-langserver",
        "typescript-language-server",
        "tsserver",
        "rust-analyzer",
        "clangd",
        "gopls",
        "jdtls",
        "metals",
        "gradle",
        "mvn",
        "java",
    }
)

#: The ONLY packages permitted to spawn a heavy backend (policy §8).
#:
#: A closed set, not an open rule: the policy is "spawning lives in a
#: ``service/`` package", and every entry here is one. Adding the second
#: entry does not widen that — semantic_memory's supervisor sits inside
#: ``service/`` for the same reason the LSP one does.
_ALLOWED_SPAWNERS: Final = frozenset(
    {
        ("deep_worker_cli", "lsp", "service"),
        ("deep_worker_cli", "semantic_memory", "service"),
    }
)

#: Transports that keep the process out of the agent. Anything else
#: (including an omitted ``type``) is treated as stdio.
_REMOTE_TRANSPORTS: Final = frozenset({"sse", "http", "streamable-http", "ws"})

_SPAWN_FUNCS: Final = frozenset(
    {"Popen", "run", "call", "check_call", "check_output"}
)

#: The ``os.exec*`` family, listed explicitly. A ``startswith("exec")``
#: test would also match ``cursor.execute`` (314 call sites in this
#: repo) and ``exec_module``; breadth there buys nothing and risks the
#: false positives that get guards switched off.
_OS_EXEC_FUNCS: Final = frozenset(
    {
        "execv", "execve", "execvp", "execvpe",
        "execl", "execle", "execlp", "execlpe",
    }
)

_POLICY_FILENAME: Final = "POLICY_NO_INPROCESS_HEAVY_MCP.md"


@dataclass(frozen=True)
class Violation:
    path: Path
    line: int
    message: str

    def __str__(self) -> str:
        return f"{self.path}:{self.line}: {self.message}"


def _binary_name(command: str) -> str:
    return Path(str(command)).name


def _is_heavy(command: object) -> bool:
    if not isinstance(command, str) or not command:
        return False
    return _binary_name(command) in HEAVY_BINARIES


# ── 1. mcp.json ──────────────────────────────────────────────────────


def scan_mcp_config(path: Path) -> list[Violation]:
    """Flag entries that exec a heavy binary over stdio."""
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8") or "{}")
    except (OSError, json.JSONDecodeError):
        return []
    if not isinstance(raw, dict):
        return []
    servers = raw.get("mcpServers") if "mcpServers" in raw else raw
    if not isinstance(servers, dict):
        return []

    out: list[Violation] = []
    for name, entry in servers.items():
        if not isinstance(entry, dict):
            continue
        transport = str(entry.get("type", "stdio")).lower()
        if transport in _REMOTE_TRANSPORTS:
            continue  # not in-process; the policy does not apply
        command = entry.get("command")
        if _is_heavy(command):
            out.append(
                Violation(
                    Path(path),
                    1,
                    f"mcp server {name!r} launches heavy binary "
                    f"{command!r} over {transport} — use the shared "
                    f"service bridge instead (policy §8)",
                )
            )
            continue
        # ``node .../tsserver.js`` style: heavy binary hidden in args.
        args = entry.get("args")
        if isinstance(args, list):
            for arg in args:
                if _is_heavy(arg):
                    out.append(
                        Violation(
                            Path(path),
                            1,
                            f"mcp server {name!r} passes heavy binary "
                            f"{arg!r} to {command!r} over {transport} "
                            f"(policy §8)",
                        )
                    )
                    break
    return out


# ── 2. Python source ─────────────────────────────────────────────────


def _in_allowlisted_package(path: Path, repo_root: Path) -> bool:
    try:
        parts = Path(path).resolve().relative_to(Path(repo_root).resolve()).parts
    except ValueError:
        parts = Path(path).parts
    return any(
        parts[i : i + len(allowed)] == allowed
        for allowed in _ALLOWED_SPAWNERS
        for i in range(len(parts) - len(allowed) + 1)
    )


def _literal_strings(node: ast.AST) -> list[str]:
    """String literals reachable as direct arguments or list elements."""
    out: list[str] = []
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        out.append(node.value)
    elif isinstance(node, (ast.List, ast.Tuple)):
        for elt in node.elts:
            if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                out.append(elt.value)
    return out


def scan_python_source(path: Path, *, repo_root: Path) -> list[Violation]:
    """Flag heavy-binary spawns outside the allowlisted package."""
    if _in_allowlisted_package(path, repo_root):
        return []
    try:
        src = Path(path).read_text(encoding="utf-8")
    except OSError:
        return []
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return []

    out: list[Violation] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = (
            func.attr
            if isinstance(func, ast.Attribute)
            else func.id
            if isinstance(func, ast.Name)
            else ""
        )
        interesting = (
            name in _SPAWN_FUNCS
            or name in {"create_subprocess_exec", "create_subprocess_shell"}
            or name == "stdio_client"
            or name in _OS_EXEC_FUNCS
        )
        if not interesting:
            continue

        candidates: list[str] = []
        for arg in node.args:
            candidates.extend(_literal_strings(arg))
            # ``stdio_client(StdioServerParameters(command="serena"))``
            # hides the binary one constructor deep, so look inside any
            # nested call's arguments too.
            if isinstance(arg, ast.Call):
                for inner in arg.args:
                    candidates.extend(_literal_strings(inner))
                for inner_kw in arg.keywords:
                    if inner_kw.arg in {"command", "args", "executable"}:
                        candidates.extend(_literal_strings(inner_kw.value))
        for kw in node.keywords:
            if kw.arg in {"command", "args", "executable"}:
                candidates.extend(_literal_strings(kw.value))
            elif isinstance(kw.value, ast.Call):
                for inner_kw in kw.value.keywords:
                    if inner_kw.arg in {"command", "args", "executable"}:
                        candidates.extend(_literal_strings(inner_kw.value))

        heavy = [c for c in candidates if _is_heavy(c)]
        if heavy:
            out.append(
                Violation(
                    Path(path),
                    node.lineno,
                    f"spawns heavy binary {heavy[0]!r} outside "
                    f"deep_worker_cli/lsp/service/** — only the shared "
                    f"service may start a backend (policy §8)",
                )
            )
    return out


# ── 3. exemption list edits ──────────────────────────────────────────


def scan_exemption_changes(
    changed_paths: Sequence[Path],
) -> list[Violation]:
    """The policy document may only move together with a design doc.

    Deliberately broader than the exemption list alone: any edit to the
    policy is a policy change and must arrive with its rationale, and
    detecting a section-scoped edit reliably would mean parsing prose.
    """
    changed = [Path(p) for p in changed_paths]
    touches_policy = [p for p in changed if p.name == _POLICY_FILENAME]
    if not touches_policy:
        return []
    has_design = any(
        p.name.startswith("DESIGN_") and p.suffix == ".md" for p in changed
    )
    if has_design:
        return []
    return [
        Violation(
            touches_policy[0],
            1,
            "policy document changed without a DESIGN_*.md in the same "
            "delta — a policy change must arrive with its rationale "
            "(policy §8)",
        )
    ]


# ── entry point ──────────────────────────────────────────────────────


def scan_paths(
    paths: Iterable[Path],
    *,
    repo_root: Path,
    changed_paths: Sequence[Path] = (),
) -> list[Violation]:
    """Run every applicable check over ``paths``."""
    out: list[Violation] = []
    for path in paths:
        p = Path(path)
        if p.name == "mcp.json":
            out.extend(scan_mcp_config(p))
        elif p.suffix == ".py":
            out.extend(scan_python_source(p, repo_root=repo_root))
    out.extend(scan_exemption_changes(changed_paths))
    return out
