"""The ``lsp bridge`` entry must not import the agent stack.

The bridge is a per-agent stdio MCP proxy to the shared LSP service. It
uses none of the agent runtime, yet it used to be reachable only through
``agent_command``, whose module-level LangChain import chain pulls
``langchain_text_splitters`` → ``transformers`` → ``torch`` eagerly.
That cost ~4 s and ~650 MB RSS per bridge, for modules it never touches.

**These tests drive the REAL entry point: the ``deep-worker`` wrapper
script, invoked with exactly the argv that the MCP config records.**

That distinction is the whole point of this file. An earlier version
drove ``deep_worker_cli.cli.router:deep_worker_main`` — the declared
console-script — and passed while production was still loading torch.
It measured an entry nothing executes: the package is not installed
here (``pip install -e .`` is forbidden, see ``deep_worker/CLAUDE.md``
§1), so ``[project.scripts]`` never materialises, and the real
``deep-worker`` on PATH is a tracked shell wrapper that execs
``python3 -m deep_worker_cli.cli.agent_command`` directly, bypassing the
router entirely. Measured on 2026-10-09: eight live bridges, 162 heavy
lines in ``/proc/<pid>/maps`` each, RSS 263–663 MB.

Two detection axes are used, because neither alone is sufficient:

* ``/proc/<pid>/maps`` — catches native libraries actually mapped into
  the address space (torch/nvidia/cuda/cublas/cudnn/tokenizers ``.so``).
* ``sys.modules`` — catches pure-Python packages that never appear as a
  mapping at all. ``transformers`` is exactly that case: it is pure
  Python, and ``grep -ci transformers /proc/<pid>/maps`` on a live heavy
  bridge returns **0**. A maps-only check is blind to it.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import signal
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

#: Top-level packages the thin bridge path must never pull in. Each is a
#: distinct failure mode, not a variation of one: the ML stack
#: (torch/transformers/nvidia/tokenizers) is the expensive tail, while
#: langchain/langgraph/deepagents are the agent stack that reaches it.
#:
#: NOTE: ``transformers`` is detectable ONLY via ``sys.modules`` — it is
#: pure Python and contributes no entry to ``/proc/<pid>/maps``. Reading
#: a zero from the maps axis as "not loaded" would be wrong.
FORBIDDEN = (
    "torch",
    "transformers",
    "tokenizers",
    "nvidia",
    "langchain",
    "langchain_core",
    "langgraph",
    "deepagents",
)

#: Heavy native libraries, as matched against ``/proc/<pid>/maps``. This
#: is the exact token set the measurements in this file quote, so the
#: number in a failure message is reproducible by hand:
#:     grep -cEi 'torch|nvidia|cuda|cublas|cudnn|tokenizers' /proc/<pid>/maps
#: The metric is MAP LINES, not libraries: one ``.so`` yields several
#: segments, so ~38 unique heavy paths show up as ~162 lines.
HEAVY_MAPS_RE = re.compile(r"torch|nvidia|cuda|cublas|cudnn|tokenizers", re.I)

_REPO_ROOT = Path(__file__).resolve().parents[3]

#: The physical wrapper inside THIS tree. Tests prefer the wrapper found
#: on PATH (that is what the config's bare ``deep-worker`` resolves to),
#: and fall back to this one when PATH has no entry — e.g. on CI.
_TREE_WRAPPER = _REPO_ROOT / "deep-worker"

_BRIDGE_TIMEOUT_S = 120

#: The mapping table must hold still this long (wall clock) before a
#: measurement counts as settled.
_SETTLE_QUIET_S = 1.5

#: ...and no measurement may settle before this floor, whatever the
#: table looks like. Measured on this host: a heavy bridge's first
#: torch/cuda mapping appears at ~1.3 s, so a floor below that would let
#: "not yet importing" pass as "imports nothing".
_SETTLE_FLOOR_S = 4.0


def _wrapper_on_path() -> str | None:
    """The ``deep-worker`` a bare config command would actually run."""
    return shutil.which("deep-worker")


def _resolve_wrapper() -> str:
    """The wrapper belonging to THIS tree, which is what is under test.

    The config records a bare ``deep-worker``, and on a developer host
    the PATH entry is a symlink into whichever tree is installed — for
    production that is the master checkout, not this one. Measuring the
    PATH wrapper would therefore grade a different tree's code: before
    the fast-forward it reports the old behaviour, and the suite would
    fail for a reason that has nothing to do with the commit under test
    (and, worse, would start passing later without any change here).

    So the tree-local wrapper is preferred, and the PATH one is used
    only as a fallback. ``test_path_wrapper_is_this_tree_or_known`` below
    records whether the two currently agree, so a stale production
    wrapper stays visible instead of silently unmeasured.
    """
    if _TREE_WRAPPER.is_file():
        return str(_TREE_WRAPPER)
    found = _wrapper_on_path()
    if found:
        return found
    pytest.skip("no deep-worker wrapper available to exercise")


def _probe_env() -> dict[str, str]:
    """Environment for a wrapper launch.

    ``PYTHONPATH`` is deliberately cleared: the wrapper is responsible
    for setting it, and inheriting pytest's own would mask a wrapper
    that fails to. ``DEEP_WORKER_*`` tree selectors are cleared so the
    layout under test is chosen by the test, not by the ambient session.
    """
    env = {
        k: v
        for k, v in os.environ.items()
        if k not in {"PYTHONPATH", "DEEP_WORKER_DEV_TREE", "DEEP_WORKER_TREE_ROOT"}
    }
    env["PYTHONUNBUFFERED"] = "1"
    return env


def _measure_bridge(argv: list[str], env: dict[str, str]) -> dict[str, object]:
    """Launch a bridge, measure it from the OUTSIDE, then kill it.

    The bridge is an MCP stdio server: it starts, waits on stdin and
    never exits on its own. So it is launched with a live stdin pipe and
    inspected via procfs while it runs.

    Waiting is event-driven on observable state — the maps file growing
    quiet — not a fixed sleep, and the whole thing is bounded. The
    process is killed by process GROUP in ``finally`` so a wrapper that
    exec'd a child cannot leak a 650 MB bridge into the test host.
    """
    proc = subprocess.Popen(
        argv,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        cwd=str(_REPO_ROOT),
        start_new_session=True,
    )
    try:
        maps_path = Path(f"/proc/{proc.pid}/maps")
        start = os.times().elapsed
        deadline = start + _BRIDGE_TIMEOUT_S
        last = -1
        quiet_since = start
        heavy_lines = 0
        settled = False
        while True:
            now = os.times().elapsed
            if now >= deadline:
                break
            if proc.poll() is not None:
                break
            try:
                text = maps_path.read_text()
            except OSError:
                break
            heavy_lines = sum(1 for ln in text.splitlines() if HEAVY_MAPS_RE.search(ln))
            total = len(text.splitlines())
            # Settled once the mapping table has stopped moving for a
            # WALL-CLOCK span, and not before a floor has elapsed.
            #
            # Counting identical observations instead of measuring time
            # is what broke an earlier version of this helper: the loop
            # spun tens of thousands of times within 50 ms, hit its
            # "stable" threshold before the child had executed any
            # bytecode, and reported zero heavy mappings for a process
            # that goes heavy at ~1.3 s. Every test then passed on the
            # unfixed wrapper. A zero observed before the import phase
            # is not a thin process — it is an unmeasured one.
            if total == last:
                if now - quiet_since >= _SETTLE_QUIET_S and now - start >= _SETTLE_FLOOR_S:
                    settled = True
                    break
            else:
                last = total
                quiet_since = now
        if not settled and proc.poll() is None:
            # Never let "we stopped looking" masquerade as "nothing was
            # loaded": an unsettled measurement is not evidence of a
            # thin process, so say so instead of returning a tidy zero.
            return {
                "heavy_map_lines": heavy_lines,
                "rss_kb": 0,
                "pss_kb": 0,
                "alive": True,
                "returncode": None,
                "settled": False,
                "stderr": "",
            }
        rss_kb = pss_kb = 0
        try:
            for line in Path(f"/proc/{proc.pid}/smaps_rollup").read_text().splitlines():
                if line.startswith("Rss:"):
                    rss_kb = int(line.split()[1])
                elif line.startswith("Pss:"):
                    pss_kb = int(line.split()[1])
        except OSError:
            pass
        stderr_tail = ""
        if proc.poll() is not None:
            try:
                stderr_tail = (proc.stderr.read() or "")[-600:]
            except (OSError, ValueError):
                stderr_tail = ""
        return {
            "heavy_map_lines": heavy_lines,
            "rss_kb": rss_kb,
            "pss_kb": pss_kb,
            "alive": proc.poll() is None,
            "returncode": proc.poll(),
            "settled": settled,
            "stderr": stderr_tail,
        }
    finally:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            proc.kill()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            pass


def _bridge_argv(wrapper: str, *, dev: bool) -> list[str]:
    """Exactly the argv the MCP config records, optionally with --dev.

    ``build_bridge_mcp_entry`` writes ``command: "deep-worker"`` and
    ``args: ["lsp", "bridge", "--project-root", <root>]``; this mirrors
    it so the test cannot drift from the shipped config form.
    """
    argv = [wrapper, "lsp", "bridge", "--project-root", str(_REPO_ROOT)]
    if dev:
        argv.insert(1, "--dev")
    return argv


@pytest.mark.parametrize("dev", [False, True], ids=["master-layout", "dev-layout"])
def test_real_wrapper_bridge_is_free_of_native_ml_libs(dev):
    """The config's own command must not map the native ML stack.

    This is the test the previous version of this file should have been:
    it runs the wrapper from PATH with the recorded argv and reads the
    resulting process's ``/proc/<pid>/maps``. It FAILS on the pre-fix
    wrapper, where the exec target is ``agent_command`` and ~162 heavy
    lines appear.
    """
    wrapper = _resolve_wrapper()
    env = _probe_env()
    if dev:
        # Point --dev at THIS tree so the test never depends on the
        # developer's real dev checkout being present or on
        # whatever state it happens to be in.
        env["DEEP_WORKER_DEV_TREE_ROOT_OVERRIDE"] = str(_REPO_ROOT.parents[1])
    verdict = _measure_bridge(_bridge_argv(wrapper, dev=dev), env)
    # A zero from a process that never got measured is not evidence.
    assert verdict["settled"] or verdict["returncode"] is not None, (
        "bridge mappings never settled within the budget, so this "
        "measurement cannot support a verdict either way "
        f"(last count {verdict['heavy_map_lines']})"
    )
    assert verdict["heavy_map_lines"] == 0, (
        "the real bridge command mapped native ML libraries: "
        f"{verdict['heavy_map_lines']} heavy lines in /proc/<pid>/maps "
        f"(regex {HEAVY_MAPS_RE.pattern}), RSS {verdict['rss_kb']} kB, "
        f"PSS {verdict['pss_kb']} kB; alive={verdict['alive']} "
        f"rc={verdict['returncode']}"
    )


def test_maps_detector_sees_the_ml_stack_when_present(tmp_path):
    """Positive control for the maps axis.

    Without this, a probe that measured nothing — a wrapper that died
    instantly, a typo in the regex — would make the test above pass
    vacuously. Importing torch is the very thing the fix avoids, so it
    must trip the detector.

    The probe body lives in a FILE rather than ``-c``: an inline source
    string has to survive this file's own quoting, and when it does not,
    the child dies on a SyntaxError and reports zero heavy mappings —
    which is indistinguishable from "the detector found nothing". That
    exact failure happened while writing this test, so the probe now
    also has to announce readiness before it is measured.
    """
    probe = tmp_path / "torch_probe.py"
    probe.write_text(
        textwrap.dedent(
            """
            import sys
            import time

            import torch  # noqa: F401

            sys.stderr.write("READY\\n")
            sys.stderr.flush()
            time.sleep(600)
            """
        )
    )
    env = _probe_env()
    verdict = _measure_bridge([sys.executable, str(probe)], env)
    if verdict["returncode"] not in (None, 0):
        pytest.skip(
            f"torch probe could not run: rc={verdict['returncode']} "
            f"{verdict['stderr']}"
        )
    assert verdict["alive"], (
        "torch probe exited before it could be measured, so a zero here "
        f"would prove nothing: {verdict['stderr']}"
    )
    assert verdict["heavy_map_lines"] > 0, (
        "maps detector saw no heavy libraries in a process that imported "
        "torch — the detector itself is broken, so a zero elsewhere "
        "proves nothing"
    )


@pytest.mark.parametrize("dev", [False, True], ids=["master-layout", "dev-layout"])
def test_bridge_process_sys_modules_has_no_agent_stack(dev, tmp_path):
    """The bridge process's own ``sys.modules`` must stay agent-free.

    Separate axis from maps, and not redundant with it: ``transformers``
    is pure Python and contributes zero map lines, so only this axis can
    observe it. Likewise ``langchain``/``langgraph``/``deepagents``.

    The bridge is driven through the real wrapper; the module census is
    taken inside that process via a sitecustomize shim injected on
    PYTHONPATH, which dumps ``sys.modules`` at interpreter exit. The
    shim cannot mask the defect: it only observes, and it runs in the
    process the wrapper actually exec'd.
    """
    wrapper = _resolve_wrapper()
    out = tmp_path / "modules.json"
    shim_dir = tmp_path / "shim"
    shim_dir.mkdir()
    # The census is taken from a WATCHDOG THREAD, not atexit.
    #
    # atexit does not run here: the bridge is an MCP server that never
    # returns on its own, so the measurement ends with SIGKILL, and
    # SIGKILL runs no handlers. An atexit-based shim therefore produced
    # no file at all, the test skipped, and two assertions that should
    # have caught a langchain-loading bridge passed silently.
    #
    # The thread samples after the import phase and keeps overwriting,
    # so the file always reflects the most complete module set observed.
    (shim_dir / "sitecustomize.py").write_text(
        textwrap.dedent(
            f"""
            import json, sys, threading

            _OUT = {str(out)!r}

            def _sample():
                while True:
                    try:
                        tops = sorted({{m.split('.')[0] for m in sys.modules}})
                        tmp = _OUT + ".tmp"
                        with open(tmp, "w") as fh:
                            json.dump({{"top": tops, "total": len(sys.modules)}}, fh)
                        import os
                        os.replace(tmp, _OUT)
                    except Exception:
                        pass
                    threading.Event().wait(0.25)

            threading.Thread(target=_sample, daemon=True).start()
            """
        )
    )
    env = _probe_env()
    # The wrapper PREPENDS its own entries to PYTHONPATH, so the shim
    # stays importable while the wrapper's paths still win for real
    # packages.
    env["PYTHONPATH"] = str(shim_dir)
    if dev:
        env["DEEP_WORKER_DEV_TREE_ROOT_OVERRIDE"] = str(_REPO_ROOT.parents[1])
    _measure_bridge(_bridge_argv(wrapper, dev=dev), env)
    # A missing census is a FAILURE, not a skip. Skipping here is how
    # this axis went green against a bridge that demonstrably imported
    # langchain: no file, no assertion, no signal.
    assert out.is_file(), (
        "no sys.modules census was produced, so this axis measured "
        "nothing; treat as failure rather than silently passing"
    )
    census = json.loads(out.read_text())
    loaded = sorted(set(census["top"]) & set(FORBIDDEN))
    assert loaded == [], (
        "bridge process imported forbidden packages: "
        f"{loaded} (total modules={census['total']})"
    )


def _run_wrapper(wrapper: str, args: list[str], env: dict[str, str]):
    """Run the wrapper to completion and capture its result."""
    return subprocess.run(
        [wrapper, *args],
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
        env=env,
        cwd=str(_REPO_ROOT),
        timeout=600,
    )


@pytest.mark.parametrize("dev", [False, True], ids=["master-layout", "dev-layout"])
def test_non_bridge_subcommands_still_take_the_normal_path(dev):
    """Intercepting ``lsp bridge`` must not divert anything else.

    The wrapper is the shared launcher for every agent on the host, so
    a fix that makes the bridge thin while changing where ANY other
    invocation goes would trade one defect for a worse one. These cases
    must keep reaching ``agent_command`` exactly as before:

    * a sibling ``lsp`` subcommand (``service-status``) — proves the
      interception keys on the full ``lsp bridge`` pair and not merely
      on the word ``lsp``;
    * an unrecognised option — proves argument forwarding is unchanged,
      since the error text comes from ``agent_command``'s own parser.
    """
    wrapper = _resolve_wrapper()
    env = _probe_env()
    if dev:
        env["DEEP_WORKER_DEV_TREE_ROOT_OVERRIDE"] = str(_REPO_ROOT.parents[1])
    prefix = ["--dev"] if dev else []

    # ``service-status`` reports "not running" with exit code 1 — the
    # codes below are the measured contract on this host, pinned so a
    # wrapper that rewrote or swallowed them would be caught.
    sibling = _run_wrapper(
        wrapper, [*prefix, "lsp", "service-status", "--project-root", "/tmp"], env
    )
    assert "LSP service for /tmp" in sibling.stdout, (
        "sibling lsp subcommand did not reach its normal handler: "
        f"rc={sibling.returncode} {sibling.stdout[-300:]!r}"
    )
    assert sibling.returncode == 1, (
        "sibling lsp subcommand's exit code changed: expected 1 for "
        f"'not running', got {sibling.returncode}"
    )

    # An unknown flag must still be rejected BY agent_command's parser,
    # which proves the args reached it untouched.
    bogus = _run_wrapper(wrapper, [*prefix, "--definitely-not-a-flag"], env)
    combined = bogus.stdout + bogus.stderr
    assert "unrecognized arguments" in combined, (
        "unrecognised flag did not reach the normal argument parser: "
        f"rc={bogus.returncode} {combined[-400:]!r}"
    )
    assert bogus.returncode == 2, (
        f"argparse exit code changed: expected 2, got {bogus.returncode}"
    )


def test_wrapper_preserves_exit_codes_on_the_normal_path():
    """Exit codes must survive the interception unrewritten.

    An invalid ``lsp`` subcommand is rejected by the same parser that
    accepts ``bridge``, so this is the closest neighbour to the
    intercepted case: if interception were keyed too loosely, this call
    would be diverted and its argparse failure would disappear.
    """
    wrapper = _resolve_wrapper()
    env = _probe_env()
    failing = _run_wrapper(wrapper, ["lsp", "no-such-subcommand"], env)
    combined = failing.stdout + failing.stderr
    assert failing.returncode == 2, (
        "an invalid lsp subcommand should exit 2 from argparse; got "
        f"{failing.returncode}: {combined[-400:]!r}"
    )
    assert "invalid choice" in combined, (
        f"did not reach the lsp argument parser: {combined[-400:]!r}"
    )


def test_path_wrapper_is_thin_once_it_matches_this_tree():
    """The installed wrapper must be thin — once it carries this fix.

    The config's bare ``deep-worker`` resolves through PATH, so that
    binary is what production actually runs. On a developer host it
    points at the master checkout, which only receives this change at
    the fast-forward; until then a difference is expected rather than a
    defect, and asserting on it would make this suite fail for the state
    of another tree.

    This test therefore asserts only the implication: IF the PATH
    wrapper is byte-identical to this tree's, it must be thin. After the
    FF that becomes a real assertion on production's launcher; before
    it, the mismatch is reported so it cannot pass unnoticed.
    """
    found = _wrapper_on_path()
    if not found:
        pytest.skip("no deep-worker on PATH")
    path_bytes = Path(found).resolve().read_bytes()
    if path_bytes != _TREE_WRAPPER.read_bytes():
        pytest.skip(
            f"PATH wrapper ({Path(found).resolve()}) differs from this "
            "tree's; it carries this fix only after the fast-forward"
        )
    verdict = _measure_bridge(
        [found, "lsp", "bridge", "--project-root", str(_REPO_ROOT)], _probe_env()
    )
    assert verdict["heavy_map_lines"] == 0, (
        "the installed wrapper matches this tree yet still maps the ML "
        f"stack: {verdict['heavy_map_lines']} heavy lines, "
        f"RSS {verdict['rss_kb']} kB"
    )


def test_config_entry_keeps_the_wrapper_command_and_placeholder():
    """The recorded entry must stay ``deep-worker lsp bridge …``.

    The fix lives in the wrapper precisely so the config form does NOT
    have to change, which is what makes every already-written entry
    become thin without migration. Pinning the form here is what makes
    that claim checkable, and it guards the reverse mistake: switching
    the entry to ``python -m …`` or a console script would be
    uncallable on this host, where the package is not installed and
    ``[project.scripts]`` never materialises.

    The user-scope form must also keep the ``${agent:working_dir}``
    placeholder literally — a shared config that names one project
    silently makes it everybody's navigation target.
    """
    from deep_worker_cli.external_mcp.config import AGENT_WORKING_DIR_PLACEHOLDER
    from deep_worker_cli.lsp.bridge import build_bridge_mcp_entry

    literal = build_bridge_mcp_entry(workspace_root=_REPO_ROOT)
    assert literal["command"] == "deep-worker", (
        f"bridge entry no longer invokes the wrapper: {literal['command']!r}"
    )
    assert literal["args"][:3] == ["lsp", "bridge", "--project-root"], literal["args"]
    assert literal["args"][3] == str(_REPO_ROOT.resolve())

    following = build_bridge_mcp_entry(follow_agent_cwd=True)
    assert following["args"][3] == AGENT_WORKING_DIR_PLACEHOLDER, (
        "user-scope entry lost the ${agent:working_dir} placeholder: "
        f"{following['args']!r}"
    )
    assert following["command"] == "deep-worker"


def test_sys_modules_detector_sees_the_agent_stack(tmp_path):
    """Positive control for the sys.modules axis.

    Importing ``agent_command`` is the path the fix avoids; its module
    graph must trip the census. This also pins the asymmetry that makes
    two axes necessary: ``transformers`` shows up HERE while being
    invisible to the maps axis.
    """
    out = tmp_path / "modules.json"
    src = (
        "import json, sys\n"
        "import deep_worker_cli.cli.agent_command  # noqa: F401\n"
        "tops = sorted({m.split('.')[0] for m in sys.modules})\n"
        f"open({str(out)!r}, 'w').write(json.dumps(tops))\n"
    )
    env = _probe_env()
    env["PYTHONPATH"] = os.pathsep.join(
        [str(_REPO_ROOT), str(_REPO_ROOT.parent / "deepagents"), str(_REPO_ROOT.parent)]
    )
    proc = subprocess.run(
        [sys.executable, "-c", src],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(_REPO_ROOT),
        timeout=600,
    )
    if proc.returncode != 0 or not out.is_file():
        pytest.skip(f"agent stack not importable here: {proc.stderr[-400:]}")
    tops = set(json.loads(out.read_text()))
    assert "langchain" in tops, (
        f"detector failed to observe the agent stack it bans; saw {sorted(tops & set(FORBIDDEN))}"
    )
