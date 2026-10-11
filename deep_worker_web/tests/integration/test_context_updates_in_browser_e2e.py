"""Context never refreshes (a related internal incident analysis, not included here) — browser-level guard.

Why this file exists
--------------------
The backend fix (refresh the mirror on ``status_change``) was verified
with a raw WebSocket client and green node tests, and STILL the operator
saw a frozen ``Context`` field. Reason: a second, independent defect in
``main.js`` that every node test structurally could not see.

``main.js`` used to declare

    let selectedAgent = (globalThis.selectedAgent = null);

which copies the value ONCE at load. A ``let`` at module scope creates no
``window`` property, so ``selectAgent()``'s bare ``selectedAgent = name``
updated only the module binding while ``globalThis.selectedAgent`` stayed
``null`` forever — and every handler guarding on
``globalThis.selectedAgent`` early-returned in a real browser.

Every node test pre-seeded ``globalThis.selectedAgent = AGENT`` by hand
and none ever called ``selectAgent()``, so they set up precisely the
state a browser never reaches and stayed green through the whole bug.

Therefore this guard MUST run in a real browser and MUST select the
agent the way a user does — by clicking the agent row. It asserts on the
rendered DOM text, not on internal state.

Requires a running server (``DWW_E2E_SERVER``, default 127.0.0.1:7887)
that serves the CURRENT ``main.js``. Marked ``ui``; selected with
``-m ui``.
"""

from __future__ import annotations

import glob
import json
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Any

import pytest

playwright_sync = pytest.importorskip("playwright.sync_api")
sync_playwright = playwright_sync.sync_playwright

SERVER = os.environ.get("DWW_E2E_SERVER", "http://127.0.0.1:7887")
# Overridable so a RED mutation run (where the UI never updates and the
# test must burn the whole budget before failing) stays practical.
BUDGET_S = float(os.environ.get("DWW_CTX_BUDGET_S", "240"))

# "1234", "12.3k", "1.2M" — whatever formatAgentContext renders, we only
# need the leading numeric token to prove the value CHANGED.
_NUM_RE = re.compile(r"([0-9][0-9.,]*)\s*([kKmM])?")


def _find_chrome() -> str | None:
    base = os.path.expanduser("~/.cache/ms-playwright")
    for pat in (
        f"{base}/chromium-*/chrome-linux64/chrome",
        f"{base}/chromium-*/chrome-linux/chrome",
    ):
        hits = sorted(glob.glob(pat))
        if hits:
            return hits[-1]
    return None


def _parse_ctx(text: str) -> float | None:
    """Best-effort numeric value out of the rendered Context span."""
    if not text:
        return None
    m = _NUM_RE.search(text.replace(" ", ""))
    if not m:
        return None
    try:
        val = float(m.group(1).replace(",", ""))
    except ValueError:
        return None
    suffix = (m.group(2) or "").lower()
    if suffix == "k":
        val *= 1_000
    elif suffix == "m":
        val *= 1_000_000
    return val


@pytest.mark.ui
@pytest.mark.timeout(int(BUDGET_S) + 60)
def test_selected_agent_is_visible_on_globalthis_after_clicking_a_row() -> None:
    """The precise regression: clicking an agent row must make the
    selection observable at ``globalThis.selectedAgent``.

    This is the single line that broke production while every node test
    stayed green. Kept as its own case so a failure names the cause
    directly instead of surfacing as "the number did not change".
    """
    chrome = _find_chrome()
    with sync_playwright() as p:
        launch_kwargs: dict[str, Any] = {"headless": True}
        if chrome:
            launch_kwargs["executable_path"] = chrome
        browser = p.chromium.launch(**launch_kwargs)
        try:
            page = browser.new_context().new_page()
            page.set_default_timeout(15_000)
            page.goto(f"{SERVER}/")
            page.wait_for_load_state("networkidle")
            page.wait_for_timeout(1200)

            items = page.locator(".agent-item")
            if items.count() == 0:
                pytest.skip("no agents on the running server")

            # Before any click the property must exist AND be null —
            # proving the declaration itself is present and that the
            # assertion after the click measures the click, not a value
            # that was already sitting there.
            assert page.evaluate("'selectedAgent' in globalThis") is True, (
                "globalThis.selectedAgent is not declared at all"
            )
            assert page.evaluate("globalThis.selectedAgent") is None, (
                "globalThis.selectedAgent is already set before any click — "
                "the post-click assertion would not be measuring the click"
            )

            items.first.click()
            page.wait_for_timeout(600)

            sel = page.evaluate("globalThis.selectedAgent")
            assert sel, (
                "clicking an agent row did not publish the selection to "
                "globalThis.selectedAgent — every broadcast handler that "
                "guards on it will early-return (the 2026-08-22 defect)"
            )

            # And it must agree with what the UI shows as selected.
            shown = page.evaluate(
                "document.querySelector('.agent-item.selected')?.innerText || ''"
            )
            assert sel in shown, (
                f"globalThis.selectedAgent={sel!r} does not match the "
                f"highlighted row {shown!r}"
            )
        finally:
            browser.close()


@pytest.mark.ui
@pytest.mark.timeout(int(BUDGET_S) + 60)
def test_context_span_updates_in_browser_on_a_real_turn() -> None:
    """End-to-end, no mocks: a real CLI-driven turn must move the
    rendered ``Context`` text with NO operator action (no F5).

    Spawns its own throwaway agent so the assertion does not depend on
    someone else's traffic, and drives it from the CLI so the turn is
    NOT web-initiated — the exact case that was broken.
    """
    chrome = _find_chrome()
    agent = f"ctxui{os.getpid()}"
    deadline = time.monotonic() + BUDGET_S
    started = False

    try:
        # ── start a real agent ────────────────────────────────────────
        rc = subprocess.run(
            ["deep-worker", agent, "--provider", "minimax", "-l", "100k",
             "-q", "Say the word one."],
            capture_output=True, text=True, timeout=120,
        )
        if rc.returncode != 0:
            pytest.skip(f"could not start agent: {rc.stdout} {rc.stderr}")
        started = True

        stats = (
            Path.home() / ".deep_worker" / "agents" / agent / "token_stats.jsonl"
        )
        while time.monotonic() < deadline and not stats.exists():
            time.sleep(2)
        if not stats.exists():
            pytest.skip("agent produced no token_stats.jsonl")

        def _agent_rows() -> list[dict[str, Any]]:
            out = []
            for line in stats.read_text().splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                if r.get("kind") == "agent":
                    out.append(r)
            return out

        while time.monotonic() < deadline and not _agent_rows():
            time.sleep(2)
        if not _agent_rows():
            pytest.skip("no kind=agent usage rows yet")

        with sync_playwright() as p:
            launch_kwargs: dict[str, Any] = {"headless": True}
            if chrome:
                launch_kwargs["executable_path"] = chrome
            browser = p.chromium.launch(**launch_kwargs)
            try:
                page = browser.new_context().new_page()
                page.set_default_timeout(20_000)
                page.goto(f"{SERVER}/")
                page.wait_for_load_state("networkidle")

                # Find and click OUR agent's row — exactly what a user does.
                row = page.locator(".agent-item", has_text=agent)
                for _ in range(30):
                    if row.count() > 0:
                        break
                    page.wait_for_timeout(1000)
                    row = page.locator(".agent-item", has_text=agent)
                if row.count() == 0:
                    pytest.skip(f"agent {agent} never appeared in the UI list")
                row.first.click()
                page.wait_for_timeout(1200)

                ctx = page.locator("#agent-context")
                before_text = ctx.inner_text()
                before_val = _parse_ctx(before_text)
                assert before_val is not None, (
                    f"Context span shows no number before the turn: "
                    f"{before_text!r}"
                )

                rows_before = len(_agent_rows())

                # Plant a no-reload witness BEFORE the turn. A page
                # navigation/reload creates a fresh JS realm and wipes
                # this property, so its survival past the assertion below
                # is what proves the update arrived live rather than via
                # a reload. (Setting it *after* the wait and reading it
                # back would be a tautology — it can never fail, and that
                # is the same unearned-evidence class as §3's
                # "painter → DOM: OK".)
                page.evaluate("globalThis.__ctx_probe_marker = 1")

                # ── drive a REAL turn from the CLI (not the web UI) ────
                subprocess.run(
                    ["dwcli", agent, "-p",
                     "Name three colors and three animals, one per line."],
                    capture_output=True, text=True, timeout=120,
                )

                # Wait for the provider to actually write a new usage row;
                # only then may we demand the UI to have moved.
                while time.monotonic() < deadline and len(_agent_rows()) <= rows_before:
                    time.sleep(2)
                rows_after = _agent_rows()
                if len(rows_after) <= rows_before:
                    pytest.skip("provider wrote no new usage row for the turn")
                expected = rows_after[-1].get("prompt_tokens")

                # ── the assertion: DOM changes with NO reload ──────────
                after_text = before_text
                after_val = before_val
                while time.monotonic() < deadline:
                    after_text = ctx.inner_text()
                    after_val = _parse_ctx(after_text)
                    if after_val is not None and after_val != before_val:
                        break
                    page.wait_for_timeout(1000)

                assert after_val is not None and after_val != before_val, (
                    "the Context span did NOT change after a real turn "
                    "without a page reload — the operator sees a frozen "
                    f"number. before={before_text!r} after={after_text!r} "
                    f"expected≈{expected}"
                )

                # The rendered number must correspond to the JSONL truth
                # (allow formatter rounding, e.g. 16707 -> "16.7k").
                assert expected is not None
                assert abs(after_val - float(expected)) <= max(
                    100.0, float(expected) * 0.01
                ), (
                    f"Context renders {after_text!r} (~{after_val}) but the "
                    f"provider reported prompt_tokens={expected}"
                )

                # And it happened with NO reload: the witness planted
                # before the turn must still be there. If anything had
                # navigated the page, the JS realm would be new and this
                # would read back ``None``.
                assert page.evaluate("globalThis.__ctx_probe_marker") == 1, (
                    "the no-reload witness planted before the turn is gone — "
                    "the page was navigated, so this run does NOT prove the "
                    "Context span updates live"
                )
            finally:
                browser.close()
    finally:
        if started:
            subprocess.run(["dwcli", "--stop", agent],
                           capture_output=True, timeout=60)
            time.sleep(2)
            import shutil
            shutil.rmtree(
                Path.home() / ".deep_worker" / "agents" / agent,
                ignore_errors=True,
            )
            for pat in (f"{agent}.*",):
                for f in glob.glob(
                    str(Path.home() / ".deep_worker" / "logs" / pat)
                ):
                    try:
                        os.unlink(f)
                    except OSError:
                        pass
