"""Honest browser e2e: the Full Dialog tab is a STATIC, ISOLATED snapshot.

DESIGN_FULL_DIALOG_STATIC_SNAPSHOT.md §4 test 3 / AC1–AC3.

Drives the REAL UI in a real headless Chromium against a running dev server
(default 127.0.0.1:7887, override with ``DWW_E2E_SERVER``): select an agent,
click ``#full-dialog-btn``, then prove in the opened tab that

* ``window.opener === null`` (noopener ⇒ own browsing-context group),
* ``document.scripts.length === 0`` (no JS at all),
* ZERO network requests are issued after ``load`` while agents keep working,
* ``document.body.innerText`` is byte-identical across samples spanning ≥ 6 s,
* the first line is the snapshot banner (agent, verbosity, click-time stamp),
* the opener (main) tab keeps answering JS in < 100 ms the whole time.

Marked ``ui`` (needs a live server + Playwright chromium) so it is excluded
from the default hermetic run:

    python3 -m pytest tests/integration/test_full_dialog_tab_is_static.py -m ui -s

Bounded: hard 60 s budget, browser always closed in ``finally``.
"""
from __future__ import annotations

import glob
import os
import re
import time
from typing import Any

import pytest

playwright_sync = pytest.importorskip("playwright.sync_api")
sync_playwright = playwright_sync.sync_playwright

SERVER = os.environ.get("DWW_E2E_SERVER", "http://127.0.0.1:7887")
BUDGET_S = 60.0
WATCH_S = 6.0
BANNER_RE = re.compile(
    r"^=== Full Dialog snapshot: agent=(\S+)\s+verbosity=(\S+)\s+"
    r"taken=\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}[+-]\d{2}:\d{2} ===$"
)


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


@pytest.mark.ui
@pytest.mark.timeout(int(BUDGET_S) + 15)
def test_full_dialog_tab_is_static_and_isolated() -> None:
    deadline = time.monotonic() + BUDGET_S
    chrome = _find_chrome()
    browser = None
    with sync_playwright() as p:
        try:
            launch_kwargs: dict[str, Any] = {"headless": True}
            if chrome:
                launch_kwargs["executable_path"] = chrome
            browser = p.chromium.launch(**launch_kwargs)
            context = browser.new_context()
            page = context.new_page()
            page.set_default_timeout(15_000)
            page.goto(f"{SERVER}/")
            page.wait_for_load_state("networkidle")
            page.wait_for_timeout(1500)

            items = page.locator(".agent-item")
            if items.count() == 0:
                pytest.skip("no agents available on the running server")
            items.first.click()
            page.wait_for_timeout(800)
            page.locator("#full-dialog-verbosity").select_option("full")

            t_click = time.time()
            with context.expect_page() as np_info:
                page.locator("#full-dialog-btn").click()
            dlg = np_info.value
            dlg_requests: list[str] = []
            def _on_request(r: Any) -> None:
                dlg_requests.append(str(r.url))

            dlg.on("request", _on_request)
            dlg.wait_for_load_state("load")
            t_loaded = time.time()

            # --- isolation & script-freeness -------------------------------
            assert dlg.evaluate("window.opener === null"), "opener leaked: tab shares the main page's browsing-context group"
            assert dlg.evaluate("document.scripts.length") == 0, "Full Dialog tab must contain no <script>"
            assert dlg.url.startswith(f"{SERVER}/api/agents/") and "/dialog-text?" in dlg.url, dlg.url

            # --- banner --------------------------------------------------------
            text0 = dlg.evaluate("document.body.innerText")
            first = text0.split("\n", 1)[0]
            m = BANNER_RE.match(first)
            assert m, f"first line is not the snapshot banner: {first!r}"
            assert m.group(2) == "full", first
            second = text0.split("\n", 2)[1]
            assert second.startswith("=== Static snapshot") and "never updates" in second, second

            # --- static over time + main tab responsive ------------------------
            samples = [text0]
            worst_main_ms = 0.0
            n_probes = 0
            while time.time() - t_loaded < WATCH_S and time.monotonic() < deadline:
                t = time.perf_counter()
                assert page.evaluate("1+1") == 2
                worst_main_ms = max(worst_main_ms, (time.perf_counter() - t) * 1000)
                n_probes += 1
                dlg.wait_for_timeout(1000)
                samples.append(dlg.evaluate("document.body.innerText"))
            assert len(samples) >= 3, "watch window too short"
            assert all(s == text0 for s in samples), "Full Dialog content CHANGED after load — it must be a static snapshot"
            assert dlg_requests == [], f"Full Dialog tab issued network requests after load: {dlg_requests[:5]}"
            assert worst_main_ms < 100.0, f"main tab stalled {worst_main_ms:.0f} ms while dialog tab open"
            print(
                f"\n[static-tab] agent={m.group(1)} chars={len(text0)} samples={len(samples)} "
                f"dialog_requests_after_load={len(dlg_requests)} main_probe_worst={worst_main_ms:.1f}ms "
                f"probes={n_probes} click→load={t_loaded - t_click:.2f}s"
            )
        finally:
            if browser is not None:
                browser.close()
