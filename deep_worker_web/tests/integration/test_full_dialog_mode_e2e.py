"""Honest browser e2e for the Full Dialog ACTION MODE (tab / copy / download).

Drives the REAL UI in headless Chromium against a running dev server
(default 127.0.0.1:7887, override with ``DWW_E2E_SERVER``):

* mode ``copy`` + click  → NO new page; clipboard text starts with the snapshot
  banner (nothing rendered into any DOM);
* mode ``download`` + click → a download whose suggested filename is
  ``dialog_<agent>_<verbosity>_….txt``, NO new page;
* the mode survives a reload (localStorage) and an agent switch;
* the Stop TM button label is "⏹️ Stop TM" after a status update.

Bounded: hard budget, every page/context closed in ``finally``. Clipboard
permissions are granted on a FRESH context (Playwright requirement); the
150 KB ``writeText`` can take > 1 s, timeouts are generous.
Marked ``ui`` — excluded from the hermetic unit run:

    DWW_E2E_SERVER=http://127.0.0.1:7888 python3 -m pytest \
        tests/integration/test_full_dialog_mode_e2e.py -m ui -s
"""
from __future__ import annotations

import glob
import os
import re
from typing import Any

import pytest

playwright_sync = pytest.importorskip("playwright.sync_api")
sync_playwright = playwright_sync.sync_playwright

SERVER = os.environ.get("DWW_E2E_SERVER", "http://127.0.0.1:7887")
BUDGET_S = 90


def _find_chrome() -> str | None:
    base = os.path.expanduser("~/.cache/ms-playwright")
    for pat in (f"{base}/chromium-*/chrome-linux64/chrome", f"{base}/chromium-*/chrome-linux/chrome"):
        hits = sorted(glob.glob(pat))
        if hits:
            return hits[-1]
    return None


def _select_first_agent(page: Any) -> str:
    page.goto(f"{SERVER}/", timeout=20000)
    page.wait_for_load_state("networkidle", timeout=20000)
    page.wait_for_timeout(1500)
    items = page.locator(".agent-item")
    if items.count() == 0:
        pytest.skip("no agents available on the running server")
    items.first.click()
    page.wait_for_timeout(800)
    return page.locator("#selected-agent-name").text_content() or ""


@pytest.mark.ui
@pytest.mark.timeout(BUDGET_S + 15)
def test_full_dialog_modes_through_real_ui() -> None:
    chrome = _find_chrome()
    with sync_playwright() as p:
        kw: dict[str, Any] = {"headless": True}
        if chrome:
            kw["executable_path"] = chrome
        browser = p.chromium.launch(**kw)
        # Fresh context WITH clipboard permissions (must be set at creation).
        context = browser.new_context(permissions=["clipboard-read", "clipboard-write"], accept_downloads=True)
        try:
            page = context.new_page()
            agent = _select_first_agent(page).strip()
            assert agent

            # Layout: document order button → verbosity → mode → Stats (selects are
            # nested in .full-dialog-selects, so compare positions, not siblings).
            ok = page.evaluate(
                "() => { const ids=['full-dialog-btn','full-dialog-verbosity','full-dialog-mode','stats-btn'];"
                " const els=ids.map(i=>document.getElementById(i)); if (els.some(e=>!e)) return 'missing';"
                " for (let i=0;i<els.length-1;i++) if (!(els[i].compareDocumentPosition(els[i+1]) & Node.DOCUMENT_POSITION_FOLLOWING)) return 'order:'+ids[i];"
                " return 'ok'; }"
            )
            assert ok == "ok", ok
            # §7: mode select sits directly under the verbosity select (one block).
            vb = page.locator("#full-dialog-verbosity").bounding_box(); mb = page.locator("#full-dialog-mode").bounding_box()
            assert vb and mb and abs(vb["x"] - mb["x"]) < 2 and mb["y"] > vb["y"] + vb["height"] - 1, (vb, mb)
            # §7: on a narrow viewport every header control stays inside the viewport width.
            page.set_viewport_size({"width": 600, "height": 900}); page.wait_for_timeout(300)
            overflow = page.evaluate(
                "() => Array.from(document.querySelectorAll('.agent-controls button, .agent-controls select, .agent-controls label'))"
                ".filter(e => e.offsetParent !== null).map(e => e.getBoundingClientRect())"
                ".filter(r => r.right > window.innerWidth + 0.5 || r.left < -0.5).length"
            )
            assert overflow == 0, f"{overflow} controls overflow a 600px viewport"
            page.set_viewport_size({"width": 1280, "height": 900}); page.wait_for_timeout(300)

            page.locator("#full-dialog-verbosity").select_option("full")

            # ---- copy mode: nothing opens, clipboard receives the snapshot ----
            page.locator("#full-dialog-mode").select_option("copy")
            page.evaluate("() => navigator.clipboard.writeText('SENTINEL')")
            pages_before = len(context.pages)
            page.locator("#full-dialog-btn").click()
            # wait until the toast confirms (fetch + writeText of a large body).
            page.locator(".notification", has_text="Copied").wait_for(timeout=30000)
            # The toast container announces to screen readers (NVDA reads "Copied …").
            assert page.evaluate("() => document.getElementById('notification-container').getAttribute('aria-live')") == "polite"
            assert len(context.pages) == pages_before, "copy mode must not open a page"
            clip = page.evaluate("() => navigator.clipboard.readText()")
            assert clip.startswith(f"=== Full Dialog snapshot: agent={agent}  verbosity=full"), clip[:160]
            assert "this page never updates" in clip.split("\n", 2)[1]

            # ---- download mode: a .txt download, no page ----
            page.locator("#full-dialog-mode").select_option("download")
            pages_before = len(context.pages)
            with page.expect_download(timeout=30000) as dl_info:
                page.locator("#full-dialog-btn").click()
            dl = dl_info.value
            name = dl.suggested_filename
            assert re.fullmatch(rf"dialog_{re.escape(agent)}_full_\d{{8}}-\d{{6}}\.txt", name), name
            path = dl.path()
            head = open(path, encoding="utf-8").read(200)
            assert head.startswith(f"=== Full Dialog snapshot: agent={agent}"), head
            assert len(context.pages) == pages_before, "download mode must not open a page"

            # ---- persistence: reload keeps 'download'; agent switch keeps it ----
            page.reload(); page.wait_for_load_state("networkidle", timeout=20000); page.wait_for_timeout(1500)
            assert page.locator("#full-dialog-mode").input_value() == "download"
            items = page.locator(".agent-item")
            if items.count() > 1:
                items.nth(1).click(); page.wait_for_timeout(600)
                assert page.locator("#full-dialog-mode").input_value() == "download"

            # ---- Stop TM label (JS status update must not restore the long text) ----
            page.locator(".agent-item").first.click(); page.wait_for_timeout(1500)
            label = page.locator("#stop-taskmanager-btn").text_content() or ""
            assert "Stop TM" in label and "TaskManager" not in label, label

            # ---- tab mode still the isolated static snapshot ----
            page.locator("#full-dialog-mode").select_option("tab")
            with context.expect_page(timeout=20000) as np_info:
                page.locator("#full-dialog-btn").click()
            np = np_info.value; np.wait_for_load_state("load", timeout=30000)
            assert np.evaluate("window.opener === null")
            np.close()
            print(f"\n[mode-e2e] agent={agent} clip={len(clip)} chars, download={name}")
        finally:
            context.close(); browser.close()
