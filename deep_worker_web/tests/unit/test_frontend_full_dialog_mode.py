"""Static guard: Full Dialog action mode (tab / copy / download), Stats placement,
"Stop TM" label, screen-reader-announced toasts.

DESIGN_FULL_DIALOG_ACTION_MODE.md §4.2. Pure source guards over ``static/`` — no
browser, no server — so the contract cannot silently regress. Comment-only
lines are stripped before searching (same precedent as
``tests/integration/test_no_blocking_rpyc_in_handlers.py``).
"""
from __future__ import annotations

import re
from pathlib import Path

STATIC = Path(__file__).resolve().parents[2] / "static"
MAIN_JS = STATIC / "js" / "main.js"
INDEX = STATIC / "index.html"
CSS = STATIC / "css" / "style.css"


def _js_code_lines() -> list[str]:
    """main.js with ``//`` comment-only lines removed."""
    return [ln for ln in MAIN_JS.read_text(encoding="utf-8").splitlines() if not ln.lstrip().startswith("//")]


def _function_body(src: str, name: str) -> str:
    start = src.index(f"function {name}(")
    i = src.index("{", start)
    depth = 0
    for j in range(i, len(src)):
        if src[j] == "{":
            depth += 1
        elif src[j] == "}":
            depth -= 1
            if depth == 0:
                return src[i : j + 1]
    raise AssertionError(f"unbalanced braces in {name}")


# ---------------------------------------------------------------- index.html
def test_mode_select_has_exactly_three_options_in_order() -> None:
    html = INDEX.read_text(encoding="utf-8")
    m = re.search(r'<select id="full-dialog-mode"[^>]*>(.*?)</select>', html, re.S)
    assert m, "#full-dialog-mode select missing"
    values = re.findall(r'<option value="([^"]+)"', m.group(1))
    assert values == ["tab", "copy", "download"], values


def test_stats_button_after_mode_select_not_between_button_and_verbosity() -> None:
    html = INDEX.read_text(encoding="utf-8")
    pos = {k: html.index(k) for k in ('id="full-dialog-btn"', 'id="full-dialog-verbosity"', 'id="full-dialog-mode"', 'id="stats-btn"')}
    assert pos['id="full-dialog-btn"'] < pos['id="full-dialog-verbosity"'] < pos['id="full-dialog-mode"'] < pos['id="stats-btn"'], pos


def test_selects_form_one_vertical_block() -> None:
    """§7: verbosity + mode selects are the ONLY children of .full-dialog-selects, in order."""
    html = INDEX.read_text(encoding="utf-8")
    m = re.search(r'<div class="full-dialog-selects">(.*?)</div>', html, re.S)
    assert m, ".full-dialog-selects wrapper missing"
    ids = re.findall(r'<select id="([^"]+)"', m.group(1))
    assert ids == ["full-dialog-verbosity", "full-dialog-mode"], ids
    # No other element kinds inside the block.
    assert not re.search(r"<(button|input|label|a)\b", m.group(1)), m.group(1)


def test_css_controls_wrap_and_block_is_column() -> None:
    css = CSS.read_text(encoding="utf-8")
    rules = re.findall(r"\.agent-controls\s*\{([^}]*)\}", css)
    assert len(rules) == 1, f"expected exactly one .agent-controls rule, got {len(rules)}"
    assert re.search(r"flex-wrap\s*:\s*wrap", rules[0]), rules[0]
    header = re.search(r"\.agent-header\s*\{([^}]*)\}", css)
    assert header and re.search(r"flex-wrap\s*:\s*wrap", header.group(1)), header.group(1) if header else None
    block = re.search(r"\.full-dialog-selects\s*\{([^}]*)\}", css)
    assert block and re.search(r"flex-direction\s*:\s*column", block.group(1)), block.group(1) if block else None


def test_stop_tm_label_in_html() -> None:
    html = INDEX.read_text(encoding="utf-8")
    m = re.search(r'<button id="stop-taskmanager-btn"[^>]*>([^<]*)</button>', html)
    assert m and "Stop TM" in m.group(1) and "TaskManager" not in m.group(1), m.group(0) if m else None


# ------------------------------------------------------------------- main.js
def test_mode_persistence_key_and_restore() -> None:
    src = "\n".join(_js_code_lines())
    assert "FULL_DIALOG_MODE_KEY" in src
    assert re.search(r"FULL_DIALOG_MODE_KEY\s*=\s*['\"]deep_worker_web\.dialog_mode['\"]", src)
    assert re.search(r"localStorage\.setItem\(\s*FULL_DIALOG_MODE_KEY", src), "mode must be persisted"
    assert re.search(r"localStorage\.getItem\(\s*FULL_DIALOG_MODE_KEY", src), "mode must be restored"
    assert "function restoreFullDialogMode(" in src and "function getFullDialogMode(" in src
    assert re.search(r"FULL_DIALOG_MODE_VALUES\s*=\s*\[\s*'tab'\s*,\s*'copy'\s*,\s*'download'\s*\]", src)


def test_open_full_dialog_dispatches_and_keeps_noopener_for_tab() -> None:
    src = "\n".join(_js_code_lines())
    body = _function_body(src, "openFullDialog")
    assert "getFullDialogMode()" in body
    assert re.search(r"window\.open\(\s*url\s*,\s*'_blank'\s*,\s*'noopener,noreferrer'\s*\)", body), body
    assert "navigator.clipboard.writeText(" in body or "copyFullDialogToClipboard(" in src
    assert "download=1" in src


def test_no_dom_fallback_for_copy() -> None:
    src = "\n".join(_js_code_lines())
    assert "execCommand(" not in src, "execCommand('copy') fallback puts text into the DOM — forbidden"
    assert not re.search(r"createElement\(\s*['\"]textarea['\"]\s*\)", src), "hidden textarea fallback forbidden"


def test_stop_taskmanager_literal_gone_from_code_lines() -> None:
    offenders = [ln.strip() for ln in _js_code_lines() if "Stop TaskManager" in ln]
    assert offenders == [], offenders


def test_toast_container_is_announced() -> None:
    src = "\n".join(_js_code_lines())
    body = _function_body(src, "showNotification")
    assert "aria-live" in body and "polite" in body, "toast container must be aria-live=polite"
    assert "role" in body and "status" in body
