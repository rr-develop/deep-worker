"""Frontend guards for the agent-panel tabs (Dialog / MCP / Stats).

These assert the properties that are easy to regress silently and that
the owner's requirements depend on:

* the existing Dialog panels are still present and inside their tab;
* the inactive panel uses ``hidden`` (leaves the accessibility tree) —
  a large hidden-but-present DOM is a measured browser-freeze risk here;
* Stats is fetched on activation/refresh only — never on a timer;
* nothing in the MCP tab polls or probes implicitly;
* new DOM is built with textContent, not innerHTML (server names, URLs
  and commands are user-supplied).
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

STATIC = Path(__file__).resolve().parents[2] / "static"
INDEX_HTML = STATIC / "index.html"
MAIN_JS = STATIC / "js" / "main.js"


@pytest.fixture(scope="module")
def html() -> str:
    return INDEX_HTML.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def js() -> str:
    return MAIN_JS.read_text(encoding="utf-8")


# ── tab scaffolding ──────────────────────────────────────────────────


def test_tablist_exists_with_three_tabs(html: str) -> None:
    assert 'role="tablist"' in html
    for tab in ("tab-btn-dialog", "tab-btn-mcp", "tab-btn-stats"):
        assert f'id="{tab}"' in html


def test_each_tab_button_controls_its_panel(html: str) -> None:
    for name in ("dialog", "mcp", "stats"):
        assert f'aria-controls="tab-{name}"' in html
        assert f'id="tab-{name}"' in html


def test_panels_have_tabpanel_role(html: str) -> None:
    assert html.count('role="tabpanel"') == 3


def test_inactive_panels_use_hidden_attribute(html: str) -> None:
    """``hidden`` (not a display class) so the panel leaves the a11y tree."""
    mcp = html.index('id="tab-mcp"')
    stats = html.index('id="tab-stats"')
    # The attribute appears within the opening tag of each panel.
    assert "hidden" in html[mcp:mcp + 400]
    assert "hidden" in html[stats:stats + 400]


def test_dialog_panel_is_active_by_default(html: str) -> None:
    dialog = html.index('id="tab-dialog"')
    opening_tag = html[dialog:html.index(">", dialog)]
    assert "hidden" not in opening_tag


# ── existing functionality preserved ─────────────────────────────────


def test_existing_dialog_panels_are_inside_the_dialog_tab(html: str) -> None:
    start = html.index('id="tab-dialog"')
    end = html.index('id="tab-mcp"')
    dialog_section = html[start:end]

    for marker in (
        'id="last-error-panel"',
        'id="approval-panel"',
        'class="last-response-panel"',
        'class="message-input-panel"',
        'id="message-input"',
        'id="send-btn"',
    ):
        assert marker in dialog_section, f"{marker} must stay in the Dialog tab"


def test_header_controls_stay_outside_the_tabs(html: str) -> None:
    """Stop/Restart/Interrupt must be reachable from every tab."""
    tabs_start = html.index('class="agent-tabs"')
    header = html[:tabs_start]

    for marker in (
        'id="stop-agent-btn"',
        'id="restart-agent-btn"',
        'id="interrupt-btn"',
        'id="full-dialog-btn"',
    ):
        assert marker in header, f"{marker} must stay outside the tab panels"


def test_stats_header_button_is_preserved(html: str) -> None:
    """§7.3: the popup button stays; the tab is an addition, not a swap."""
    assert 'id="stats-btn"' in html
    assert "openAgentStats" in html


# ── no polling, no implicit probing ──────────────────────────────────


def _function_body(source: str, name: str) -> str:
    start = source.index(f"function {name}(")
    depth = 0
    for idx in range(source.index("{", start), len(source)):
        if source[idx] == "{":
            depth += 1
        elif source[idx] == "}":
            depth -= 1
            if depth == 0:
                return source[start:idx + 1]
    raise AssertionError(f"could not extract {name}")


def test_no_timer_drives_the_new_tabs(js: str) -> None:
    for fn in ("loadMcpTab", "loadStatsTab", "loadMcpCatalog", "switchAgentTab"):
        body = _function_body(js, fn)
        assert "setInterval" not in body, f"{fn} must not poll"
        assert "setTimeout" not in body, f"{fn} must not schedule work"


def test_selecting_an_agent_does_not_fetch_stats(js: str) -> None:
    """Requirement: Stats must not recompute continuously — selecting an
    agent must not trigger it unless that tab is the active one."""
    body = _function_body(js, "selectAgent")
    assert "loadStatsTab" in body
    # It must be guarded by the active-tab check, not called unconditionally.
    assert re.search(r"if\s*\(\s*activeTab\s*===\s*'stats'\s*\)\s*loadStatsTab", body)


def test_stats_refresh_is_explicit(js: str) -> None:
    body = _function_body(js, "setupStatsTab")
    assert "stats-refresh-btn" in body
    assert "loadStatsTab" in body


def test_switch_persists_active_tab(js: str) -> None:
    body = _function_body(js, "switchAgentTab")
    assert "localStorage.setItem" in body


def test_switch_sets_hidden_and_aria_selected(js: str) -> None:
    body = _function_body(js, "switchAgentTab")
    assert "hidden" in body
    assert "aria-selected" in body


def test_mcp_tab_loads_only_once_per_agent(js: str) -> None:
    body = _function_body(js, "switchAgentTab")
    # The selected agent MUST be referenced as ``globalThis.selectedAgent``
    # since a related internal incident analysis (not included here): there is a single
    # storage slot, and a bare ``selectedAgent`` here would be a read of a
    # module-local binding that no longer exists — silently undefined, so
    # the once-per-agent gate would compare against undefined forever.
    #
    # The required form is deliberate, not incidental. After this incident
    # the IDENTIFIER IS THE BUG, so an optional ``globalThis.`` group would
    # accept the exact broken spelling this guard exists to reject.
    assert re.search(
        r"mcpTabLoadedFor\s*!==\s*globalThis\.selectedAgent", body
    ), body
    assert re.search(
        r"statsTabLoadedFor\s*!==\s*globalThis\.selectedAgent", body
    ), body


# ── XSS-safe rendering ───────────────────────────────────────────────


def test_new_render_functions_avoid_innerhtml(js: str) -> None:
    for fn in (
        "renderMcpAttached",
        "renderMcpCatalog",
        "renderStatsTab",
        "mcpEl",
    ):
        body = _function_body(js, fn)
        assert "innerHTML" not in body, f"{fn} must not use innerHTML"


def test_mcp_element_helper_uses_textcontent(js: str) -> None:
    assert "textContent" in _function_body(js, "mcpEl")


# ── tool counts are never invented ───────────────────────────────────


def test_attached_tool_count_requires_live_data(js: str) -> None:
    """§7.2: a count is rendered only for a server the agent reports."""
    body = _function_body(js, "renderMcpAttached")
    assert "if (attached)" in body
    assert "liveByServer[srv.name].length" in body


def test_catalog_tool_count_requires_tools_hint(js: str) -> None:
    body = _function_body(js, "renderMcpCatalog")
    assert "Array.isArray(entry.tools_hint)" in body


def test_catalog_render_does_not_fetch(js: str) -> None:
    """Rendering must never open a connection to an MCP server."""
    assert "fetch(" not in _function_body(js, "renderMcpCatalog")
    assert "fetch(" not in _function_body(js, "renderMcpAttached")


# ── form ─────────────────────────────────────────────────────────────


def test_form_offers_both_transports(html: str) -> None:
    assert 'value="stdio"' in html
    assert 'value="http"' in html


def test_form_mentions_env_placeholder_for_secrets(html: str) -> None:
    assert "${env:" in html


def test_auto_approve_is_not_checked_by_default(html: str) -> None:
    idx = html.index('id="mcp-f-auto-approve"')
    tag = html[html.rindex("<input", 0, idx):html.index(">", idx)]
    assert "checked" not in tag
