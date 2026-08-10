"""Auto-detection of project languages and managing them.

See ``docs/DESIGN_LSP_LANGUAGES.md``.

Why this exists: on first activation Serena enables EXACTLY ONE language —
the most frequent one (``serena_config.py:376-382``) — and adds the rest
only in interactive mode, which the MCP server does not have. On a project
with kotlin 65% / python 32% this silently dropped the whole Python backend,
and the agent got ``Active languages: ['kotlin']``.

Pinned here:

* a 10% threshold of RECOGNIZED sources;
* merging never removes already-configured languages;
* the ``project.yml`` editor is strict and fail-loud: ambiguity →
  refusal, not file corruption;
* writes are atomic, permissions 0600/0700, template comments intact.
"""
from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest
import yaml

from deep_worker_cli.lsp.languages import (
    LanguageConfigError,
    _LANGUAGE_THRESHOLD_PCT,
    apply_language_changes,
    detected_languages,
    merge_languages,
    read_languages,
    validate_language_name,
    write_languages,
)


def _write(path: Path, text: str, mode: int = 0o600) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    path.write_text(text)
    path.chmod(mode)
    return path


_BLOCK_CONFIG = """\
# the name by which the project can be referenced
project_name: "demo"

# list of languages for which language servers are started
#   kotlin              python              rust
languages:
- kotlin

# the encoding used by text files in the project
encoding: "utf-8"
"""

_INLINE_CONFIG = """\
project_name: "demo"
languages: ["python"]
encoding: "utf-8"
"""


# ── §4.2: threshold ──────────────────────────────────────────────────────


def test_detect_applies_ten_percent_threshold() -> None:
    """The threshold from the user requirement: 10%."""
    composition = {"kotlin": 65.42, "python": 31.83, "bash": 2.68}
    assert detected_languages(composition) == ["kotlin", "python"]
    assert _LANGUAGE_THRESHOLD_PCT == 10.0


def test_detected_sorted_by_pct_then_priority() -> None:
    """A6: the order is deterministic, not whatever the dict walk yields.

    For Serena the first language in the list is the default language, so a
    random order would change behavior from run to run.
    """
    composition = {"python": 20.0, "kotlin": 60.0, "go": 15.0}
    assert detected_languages(composition) == ["kotlin", "python", "go"]


def test_empty_composition_detects_nothing() -> None:
    assert detected_languages({}) == []


# ── §4.2: merge ────────────────────────────────────────────────────


def test_merge_keeps_existing_and_adds_detected() -> None:
    assert merge_languages(["kotlin"], ["kotlin", "python"]) == [
        "kotlin",
        "python",
    ]


def test_merge_never_removes_configured_language() -> None:
    """The user may have added a language by hand — we do not drop it."""
    assert merge_languages(["kotlin", "go"], ["kotlin"]) == ["kotlin", "go"]


def test_merge_preserves_existing_order() -> None:
    """The first language is Serena's default language; it must not be reordered."""
    assert merge_languages(["go", "kotlin"], ["kotlin", "python"]) == [
        "go",
        "kotlin",
        "python",
    ]


# ── §4.4 / A2: reading and the strict editor ─────────────────────────────


def test_read_block_list(tmp_path: Path) -> None:
    cfg = _write(tmp_path / ".serena" / "project.yml", _BLOCK_CONFIG)
    assert read_languages(cfg) == ["kotlin"]


def test_inline_languages_list_is_supported(tmp_path: Path) -> None:
    """A2: Serena's template writes languages inline — the format is mandatory."""
    cfg = _write(tmp_path / ".serena" / "project.yml", _INLINE_CONFIG)
    assert read_languages(cfg) == ["python"]
    write_languages(cfg, ["python", "kotlin"])
    assert read_languages(cfg) == ["python", "kotlin"]
    assert yaml.safe_load(cfg.read_text())["project_name"] == "demo"


def test_write_preserves_comments_and_other_keys(tmp_path: Path) -> None:
    cfg = _write(tmp_path / ".serena" / "project.yml", _BLOCK_CONFIG)
    write_languages(cfg, ["kotlin", "python"])
    text = cfg.read_text()
    assert "# the name by which the project can be referenced" in text
    assert "#   kotlin              python              rust" in text
    assert "# the encoding used by text files in the project" in text
    assert 'encoding: "utf-8"' in text
    assert read_languages(cfg) == ["kotlin", "python"]


def test_duplicate_top_level_key_refused(tmp_path: Path) -> None:
    """A2: ambiguity → refusal, the file is left untouched."""
    src = 'project_name: "d"\nlanguages:\n- kotlin\nlanguages:\n- python\n'
    cfg = _write(tmp_path / ".serena" / "project.yml", src)
    before = cfg.read_bytes()
    with pytest.raises(LanguageConfigError):
        write_languages(cfg, ["kotlin", "python"])
    assert cfg.read_bytes() == before


def test_nested_or_commented_languages_not_matched(tmp_path: Path) -> None:
    """A2: `languages:` inside a nested map or in a comment is not a target."""
    src = (
        'project_name: "d"\n'
        "# languages:\n"
        "nested:\n"
        "  languages:\n"
        "  - kotlin\n"
    )
    cfg = _write(tmp_path / ".serena" / "project.yml", src)
    before = cfg.read_bytes()
    with pytest.raises(LanguageConfigError):
        write_languages(cfg, ["python"])
    assert cfg.read_bytes() == before


def test_comment_inside_block_sequence_refused(tmp_path: Path) -> None:
    """A2: a comment does NOT terminate a YAML sequence.

    A naive scan "up to the first non-item line" would cut the node at the
    comment and leave the trailing ``- python`` alive: PyYAML would still see
    python even though the command reported a removal. We cannot relocate the
    comment unambiguously → refusal without writing.
    """
    src = (
        'project_name: "d"\n'
        "languages:\n"
        "- kotlin\n"
        "# kept note\n"
        "- python\n"
        'encoding: "utf-8"\n'
    )
    cfg = _write(tmp_path / ".serena" / "project.yml", src)
    before = cfg.read_bytes()
    with pytest.raises(LanguageConfigError):
        write_languages(cfg, ["kotlin"])
    assert cfg.read_bytes() == before
    assert read_languages(cfg) == ["kotlin", "python"]


def test_blank_line_inside_block_sequence_removes_all_items(
    tmp_path: Path,
) -> None:
    """A2: a blank line inside a sequence carries no semantics.

    The node is consumed whole up to the next top-level key, so the removal
    really removes — no tail is left behind.
    """
    src = (
        'project_name: "d"\n'
        "languages:\n"
        "- kotlin\n"
        "\n"
        "- python\n"
        'encoding: "utf-8"\n'
    )
    cfg = _write(tmp_path / ".serena" / "project.yml", src)
    write_languages(cfg, ["kotlin"])
    assert read_languages(cfg) == ["kotlin"]
    assert yaml.safe_load(cfg.read_text())["encoding"] == "utf-8"


def test_unexpected_line_inside_block_sequence_refused(tmp_path: Path) -> None:
    """A2: any unexpected line inside the node → refusal, not guessing."""
    src = 'project_name: "d"\nlanguages:\n- kotlin\n  weird continuation\n'
    cfg = _write(tmp_path / ".serena" / "project.yml", src)
    before = cfg.read_bytes()
    with pytest.raises(LanguageConfigError):
        write_languages(cfg, ["python"])
    assert cfg.read_bytes() == before


def test_malformed_yaml_refused(tmp_path: Path) -> None:
    cfg = _write(tmp_path / ".serena" / "project.yml", "a: [unclosed\n")
    before = cfg.read_bytes()
    with pytest.raises(LanguageConfigError):
        write_languages(cfg, ["python"])
    assert cfg.read_bytes() == before


def test_missing_key_in_existing_file_refused(tmp_path: Path) -> None:
    cfg = _write(tmp_path / ".serena" / "project.yml", 'project_name: "d"\n')
    before = cfg.read_bytes()
    with pytest.raises(LanguageConfigError):
        write_languages(cfg, ["python"])
    assert cfg.read_bytes() == before


def test_trailing_newline_preserved(tmp_path: Path) -> None:
    """A2: the presence/absence of the final newline is preserved."""
    cfg = _write(
        tmp_path / ".serena" / "project.yml",
        'project_name: "d"\nlanguages:\n- kotlin',
    )
    write_languages(cfg, ["kotlin", "python"])
    assert not cfg.read_text().endswith("\n\n")
    assert read_languages(cfg) == ["kotlin", "python"]


# ── A3: permissions and atomicity ──────────────────────────────────────────


def test_write_preserves_file_mode(tmp_path: Path) -> None:
    cfg = _write(tmp_path / ".serena" / "project.yml", _BLOCK_CONFIG, mode=0o600)
    write_languages(cfg, ["kotlin", "python"])
    assert stat.S_IMODE(cfg.stat().st_mode) == 0o600


def test_write_does_not_widen_permissions(tmp_path: Path) -> None:
    cfg = _write(tmp_path / ".serena" / "project.yml", _BLOCK_CONFIG, mode=0o640)
    write_languages(cfg, ["kotlin", "python"])
    assert stat.S_IMODE(cfg.stat().st_mode) == 0o640


def test_write_failure_leaves_original_byte_identical(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A3: a write failure leaves neither corruption nor temporary garbage."""
    cfg = _write(tmp_path / ".serena" / "project.yml", _BLOCK_CONFIG)
    before = cfg.read_bytes()

    def _boom(*_a: object, **_k: object) -> None:
        raise OSError("disk on fire")

    monkeypatch.setattr(os, "replace", _boom)
    with pytest.raises(OSError):
        write_languages(cfg, ["kotlin", "python"])
    assert cfg.read_bytes() == before
    leftovers = [p.name for p in cfg.parent.iterdir() if p.name != "project.yml"]
    assert leftovers == [], f"temp files left behind: {leftovers}"


def test_no_write_when_nothing_changes(tmp_path: Path) -> None:
    cfg = _write(tmp_path / ".serena" / "project.yml", _BLOCK_CONFIG)
    before_mtime = cfg.stat().st_mtime_ns
    changed = apply_language_changes(cfg, desired=["kotlin"])
    assert changed is False
    assert cfg.stat().st_mtime_ns == before_mtime


# ── §5 / A5: CLI-level operations ─────────────────────────────────────


def test_add_is_idempotent(tmp_path: Path) -> None:
    cfg = _write(tmp_path / ".serena" / "project.yml", _BLOCK_CONFIG)
    assert apply_language_changes(cfg, desired=["kotlin"]) is False
    assert read_languages(cfg) == ["kotlin"]


def test_apply_refuses_empty_result(tmp_path: Path) -> None:
    """§5: an empty list means symbolic tools are dead. That is not what we want."""
    cfg = _write(tmp_path / ".serena" / "project.yml", _BLOCK_CONFIG)
    before = cfg.read_bytes()
    with pytest.raises(LanguageConfigError):
        apply_language_changes(cfg, desired=[])
    assert cfg.read_bytes() == before


# ── A1/A5: detection and validation ────────────────────────────────────────


def test_detected_uses_priority_for_ties() -> None:
    """On ties the order is set by Serena's priority, not the dict."""
    composition = {"a_lang": 30.0, "b_lang": 30.0}
    order = detected_languages(composition, {"a_lang": 1, "b_lang": 9})
    assert order == ["b_lang", "a_lang"]


def test_validate_language_name_accepts_known() -> None:
    assert validate_language_name("python", ["python", "kotlin"]) == "python"


def test_validate_language_name_rejects_unknown() -> None:
    """A5: a typo must fail here, not at Serena startup."""
    with pytest.raises(LanguageConfigError) as ei:
        validate_language_name("pyton", ["python", "kotlin"])
    assert "pyton" in str(ei.value)
    assert "python" in str(ei.value), "a hint about the similar language is required"


def test_probe_reports_missing_serena(monkeypatch: pytest.MonkeyPatch) -> None:
    """Serena lives in a separate venv; its absence is a clear error."""
    import deep_worker_cli.lsp.languages as mod

    monkeypatch.setattr(mod, "serena_python", lambda *_a, **_k: None)
    with pytest.raises(LanguageConfigError) as ei:
        mod.language_composition("/tmp")
    assert "serena is not installed" in str(ei.value)


def test_new_file_is_created_private(tmp_path: Path) -> None:
    """A3/policy: a new config is 0600 in a 0700 directory, no 644/777.

    A separate test from ``test_write_preserves_file_mode``: there the file
    already exists and the mode is taken from it. The "file did not exist"
    branch must be pinned explicitly, or the default could widen unnoticed.
    """
    import deep_worker_cli.lsp.languages as mod

    cfg = tmp_path / ".serena" / "project.yml"
    mod._atomic_write(cfg, 'project_name: "d"\nlanguages:\n- kotlin\n')
    assert stat.S_IMODE(cfg.stat().st_mode) == 0o600
    assert stat.S_IMODE(cfg.parent.stat().st_mode) == 0o700
