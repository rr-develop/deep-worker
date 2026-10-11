"""Grammar and parsing tests for ``--turn-ending-tool`` spec strings.

See ``deep_worker_cli/docs/DESIGN_CLI_TURN_ENDING_TOOL_FLAG.md`` §3.2.

Two forms only:

    <name>          -> default predicate ``bool(d.get("success"))``
    <name>:<field>  -> predicate ``d.get("<field>") is True``

Everything else is a hard error at the boundary where the string enters
the system, never later.
"""

import pytest

from deep_worker_cli.middleware.tool_completion import (
    DuplicateTurnEndingToolError,
    TurnEndingTool,
    TurnEndingToolFlagError,
    _canonical_turn_ending_spec_str,
    _parse_turn_ending_tool_specs,
    _validate_turn_ending_spec_str,
)


# ----------------------------------------------------------------------
# Level 1 — single-string grammar
# ----------------------------------------------------------------------


def test_validator_bare_name_returns_name_and_none():
    assert _validate_turn_ending_spec_str("foo") == ("foo", None)


def test_validator_name_and_field():
    assert _validate_turn_ending_spec_str("foo:bar") == ("foo", "bar")


def test_bare_name():
    """``"foo"`` parses to the default-predicate declaration."""
    (spec,) = _parse_turn_ending_tool_specs(["foo"])
    assert isinstance(spec, TurnEndingTool)
    assert spec.tool_name == "foo"
    assert spec.echo_field is None
    assert spec.is_success({"success": 1}) is True
    assert spec.is_success({"success": 0}) is False
    assert spec.is_success({}) is False


def test_name_and_field():
    """``"foo:bar"`` uses strict identity against ``True``."""
    (spec,) = _parse_turn_ending_tool_specs(["foo:bar"])
    assert spec.tool_name == "foo"
    assert spec.echo_field is None
    assert spec.is_success({"bar": True}) is True
    for payload in ({"bar": False}, {"bar": 1}, {"bar": "true"}, {}, {"bar": None}):
        assert spec.is_success(payload) is False, payload


def test_predicate_uses_get_not_bracket():
    """A payload without the key must not raise ``KeyError``."""
    (spec,) = _parse_turn_ending_tool_specs(["foo:document_closed"])
    assert spec.is_success({"unrelated": True}) is False


def test_late_binding_across_multiple_specs():
    """Each predicate closes over its OWN field (default-arg binding)."""
    spec_a, spec_b = _parse_turn_ending_tool_specs(["a:x", "b:y"])
    assert spec_a.is_success({"x": True, "y": False}) is True
    assert spec_a.is_success({"x": False, "y": True}) is False
    assert spec_b.is_success({"x": False, "y": True}) is True
    assert spec_b.is_success({"x": True, "y": False}) is False


def test_empty_name_rejected():
    with pytest.raises(TurnEndingToolFlagError):
        _validate_turn_ending_spec_str(":x")


def test_empty_spec_rejected():
    with pytest.raises(TurnEndingToolFlagError):
        _validate_turn_ending_spec_str("")


def test_empty_field_rejected():
    with pytest.raises(TurnEndingToolFlagError):
        _validate_turn_ending_spec_str("foo:")


def test_dotted_field_rejected():
    with pytest.raises(TurnEndingToolFlagError) as exc:
        _validate_turn_ending_spec_str("foo:a.b")
    assert "nested" in str(exc.value).lower()


def test_extra_colon_rejected():
    with pytest.raises(TurnEndingToolFlagError):
        _validate_turn_ending_spec_str("foo:bar:baz")


@pytest.mark.parametrize(
    "raw",
    [" foo", "foo ", " foo ", "foo: bar", "foo :bar", "a b", "foo:ba r", "\tfoo"],
)
def test_whitespace_rejected(raw):
    with pytest.raises(TurnEndingToolFlagError):
        _validate_turn_ending_spec_str(raw)


def test_non_string_rejected():
    with pytest.raises(TurnEndingToolFlagError):
        _validate_turn_ending_spec_str(None)  # type: ignore[arg-type]


def test_multiple_specs_returned_in_order():
    specs = _parse_turn_ending_tool_specs(["a", "b:x"])
    assert [s.tool_name for s in specs] == ["a", "b"]


def test_empty_input_yields_empty_list():
    assert _parse_turn_ending_tool_specs([]) == []


def test_parse_reports_first_invalid_spec():
    with pytest.raises(TurnEndingToolFlagError) as exc:
        _parse_turn_ending_tool_specs(["ok", "bad:spec:trap", "also:bad:trap"])
    assert "bad:spec:trap" in str(exc.value)


# ----------------------------------------------------------------------
# argparse ``type=`` handler — raw strings on the namespace
# ----------------------------------------------------------------------


def test_canonical_returns_the_raw_string():
    assert _canonical_turn_ending_spec_str("foo:bar") == "foo:bar"
    assert isinstance(_canonical_turn_ending_spec_str("foo"), str)


def test_canonical_raises_on_invalid():
    import argparse

    with pytest.raises(TurnEndingToolFlagError):
        _canonical_turn_ending_spec_str("foo:bar:baz")
    # argparse turns ArgumentTypeError into a clean usage error, so the
    # flag error MUST remain a subclass of it.
    assert issubclass(TurnEndingToolFlagError, argparse.ArgumentTypeError)


# ----------------------------------------------------------------------
# Level 2 — full-list constraints
# ----------------------------------------------------------------------


def test_duplicate_names_rejected():
    with pytest.raises(DuplicateTurnEndingToolError) as exc:
        _parse_turn_ending_tool_specs(["foo", "foo"])
    assert "foo" in str(exc.value)


def test_bare_and_field_form_of_same_name_collide():
    """``foo`` and ``foo:done`` are the SAME tool_name — a duplicate."""
    with pytest.raises(DuplicateTurnEndingToolError) as exc:
        _parse_turn_ending_tool_specs(["foo", "foo:done"])
    assert "foo" in str(exc.value)


def test_builtin_name_rejected():
    from deep_worker_cli.middleware.send_message_completion import TOOL_NAME

    with pytest.raises(DuplicateTurnEndingToolError) as exc:
        _parse_turn_ending_tool_specs([TOOL_NAME])
    message = str(exc.value)
    assert TOOL_NAME in message
    assert "built-in" in message


def test_builtin_name_with_field_still_rejected():
    from deep_worker_cli.middleware.send_message_completion import TOOL_NAME

    with pytest.raises(DuplicateTurnEndingToolError):
        _parse_turn_ending_tool_specs([f"{TOOL_NAME}:success"])


def test_distinct_names_pass_full_list_validation():
    specs = _parse_turn_ending_tool_specs(["foo", "bar:x"])
    assert [s.tool_name for s in specs] == ["foo", "bar"]
