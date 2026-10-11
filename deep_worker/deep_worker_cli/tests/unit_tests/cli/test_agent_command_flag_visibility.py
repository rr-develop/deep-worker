"""Guards against silently losing the ``--turn-ending-tool`` flag.

A flag can disappear in two ways that unit tests of the surrounding
logic would not notice: removal from the parser (help no longer lists
it) and removal from ``_explicit_cli_flags``'s detector parser (restart
override detection silently stops seeing it).
"""

from deep_worker_cli.cli.agent_command import _explicit_cli_flags, create_parser


def test_flag_visible_in_argparse_help():
    assert "--turn-ending-tool" in create_parser().format_help()


def test_flag_registered_in_detector():
    explicit = _explicit_cli_flags(["a", "--turn-ending-tool", "x"])
    assert "turn_ending_tools" in explicit


def test_detector_silent_when_flag_absent():
    assert "turn_ending_tools" not in _explicit_cli_flags(["a"])
