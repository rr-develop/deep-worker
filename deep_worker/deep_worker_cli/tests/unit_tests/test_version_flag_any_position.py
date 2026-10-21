"""``--version`` answers wherever it sits in argv, naming this product.

Why this exists: the router honoured the flag only as ``argv[0]``, so
``deep-worker <agent> --version`` fell through to the agent parser, whose
own ``--version`` action still carried the ancestor's program name. The
same flag therefore printed two different product names depending on its
position. Both the position and the name are pinned here.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pytest

from deep_worker_cli._version import __version__
from deep_worker_cli.cli import router

_CLI_ROOT = Path(__file__).parent.parent.parent


@pytest.mark.parametrize(
    "argv",
    [
        pytest.param(["--version"], id="first-long"),
        pytest.param(["-V"], id="first-short"),
        pytest.param(["someagent", "--version"], id="after-agent-name"),
        pytest.param(["someagent", "-V"], id="after-agent-name-short"),
        pytest.param(["--web", "--version"], id="after-web-flag"),
        pytest.param(["--provider", "deepseek", "--version"], id="after-option-value"),
    ],
)
def test_version_is_served_at_any_position(
    argv: list[str], capsys: pytest.CaptureFixture[str]
) -> None:
    """The dispatcher claims the flag anywhere and prints this product's name."""
    code = router._dispatch_version(argv)

    assert code == 0, f"{argv} was not recognised as a version request"
    out = capsys.readouterr().out
    assert out.strip() == f"deep-worker {__version__}"


@pytest.mark.parametrize(
    "argv",
    [
        pytest.param([], id="empty"),
        pytest.param(["someagent"], id="agent-name-only"),
        pytest.param(["version"], id="agent-literally-called-version"),
        pytest.param(["--web"], id="web"),
        pytest.param(["--", "--version"], id="after-double-dash"),
        pytest.param(["--", "-V"], id="after-double-dash-short"),
        pytest.param(["--", "someagent", "--version"], id="double-dash-then-more"),
    ],
)
def test_argv_without_the_flag_is_left_alone(argv: list[str]) -> None:
    """Without the flag the dispatcher declines, so routing continues.

    The ``version`` case is the reason the feature is a flag and not a
    positional: an agent may legitimately be named "version", and starting
    it must not be hijacked by the version report.

    The ``--`` cases are the same contract one level down. argparse treats
    everything after the separator as positional — it parses
    ``['--', '--version']`` as the agent NAME ``--version`` — so claiming
    the flag there would report the version instead of starting that
    agent. A hand-rolled argv scan does not know that rule unless it is
    told, which is what these cases pin.
    """
    assert router._dispatch_version(argv) is None


def test_double_dash_matches_argparse_semantics() -> None:
    """The ``--`` rule we implement is argparse's own, not an invention.

    Pins the upstream behaviour the dispatcher defers to, so that if a
    future argparse changed it, the mismatch surfaces here rather than as
    a silent divergence between the pre-parse scan and the real parser.
    """
    parser = argparse.ArgumentParser(prog="deep-worker")
    parser.add_argument("name", nargs="?")
    parser.add_argument("-V", "--version", action="store_true")

    namespace = parser.parse_args(["--", "--version"])

    assert namespace.name == "--version"
    assert namespace.version is False


def test_agent_parser_version_action_names_this_product() -> None:
    """The second ``--version`` site must not print the ancestor's name.

    ``_dispatch_version`` now intercepts every argv containing the flag, so
    this action is unreachable in practice — but it is one edit away from
    being reachable again, and it used to disagree with the router. Pinned
    by running the parser, not by reading the literal.
    """
    result = subprocess.run(
        [sys.executable, "-m", "deep_worker_cli.main", "--version"],
        capture_output=True,
        text=True,
        check=False,
        cwd=str(_CLI_ROOT),
    )

    assert result.returncode == 0, result.stderr
    assert f"deep-worker {__version__}" in result.stdout
    assert "deepagents" not in result.stdout.lower()
