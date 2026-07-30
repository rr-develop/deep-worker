"""CLI parsing tests for the startup query options.

See ``deep_worker/deep_worker_cli/docs/DESIGN_STARTUP_QUERY_OPTIONS.md`` §6.1.

Contract under test:

  * ``--sum-query`` / ``--sum-query-file`` and ``-q/--query`` /
    ``--query-file`` parse into namespace attributes.
  * The two members of each pair are mutually exclusive (argparse exit 2).
  * The two pairs may be combined.
  * ``resolve_startup_queries`` reads files as UTF-8, hard-fails (exit 2)
    on unreadable files and on empty / whitespace-only content.
  * Startup queries are NOT persisted into ``launch_config``.
"""

from __future__ import annotations

import pytest

from deep_worker_cli.cli.agent_command import (
    build_launch_config,
    create_parser,
    resolve_startup_queries,
)


class TestStartupQueryParsing:
    def test_parses_summarize_without_prompt(self) -> None:
        args = create_parser().parse_args(["ag", "--summarize"])
        assert args.summarize is True
        assert args.query is None

    def test_parses_summarize_with_custom_prompt(self) -> None:
        args = create_parser().parse_args(
            ["ag", "--summarize", "--query", "Focus on decisions"]
        )
        assert args.summarize is True
        assert args.query == "Focus on decisions"

    def test_parses_sum_query_text(self) -> None:
        args = create_parser().parse_args(["ag", "--sum-query", "hello"])
        assert args.sum_query == "hello"
        assert args.sum_query_file is None

    def test_parses_sum_query_file(self, tmp_path) -> None:
        f = tmp_path / "q.txt"
        f.write_text("from file", encoding="utf-8")
        args = create_parser().parse_args(["ag", "--sum-query-file", str(f)])
        assert args.sum_query_file == f
        assert args.sum_query is None

    def test_parses_plain_query_text_and_short_form(self) -> None:
        args = create_parser().parse_args(["ag", "-q", "do it"])
        assert args.query == "do it"
        args2 = create_parser().parse_args(["ag", "--query", "do it"])
        assert args2.query == "do it"

    def test_parses_query_file(self, tmp_path) -> None:
        f = tmp_path / "q.txt"
        f.write_text("plain from file", encoding="utf-8")
        args = create_parser().parse_args(["ag", "--query-file", str(f)])
        assert args.query_file == f

    def test_mutex_sum_query_pair_rejected(self) -> None:
        with pytest.raises(SystemExit) as exc:
            create_parser().parse_args(
                ["ag", "--sum-query", "a", "--sum-query-file", "/x"]
            )
        assert exc.value.code == 2

    def test_mutex_query_pair_rejected(self) -> None:
        with pytest.raises(SystemExit) as exc:
            create_parser().parse_args(["ag", "-q", "a", "--query-file", "/x"])
        assert exc.value.code == 2

    def test_pair_combination_allowed(self) -> None:
        args = create_parser().parse_args(
            ["ag", "--sum-query", "s", "-q", "p"]
        )
        assert args.sum_query == "s"
        assert args.query == "p"


class TestResolveStartupQueries:
    @pytest.mark.parametrize("sum_option", ["--sum-query", "--sum-query-file"])
    def test_summarize_rejects_pseudo_summary_options(
        self, sum_option: str, tmp_path
    ) -> None:
        value = "pseudo"
        if sum_option.endswith("-file"):
            file = tmp_path / "pseudo.txt"
            file.write_text(value, encoding="utf-8")
            value = str(file)
        args = create_parser().parse_args(
            ["ag", "--summarize", sum_option, value]
        )
        with pytest.raises(SystemExit) as exc:
            resolve_startup_queries(args)
        assert exc.value.code == 2

    def test_resolve_inline_texts(self) -> None:
        args = create_parser().parse_args(["ag", "--sum-query", "s", "-q", "p"])
        sum_query, query = resolve_startup_queries(args)
        assert sum_query == "s"
        assert query == "p"

    def test_resolve_from_files(self, tmp_path) -> None:
        sf = tmp_path / "s.txt"
        sf.write_text("sum text", encoding="utf-8")
        qf = tmp_path / "q.txt"
        qf.write_text("plain text", encoding="utf-8")
        args = create_parser().parse_args(
            ["ag", "--sum-query-file", str(sf), "--query-file", str(qf)]
        )
        sum_query, query = resolve_startup_queries(args)
        assert sum_query == "sum text"
        assert query == "plain text"

    def test_resolve_none_when_absent(self) -> None:
        args = create_parser().parse_args(["ag"])
        assert resolve_startup_queries(args) == (None, None)

    def test_file_missing_fails(self, tmp_path) -> None:
        missing = tmp_path / "nope.txt"
        args = create_parser().parse_args(["ag", "--query-file", str(missing)])
        with pytest.raises(SystemExit) as exc:
            resolve_startup_queries(args)
        assert exc.value.code == 2

    def test_file_empty_fails(self, tmp_path) -> None:
        f = tmp_path / "empty.txt"
        f.write_text("", encoding="utf-8")
        args = create_parser().parse_args(["ag", "--sum-query-file", str(f)])
        with pytest.raises(SystemExit) as exc:
            resolve_startup_queries(args)
        assert exc.value.code == 2

    def test_text_whitespace_only_fails(self) -> None:
        args = create_parser().parse_args(["ag", "--query", "   \n\t "])
        with pytest.raises(SystemExit) as exc:
            resolve_startup_queries(args)
        assert exc.value.code == 2


class TestStartupQueryNotPersisted:
    def test_startup_query_not_persisted_to_launch_config(self) -> None:
        args = create_parser().parse_args(
            ["ag", "--sum-query", "s", "-q", "p"]
        )
        cfg = build_launch_config(args)
        # None of the startup-query keys may appear in the persisted config.
        for key in ("sum_query", "sum_query_file", "query", "query_file"):
            assert key not in cfg, f"{key} must NOT be persisted to launch_config"

    def test_summarize_not_persisted_to_launch_config(self) -> None:
        args = create_parser().parse_args(
            ["ag", "--summarize", "--query", "Focus"]
        )
        cfg = build_launch_config(args)
        for key in ("summarize", "summary_prompt", "startup_summarize"):
            assert key not in cfg


class TestSummarizeDispatchMatrix:
    def test_ordinary_query_without_summarize_still_plain_query(self) -> None:
        args = create_parser().parse_args(["ag", "--query", "hello"])
        assert args.summarize is False
        sum_query, query = resolve_startup_queries(args)
        assert sum_query is None
        assert query == "hello"

    def test_resolve_summarize_keeps_prompt_as_query_text(self) -> None:
        args = create_parser().parse_args(
            ["ag", "--summarize", "--query", "Focus"]
        )
        sum_query, query = resolve_startup_queries(args)
        assert sum_query is None
        assert query == "Focus"
