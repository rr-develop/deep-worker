# DESIGN: Safety guard for `deep-worker --remove-before` with future cutoff

**Status:** ✅ IMPLEMENTED
**Author:** lc6_fix
**Date:** 2026-04-23
**Motivated by:** 2026-04-23 incident — `deep-worker --remove-before 2030-01-01`
ran against real `$HOME` during an ad-hoc integration probe, deleted 24
threads / 13580 checkpoints from every `sessions_*.db` file in the legacy
ancestor tree (since removed — POLICIES §16), across every agent. Data was
not recoverable through the live SQL table
(though raw pages survive in the file as free pages). Root cause: the CLI
happily accepts any ISO-8601 or relative date without sanity-checking that
it is in the past — `2030-01-01`, being in the future, matches every
existing thread's `updated_at`, effectively meaning "delete everything".

---

## 1. Problem

`deep-worker --remove-before DATE` is a globally-scoped destructive
command. It sweeps every per-agent checkpoint DB under `~/.deep_worker/`
plus (after commit `41703a1`) every `~/.deep_worker/<agent>/summaries/`
file, dropping anything whose `updated_at < DATE`. At the time of the
incident the sweep also covered the legacy ancestor tree's `sessions_*.db`
and global `sessions.db`, which have since been removed (POLICIES §16).

The CLI today has only one guard: `--dry-run`. It has **no** guard for
the common mistake of supplying a future cutoff, which is semantically
equivalent to "nuke everything", yet indistinguishable from a plausible
typo or copy-paste error.

`DATE` parsing already accepts both absolute ISO-8601 (`2030-01-01`) and
relative spellings (`30d`, `1y`). A user who types `1y` when they meant
`-1y`, or a script that constructs a date arithmetically and overshoots,
or an agent (me) that uses `2030-01-01` in a "clearly far-future, should
match every test fixture" test script — all produce the same catastrophic
outcome.

---

## 2. Goals

1. When the parsed cutoff is **in the future** (`cutoff > now`), abort by
   default with a clear error message explaining:
   - the parsed cutoff timestamp,
   - why it is blocked ("future cutoff would delete ALL data"),
   - the opt-out flag.
2. Provide an explicit opt-out flag `--force` to permit the operation for
   users who genuinely want to delete everything (e.g. total reset).
3. Preserve all existing behaviour for past cutoffs (no guard triggers,
   no behavioural change, no new prompts). `--dry-run` continues to work
   unchanged.
4. Make the guard impossible to bypass by accident — subprocess invokers
   in tests are covered (flag must be present in `argv`, otherwise abort
   even in `--dry-run`).
5. Surface the warning in help text and README so readers learn about
   the guard before running the command.

### Non-goals

- No interactive y/N confirmation prompt. This command is used in
  scripts (manual ops, CI, CLI invocations); blocking stdin is hostile
  and would break non-TTY callers. A flag-based opt-out is simpler and
  more predictable.
- No guard on past cutoffs, even very-old ones. `--remove-before 1y`
  (year ago) is legitimate and common; we don't second-guess it.
- Not touching `remove_threads_before` in `sessions.py` — that is a
  library function used both from CLI and tests. The guard lives in the
  CLI layer (`remove_before_command` in `cli/agent_command.py`) where
  user intent is expressed. Tests that call `remove_threads_before`
  directly (with explicit `db_path=tmp_path/test.db`) are already safe
  and should not be burdened.

---

## 3. Design

### 3.1 Where the guard lives

In `deep_worker_cli/cli/agent_command.py`, function
`remove_before_command`. That is the single entry point from CLI
`main()` dispatch. It already parses the date via
`deep_worker_cli.sessions.parse_date`, so it has the parsed `datetime`
BEFORE calling any destructive function.

The guard is inserted immediately after `parse_date`, BEFORE the call
to `remove_threads_before(...)` and BEFORE `prune_summaries_before(...)`.

```
    cutoff_dt = parse_date(date_str)
    now = datetime.now(timezone.utc)
    if cutoff_dt > now and not force:
        console.print(red("refusing future-cutoff ..."))
        return 2   # distinct exit code — not 0, not 1
    # ... existing logic unchanged
```

### 3.2 New `--force` argparse flag

Added to the parser next to `--dry-run`:

```
parser.add_argument(
    "--force",
    action="store_true",
    dest="force",
    help=(
        "Required to confirm --remove-before when the cutoff is in the "
        "future (which would delete ALL data). Harmless for past cutoffs."
    ),
)
```

`main()` forwards `args.force` into `remove_before_command(date_str,
dry_run, force)`.

### 3.3 `--remove-before` help-text update

Extended to surface the guard and the destructive scope:

```
--remove-before DATE
    Remove conversation data older than DATE (ISO 8601 or relative like
    30d, 1y). Prunes checkpoint threads AND offloaded summary sections
    under ~/.deep_worker/<agent>/summaries/ across ALL agents, not just
    one. If DATE resolves to a future timestamp, the command refuses by
    default and requires --force to proceed (since a future cutoff means
    "delete everything"). Always pair with --dry-run the first time.
```

### 3.4 Exit codes

- `0` — operation succeeded (or nothing to do).
- `1` — real error (invalid date, I/O failure).
- `2` — future cutoff without `--force` (distinct so shell scripts can
  detect the guard vs a parse error).

### 3.5 Edge cases

| Case | Behaviour |
|---|---|
| `--remove-before 30d` | past → proceeds (unchanged) |
| `--remove-before 2026-01-01` (in the past at run time) | proceeds (unchanged) |
| `--remove-before 2030-01-01` (future) | aborts with exit 2, clear message |
| `--remove-before 2030-01-01 --force` | proceeds |
| `--remove-before 2030-01-01 --dry-run` | aborts with exit 2 **still** — dry-run is not an opt-out. A dry-run of "would delete everything" is still a sign of a bug, and forcing `--force` to be present flushes the confusion out. |
| `--remove-before 2030-01-01 --dry-run --force` | proceeds in dry-run mode |
| Malformed DATE | unchanged — parse_date raises, existing error path |
| Relative `1y` interpreted by parse_date as past (default semantics — "1 year ago") | proceeds |

### 3.6 Constraints compliance

- **Async preserved**: `remove_before_command` stays `async`. The guard
  is a sync check (compare two datetimes) inserted ahead of the async
  work; it does not convert anything.
- **No polling**: the guard is a single comparison, not a loop.
- **No new threads**: none introduced.
- **Event-driven**: nothing blocking added to any event loop — the
  comparison happens before the first await.
- **Cache untouched**: the guard runs at the CLI boundary, before any
  DB/backend/cache interaction.

---

## 4. Test plan (TDD)

File: `deep_worker_cli/tests/unit/cli/test_remove_before_future_guard.py`.

1. `test_future_cutoff_without_force_is_blocked` — call
   `remove_before_command("2030-01-01", dry_run=False, force=False)` in
   an empty tmp-HOME → returns `2`, prints a message containing
   `"future"` and `"--force"`, and no DB / summary file is touched.

2. `test_future_cutoff_without_force_blocked_even_in_dry_run` — as above
   with `dry_run=True` → still returns `2`. Dry-run is not an opt-out.

3. `test_future_cutoff_with_force_proceeds` — with `force=True` → runs
   the underlying operation (exit `0`); since we seed nothing, no data
   is removed; the assertion is just that it did NOT abort.

4. `test_past_cutoff_no_guard` — `"2020-01-01"` (past) with
   `force=False` → proceeds normally, exit `0`.

5. `test_relative_past_cutoff_no_guard` — `"30d"` with `force=False` →
   proceeds normally.

6. `test_future_cutoff_error_message_mentions_parsed_timestamp` —
   error text includes the ISO-formatted parsed cutoff so the user can
   verify the misparse immediately.

All six tests run in `monkeypatch.setenv("HOME", str(tmp_path))` so they
do not touch the real `~/.deep_worker`. No subprocess, no RPyC, no LLM.

---

## 5. Rollback

Revert the single commit. The guard is entirely additive; the existing
code path is unchanged when `force=True` or `cutoff <= now`.

---

## 6. Acceptance checklist

- [ ] `--force` flag present in `--help` output, described clearly.
- [ ] `--remove-before` help-text mentions the global destructive scope
      and the future-cutoff guard.
- [ ] 6 new unit tests pass.
- [ ] Full test suite stays green.
- [ ] README "Thread management commands" block warns about
      `~/.deep_worker/` global scope and forbids ad-hoc invocation
      outside `HOME`-isolated tests.

---

**End of design.**
