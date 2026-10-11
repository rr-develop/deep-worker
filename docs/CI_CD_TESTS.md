# CI/CD Test Rules — GitHub Actions

> **Audience**: every automated agent and every human contributor.
>
> **Why this document exists**: the Unit tests workflow fails on commits that take host-specific assumptions for granted. The two regressions that produced the August-2026 red-pipeline cycle were:
>
> 1. a test that hardcoded an absolute developer-workstation path as a git tree path (developer workstation only — CI runners use `$GITHUB_WORKSPACE`),
> 2. a test that read `iter_legacy_thread_db_paths()` against a pre-v6 contract that the v6 storage migration had already invalidated.
>
> Both passed locally because the developer host satisfies the assumptions; both failed in CI because the assumptions are not portable. This document makes every portability rule explicit so future contributors do not repeat the mistake.

---

## 1. Where the pipelines live

| Pipeline | File | Triggers | Purpose |
|---|---|---|---|
| Unit tests | `.github/workflows/unittest.yml` | `push` + `pull_request` on `master` and `main` | full `pytest tests/unit_tests` matrix on Python 3.11 and 3.12 |
| Codespell | `.github/workflows/codespell.yml` | `push` + `pull_request` on `master` and `main` | spelling check, `.codespellrc` at repo root |

The pre-commit hook (`.githooks/pre-commit`) is **local** — it runs `pyright strict` on provider packages, the secret scan, `.venv` / forbidden-lock-file enforcement, and the provider-manifest invariant. It does **not** run pytest (pyright is slow enough; pytest is the responsibility of the GitHub workflow).

**Both** pipelines must be green for a push to master to be considered acceptable. Bypassing either is forbidden (see [`POLICIES.md`](./POLICIES.md) §5 and `CLAUDE.md` §4 — `--no-verify` is an absolute prohibition).

---

## 2. What the Unit tests workflow actually does

```yaml
working-directory: deep_worker/deep_worker_cli
steps:
  - pip install --upgrade pip
  - pip install -e .
  - pip install pytest pytest-asyncio responses pytest-mock
  - env: DEEPSEEK_API_KEY=test-key-not-used-by-unit-tests
    run: pytest tests/unit_tests -q
```

The package is installed editable (one .pth file, no copy). `DEEPSEEK_API_KEY` is a dummy — `AgentConfig` validates the key at construction time, and unit tests construct configs without one, so no test makes a real API call. **Do not change this env var name or value format**; tests that read it will silently break.

### Reproducing CI locally — the exact command set

```bash
# From the repo root (NOT inside a .venv, NOT with pip-installed deps).
cd <repo-root>/deep_worker/deep_worker_cli

# Equivalent of: pip install -e .  + deps
# (the project runs from source — see CLAUDE.md §1 — but the editable
# install only places a .pth; runtime imports still resolve to source.)
DEEPSEEK_API_KEY=test-key-not-used-by-unit-tests \
    python3 -m pytest tests/unit_tests -q
```

If your local run is green and CI is red, the regression is in your portability assumptions (see §3 below), not in the workflow.

---

## 3. The portability rules every CI test must obey

A unit test that passes on a developer workstation but fails on a CI runner is the **most expensive** class of bug here — it turns the entire push-gate red, blocks merges, and produces no useful diagnostic at the developer side. These rules make it impossible to land one.

### 3.1 Never hardcode host-specific paths

The CI runner's workspace is at:

```
/home/runner/work/<repo>/<repo>
```

…**not** at any developer-workstation layout. The git tree path that the runner sees is whatever `actions/checkout@v4` puts at `$GITHUB_WORKSPACE`. Tests that need "any real git tree" must discover it dynamically.

❌ Forbidden:

```python
def test_git_commit_in_real_tree() -> None:
    sha, _ = _get_git_commit_info("/absolute/dev/path")
    assert sha != "unknown"
```

✅ Correct — discover via subprocess:

```python
def test_git_commit_in_real_tree() -> None:
    discovered = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        capture_output=True, text=True, timeout=5,
    ).stdout.strip()
    assert discovered, "must run from inside a git working tree"
    sha, _ = _get_git_commit_info(discovered)
    assert sha != "unknown"
```

`subprocess.run(["git", ...])` works on CI runners (Ubuntu-latest has `git` pre-installed) and on every developer host. The cost is one extra subprocess per test that needs it.

Other host-specific paths that have already been bitten or should be considered suspect:

- Absolute developer-workstation checkout paths
- Per-agent state directories under the user's home directory
- `/tmp/<specific-name>` (use `tmp_path` fixture or `tempfile.mkdtemp()`)
- Any absolute path that doesn't come from `tempfile`, `tmp_path`, or `pathlib.Path.cwd()`

### 3.2 Tests must match the current code contract, not last week's

When a function's return shape changes, the test is the source of truth for the new contract. If you change a function's signature or return value, update the tests in the same commit. **A test that says `p.parent.name == "legacy_agent"` while the function now returns the agent dir directly is a red-pipeline bug, not a passing test.**

This is what happened with `iter_legacy_thread_db_paths()`: the v6 migration (commit `1492143`) changed the function to return agent dirs instead of `threads.db` paths, but `test_iter_legacy_thread_db_paths_returns_only_legacy` was not updated. The test passed locally because the developer's fixture happened to satisfy the old expectations, then failed on every CI run for weeks.

Rule: **if you change a public function's return type or semantics, grep for callers and update them, including tests**, before the commit.

### 3.3 Network and external processes

By design, the unit test suite makes **no network calls and no subprocess invocations of external long-running processes**. The dummy `DEEPSEEK_API_KEY` is enough to keep `AgentConfig` construction from raising; nothing in `tests/unit_tests` should call a real provider API.

If your test needs network or a daemon, it does not belong in `tests/unit_tests`. Move it to `tests/integration_tests/` and mark it with `@pytest.mark.integration` (the workspace filters those out by default).

### 3.4 Time and randomness

Use `freezegun` or the `monkeypatch` fixture for "now"; use a fixed seed for randomness. Wall-clock and `random.random()` without a seed are the next-most-common red-pipeline cause after hardcoded paths.

### 3.5 Filesystem permissions

`POLICIES.md` §4 governs: files `0o600`, dirs `0o700`. Tests that create files in `tmp_path` must respect the same defaults — if your test mode is wrong, the pre-commit hook will catch it, but CI may also catch a different aspect of the same bug.

---

## 4. Reading the CI failure logs (the right way)

Use the `gh` CLI from any host with valid GitHub auth. The patterns below have solved every red-CI cycle the project has seen.

```bash
# List recent runs on this repo
gh run list --repo <owner>/<repo> --limit 5

# Get the failure summary (which step went red)
gh run view <run-id> --repo <owner>/<repo>

# Pull the failing job's full log (use the job id from `gh run view`)
gh run view <run-id> --repo <owner>/<repo> --log --job <job-id>

# Watch a run live (useful right after `git push`)
gh run watch <run-id> --repo <owner>/<repo> --exit-status
```

`gh auth status` must show a logged-in github.com account with access to the repository.

**Always look at `gh run view --log --job <id>` before guessing**. A red pipeline from a portability bug has a very specific failure shape: the suite reaches ~99 % completion, then a single test fails with an obvious assertion mismatch (e.g. `'unknown' != 'unknown'`). If you see that pattern, the bug is in test portability, not in your code change — fix the test, not the code.

---

## 5. Procedure when CI is red

1. **Pull the log** with `gh run view … --log --job <id>`. Confirm the failure is in this repo (not a flaky upstream or a runner problem — the matrix Python 3.11 vs 3.12 will tell you; if both versions fail the same way, it's our bug).
2. **Reproduce locally** with the exact §2 command. If local is green, your fix for a previous red cycle was incomplete — go back to step 1.
3. **Identify the portability violation** with the §3 rules.
4. **Fix the test**, not the workflow. Workflow edits require a maintainer sign-off and are almost never the right answer.
5. **Re-run local** to the same green. Then `git push` — the push will trigger the next workflow run.
6. **Watch the run** with `gh run watch` and confirm green before reporting done.

---

## 6. Escalation paths

| Symptom | First action | Where to ask |
|---|---|---|
| One test fails the same way on 3.11 and 3.12 | Read the log, apply §3 | This document (most cases) |
| Pre-commit hook blocks locally | Read `.githooks/pre-commit`, do **not** use `--no-verify` | [`CLAUDE.md`](../deep_worker/CLAUDE.md) §4 + [`POLICIES.md`](./POLICIES.md) §5 |
| pyright strict reports an error in a provider package | Fix the type error or remove the dead import — never `# type: ignore` without justification | `POLICIES.md` §5 |
| Workflow file looks wrong | Stop, ask the maintainer before editing `.github/workflows/*.yml` | direct ping |
| CI runner itself has a problem (`apt` lock, OOM, disk) | Re-run the workflow (`gh run rerun`) | infrastructure, not code |
| Tests pass locally, **red in CI**, and §3 rules don't surface a violation | Capture full log, ask for a second pair of eyes | escalate to a reviewer agent |

---

## 7. Quick checklist before every push that touches a test file

```text
[ ] Test creates no host-specific path (use tmp_path, subprocess, Path.cwd()).
[ ] Test asserts on the current contract of any function it calls.
[ ] Test that needs "a git tree" discovers it via subprocess, not literal.
[ ] Test makes no real network call.
[ ] DEEPSEEK_API_KEY dummy is preserved in any subprocess env it sets.
[ ] Pre-commit hook (pyright strict on touched packages) passes locally.
[ ] Local `pytest tests/unit_tests -q` shows 0 failures, no new skips.
[ ] After push, `gh run watch` confirms green before reporting done.
```

Following this list is enough to keep the pipeline green through future refactors. If you find a portability rule not covered here, add it in the same PR — that's how this document stays current.
