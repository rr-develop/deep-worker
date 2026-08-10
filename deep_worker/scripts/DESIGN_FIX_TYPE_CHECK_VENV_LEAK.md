# DESIGN — forbidden `.venv` left behind by repository-provided commands

Status: revision 5 — SIGNED OFF (5 review rounds)
Scope: `deep_worker/scripts/` (new wrappers), both package `Makefile`s, docs
Branch: `dev`

## 1. User-visible symptom

A commit is rejected by the pre-commit hook with:

```
🔍 Checking for forbidden .venv directories...
❌ ERROR: Found forbidden virtual environment in <repo-root>:
<repo-root>/deep_worker/deepagents/.venv
COMMIT BLOCKED: Virtual environments detected!
```

…even though the developer never created a virtual environment on purpose.

Observed live on 2026-08-05 while committing the image-eviction fix.

## 2. Root cause (reproduced, not inferred)

### 2.1 The hook itself is innocent

`deep_worker/scripts/pre-commit-type-check.sh` is a sanctioned `uv run` site.
It creates a project `.venv` and cleans up after itself:

```bash
trap cleanup_venvs EXIT INT TERM
```

**That trap works.** Verified both paths:

| Path | `.venv` afterwards |
|---|---|
| type check passes (`rc=0`) | none |
| type check fails (`rc=1`, deliberate type error injected) | none |

### 2.2 The leak is in what the repository tells humans to run

Every *other* repository-provided command that shells out to `uv run` has no
cleanup at all. `git grep -n "uv run"` over tracked files finds three classes:

**(a) The hook's own remediation output** — `pre-commit-type-check.sh:165,168`:

```
  3. Run type checks manually to verify:
     - cd .../deep_worker_cli && uv run --all-groups ty check deep_worker_cli
     - cd .../deepagents && uv run --all-groups ty check deepagents
```

Reproduced directly:

```
$ cd deep_worker/deepagents
$ find . -maxdepth 1 -name .venv     # (nothing)
$ uv run --all-groups ty check deepagents
rc=0
$ find . -maxdepth 1 -name .venv
./.venv                               # <-- leaked, 76 MB
```

`deep_worker/scripts/README.md:41` repeats the same bare command.

**(b) Both package `Makefile`s** — this is the part revision 1 missed, and it is
the larger surface. `deep_worker_cli/Makefile` and `deepagents/Makefile` each run
bare `uv run` in `test`, `tests`, `coverage`, `integration_test(s)`,
`test_watch`, `benchmark`, `lint*`, `format*`, `check_imports`, and (cli only)
`type`. None of them clean up. `make type` and `make test` are advertised in the
`help` target, i.e. these are commands the repository actively recommends.

This is not hypothetical: `.githooks/pre-commit:64-65` names it as a
contributing cause of the 2026-04-22 incident —

> But `uv run` in the Makefile and scripts/pre-commit-type-check.sh KEPT
> re-creating `.venv` on every developer `make test` / type check.

The hook side of that sentence was fixed (the trap). **The Makefile side never
was.** Revision 1 of this design asserted the hook was the "single sanctioned
place"; that was wrong as a statement about the repository's actual behaviour.

**(c) Non-actionable mentions** — policy prose in `.githooks/pre-commit`,
an internal service's prompt strings that *forbid* `uv run`, a comment in
`pyproject.toml`, and `deep_worker_cli/WORKING_DIR_FIX.md` (a historical
transcript of a past fix). These are not instructions to run anything now.

### 2.3 Why this is self-perpetuating

1. type check fails → developer is *told* to run the bare command (or runs
   `make type` / `make test`, which the help text advertises);
2. `.venv` appears;
3. next commit fails on the **venv check**, not on types;
4. the error now points at virtual environments the developer never knowingly
   created, obscuring the actual type error.

The developer is punished for following the tool's own printed instructions.

## 3. Requirements

R1. Every command the repository *tells a human to run* must clean up its own
    `.venv`, exactly as the hook does. This covers the `Makefile` targets, not
    just the printed hints.
R2. No weakening of the `.venv` ban. It stays a hard stop with the 2026-04-22
    incident rationale (a stale `rpyc` install shadowing a sibling project).
R3. No change to what is type-checked/tested, or to pass/fail semantics.
R4. Cleanup must survive failure and interrupt, not just the happy path.
R5. `uv run` stays confined to cleanup-safe sites. After this change the
    cleanup-safe sites are: `pre-commit-type-check.sh`, the new
    `scripts/uv-run.sh`, and `Makefile` recipes that go through it. A bare
    `uv run` anywhere else remains forbidden and is caught by a test.
R6. Cleanup must never delete a `.venv` it did not create.
R7. Concurrent sanctioned invocations against the same project root must be
    serialized, so that "this invocation created it" is a claim that can
    actually be true (R6 is unenforceable without this).
R8. An interrupt must not clean up while a surviving `uv` child can still
    recreate `.venv`.

## 4. Rejected alternatives

A. **Make the venv check auto-delete `.venv` and continue.**
   Rejected — explicitly declined by the user. It would hide a `.venv` created
   by *any* tool, and the ban is a deliberate hard stop with an incident behind
   it (R2). This is also exactly the shape of the 2026-04-22 sabotage.

B. **Tell people to `rm -rf .venv` afterwards.**
   Rejected: relies on the human remembering a second step.

C. **Drop the remediation advice entirely.**
   Rejected: iterating on type errors without a full commit is the fast path.

D. **`uv run --no-project` / `uvx`.**
   Rejected: without project dependencies `ty` reports 170+ unresolved-import
   diagnostics and the check becomes meaningless.

E. **Delete the `Makefile` targets.**
   Rejected: they are the documented developer workflow. Out of scope to
   redesign; in scope to stop them leaking (R3 — behaviour unchanged).

## 5. Design

### 5.1 One cleanup-safe seam: `scripts/uv-run.sh`

Add `deep_worker/scripts/uv-run.sh` — a thin wrapper that runs `uv run "$@"` in
the current directory with trap-based cleanup:

- resolves the project root it is invoked from (the directory containing the
  `pyproject.toml` that `uv` would use);
- **pre-existence guard (R6):** before doing anything, if `.venv` already
  exists, fail loudly with a non-zero exit and do **not** delete it. It was not
  created by us and the ban says a human must look at it. Ownership is recorded
  in a variable; the trap only removes a `.venv` this invocation created;
- `trap cleanup EXIT INT TERM` — fires on success, failure and interrupt (R4);
- exit code is the wrapped command's exit code, preserved across the trap (R3);
- works for any `uv run` payload — `ty`, `pytest`, `ruff`, `python` — so it
  covers every `Makefile` recipe, not just type checks.

#### 5.1.1 Serialization (R7)

The ownership claim in R6 is only defensible if two sanctioned invocations
cannot interleave. Without a lock, two wrappers can both observe "no `.venv`",
both run, and the first one's trap deletes the environment out from under the
second — which is also a correctness bug for the second command, not just a
cleanup bug.

The wrapper therefore takes an **exclusive `flock` on a lock file keyed by the
resolved project root**, held across the whole critical section:
pre-existence check → `uv run` → cleanup → release. `flock` blocks in the
kernel until the holder releases; it is event-driven, with no polling, and the
lock is released automatically if the process dies (including `SIGKILL`), so a
crashed wrapper cannot wedge the next one.

The lock file lives outside the project tree, so it never appears in the working
tree that the `.venv` ban scans. Its location must be safe against a hostile or
merely pre-existing path — a predictable hash under a shared `/tmp` is a classic
symlink/pre-creation target, and mode 0600 applied *after* the fact does not
help if someone else already owns the inode.

Lock directory selection, in order:

1. `${XDG_RUNTIME_DIR}` — used **only if** it is an absolute path, an existing
   directory, owned by the effective UID, and not group- or world-writable.
   (This is the normal case: systemd gives each user a private `0700` runtime
   dir.)
2. Otherwise a per-UID private directory under `${TMPDIR:-/tmp}`, named with the
   effective UID, created with mode `0700`.

In both cases the wrapper then creates its own `deep-worker-uv-locks`
subdirectory with mode `0700`.

**Every path component the wrapper creates or opens is validated, not just the
final one.** `TMPDIR` is itself attacker-controllable, and the per-UID component
under it is predictable, so it can pre-exist as a symlink, be foreign-owned, or
be permissive — validating only `deep-worker-uv-locks` while blindly trusting
its parent leaves the same hole one level up. Concretely, for **both** the
per-UID parent and the `deep-worker-uv-locks` subdirectory:

- create/open without following symlinks (`mkdir` is fail-if-exists, and the
  subsequent check is done on the path itself, not through a symlink);
- it must be a real directory, not a symlink or any other file type;
- it must be owned by the effective UID;
- it must have no group or world permission bits.

If any check fails at any level the wrapper aborts with a clear message naming
the offending path, rather than repairing it, `chmod`-ing it, or silently
falling back — a foreign-owned or permissive lock path means something is wrong
that a human should see. The `${XDG_RUNTIME_DIR}` branch is validated by the
same rules before it is accepted (step 1); failing those checks means falling
back to branch 2, not aborting, since a missing or unusable `XDG_RUNTIME_DIR` is
ordinary rather than suspicious.

The lock **filename** is the full **SHA-256** digest of the resolved project
root path. A truncated or weak digest would let two distinct roots collide onto
one lock file, which silently serializes unrelated projects against each other —
a correctness and throughput bug that would be very hard to diagnose. The full
digest is used, not a prefix.

The lock file itself is opened/created without following symlinks (`O_NOFOLLOW`
semantics) with mode `0600`. If the path exists and is not a regular file owned
by the effective UID, the wrapper aborts. A persistent empty lock file across
runs is fine and expected — the lock itself is kernel state tied to the open
file description, not to the file's contents, so nothing needs to be cleaned up
and a stale file cannot grant or withhold the lock.

#### 5.1.2 Signal and child-termination semantics (R8)

A naive `trap cleanup INT TERM` is not enough: `uv` is a child process, and on
`SIGINT` to the process group both the wrapper and `uv` receive the signal. If
the wrapper's trap wins the race and removes `.venv` while a surviving `uv`
child is still resolving dependencies, `uv` recreates `.venv` *after* cleanup
and the leak reappears — the exact failure this design is meant to remove.

Tracking only the immediate `uv` PID is **also** insufficient. `uv run` launches
descendants (`ty`, `pytest`, a resolver subprocess…). `TERM`→`KILL` of the `uv`
process alone does not reap those: a grandchild that ignores `TERM`, or is
simply slower to die, is reparented and can create `.venv` after the wrapper has
cleaned up and exited. Signalling one PID cannot express "and everything it
started".

Therefore the wrapper:

**What is *not* available.** There is no readiness event for "every member of a
process group has exited". `wait`, `waitpid` and pidfds observe the *direct
child* (the group leader) only; a leader can exit while a grandchild it spawned
is still running and has been reparented. Waiting on the leader and treating its
exit as proof the group is empty is exactly the bug. A true group-lifetime
primitive would require a cgroup scope (`systemd-run --scope`, `cgroup.events`),
which is a heavier dependency than this wrapper should take on and is not
available in every environment it must run in (containers, CI, non-systemd).

So the wrapper does **not** claim to wait for group disappearance. It uses a
protocol that is implementable in Bash and does not depend on knowing when the
group is empty:

- runs the wrapped command in **its own process group** (a new session/PGID),
  distinct from the wrapper's own group, and records that PGID. The distinct
  group is what makes "everything this invocation started" addressable as a
  unit;
- on `INT`/`TERM`, forwards the original signal to the **whole child PGID**, not
  to a single PID — including when the wrapper itself was signalled via its own
  group, since the child group does not receive that signal;
- then waits, without polling, for **either** the direct child to exit **or** a
  single bounded grace timer to fire — whichever comes first. This wait is only
  a courtesy window to let the command shut down cleanly; its outcome is not
  used as evidence about the rest of the group;
- then **unconditionally** sends `KILL` to the entire child PGID, regardless of
  whether the direct child already exited. `ESRCH` ("no such process") is a
  success, not an error — it just means nothing was left. This unconditional
  group kill is the load-bearing step: it is what actually guarantees no
  grandchild survives, since leader exit proves nothing;
- reaps the direct child, and only then removes `.venv`. Ordering is the whole
  point: nothing that could recreate `.venv` is still alive at cleanup time;
- re-raises the original signal after cleanup so the exit status is the
  conventional `128+signo` rather than a synthetic code (R3);
- the cleanup itself is idempotent, since `EXIT` fires after the signal traps.

### 5.2 `scripts/type-check.sh` — the human-facing entry point

Add `deep_worker/scripts/type-check.sh`, implemented on top of `uv-run.sh`:

- runs both packages by default;
- accepts an optional package argument (`deep_worker_cli` | `deepagents`);
- **aggregates failures like the hook does**: if the first package fails, the
  second still runs, and the final exit code is non-zero (`set -e` must not
  short-circuit this);
- rejects an unknown package argument with a non-zero exit and a clear message;
- same "not a deep_worker repository" guard as the hook.

### 5.3 Route `Makefile` recipes through the seam

In both `Makefile`s, replace bare `uv run …` with `$(UV_RUN) …` where

```make
UV_RUN := $(abspath $(dir $(lastword $(MAKEFILE_LIST)))/../scripts/uv-run.sh)
```

Wrapper discovery is thus **Makefile-relative and absolute**: it does not shell
out to `git` (no dependency on a git checkout, no subprocess per `make`), and it
does not depend on the caller's cwd. `:=` (simple-immediate) is deliberate —
`MAKEFILE_LIST` grows as other makefiles are included, so a recursive `=` would
resolve to whatever was last included at use time. The package recipes still
assume the package directory as cwd for their *payloads*, which is unchanged;
only wrapper *discovery* is made robust.

Recipe arguments, targets, help text and semantics are otherwise unchanged
(R3). `run: uvx --no-cache --reinstall .` is left alone — `uvx` runs from a
cache and does not create a project `.venv`; a test pins that assumption.

### 5.4 Point human-facing instructions at the wrapper

- `pre-commit-type-check.sh` remediation output (165/168) → suggest
  `deep_worker/scripts/type-check.sh [package]`.
- `deep_worker/scripts/README.md:41` → same.

Not touched: `deep_worker_cli/WORKING_DIR_FIX.md` — a historical transcript of
what was run at the time, not current instructions. Rewriting it would falsify
a record. It is listed in the static allow-list (§6, test 10) so the omission is
a recorded decision rather than a hole.

### 5.5 The hook keeps its own cleanup

`pre-commit-type-check.sh` is **not** rerouted through `uv-run.sh`. Per the
reviewer: a sourced/exec'd helper would add a new runtime dependency to the
commit gate, and a broken helper would then break every commit. The duplication
is deliberate and is pinned by test 12 so the two copies cannot drift.

### 5.6 Cleanup scope

Exactly the project-local `.venv` of the invoking project root. No global
sweep: deleting directories it did not create is how a cleanup script becomes a
footgun, and the repo-wide ban already catches anything else.

## 6. Test plan (TDD, RED first)

Shell-level tests asserting on the filesystem, where the defect manifests.
`uv` is stubbed via a `PATH` seam where the test needs a deterministic
pass/fail/interrupt, and exercised for real where the point is that a real
`.venv` appears and disappears.

RED reproducers (must fail before the change):
1. `uv-run.sh` and `type-check.sh` exist and are executable.
2. `make type` / `make test` in each package leave no `.venv` (currently they
   do — this is the reviewer's BLOCKER, and it must go RED first).

Wrapper behaviour:
3. No `.venv` when the wrapped command SUCCEEDS.
4. No `.venv` when the wrapped command FAILS (non-zero exit AND no `.venv`).
5. No `.venv` when INTERRUPTED (R8): signal the wrapper's **process group**
   mid-run, then assert (a) no `.venv` remains, (b) the **recorded** child PID
   and PGID are dead after the unconditional group `KILL` — checked by identity
   (the stub writes its PID/PGID to a file, and the test verifies those exact
   processes no longer exist), *not* by waiting on the group or on a pidfd,
   since per §5.1.2 neither can prove group disappearance — and (c) the exit
   status is `128+signo`. A stub `uv` that sleeps and then creates `.venv`
   *after* a delay makes the clean-before-child-dies race deterministic: with a
   naive trap this test goes RED.
5a. **Grandchild containment (R8):** the stub `uv` forks a grandchild that
   **ignores `TERM`** and creates `.venv` late, and then the stub (the group
   **leader**) **exits immediately**. The early leader exit is the point: it
   satisfies any "wait for the direct child" step instantly, so the wrapper
   proceeds while the grandchild is still alive. Assert no `.venv` remains and
   no descendant survives — again asserted by identity against the grandchild's
   recorded PID, not by any group-wide wait.
   This test is specifically constructed to make the **unconditional group
   `KILL`** load-bearing: an implementation that treats leader exit as proof the
   group is empty, or that only escalates to `KILL` when the grace timer
   expires, goes RED here while still passing test 5.
6. Exit code is the wrapped command's exit code, not the cleanup's.
7. **Pre-existing `.venv` is preserved (R6):** plant a sentinel file inside a
   pre-existing `.venv`, invoke the wrapper, assert non-zero exit AND the
   sentinel is byte-identical afterwards.
7a. **Concurrency (R7):** launch two wrappers against the same project root at
   once, with a stub `uv` that records enter/exit timestamps to a shared log.
   Assert the second does not enter `uv` until the first has cleaned up and
   released the lock (non-overlapping intervals), both processes report their
   own correct rc, and no `.venv` remains at the end. Without the lock the
   intervals overlap and this goes RED.
7b. **Hostile lock path (§5.1.1):** with `XDG_RUNTIME_DIR`/`TMPDIR` pointed at a
   scratch area, assert the wrapper refuses to proceed (non-zero, clear message,
   no `uv` invoked) for a hostile condition at **each** path level, not only the
   deepest one:
   - **per-UID parent** under `${TMPDIR:-/tmp}` pre-created as (a) a symlink,
     (b) a group/world-writable directory, (c) a non-directory file;
   - **`deep-worker-uv-locks` subdirectory** pre-created as (a) a symlink,
     (b) a group/world-writable directory;
   - **lock file** path pre-created as a symlink pointing elsewhere — and assert
     the symlink target is neither written to nor truncated.
   Validating only the deepest component passes the subdirectory cases but goes
   RED on the per-UID parent cases.
   Also assert `XDG_RUNTIME_DIR` is *ignored* (fallback to the per-UID `0700`
   directory, not an abort) when it is relative, missing, or not owned by the
   effective UID.
7c. **Lock identity (§5.1.1):** two distinct resolved project roots map to two
   distinct lock files, and the same root maps to the same lock file across
   invocations. Concretely: two wrappers against *different* roots run
   concurrently with overlapping intervals (they must **not** serialize against
   each other), while two against the same root serialize (test 7a). Guards
   against a truncated/weak digest collapsing unrelated projects onto one lock.

`type-check.sh` behaviour:
8. First package fails → second package still runs → final rc non-zero
   (stubbed `uv` seam, asserted via per-package marker files).
9. Single-package argument checks only that package; unknown argument exits
   non-zero with a clear message.

Anti-recurrence (stops the defect class, not just this instance):
10. Scan **all tracked executable/human-facing surfaces** — `Makefile`, `*.sh`,
    README/instruction Markdown — for bare `uv run` / `uv sync` / `uv venv`.
    Allow-listing is **occurrence-level, not path-level**: the fixture is a
    table of `(path, normalized line, expected count)`, and the assertion is
    equality against the whole set of found occurrences. A path-level list would
    let a future bare command added *inside* an allowed file — especially
    `pre-commit-type-check.sh` or an internal service — pass silently; occurrence-level
    means any additional occurrence, even in an allowed file, goes RED, and a
    removed one goes RED too (so the list cannot rot).
    Occurrences are classified before comparison: **executable invocations**
    (a command in a recipe or script body) vs **quoted prose** (prohibition text
    in `.githooks/pre-commit` and an internal service's prompts, the reproduction transcript in
    this design doc, the historical `WORKING_DIR_FIX.md`). The sanctioned
    executable occurrences are exactly: the two `uv run --all-groups ty check`
    lines in `pre-commit-type-check.sh`, and the single `uv run "$@"` in
    `scripts/uv-run.sh`. Run against revision-1's state, this test flags the
    Makefiles immediately.
11. Remediation output of `pre-commit-type-check.sh` and `scripts/README.md`
    reference the wrapper and contain no bare `uv run` (runtime assertion,
    separate from the static scan).
12. Trap and cleanup scope are pinned **independently per script**, because the
    two scopes are deliberately different and asserting they are "the same"
    would be false:
    - `pre-commit-type-check.sh`: has `trap cleanup_venvs EXIT INT TERM`, and
      its cleanup covers exactly the two fixed package paths
      (`deep_worker_cli/.venv`, `deepagents/.venv`), since the hook always
      checks both packages regardless of cwd.
    - `uv-run.sh`: has its trap on `EXIT INT TERM`, and its cleanup covers
      exactly one path — the `.venv` of the single resolved project root it was
      invoked for (§5.6).
    Each assertion names its own documented scope; neither is derived from the
    other.

Manual verification:
13. Real `make type`, real `make test`, real failing type check and a real
    commit end to end; `find . -type d -name .venv` empty after each.

## 7. Out of scope

- The repo-wide `.venv` ban and its enforcement (unchanged by design, R2).
- Redesigning or removing `Makefile` targets (E) — only their leak is fixed.
- `__pycache__` permissions (`deepagents/middleware/__pycache__` is mode 0777
  on this host). Pre-existing and unrelated; reported separately.
- Which packages are type-checked, and the `ty` configuration.

## 8. Risk

Low, but higher than revision 1: `Makefile` recipes are now edited, so a wrong
`UV_RUN` path would break `make test` for developers. Tests 2 and 13 exercise
the real targets rather than only asserting on text. No production code, no
hook logic, no change to what blocks a commit.
