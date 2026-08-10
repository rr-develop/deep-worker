# DESIGN: the fifth source of job-id — a shell-job does not resolve outside the `CLIShellBackend` instance

**Status:** DESIGN SIGNED OFF + CODE SIGNED OFF (2026-07-31) — shipped as `b2dffcc`  
**Date:** 2026-07-31  
**Branch:** `dev`  
**Related:** `DESIGN_FIX_JOB_ID_NAMESPACE_STALL.md` (sources 1–2),
`DESIGN_FIX_STAGE_A_FALSE_PROMISE.md` (3),
`DESIGN_FIX_SANDBOX_EXECUTE_JOB_ID.md` (4, commit `22e52be`).

## 1. Symptom

After the fix of the 4th source the notice already carries the canonical `job_N`. But:

```
wait_for_job(job_id="job_N") → Error: no background job with id 'job_N'.
```

when `execute` parked a shell command and `wait_for_job`/`kill_job`
resolve a **different** `CLIShellBackend` instance (empty `self._jobs`).

Live reproduction (unit level):

| Backend A | Backend B |
|-----------|-----------|
| `execute("sleep 30", timeout=1)` → `job_1` in `A._jobs` + `ParkedJobRegistry` | `B._jobs == {}` |
| `A.wait_for_job("job_1")` → STILL running | **`B.wait_for_job("job_1")` → no background job** |

## 2. Root cause

`ParkedJobRegistry` already stores the shell-job (`kind="shell"`), but
`CLIShellBackend.wait_for_job` / `kill_job`:

1. look up **only** `self._jobs` / `self._job_aliases`;
2. the `_resolve_parked` fallback accepts **only** `kind == "tool"`;
3. `kind="shell"` in the shared registry is **ignored**.

The registry contract ("what we advertise is what resolves") is broken for shell.
Any divergence between backend instances (factory, subagent,
middleware rebuild, unwrap error) is enough — the handle is lost.

Additionally: `_register_shared(..., handle=self)` stores the **backend**
rather than a `_BackgroundJob`, so even when reading the registry the pipes
cannot be awaited without a hop into `owner._jobs`.

## 3. Solution

1. When parking a shell job: `registry.register(..., kind="shell", handle=job)`
   where `job` is a `_BackgroundJob` with `owner` = the parking `CLIShellBackend` (for cross-instance `_drop_job`).
2. `wait_for_job` / `kill_job`: local miss → `registry.resolve` →
   - `kind=="shell"` + handle=`_BackgroundJob` → wait for / kill the job;
   - `kind=="tool"` → the existing `_wait_parked_tool` / `_kill_parked_tool`.
3. `_drop_job`: cleans owner.`_jobs`, aliases and `registry.drop` — regardless
   of which backend called finalize.

**Unchanged:** the `job_N` namespace, event-driven (no polling), no new
threads, one process-local registry, the proxy-MCP architecture.

## 4. TDD

| # | Test | RED |
|---|------|-----|
| T1 | A.execute timeout → B.wait_for_job(job_N, short) → STILL running, not error | B: no background job |
| T2 | A.execute timeout → B.kill_job(job_N) → killed, process dead | B: no background job |
| T3 | A.execute timeout → A.wait_for_job (same instance) — no regression | — |
| T4 | kind=tool via the registry still resolves (regression 1–4) | — |

## 5. Scope

**In scope:** `deep_worker_cli/backends.py`, unit tests.  
**Out of scope:** removing proxy-MCP parking; sandbox remote backends.


## 6. Implementation (files / symbols)

| File | Change |
|------|-----------|
| `deep_worker_cli/backends.py` | `_BackgroundJob.owner`; `_resolve_shell_from_registry`; `_register_shared(..., job=)`; `_drop_job` via owner |
| `tests/unit_tests/test_shell_job_cross_backend.py` | T1–T4 |
| `docs/DESIGN_FIX_SHELL_JOB_CROSS_BACKEND.md` | this document |

Resolution order in `wait_for_job` / `kill_job`:
1. `_resolve_job` → local `_jobs` / `_job_aliases` → registry `kind=shell` (handle=`_BackgroundJob`)
2. otherwise `_resolve_parked` → `kind=tool` (loci 1–4)
3. otherwise `Error: no background job`


## 7. Sign-off

- **DESIGN SIGNED OFF** + **CODE SIGNED OFF** (2026-07-31)
- Commit: `b2dffcc`
- Independent verification by reviewer: 29 unit tests green (cross-backend + regression suites)
