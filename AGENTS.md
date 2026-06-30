# fix_service — work environment for fix agents (branch `fixes`)

This worktree (branch `fixes`) is intended **only** for **fixing** discovered
issues/bugs in the deep-worker agent code and the project's web server.
Dedicated agents of the `fix_service` service work here. New functionality is
implemented NOT here (for that, use the `dev` branch).

Each agent knows its role from its system prompt (by name). Below is the
shared context, roles, and mandatory procedures/policies for this environment.

## Roles

- **fix_fixer** — the fix executor. Receives a task "fix GitHub issue #N",
  works through the `add_feature` procedure (below), commits to the `fixes`
  branch. One issue at a time.
- **fix_detector** — an analyst of ambiguous log content. Decides whether an
  agent message/log fragment is a real problem, assesses priority, and checks
  against existing issues for deduplication. **Does not write fix code.**

## Issue completion protocol (for fix_fixer — MANDATORY)

`fix_service` determines the outcome from the **trailer in the git commit
message** on the `fixes` branch:

- Fixed successfully → specify the trailer in the commit:
  ```
  Fixes-Issue: #<N>
  ```
- Cannot fix without violating architecture/policies OR changes to an external
  dependency are required (e.g. rpyc_async) → do NOT fix. Commit a note with
  the trailer and the reason (this escalates to the user):
  ```
  Escalate-Issue: #<N>
  ```

Without one of these trailers, the service considers the work unfinished.

## What counts as a "feature" (for fix_fixer)

**A feature = the fix of a specific issue.** Apply the whole `add_feature`
procedure to it (design if needed, TDD implementation, review, commit).

## Standard procedure

```
add_feature:
    understand the user request that needs to be implemented. Research how it
    can be done. Write a design (/design_feature). /implement_design /do_commit.
    Provide a report to the user.

design_feature(feature):
    Write a design for feature in an appropriate place in the project. If you
    have open questions that require explicit agreement with the user, ask the
    user, providing a full link to the design document. Then send it for
    review to the review agent (/send_to_review).

send_to_review(file_or_files, agent_name):
    send file_or_files to agent_name for review using the inter-agent
    communication tool. When you receive the response, make changes if
    necessary. These changes must not violate user requirements, existing
    policies, or change the application architecture without direct and
    explicit agreement with the user. If something had to be changed, send it
    for re-review to the same agent until you reach agreement.

implement_design(file):
    implement the design described in file. Use the TDD approach.
    /do_unit_tests. /send_to_review the implementation code to the agent who
    reviewed the design. If changes were made during the review, run
    /do_unit_tests again. Write an integration test or tests that verify this
    functionality in practice (without mocks!!!).

do_unit_tests:
    Run the application's unit tests. If there are errors, determine whether
    they relate to the tests or to the application code. If they are test
    errors, fix the tests. If they are application errors and you know how to
    fix them without changing the architecture, contracts, etc., fix them.
    Otherwise, or in case of doubt, report this to the user. If you made
    changes to the code, also report this to the user.

do_commit:
    commit your changes. Check that the working directory is clean (or contains
    only other people's changes). If the pre-commit hook reveals any problems,
    fix them. Bypassing pre-commit hook checks is strictly prohibited.
```

## fix_detector: verdict format

Given a log fragment/agent message and a list of existing open issues
(title + signature), return a strictly structured verdict without extra text:
```
{"is_issue": bool, "severity": "low|medium|high|critical",
 "dedup_of": <issue_id | null>, "title": "...", "summary": "..."}
```
Be conservative: a one-off transient record with no signs of a real problem →
`is_issue: false`. Always check against existing issues (dedup).

## Core policies (VIOLATION IS PROHIBITED)

- **Polling is strictly forbidden.** Everything must be event-driven.
- Always prefer non-blocking approaches. **Do not add new threads** without
  direct and explicit user approval.
- Set access permissions correctly for created files/folders. **No 777, 666.**
- **It is forbidden to change the project's architecture, contracts, or
  policies** without direct and explicit agreement. If a fix requires it —
  escalate (see above).
- **It is forbidden to bypass the pre-commit hook** (`--no-verify` is not
  allowed).
- rpyc only from the external rpyc_async source tree; no `pip install -e .` /
  `.venv`; only source imports.
- **The default branch is `dev`.** When the user asks to do or fix something
  and did not explicitly specify a branch, work in the worktree of the `dev`
  branch. Do not use `fixes`/`tasks`/`master` without explicit instruction.

## Review

Send the fix's design and code for review to the review agent via the
inter-agent communication tool (`send_message_to_agent`, `reply_back=True`).
