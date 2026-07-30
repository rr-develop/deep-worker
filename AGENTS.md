# Fix service — working environment for fixing agents

Agents of this service fix discovered problems/bugs in the code of the agent
platform and its web server.

**Working branch:**

- **User request (any fix/task on a direct request) → branch `dev`**
  in a dedicated worktree, then fast-forward into `master`.
- Branch `fixes` is intended **exclusively** for the automatic mode of the
  fix service (tasks from GitHub issues). It is **never** used for user
  requests — only if the user explicitly and unambiguously asked to work on it.

Each agent knows its role from its system prompt (by name). Below is the
general context, roles, and mandatory procedures/policies for this environment.

## Roles

- **fix_fixer** — the fix implementer. In automatic mode it receives the task
  "fix GitHub issue #N", works through the `add_feature` procedure (below),
  and commits to branch `fixes`. One issue at a time. On a direct user request
  it works on `dev` (see above).
- **fix_detector** — analyst of ambiguous log content. Decides whether an
  agent message/log fragment is a real problem, estimates priority, and
  cross-checks against existing issues for deduplication. **Does not write
  fix code.**

## Issue completion protocol (fix service automatic mode only)

In automatic mode the fix service determines the outcome by the **trailer in
the git commit message** on branch `fixes`:

- Fixed successfully → put the trailer in the commit:
  ```
  Fixes-Issue: #<N>
  ```
- Cannot fix without violating architecture/policies OR changes in an external
  dependency are required (e.g., a local rpyc source checkout) → do NOT make
  the fix. Commit a note with the trailer and the reason (this is an
  escalation to the user):
  ```
  Escalate-Issue: #<N>
  ```

Without one of these trailers the service will consider the work unfinished.

## What counts as a "feature" (for fix_fixer)

**Feature = fixing a specific issue.** Apply the whole `add_feature` procedure
to it (design if needed, TDD implementation, review, commit).

## Standard procedure

```
add_feature:
    understand the user request that needs to be implemented. Investigate how
    it can be done. Write a design (/design_feature). /implement_design /do_commit.
    Provide a report to the user.

design_feature(feature):
    Write a design for feature in a suitable place in the project. If you have
    open questions that require explicit agreement with the user, ask him/her
    about them, providing the full link to the design document.
    Then send it for review (/send_to_review).

send_to_review(file_or_files, agent_name):
    send file_or_files to agent_name for review, using the inter-agent
    communication tool. When you receive their response, make edits if
    necessary. These edits must not violate the user's requirements, existing
    policies, or change the application architecture without direct and
    explicit agreement with the user. If something had to be changed, send it
    for re-review to the same agent until you reach agreement.

implement_design(file):
    implement the design described in file. Use the TDD approach.
    /do_unit_tests. /send_to_review the implementation code to the agent that
    reviewed the design. If changes had to be made during the review, run
    /do_unit_tests again. Write an integration test or tests that verify this
    functionality in practice (no mocks!!!).

do_unit_tests:
    Run the application's unit tests. If there are errors, find out whether
    they relate to the tests or to the application code. If they are test
    errors, fix the tests. If they are application errors and you understand
    how to fix them without changing the architecture, contracts, etc., fix
    them. Otherwise, or in case of doubt, report this to the user. If you made
    fixes in the code, also report this to the user.

do_commit:
    commit your changes. Check that the working directory is clean (or contains
    only other people's changes). If the pre-commit hook reveals any problems,
    fix them. Bypassing the checks in the pre-commit hook is strictly
    forbidden.
```

## fix_detector: verdict format

Given a log fragment/agent message and a list of existing open issues
(title + signature), return a strictly structured verdict without extra text:
```
{"is_issue": bool, "severity": "low|medium|high|critical",
 "dedup_of": <issue_id | null>, "title": "...", "summary": "..."}
```
Be conservative: a one-off transient log entry with no signs of a real
problem → `is_issue: false`. Always cross-check against existing issues
(dedup).

## Core policies (VIOLATION FORBIDDEN)

- **Polling is strictly forbidden.** Everything must be event-driven.
- Always prefer non-blocking approaches. **Do not add new threads** without
  direct and explicit user approval.
- Set access permissions on created files/folders correctly. **No 777, 666.**
- **Changing the project architecture, contracts, or policies is forbidden**
  without direct and explicit agreement. If the fix requires it — escalate
  (see above).
- **Bypassing the pre-commit hook is forbidden** (`--no-verify` is not
  allowed).
- rpyc only from a local source checkout; no `pip install -e .` / `.venv`;
  only source-imports.
- **All fixes on a direct user request are made on branch `dev`**
  (a dedicated worktree) and then fast-forwarded into `master`. Branch
  `fixes` is only for the fix service's automatic mode; using it for user
  requests is forbidden, except when the user explicitly and unambiguously
  asked for it. Do not use `tasks`/`master` directly without an explicit
  instruction.

## Review

Send the fix design and code for review to the review agent through the
inter-agent messaging tool (`send_message_to_agent`, `reply_back=True`).
