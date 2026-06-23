# fix_service — working environment for fix agents (branch `fixes`)

This worktree (branch `fixes`) is intended **only** for **fixing** discovered
problems/bugs in the worker agent code and the web server. Dedicated agents of
the `fix_service` service work here. New functionality is NOT implemented here
(the `dev` branch is used for that).

Each agent knows its role from its system prompt (by name). Below is the shared
context, roles, and mandatory procedures/policies for this environment.

## Roles

- **fix_fixer** — the fix implementer. Receives a task to "fix GitHub issue
  #N", follows the `add_feature` procedure (below), and commits to the `fixes`
  branch. One issue at a time.
- **fix_detector** — the analyst of ambiguous log content. Determines whether
  an agent message/log fragment is a real problem, assesses priority, and
  checks against existing issues for deduplication. **Does not write fix code.**

## Issue completion protocol (for fix_fixer — MANDATORY)

`fix_service` determines the outcome from the **trailer in the git commit
message** on the `fixes` branch:

- Successfully fixed → include this trailer in the commit:
  ```
  Fixes-Issue: #<N>
  ```
- If you cannot fix it without violating architecture/policies OR changes to an
  external dependency are required (for example an external rpyc source
  checkout) → do NOT make the fix. Commit a note with the trailer and the
  reason (this is an escalation to the user):
  ```
  Escalate-Issue: #<N>
  ```

Without one of these trailers, the service will consider the work incomplete.

## What counts as a "feature" (for fix_fixer)

**Feature = fixing a specific issue.** Apply the full `add_feature` procedure to
it (design if necessary, TDD implementation, review, commit).

## Standard procedure

```
add_feature:
    understand the user request that needs to be implemented. Research how it
    can be done. Write a design (/design_feature). /implement_design /do_commit.
    Provide a report to the user.

design_feature(feature):
    Write a design for feature in an appropriate place in the project. If you
    have open questions that require explicit approval from the user, ask the
    user, providing them with the full link to the design document.
    Then send it for review to a designated review agent (/send_to_review).

send_to_review(file_or_files, agent_name):
    send file_or_files to agent agent_name for review using the inter-agent
    communication tool. When you receive its response, make changes if
    necessary. These changes must not violate the user's requirements, existing
    policies, or change the application architecture without direct and
    explicit approval from the user. If something had to be changed, send it
    for a second review to the same agent until you reach agreement.

implement_design(file):
    implement the design described in file. Use the TDD approach.
    /do_unit_tests. /send_to_review the implementation code to the agent who
    reviewed the design. If the review required changes, run /do_unit_tests
    again. Write an integration test or tests that verify this functionality in
    practice (no mocks!!!).

do_unit_tests:
    Run the application's unit tests. If there are errors, determine whether
    they relate to the tests or to the application code. If they are test
    errors, fix the tests. If they are application errors and you know how to
    fix them without changing the architecture, contracts, etc., fix them.
    Otherwise, or in case of doubt, report this to the user. If you made
    changes to the code, also report this to the user.

do_commit:
    commit your changes. Make sure the working directory is clean (or contains
    only other people's changes). If the pre-commit hook reveals any problems,
    fix them. Bypassing pre-commit hook checks is strictly forbidden.
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

## Core policies (MUST NOT BE VIOLATED)

- **Polling is strictly forbidden.** Everything must be event-driven.
- Always prefer non-blocking approaches. **Do not add new threads** without
  direct and explicit approval from the user.
- Set correct access permissions for created files/folders. **No 777 or 666.**
- **Changing the project architecture, contracts, or policies is forbidden**
  without direct and explicit approval. If a fix requires it — escalate (see
  above).
- **Bypassing the pre-commit hook is forbidden** (`--no-verify` is not
  allowed).
- rpyc only from an external rpyc source checkout; no `pip install -e .` /
  `.venv`; only source imports.

## Review

Send the fix design and code for review to a designated review agent via the
inter-agent communication tool (`send_message_to_agent`, `reply_back=True`).
