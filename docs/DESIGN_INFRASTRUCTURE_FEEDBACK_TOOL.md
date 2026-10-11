# Design: universal infrastructure feedback tool

## Goal

Every deep-worker agent, independent of model provider and including its
TaskManager, receives a default tool named `report_infrastructure_feedback`.
Agents must use it when infrastructure behaves incorrectly, hangs, returns no
result, is unnecessarily slow, or otherwise impedes work. They may also use it
to propose improvements to the infrastructure used by agents.

Reports are stored locally for later human review. This feature does not
cluster reports, create GitHub issues, start fixers, open pull requests, or
perform any background processing.

## Tool contract

The factory binds the reporting agent's identity; the model cannot forge it.
Arguments:

- `kind`: `problem` or `improvement`;
- `component`: short infrastructure component/tool name;
- `summary`: concise symptom or proposal;
- `details`: concrete observations, including timings, operations and errors;
- `expected_behavior`: optional expected behavior.

All strings have bounded lengths. The description explicitly tells agents:

- they **must** call the tool after encountering suboptimal infrastructure;
- they **may** submit improvement wishes;
- it is only a report and does not request or authorize a code change.
- reports must never contain credentials, API keys, tokens, passwords, raw
  environment dumps, or full logs. Agents include only the minimal diagnostic
  excerpt needed to understand the problem.

Success returns a generated report id. Reporting is explicit, never automatic:
the tool does not inspect conversations or logs.

Before persistence, every user-controlled string passes through deterministic,
bounded sanitization. Known credential shapes (authorization/bearer headers,
common API-key/token/password assignments, JWTs and private-key blocks) are
replaced with `[REDACTED]`; output is length-bounded again after redaction.
Sanitization is defense in depth, not permission to submit secrets.

## Storage

Host-global append-only JSONL:

```
~/.deep_worker/infrastructure_feedback/reports.jsonl
```

The root is obtained from `deep_worker_home()`; no host path is hard-coded.
A report contains schema version, UUID, UTC timestamp, bound agent name, kind
and supplied fields.

Storage is fail-closed:

1. Traverse from the trusted user home descriptor-relative. Open each existing
   directory component with `dir_fd`/`openat` plus `O_DIRECTORY` and
   `O_NOFOLLOW` where available, then validate its held descriptor with
   `fstat`: directory, correct effective uid and safe mode. Never resume
   security-sensitive traversal from a pathname after validation. Without
   `O_NOFOLLOW`, every component uses descriptor-relative `lstat` before
   `open`; symlinks and non-directories are rejected, and the pre-open
   `st_dev/st_ino` must equal the opened descriptor's `fstat` values.
2. Create the dedicated directory relative to its held parent descriptor with
   `0700`, open it descriptor-relative, validate owner/type/mode, and retain
   that directory fd until the report file is closed. Because only the owner
   can modify this held `0700` directory, its entries cannot be swapped by
   another uid between validation and file open. Unsafe existing directories
   are rejected, not repaired.
3. Initial file creation is a distinct branch: relative to the held dedicated
   directory fd, call `open(..., O_APPEND|O_CREAT|O_EXCL|O_WRONLY, 0600,
   dir_fd=...)`, adding `O_NOFOLLOW` when available. Perform no write until
   the returned fd passes all checks.
4. If creation returns `EEXIST`, enter the existing-file branch: relative
   `lstat` rejects symlinks/non-regular files; descriptor-relative `open`
   uses `O_APPEND|O_WRONLY` and `O_NOFOLLOW` where available; `fstat`
   must match the pre-open `st_dev/st_ino`.
5. In both branches, `fstat` must report a regular file owned by the current
   effective uid with link count one. Apply and verify `fchmod(0600)` before
   writing. Platforms lacking `O_NOFOLLOW` remain safe because creation uses
   `O_EXCL`, while existing files use relative `lstat` plus inode comparison
   inside the held owner-only directory.

Any unsafe path or failed check closes the descriptor and returns a structured
error. Each sanitized, bounded report is serialized to one JSON line and
appended with one `os.write`. There is no polling, watcher, background task,
new thread, network access, issue creation, or PR creation.

## Universal wiring

The feedback tool is a common/default agent capability, not a web capability.
Introduce a dedicated `assemble_common_tools(agent_name)` helper and combine
its result with optional web tools. Disabling web search or every web policy
must not remove feedback reporting.

- Main agent: add `assemble_common_tools(self.name)` independently of the
  `web_tools` collection and before provider-specific additions.
- TaskManager: add `assemble_common_tools(f"{agent_name}:tm")` independently
  of `assemble_manager_web_tools`.
- Subagents inherit the main bound tool list through `create_deep_agent`.
- Provider-specific native/proxy layers remain unchanged. The common
  `create_cli_agent` funnel binds the tool for every provider.

## Tests

TDD unit tests:

1. schema rejects empty/oversized fields and invalid kinds;
2. stored record contains bound identity, UUID and UTC timestamp;
3. directory/file modes are `0700`/`0600`;
4. two independent tool instances append two intact JSONL records;
5. parameterize the complete registered provider set and assert the production
   main-agent assembly contains the tool for every provider;
6. repeat the complete provider matrix for TaskManager, with identity
   `<agent>:tm`;
7. assert general-purpose and configured subagents inherit the common tool for
   every provider through the production `create_cli_agent` funnel;
8. disable web search/web capabilities and assert the feedback tool remains;
9. unsafe directory, symlink, non-regular target, wrong owner and inode-swap
   cases fail closed without writing;
10. with `O_NOFOLLOW` disabled, every existing directory component is covered
   by relative `lstat`/open/`fstat` inode comparison tests, including a
   simulated component swap;
11. known credential patterns are redacted and bounded before persistence;
12. the tool description contains the mandatory problem-reporting and optional
   improvement-reporting policy.

Integration test without mocks:

- monkeypatch the live `deep_worker_cli.storage_paths.deep_worker_home`
  function to an isolated root (changing `HOME` alone is insufficient);
  construct the real tool and invoke it through LangChain's
  `BaseTool.ainvoke`; read the real JSONL, validate redaction, contents and
  permissions; invoke a second real instance and verify both records remain
  intact. No model/API/network dependency is needed.

## Non-goals

- clustering, ranking, deduplication or UI;
- automatic log capture;
- automatic GitHub issues, fixes, branches, commits or pull requests;
- polling, new threads, or a daemon/service.
