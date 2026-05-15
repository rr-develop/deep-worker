# Audit: what is blocked in claude_code_provider's `BUILT_IN_TOOLS` and what is not

**Date:** 2026-05-14
**Sources:**

* Official reference: <https://docs.claude.com/en/docs/claude-code/tools-reference>
* Changelog: <https://github.com/anthropics/claude-code/blob/main/CHANGELOG.md>
* Release dates: npm registry for `@anthropic-ai/claude-code`
* Installed local version: `2.1.140` (claude.exe, 2026-05-12) — this is the CLI version our agents currently run through.

**Context:** `claude_code_provider/client_manager.py::BUILT_IN_TOOLS` is the list of tool names we pass via `ClaudeAgentOptions.disallowed_tools` to the `claude` subprocess. The goal is that all tool calls go through the LangGraph ToolNode rather than the CLI's built-in implementations.

---

## 1. Currently in `BUILT_IN_TOOLS` (25 names)

| # | Tool | In the official reference? | Purpose | First appeared |
|---|---|---|---|---|
| 1 | `Read` | ✅ | Read a file | ≤ 0.2.18 (2025-02-26) — since the public release |
| 2 | `Write` | ✅ | Create/overwrite a file | ≤ 0.2.18 |
| 3 | `Edit` | ✅ | Targeted edit of an existing file | ≤ 0.2.18 |
| 4 | `MultiEdit` | ⚠️ Stale | Was the tool for a series of edits in one file. No longer in the reference — replaced by `Edit` with `replace_all` and multi-pass | Old. Existed in 1.x, disappeared ≈ 2.0 |
| 5 | `Bash` | ✅ | Run shell commands | ≤ 0.2.18 |
| 6 | `Glob` | ✅ | Find files by pattern | ≤ 0.2.18 |
| 7 | `Grep` | ✅ | Search file contents | ≤ 0.2.18 |
| 8 | `Task` | ⚠️ Stale | Was the tool for launching a subagent. It is now officially called **`Agent`**. The name `Task` survives as an alias in the binary's string table, but strictly speaking it is outdated | Old |
| 9 | `TodoWrite` | ✅ | Works with the session checklist. **Marked deprecated** — replaced by `TaskCreate/Get/List/Update`, but still the default for `claude -p` and the SDK | ≤ 0.2.18 |
| 10 | `Skill` | ✅ | Run a skill (markdown workflow) | 1.0.83 (2025-08-09) |
| 11 | `NotebookEdit` | ✅ | Edit a Jupyter notebook cell | 1.0.x |
| 12 | `WebFetch` | ✅ | Fetch a URL + extraction via a mini-model | ≤ 0.2.18 |
| 13 | `WebSearch` | ✅ | Search in Anthropic's web-search backend | 1.0.x |
| 14 | `BashOutput` | ⚠️ Stale | Was the tool for polling background bash output. Superseded in **2.0.64** (2025-12-10) by the unified `TaskOutput`, which itself is now deprecated in favor of `Read` | 1.0.x → removed in 2.0.64 |
| 15 | `KillShell` | ⚠️ Stale | Was used to kill a background bash task. The current official name is `TaskStop` | 1.0.x → renamed |
| 16 | `AskUserQuestion` | ✅ | Multiple-choice questions to the user | 1.0.x |
| 17 | `SlashCommand` | ⚠️ Stale | The name `SlashCommand` is not in the current table. Slash commands are now handled as "bundled skills", not via a separate tool. See `Skill` | old |
| 18 | `LS` | ⚠️ Stale | Old tool for listing directories. Removed — the model now uses `Bash` (`ls`) | old |
| 19 | `ToolSearch` | ✅ | Beta tool for deferred schema loading (`tool_search_tool_bm25_20251119`) — **intentionally** in our list, see the previous audit | 2.1.119 (2026-04-23) |
| 20 | `CronCreate` | ✅ | Schedules a prompt on a cron schedule. Blocked — see the previous commit | 2.1.72 (2026-03-09) |
| 21 | `CronDelete` | ✅ | Deletes a cron job | 2.1.72 |
| 22 | `CronList` | ✅ | Lists cron jobs | 2.1.72 |
| 23 | `Monitor` | ✅ | Background watch script, streams output lines mid-conversation | 2.1.98 (≈ 2026-04-07) |
| 24 | `PushNotification` | ✅ | Desktop/push alert when a long-running task finishes | 2.1.x |
| 25 | `RemoteTrigger` | ✅ | Claude.ai Routines (async callback) | 2.1.x |

**Stale entries** (`MultiEdit`, `Task`, `BashOutput`, `KillShell`, `SlashCommand`, `LS`) are technical debt. They do not affect functionality (the CLI ignores unknown names in `disallowedTools`), but they clutter the list. Candidates for a separate cleanup commit.

---

## 2. **NOT** in `BUILT_IN_TOOLS` (19 official names missing from our list)

Each name was cross-checked against the official reference. Divided into four groups by behavior.

### 2.1. Group A — should be ADDED (replace outdated names)

They replace the stale entries in our list and block the same class of behavior. Without them, the exact problem we blocked the corresponding old tool for would return.

| Tool | Purpose | First appeared | Why it must be blocked |
|---|---|---|---|
| **`Agent`** | Launch a subagent (via the Agent tool / Task tool — they are the same thing). | ≤ 0.2.18 (old name `Task`); renamed to `Agent` in the reference. The change is mentioned in **1.0.64** (2025-07-30). | `Task` (the stale alias) is already blocked. But **`Agent`** is the current name of the same tool in the newest CLI versions. Without it, subagents would bypass LangGraph. **Critical**. |
| **`TaskStop`** | Kills a background bash task by ID. | 2.1.30 (2026-02-03; previously `KillShell`). | `KillShell` (stale) is blocked. Without `TaskStop`, the model in new CLIs would kill a background task bypassing LangGraph. |
| **`TaskOutput`** | (Deprecated) Gets the output of a background task. The modern path is `Read` on the output file. | 2.0.64 (2025-12-10; previously `BashOutput`). | `BashOutput` is blocked. Anthropic itself deprecated it, but the name still works; we block it for symmetry. |

### 2.2. Group B — should be ADDED (relatives of already-blocked async tools)

I blocked `CronCreate/Delete/List`, `Monitor`, `PushNotification`, and `RemoteTrigger` in a previous commit, but did not consider the task-checklist tools. Some of them are **synchronous** and safe, but one — `TaskCreate` — can indirectly create "deferred" state. Details below.

| Tool | Purpose | First appeared | Decision |
|---|---|---|---|
| **`TaskCreate`** | Adds an item to the session task list. | 2.1.20 (2026-01-27) | The tool itself is **synchronous** (it just adds an entry to the checklist). But `TaskCreate` is also used by the CLI to **create background tasks** through the same call with a flag. See below. **Decision: block** for safety — we want checklists to go through `TodoWrite` (which is already blocked for us → so the model is forced to write the plan as text). The reference itself recommends using the new `Task*` tools instead of `TodoWrite`, but for us that adds no functionality. |
| **`TaskGet`** | Gets task details from the checklist. | 2.1.20 (with TaskCreate) | Pure read, synchronous. Could be left **unblocked** — but for symmetry with `TaskCreate` it is better to block. |
| **`TaskList`** | Lists checklist tasks. | 2.1.20 | Pure read. Similarly — for symmetry. |
| **`TaskUpdate`** | Updates a task's status/details in the checklist. | 2.1.20 | Synchronous. For symmetry. |

**Important clarification:** all four `Task*` tools (Create/Get/List/Update) are a **new family** of tools that replaced `TodoWrite`. They are **fully synchronous**: just reading/writing a JSON file in the session directory. They do **not** create a "ready while waiting" state by themselves. But they **bypass** our `TodoMiddleware` (if any), so blocking them is worthwhile for consistency — the agent's checklist should go through LangGraph state, not CLI-local JSON.

### 2.3. Group C — should be ADDED (can cause `ready while waiting`)

| Tool | Purpose | First appeared | Symptom |
|---|---|---|---|
| **`SendMessage`** | Sends a message to a teammate in an agent team. After sending, the session parks in idle until the teammate replies. | 2.1.32 (2026-02-05; agent teams) | Works **only** if `CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS=1` is set. We do not set it. But we still have no protection — if someone sets the env var, the tool activates and breaks the LangGraph loop. **Block for safety**. |
| **`TeamCreate`** | Creates an agent team. | 2.1.32 (2026-02-05) | Same — requires EXPERIMENTAL_AGENT_TEAMS. Creating a team is async coordination with other `claude` processes. Incompatible with the single-process LangGraph loop. |
| **`TeamDelete`** | Deletes an agent team. | 2.1.32 (with TeamCreate) | Synchronous by itself, but only makes sense paired with TeamCreate. Block for symmetry. |

### 2.4. Group D — no need to block (safe synchronous)

| Tool | Purpose | First appeared | Why keep it |
|---|---|---|---|
| `EnterPlanMode` | Switches permission mode to "plan mode" — the model only reads and proposes a plan. | 1.0.33 (2025-06-23) | Synchronous permissionMode switch. No background effects. Useful for an agent that wants to "plan first". Does not break LangGraph. |
| `ExitPlanMode` | Exits plan mode with an approved plan. | 1.0.33 | Same. |
| `EnterWorktree` | Creates a git worktree and switches CWD into it. | 2.1.72 (2026-03-09) | Synchronous. Useful. **Note:** it changes the CLI session's CWD, but this does not break LangGraph — our working_dir is determined in Python (`AgentConfig`). Re-check that our FS middleware is not lost after EnterWorktree. If it is lost — block it, but that is a separate problem. For now the standard: leave it allowed. |
| `ExitWorktree` | Exits a worktree session. | 2.1.72 | Paired with EnterWorktree. |
| `LSP` | Code intelligence (go-to-definition, find references, hover docs). Works only if the LSP plugin for the language is installed. | 2.0.74 (2025-12-19) | Synchronous inline call to the language server. No async effects. **Useful** — if the plugin is installed, the agent gets type-aware navigation. Keep. |
| `PowerShell` | Bash alternative for Windows / opt-in on Linux. | 2.1.70 (2026-03-06) | Synchronous, like Bash. **But:** `Bash` is already blocked for us (execution goes through LangGraph). Logically `PowerShell` should also be blocked. **Candidate for group A**. |
| `ListMcpResourcesTool` | Lists MCP resources (not tools, but resources — data exposed by MCP servers). | 1.0.27 (2025-06-17) | Synchronous list read. Safe. We have no MCP resources through ProxyMCPServer, but if we ever do — this tool is useful. Keep. |
| `ReadMcpResourceTool` | Reads a specific MCP resource. | 1.0.27 (with ListMcpResourcesTool) | Same. Safe. |
| `ShareOnboardingGuide` | Loads `ONBOARDING.md` and returns a share link. Part of the `/team-onboarding` skill. | 2.1.101 (2026-04-10) | Synchronous HTTP upload to Anthropic infrastructure. No async effects. Not used by us (Claude Code on the Web feature). Keep. |

**Amendment to the `PowerShell` decision:** on closer inspection this violates our principle ("all command execution goes through LangGraph"). Moving it to group A.

---

## 3. Summary: what is proposed to add

Final delta to `BUILT_IN_TOOLS` after this audit (11 names):

```python
# Group A — modern names for already-blocked tools
"Agent",          # modern name for Task (subagent-spawn)
"TaskStop",       # modern name for KillShell
"TaskOutput",     # modern name for BashOutput
"PowerShell",     # parallel Bash for Windows / opt-in Linux
# Group B — the new task-checklist family (replaces TodoWrite)
"TaskCreate",
"TaskGet",
"TaskList",
"TaskUpdate",
# Group C — agent teams (requires EXPERIMENTAL_AGENT_TEAMS=1, but we defend anyway)
"SendMessage",
"TeamCreate",
"TeamDelete",
```

And separately — cleanup candidates (stale names; can be removed, but safe to keep):

```python
# Can be removed — they do not exist in the modern CLI
"MultiEdit", "Task", "BashOutput", "KillShell", "SlashCommand", "LS"
```

I do **not propose removing the stale names in the same commit as adding new ones** — if someone happens to use an old CLI version (not our case — everyone is on 2.1.140), blocking them still matters. The best option is to keep the stale names as-is for backward compatibility and add the new names. It is still just a list of strings; size is not critical.

---

## 4. What this audit does **not** cover

* **MCP resources** (`ListMcpResourcesTool`/`ReadMcpResourceTool`): we need to decide whether to route them through ProxyMCPServer or leave them as-is. The current ProxyMCPServer implementation proxies only **tools**, not resources. If MCP resources ever appear in LangGraph, these two tools may be needed or, on the contrary, get in the way. For now — leave them.
* **Plugin-system tools** — if a Claude Code plugin installs its own tool via MCP, the name will be different (`mcp__<plugin>__<tool>`). They are not covered by our list, but they are also not needed — only our own `deepworker` ProxyMCP server runs.
* **`SkillSlashCommand`** (slash commands as part of a skill) — not a separate tool in the modern reference, but part of the `Skill` tool. Already blocked via `Skill`.
* **Harness-level tools** (`ScheduleWakeup`, `EnterPlanMode/ExitPlanMode/ExitWorktree` for Claude Code on the Web) — exposed only by the claude.ai runtime for remote agents (for example, the agent writing this document). They will **not** reach our subprocess via the SDK. No need to check or block them.

---

## 5. Implementation

**Implemented:** 2026-05-14; see the commit in the dev branch's git log.

* Added the 11 names of groups A/B/C to `claude_code_provider/client_manager.py::BUILT_IN_TOOLS` (see § 3).
* Added a parametrized test `test_audit_new_tool_disallowed` to `claude_code_provider/tests/test_tool_search_disallowed.py` that checks each of the 11 names is present in `BUILT_IN_TOOLS`.
* The stale entries (`MultiEdit`, `Task`, `BashOutput`, `KillShell`, `SlashCommand`, `LS`) are **kept** for backward compatibility with older CLIs: the CLI ignores unknown names in `--disallowedTools`, so there is no penalty.
* All 131 unit tests in `claude_code_provider/` are green.
* The integration test `test_tool_schemas_loaded_at_start.py` (a real dev-tree agent via claude-code) confirms that tool schemas remain loaded after the fix.

Further tasks are **not** part of this PR and are described in § 4 ("What this audit does not cover"):

* The behavior of `EnterWorktree` with respect to `FilesystemMiddleware.working_dir` (cwd switch inside the CLI). Currently remains allowed.
* MCP-resources tools (`ListMcpResourcesTool`, `ReadMcpResourceTool`) — left allowed until ProxyMCPServer starts handling the resource level.
