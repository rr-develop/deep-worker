# TASK (verbatim, from the owner)

Saved verbatim so it survives summarization. DO NOT edit the text of the
task — only append status below.

## Statement

```
Right now the project_memory and history_memory tools pull very heavy dependencies into the agents' runtime. Among other things, this violates our policy against doing so: heavy tools should get a separate process serving many agents (that policy appeared later than these tools themselves). I want to build this as a separate service that will pull in the heavy libraries such as PyTorch and the like.
This should be somewhat similar to another project, where a heavy neo4j runs in a separate service process and other agents simply reach it through the corresponding tools, which do not pull in that dependency themselves.
Unlike that project, I want to keep these tools native to deep-worker (not MCP). In other respects they should behave similarly. Moreover, I want to extend them so they can be used with any repository (project_memory) and with any agent (history_memory).
By default, project_memory does not specify a particular project (repository), as it does now, and in this case the tool works with the agent's current project (its working directory). If there is no index there, the agent gets an informative error saying that there is no index in the agent's directory, but that it can use this tool for directories where such an index exists by specifying them explicitly.
Similarly, an agent can use history_memory by default for its own memory (without specifying an agent name) and, by specifying the name of a particular agent, work with that agent's memory.
Security questions are out of scope for this task and its implementation. For now we work in a trusted environment and, if the need arises, we will address them later.
Otherwise, using these tools should look about the same to agents as it does now.
```

## Invariants derived from the statement

1. The tools remain **native** to deep-worker, NOT MCP.
2. The heavy stack (torch / llama-index / LanceDB) is not imported in
   the agent process — neither for search nor for indexing.
3. `project_memory` without an argument = the agent's current working directory;
   with `repo_path` = any repository.
4. `history_memory` without an argument = the agent's own memory;
   with `agent_name` = the memory of any agent.
5. No index → an **informative error** with a hint to specify the target explicitly.
   Do NOT hide the tool (see design §5.0, blocker B1).
6. For the agent, usage looks as it does now: the format of successful output
   does not change.
7. Security is out of scope — trusted environment.

## Procedure (`/add_feature`)

`/check_branches` → design → review → TDD → unit tests →
integration tests without mocks → `/do_commit` → FF into master.
Each commit is a separate sign-off request with sha256, mutation
results, and test runs. Commit only on explicit sign-off.

## Status

- [x] `/check_branches`
- [x] Design — `docs/DESIGN_SEMANTIC_MEMORY_SHARED_SERVICE.md`, commit `c779f30`
- [x] Design sign-off (2 rounds)
- [x] **Slices 1–3** — proto + registry + supervisor, commit `f5603a6`, 37 tests
- [x] **Slice 4** — service: engine + grpc_server + entrypoint, commit `8b934bc`.
      Executor for both Search AND Index, per-tenant lock, fail-loud on
      `INDEX_KIND_UNKNOWN`, idle watchdog, lazy model. Output is byte-identical
      to the in-process tool (verified on a live index).
- [x] **Fix after slice 4** — keepalive via a queue (it violated the polling
      policy), `_ALLOWED_SPAWNERS` + pinning of the allowlist
- [ ] **Slice 5** — thin client
- [ ] **Slice 6** — rewiring the tools (ALWAYS registered, B1)
- [ ] **Slice 7** — integration tests without mocks
- [ ] FF into master
