# Investigation: Codex Provider Full History Gaps

Date: 2026-05-16

Scope:

- `codex_provider`
- `deep_worker/deep_worker_cli`
- the web app

User-visible symptom:

When requesting Full history for a Codex-backed deep-worker session in
the web app, the UI shows user prompts, final assistant answers, and
sometimes TaskManager prompts, but not the complete tool-call / tool-result /
intermediate event timeline.

## Executive Finding

The symptom is real. There are two separate facts:

1. Proxy tool calls and proxy tool results can be present in LangGraph
   `state.messages`, and the raw history serializer preserves them.
2. The web app Full Dialog is not a raw LangGraph message timeline. It
   deliberately drops `ToolMessage` payloads and only renders compact
   `AIMessage.tool_calls` summaries.

There is also a Codex-provider-specific gap:

3. Codex CLI JSONL events classified as `GenericItem` (`reasoning`,
   `file_change`, `web_search`, `todo_list`, and other unknown item types) are
   parsed but are not converted into LangChain messages. They therefore do not
   enter LangGraph `state.messages` and cannot appear in raw history.

Under the stated project requirement "all events must pass through the
LangGraph message loop", item 3 is a critical provider bug. Item 2 is a
critical history/UI presentation bug: data that exists in raw history is hidden
by the web Full Dialog formatter.

## Fact 1: The Agent Uses LangGraph Checkpointed Messages As Raw History

`deep_worker_cli.agent.create_agent()` creates the deep agent with a
checkpointer:

- `deep_worker/deep_worker_cli/deep_worker_cli/agent.py:815`
- `deep_worker/deep_worker_cli/deep_worker_cli/agent.py:822`

`AgentProcess._handle_get_history()` retrieves the current LangGraph state via
`agent.aget_state(config)` and reads `state.values["messages"]`:

- `deep_worker/deep_worker_cli/deep_worker_cli/process/agent_process.py:924`
- `deep_worker/deep_worker_cli/deep_worker_cli/process/agent_process.py:936`

It then serializes exactly the selected messages:

- `deep_worker/deep_worker_cli/deep_worker_cli/process/agent_process.py:953`

Therefore, the raw history source is LangGraph checkpoint state, not WebSocket
notifications and not the web app formatter output.

## Fact 2: Raw History Serialization Preserves Proxy Tool Calls And Tool Results

`MessageDictCache.serialize_message()` preserves:

- `AIMessage.content`
- `AIMessage.tool_calls`
- `ToolMessage.content`
- `ToolMessage.tool_call_id`
- `ToolMessage.name`
- `ToolMessage.status`

Evidence:

- `deep_worker/deep_worker_cli/deep_worker_cli/process/message_dict_cache.py:58`
- `deep_worker/deep_worker_cli/deep_worker_cli/process/message_dict_cache.py:76`
- `deep_worker/deep_worker_cli/deep_worker_cli/process/message_dict_cache.py:87`

Local proof command executed during this investigation:

```bash
python3 - <<'PY'
from langchain_core.messages import AIMessage, ToolMessage, HumanMessage
from deep_worker_cli.process.message_dict_cache import serialize_message
msgs = [
    HumanMessage(content='do x', id='h1'),
    AIMessage(content='', id='a1', tool_calls=[{'id':'tc1','name':'execute','args':{'command':'echo hi'}}]),
    ToolMessage(content='hi\n', id='t1', tool_call_id='tc1', name='execute', status='success'),
]
for msg in msgs:
    print(serialize_message(msg))
PY
```

Observed output:

```text
{'type': 'HumanMessage', 'id': 'h1', 'content': 'do x'}
{'type': 'AIMessage', 'id': 'a1', 'content': '', 'tool_calls': [{'id': 'tc1', 'name': 'execute', 'args': {'command': 'echo hi'}}]}
{'type': 'ToolMessage', 'id': 't1', 'content': 'hi\n', 'tool_call_id': 'tc1', 'name': 'execute', 'status': 'success'}
```

Conclusion: if an `AIMessage.tool_calls` or `ToolMessage` is in LangGraph
`state.messages`, raw history preserves it.

## Fact 3: Codex Proxy Tool Execution Is Routed Through LangGraph

The Codex proxy MCP server explicitly does not execute tools. Its handler parks
a Future and waits for `CodexChatModel.ainvoke()` to resolve it after a matching
LangGraph `ToolMessage` appears:

- `codex_provider/proxy_mcp_server.py:3`
- `codex_provider/proxy_mcp_server.py:5`
- `codex_provider/proxy_mcp_server.py:189`
- `codex_provider/proxy_mcp_server.py:201`
- `codex_provider/proxy_mcp_server.py:206`

When Codex starts a proxy MCP call, `CodexChatModel.ainvoke()` waits for the
registered pending call and returns an `AIMessage` with `tool_calls`:

- `codex_provider/chat_model.py:401`
- `codex_provider/chat_model.py:414`
- `codex_provider/chat_model.py:425`
- `codex_provider/chat_model.py:466`

On the next invocation, when LangGraph supplies the `ToolMessage`,
`CodexChatModel.ainvoke()` resolves the parked proxy MCP call:

- `codex_provider/chat_model.py:342`
- `codex_provider/chat_model.py:343`
- `codex_provider/chat_model.py:344`

Existing integration proof:

- `codex_provider/tests/test_integration.py:150`
- `codex_provider/tests/test_integration.py:169`
- `codex_provider/tests/test_integration.py:179`
- `codex_provider/tests/test_integration.py:183`

Focused tests run during this investigation:

```bash
python3 -m pytest \
  codex_provider/tests/test_exec_runner.py::TestParseEvent::test_item_completed_unknown_type_is_passed_through \
  codex_provider/tests/test_chat_model.py::TestAinvokeToolCalls::test_tool_call_breaks_with_pending_calls \
  codex_provider/tests/test_chat_model.py::TestAinvokeToolCalls::test_toolmessage_resolves_future_not_in_prompt \
  -q
```

Observed result:

```text
3 passed
```

Conclusion: proxy tool call and proxy tool result routing is designed to pass
through LangGraph messages.

## Fact 3.1: End-To-End deep-worker Codex History Preserves Proxy Tool Calls

After the initial report, a live integration probe was run with a dedicated
test agent. No production agent received a message.

Test agent:

- `test_codex_history_filetool_1778967768`
- provider: `codex`
- working directory: `/tmp/test_codex_history_filetool_1778967768`
- prompt: read a temporary secret file using `read_file`

The file contained a random value that was not included in the prompt:

```text
DW_CODEX_FILE_SECRET_4cd60393096efc37
```

The agent answered with that value, and offline history showed four messages:

```text
[USER]
Read the file at /tmp/test_codex_history_filetool_1778967768/secret.txt using
the read_file tool. Do not guess and do not use shell. Return only the exact
file contents.

[ASSISTANT]

[TOOL]
     1  DW_CODEX_FILE_SECRET_4cd60393096efc37

[ASSISTANT]
DW_CODEX_FILE_SECRET_4cd60393096efc37
```

The same test's JSON history contained:

```text
tool_calls=[{'name': 'read_file', ...}]
tool_call_id='read_file_1_...'
```

This manual probe was converted into a reproducible pytest test:

- the Codex history round-trip test under `codex_provider/tests/`

The test starts a fresh live `deep-worker --provider codex` agent, creates a
secret file under `tmp_path`, asks the agent to read it with `read_file`, and
asserts that:

- the response contains the secret;
- `deep-worker --history <agent>` contains `[TOOL]`;
- text history contains `read_file`;
- JSON history contains `tool_calls`;
- JSON history contains `tool_call_id`.

Focused run:

```bash
CODEX_INTEGRATION_TESTS=1 python3 -m pytest \
  codex_provider/tests -k "codex_history" -q
```

Observed result:

```text
1 passed in 27.06s
```

Conclusion: when a Codex-backed deep-worker agent actually uses a proxy
deep-worker tool, the call and result do persist into LangGraph history and are
visible through `deep-worker --history`.

## Fact 4: Codex GenericItem Events Do Not Enter LangGraph Messages

Codex JSONL parsing maps unknown item types to `GenericItem`:

- `codex_provider/exec_runner.py:167`
- `codex_provider/exec_runner.py:181`

The code comment names examples:

- `codex_provider/exec_runner.py:88`
- `codex_provider/exec_runner.py:89`

`parse_event()` parses `item.completed` and `item.updated` into item events:

- `codex_provider/exec_runner.py:143`
- `codex_provider/exec_runner.py:154`

But `CodexChatModel.ainvoke()` only converts two Codex item classes into
LangChain-visible output:

- `AgentMessageItem` -> `final_text`
- proxy `McpToolCallItem` -> `AIMessage.tool_calls`

Evidence:

- `codex_provider/chat_model.py:439`
- `codex_provider/chat_model.py:441`
- `codex_provider/chat_model.py:443`
- `codex_provider/chat_model.py:452`
- `codex_provider/chat_model.py:466`
- `codex_provider/chat_model.py:468`

The explicit code path for `GenericItem` is a no-op:

```python
# else (GenericItem): reasoning / file_change /
# web_search / etc. — no deep-worker action.
```

Local proof command executed during this investigation:

```bash
python3 - <<'PY'
import asyncio
from codex_provider.chat_model import CodexChatModel
from codex_provider.exec_runner import GenericItem, ItemCompletedEvent, AgentMessageItem, TurnCompletedEvent
from codex_provider.tests.test_chat_model import _FakeExecRunner
from langchain_core.messages import HumanMessage

async def main():
    runner = _FakeExecRunner()
    runner.queue_events([
        ItemCompletedEvent(item=GenericItem(id='g1', type='reasoning', payload={'text':'hidden reasoning'})),
        ItemCompletedEvent(item=AgentMessageItem(id='a1', text='visible final')),
        TurnCompletedEvent(),
    ])
    m = CodexChatModel()
    m._exec_runner = runner
    r = await m.ainvoke([HumanMessage(content='x')], config={'configurable': {'thread_id': 'proof_generic_ignored'}})
    print({'content': r.content, 'tool_calls': r.tool_calls})

asyncio.run(main())
PY
```

Observed output:

```text
{'content': 'visible final', 'tool_calls': []}
```

Conclusion: the `reasoning` `GenericItem` did not become a LangChain message,
did not become an `AIMessage`, and would not be present in LangGraph
`state.messages`. This is a confirmed provider-level history gap.

## Fact 5: Codex Provider Does Not Implement LangChain Streaming

The provider module states that only `ainvoke` is real and streaming methods
are not implemented:

- `codex_provider/chat_model.py:1`
- `codex_provider/chat_model.py:5`
- `codex_provider/chat_model.py:6`

Conclusion: Codex CLI `item.updated` / streaming progress events are not
emitted as LangChain streaming messages by this provider.

## Fact 6: Web App Full Dialog Is A Filtered Formatter, Not Raw History

The Full Dialog button opens:

- `<web-app-dir>/static/js/main.js:313`
- `<web-app-dir>/static/js/main.js:318`

That endpoint is:

- `<web-app-dir>/app/routers/agents.py:543`
- `<web-app-dir>/app/routers/agents.py:546`

It calls `AgentService.get_agent_dialog_formatted()`:

- `<web-app-dir>/app/routers/agents.py:601`
- `<web-app-dir>/app/services/agent_service.py:1266`

`get_agent_dialog_formatted()` fetches raw history from the agent, then passes
it through `split_into_turns()` and `render_dialog()`:

- `<web-app-dir>/app/services/agent_service.py:1324`
- `<web-app-dir>/app/services/agent_service.py:1334`
- `<web-app-dir>/app/services/agent_service.py:1360`
- `<web-app-dir>/app/services/agent_service.py:1361`
- `<web-app-dir>/app/services/agent_service.py:1362`

`split_into_turns()` handles `HumanMessage` and `AIMessage`, records
`AIMessage.tool_calls`, and deliberately does not render `ToolMessage`
payloads:

- `<web-app-dir>/app/services/dialog_formatter.py:167`
- `<web-app-dir>/app/services/dialog_formatter.py:181`
- `<web-app-dir>/app/services/dialog_formatter.py:186`
- `<web-app-dir>/app/services/dialog_formatter.py:190`

`full` mode renders only text and tool-call summary entries from
`Turn.stream`:

- `<web-app-dir>/app/services/dialog_formatter.py:215`
- `<web-app-dir>/app/services/dialog_formatter.py:219`
- `<web-app-dir>/app/services/dialog_formatter.py:222`

Local proof command executed during this investigation:

```bash
python3 - <<'PY'
import sys
sys.path.insert(0, '/path/to/dev-tree/<web-app-dir>')
from app.services.dialog_formatter import split_into_turns, render_dialog
history = [
    {'type':'HumanMessage','id':'h1','content':'do x'},
    {'type':'AIMessage','id':'a1','content':'','tool_calls':[{'id':'tc1','name':'execute','args':{'command':'echo hi'}}]},
    {'type':'ToolMessage','id':'t1','content':'hi\n','tool_call_id':'tc1','name':'execute','status':'success'},
    {'type':'AIMessage','id':'a2','content':'done'},
]
print(render_dialog(split_into_turns(history), 'full'))
PY
```

Observed output:

```text
--- User ---
do x

--- Assistant (tool: execute) ---
→ execute({"command": "echo hi"})

--- Assistant ---
done
```

The `ToolMessage.content == "hi\n"` result is absent from Full Dialog output.

Conclusion: even when raw history contains `ToolMessage`, web Full Dialog
hides it by design.

## Fact 6.1: The Production Codex Agent Has No Persisted Proxy Tool Calls

The production Codex agent was not sent any messages during this
investigation. Only offline history and logs were read.

Offline state check:

```bash
python3 - <<'PY'
import asyncio
from deep_worker_cli.sessions import get_thread_history

async def main():
    res = await get_thread_history('agent_a', format='text')
    msgs = res['messages']
    print('success', res['success'], 'count', len(msgs))
    ai_with_tool_calls = 0
    tool_msgs = 0
    for m in msgs:
        if getattr(m, 'tool_calls', None):
            ai_with_tool_calls += 1
        if getattr(m, 'type', None) == 'tool' or m.__class__.__name__ == 'ToolMessage':
            tool_msgs += 1
    print('ai_with_tool_calls', ai_with_tool_calls)
    print('tool_msgs', tool_msgs)

asyncio.run(main())
PY
```

Observed output:

```text
success True count 15
ai_with_tool_calls 0
tool_msgs 0
```

`deep-worker --history agent_a --format json` also shows every persisted
assistant message with `tool_calls=[]` and no `ToolMessage` entries.

The production agent's log contains no `Returning AIMessage with ... tool_calls`
entries. It contains TaskManager iterations and MCP streamable HTTP
`ClosedResourceError` noise:

- `<agent-data-dir>/logs/agent_a.err.log`

Conclusion: for this agent, the missing tool calls are not just a display
problem. The persisted LangGraph checkpoint genuinely has zero proxy tool
calls and zero `ToolMessage`s.

Combined with Fact 3.1, this proves a narrower root cause:

- Codex provider can persist proxy tool calls.
- The agent did not emit proxy tool calls into LangGraph for those turns.
- Therefore the agent's history cannot show deep-worker tool calls, because
  none exist in its checkpoint.

Additional local Codex session logs matching the agent's prompts were found
under `~/.codex/sessions/2026/05/16/`:

- `rollout-2026-05-16T18-32-04-019e31a1-71e0-7de0-b9fa-3e46b330ee73.jsonl`
- `rollout-2026-05-16T18-34-03-019e31a3-437c-7042-8f80-94c08d69571c.jsonl`

Counts from those files:

```text
rollout-2026-05-16T18-32-04-019e31a1-71e0-7de0-b9fa-3e46b330ee73.jsonl
"type":"function_call" 331
"name":"exec_command" 250
"name":"apply_patch" 35
"name":"write_stdin" 56
"mcp_tool_call" 0
"type":"reasoning" 154

rollout-2026-05-16T18-34-03-019e31a3-437c-7042-8f80-94c08d69571c.jsonl
"type":"function_call" 27
"name":"exec_command" 27
"mcp_tool_call" 0
"type":"reasoning" 21
```

Conclusion: the matching Codex session logs contain Codex-native
`function_call` and `reasoning` events, not `mcp_tool_call` events from the
deep-worker proxy. These events are outside the provider's proxy-tool mapping
and are not represented as LangGraph `AIMessage.tool_calls` / `ToolMessage`s in
the agent's checkpoint.

## Fact 7: Web Main History View Also Filters To Last Text Answer

The web app has a raw history endpoint:

- `<web-app-dir>/app/routers/agents.py:166`

`AgentService.get_agent_history()` forwards raw agent history over WebSocket as
`history_response`:

- `<web-app-dir>/app/services/agent_service.py:1011`
- `<web-app-dir>/app/services/agent_service.py:1057`
- `<web-app-dir>/app/services/agent_service.py:1062`

But the main UI display function only searches for the last `AIMessage` with
non-empty text:

- `<web-app-dir>/static/js/main.js:364`
- `<web-app-dir>/static/js/main.js:368`
- `<web-app-dir>/static/js/main.js:372`

Conclusion: the main web panel cannot be used as evidence that raw history is
missing tool messages; it intentionally displays only the last text response.

## Fact 8: Realtime WebSocket Tool Events Are Notifications, Not History

In non-streaming invoke mode, `AgentProcess._handle_invoke()` converts new
LangGraph messages into realtime notifications:

- `deep_worker/deep_worker_cli/deep_worker_cli/process/agent_process.py:1521`
- `deep_worker/deep_worker_cli/deep_worker_cli/process/agent_process.py:1535`
- `deep_worker/deep_worker_cli/deep_worker_cli/process/agent_process.py:1542`
- `deep_worker/deep_worker_cli/deep_worker_cli/process/agent_process.py:1555`
- `deep_worker/deep_worker_cli/deep_worker_cli/process/agent_process.py:1574`

These notifications go through `notify_callback`; they are not themselves
added to LangGraph state. The persisted source remains `result["messages"]` /
checkpointed `state.messages`.

In streaming mode, `process_message_chunk()` also sends realtime notifications
for `ToolMessage` and content blocks:

- `deep_worker/deep_worker_cli/deep_worker_cli/process/stream_processor.py:129`
- `deep_worker/deep_worker_cli/deep_worker_cli/process/stream_processor.py:177`
- `deep_worker/deep_worker_cli/deep_worker_cli/process/stream_processor.py:182`
- `deep_worker/deep_worker_cli/deep_worker_cli/process/stream_processor.py:248`

Conclusion: WebSocket `tool_call` / `tool_result` events are derived display
events. They are not a separate persistence mechanism.

## Fact 9: Documentation Currently Overstates Full Dialog Fidelity

The user guide says:

- `<web-app-dir>/docs/USER_GUIDE.md:88`
- `<web-app-dir>/docs/USER_GUIDE.md:89`

But the formatter intentionally drops `ToolMessage` payloads:

- `<web-app-dir>/app/services/dialog_formatter.py:190`

The design document also says `ToolMessage payload is not rendered`:

- `<web-app-dir>/docs/DESIGN_FULL_DIALOG_FINAL_RESPONSE_PAIRS.md:360`

Conclusion: documentation contains an internal contradiction. The implementation
matches the design document's filtering behavior, not the user guide's "No
message is dropped" claim.

## Confirmed Gaps

### Gap A: Codex GenericItem Events Are Not Persisted

Severity: critical under the "all events must pass through LangGraph message
loop" requirement.

Confirmed affected event classes:

- `reasoning`
- `file_change`
- `web_search`
- `todo_list`
- any future Codex JSONL item type that falls into `GenericItem`

Confirmed mechanism:

- parsed as `GenericItem`
- ignored in `CodexChatModel.ainvoke()`
- never emitted as any LangChain message
- therefore never checkpointed into LangGraph `state.messages`

### Gap B: Full Dialog Drops ToolMessage Results

Severity: critical for post-mortem history correctness.

Confirmed mechanism:

- raw history can contain `ToolMessage`
- `split_into_turns()` does not add `ToolMessage` to the render stream
- `render_dialog(..., "full")` cannot display the tool result because the
  formatter already discarded it

### Gap C: Full Dialog Is Misnamed For Raw Auditing

Severity: high.

Confirmed mechanism:

- `Full (with tool calls)` means "all assistant text plus compact tool call
  summaries"
- it does not mean "all LangGraph messages"
- it does not include raw `ToolMessage` payloads, message ids, status fields,
  or all provider events

## Non-Gaps Confirmed

### Proxy Tool Round Trip

The Codex proxy tool round trip is not direct execution in the proxy MCP
server. The proxy waits for a LangGraph `ToolMessage` before returning to Codex.
This path is covered by existing integration tests and focused tests.

### Raw History Serializer

The raw serializer does not drop `AIMessage.tool_calls` or `ToolMessage` fields.
If these objects are absent in a raw history payload, the loss happened before
serialization.

## Minimal Required Fix Direction

This section is not an implementation plan; it records what the facts imply.

1. Codex provider must convert every relevant Codex JSONL event into a
   LangChain message or a documented LangGraph-persisted event representation.
   `GenericItem` cannot remain a no-op if the project invariant is "all events
   pass through LangGraph".
2. The web app needs a true raw timeline history view separate from the
   current readable dialog formatter, or `full` must be redefined to render
   every serialized raw message.
3. The formatter must not silently drop `ToolMessage` in the mode advertised as
   full history.
4. Documentation must stop claiming "No message is dropped" until the raw
   timeline invariant is actually true.

## Follow-Up Fix Applied On 2026-05-17

The Codex provider was changed so native Codex tools that overlap the
deep-worker base tool set are no longer accepted as a hidden execution path.

Implemented provider-side changes:

- `CodexExecRunner` now passes Codex CLI's real top-level
  `disabled_tools=[...]` config instead of the ineffective
  `tools.<name>=false` shape.
- `DEFAULT_DISALLOWED_TOOLS` now includes exact native Codex names observed in
  session logs and JSONL streams, including `command_execution`,
  `exec_command`, `apply_patch`, `write_stdin`, `read_file`, `write_file`,
  `edit_file`, `tool_search`, and agent-control names.
- `CodexChatModel` injects an explicit deep-worker tool contract into Codex
  base instructions.
- `CodexChatModel` rejects native tool-like `GenericItem` types
  (`command_execution`, `file_change`, `web_search`, `tool_search`,
  `custom_tool_call`, `function_call`, `function_call_output`) if Codex still
  emits them. The turn fails instead of silently bypassing LangGraph history.

Regression tests added/updated:

- unit test for `disabled_tools=[...]` argv generation;
- unit tests requiring exact native tool names in the deny-list;
- unit test proving native `command_execution` is rejected;
- live deep-worker Codex history test proving a real Codex-backed test agent
  uses the proxy `read_file` tool and persists `[TOOL]`, `tool_calls`, and
  `tool_call_id` in offline history.

Observed verification after the fix:

```text
python3 -m pytest codex_provider/tests -q
105 passed, 5 skipped

CODEX_INTEGRATION_TESTS=1 python3 -m pytest \
  codex_provider/tests/test_integration.py -q
4 passed
```

Residual non-Codex-provider issue: the web app's readable Full Dialog
formatter still drops raw `ToolMessage` payloads from its formatted view. Raw
offline history does preserve them when they exist.

## Commands Run

```bash
git status --short
rg -n "get_agent_dialog_formatted|Full Dialog|full-dialog|dialog_formatted|formatted.*dialog|history.*format|Full|raw history|ToolMessage|tool_calls" <web-app-dir> deep_worker/deep_worker_cli
rg -n "def exposed_get_history|_handle_get_history|get_history\\(|message_to_dict|messages_to_dict|ToolMessage|AIMessage|tool_calls|astream|stream_mode|broadcast" deep_worker/deep_worker_cli/deep_worker_cli
python3 -m pytest codex_provider/tests/test_exec_runner.py::TestParseEvent::test_item_completed_unknown_type_is_passed_through codex_provider/tests/test_chat_model.py::TestAinvokeToolCalls::test_tool_call_breaks_with_pending_calls codex_provider/tests/test_chat_model.py::TestAinvokeToolCalls::test_toolmessage_resolves_future_not_in_prompt -q
```

Plus the two local Python proof snippets embedded above.
