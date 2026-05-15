# Deep-Worker-Web User Guide

## Overview

Deep-Worker-Web is a web-based user interface for managing and interacting with deep-worker agents running on your local machine.

## Quick Start

### Prerequisites

1. Python 3.12+
2. Running deep-worker agents (registered in ProcessRegistry)
3. Dependencies installed: `pip install -r requirements.txt`

### Starting the Server

```bash
cd deep_worker_web
python3 run.py
```

The server will start on `http://127.0.0.1:7887` (localhost only for security).

### Accessing the UI

Open your browser and navigate to:
```
http://127.0.0.1:7887
```

## Features

### Agent List

The left panel shows all discovered deep-worker agents with:
- Agent name
- Current status (idle, running, waiting, error)
- AI provider

Click on an agent to select it for interaction.

### Agent Interaction

After selecting an agent, you can:

1. **View Status**: See detailed status including:
   - Current status
   - AI provider and model
   - Last activity

2. **View Last Response**: The most recent assistant message is displayed in the "Last Response" panel.

3. **Send Messages**: 
   - Type your message in the input area
   - Click "Send Message" or press `Ctrl+Enter`
   - The message is sent to the agent via RPyC

4. **Full Dialog**: Pick a verbosity level from the dropdown next to
   "📜 Full Dialog", then click the button to open the conversation
   history in a new tab.

   The dialog shows pairs of *user-side message → assistant final answer*
   in chronological order. **Four sources** of user-side messages all
   appear as first-class turns and are never filtered out:

   - **`--- User ---`** — what you typed in dwcli or the web UI.
   - **`--- TaskManager ---`** — corrective prompts the agent's
     in-process TaskManager sent to keep work on track. The
     `TaskManager:>` prefix is stripped from the body.
   - **`--- From agent:<name> ---`** — messages received from another
     agent via `send_message_to_agent()`. The `agent:<name>:> ` prefix
     is stripped from the body.
   - **`--- Summarization ---`** — synthetic injections created when
     the model compacts its own context (body starts with
     `Context Extraction Assistant…` or `You are in the middle of a
     conversation that has been summarized…`). These are NOT real user
     messages, but they are genuine drivers of the agent's next steps
     (they carry the distilled state the agent works from after the
     compaction). Showing them with a distinct header lets you see
     *why* the agent's behaviour pivots after a context compaction
     without confusing them with real user input. Design:
     `docs/DESIGN_SUMMARIZATION_VISIBLE.md`.

   If the agent has not yet produced a text reply for the most recent
   user-side message, that turn renders with `(no response yet)` so you
   can see what the agent was asked even before it answers.

   No message is dropped — Full Dialog is a faithful, post-mortem-grade
   record of everything the agent saw and said.

   ### Verbosity levels

   | Level | What you see |
   |---|---|
   | **Final answers only** *(default)* | One assistant block per turn — the **last** text-bearing reply, i.e. the genuine final answer. |
   | **All assistant text** | Every text-bearing assistant reply within the turn, in chronological order. Useful when the agent narrates intermediate progress. |
   | **Full (with tool calls)** | Same as *All assistant text*, plus a compact one-line summary for every tool call (`→ tool_name(args)`, args truncated to 200 chars). |

   You can also hit the endpoint directly:

   ```
   GET /api/agents/{name}/dialog-text                  # final (default)
   GET /api/agents/{name}/dialog-text?verbosity=text
   GET /api/agents/{name}/dialog-text?verbosity=full
   ```

   The selected verbosity is remembered between page loads (stored in
   browser `localStorage`). The endpoint always returns the **entire**
   conversation — there is no `limit` knob and no artificial cap. The
   only ceiling is the agent's own context window.

5. **Stop TaskManager**: Click "⏹️ Stop TaskManager" to stop the agent's TaskManager.

6. **Stop Agent** / **Restart Agent**: Click "⏹ Stop" or "🔄 Restart" in
   the agent header. Both return instantly; the actual work happens in
   the background and you'll see toast notifications as it progresses
   (`Stop requested → Escalated to SIGKILL → Stop ok`). If the agent
   refuses to die gracefully within the grace window (default 5 s), the
   web server automatically escalates to SIGKILL — no manual
   intervention needed. You can also tick agents in the left-hand
   checkbox list and use the batch toolbar to stop / restart many at
   once; they run in parallel, not sequentially, and each gets its own
   notification plus an aggregate "Batch stop: 3/3".

   In the rare case where even SIGKILL can't kill a process (kernel
   state `D` / `T`), you'll get an error toast with the exact `kill -9 <pid>`
   command to run manually — the system never fails silently.

### Real-Time Updates

The UI uses WebSocket for real-time updates:
- New agent messages appear automatically
- Status changes are reflected immediately
- New agents are detected automatically
- Terminated agents are removed from the list

### Non-Blocking Architecture

**Important**: All operations in deep-worker-web use a fire-and-forget pattern:

- **Instant responses**: The UI never freezes waiting for agent responses
- **Background processing**: All RPC calls execute in the background
- **Notifications**: Success and error messages appear as toast notifications
- **Timeout protection**: Operations have automatic timeouts to prevent hanging

**What this means for you:**
- When you send a message, the UI responds immediately
- The actual agent response arrives via WebSocket (usually within seconds)
- If an agent is slow or hung, you'll see an error notification after the timeout
- The UI remains fully responsive even when agents are processing

**Error Notifications:**
- **Timeout errors**: "Agent not responding (timeout)" - agent took too long
- **Connection errors**: "Connection to agent lost" - agent process died
- **Agent errors**: Specific error messages from the agent

All notifications appear in the top-right corner and auto-dismiss after 5 seconds.

## Security

⚠️ **CRITICAL SECURITY NOTICE**

Deep-Worker-Web is designed for **LOCAL USE ONLY** and MUST run on:
- Host: `127.0.0.1` (localhost) ONLY
- Port: `7887` ONLY

**DO NOT:**
- Bind to `0.0.0.0` or any other address
- Expose to external networks
- Use over the internet
- Change the host or port

The application has no authentication and provides direct access to agent capabilities. It is designed for a trusted local development environment only.

## Troubleshooting

### No Agents Showing

1. Verify agents are running:
   ```bash
   cat /tmp/deep_worker_registry.json
   ```

2. Check agent processes are alive:
   ```bash
   ps aux | grep deep_worker
   ```

3. Restart the deep-worker-web server

### WebSocket Connection Failed

1. Check if server is running on correct port:
   ```bash
   netstat -an | grep 7887
   ```

2. Check browser console for errors (F12)

3. Verify firewall is not blocking localhost connections

### Messages Not Sending

1. Verify agent has RPyC server running
2. Check agent logs for errors
3. Verify ProcessRegistry has correct RPyC port

## API Documentation

For API details, see [API.md](API.md).

## Development

For development and testing:

```bash
# Run tests
python3 -m pytest tests/

# Run with auto-reload (uvicorn WatchFiles). Opt-in only — the default
# is no-reload because WatchFiles runs the app inside a
# multiprocessing.spawn subprocess whose long-term memory profile is
# harder to reason about. See docs/DESIGN_AGENT_SERVICE_MEMORY_FIX.md.
export DEEP_WORKER_WEB_RELOAD=1
python3 run.py

# Run with debug logging (does NOT imply reload):
export DEEP_WORKER_WEB_DEBUG=true
python3 run.py
```

## Support

For issues and questions, refer to:
- [DESIGN.md](DESIGN.md) - Architecture and design decisions
- [API.md](API.md) - API reference
- Deep-worker documentation
