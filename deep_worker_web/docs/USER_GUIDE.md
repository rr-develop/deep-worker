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
   in chronological order. **Three sources** of user-side messages all
   appear as first-class turns and are never filtered out:

   - **`--- User ---`** — what you typed in dwcli or the web UI.
   - **`--- TaskManager ---`** — corrective prompts the agent's
     in-process TaskManager sent to keep work on track. The
     `TaskManager:>` prefix is stripped from the body.
   - **`--- From agent:<name> ---`** — messages received from another
     agent via `send_message_to_agent()`. The `agent:<name>:> ` prefix
     is stripped from the body.

   If the agent has not yet produced a text reply for the most recent
   user-side message, that turn renders with `(no response yet)` so you
   can see what the agent was asked even before it answers.

   The only thing filtered is the model's own context-summarization
   injection (`Context Extraction Assistant…` / `You are in the middle
   of a conversation that has been summarized…`) — that is internal
   bookkeeping, not a real user turn.

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

# Run with auto-reload
python3 -m app.main

# Run with debug logging
export DEBUG=true
python3 run.py
```

## Support

For issues and questions, refer to:
- [DESIGN.md](DESIGN.md) - Architecture and design decisions
- [API.md](API.md) - API reference
- Deep-worker documentation
