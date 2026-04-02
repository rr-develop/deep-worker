# dwtmux - Tmux Session Manager for Deep-Worker Agents

## Overview

`dwtmux` is a command-line utility for managing tmux sessions with deep-worker agents. It provides seamless integration between tmux terminal multiplexer and the `dwcli` interactive client, allowing you to work with agents in persistent terminal sessions.

## Installation

The `dwtmux` command is automatically installed as a global command when you set up the deep-worker-cli environment.

### Manual Installation

If needed, you can manually install the command:

```bash
# Make script executable (run from the repository root)
chmod +x deep_worker/deep_worker_cli/dwtmux

# Create symlink for global access
ln -sf "$(pwd)/deep_worker/deep_worker_cli/dwtmux" ~/.local/bin/dwtmux
```

**IMPORTANT**: The script in `deep_worker/deep_worker_cli/dwtmux` is the **single source of truth**. The symlink in `~/.local/bin/dwtmux` always points to the repository version, ensuring that any updates to the script are immediately available without additional copying.

## Usage

```bash
dwtmux <agent-name>
```

### Arguments

- `<agent-name>` - Name of the deep-worker agent to interact with

### Options

- `-h, --help` - Show help message and exit

## Behavior

The `dwtmux` command intelligently manages tmux sessions based on the current state:

### 1. Tmux Session Exists

If a tmux session with the agent's name already exists:
- **Action**: Attach to the existing session
- **Output**: `Attaching to existing tmux session '<agent-name>'...`

### 2. No Tmux Session, Agent Running

If no tmux session exists but the agent is running:
- **Action**: Create a new tmux session and launch `dwcli <agent-name>` inside it
- **Output**: `Creating new tmux session '<agent-name>' with dwcli...`

### 3. Agent Not Running

If the agent is not running:
- **Action**: Display error message (identical to `dwcli` error)
- **Output**:
  ```
  Error: Agent '<agent-name>' not found

  Available agents:
    - agent1
    - agent2

  Start an agent with: deep-worker <agent-name>
  ```

## Examples

### Example 1: First Connection

```bash
# Start an agent in daemon mode
$ deep-worker my_agent &

# Connect via dwtmux (creates new tmux session)
$ dwtmux my_agent
Creating new tmux session 'my_agent' with dwcli...
# Now in tmux session with dwcli interactive interface
```

### Example 2: Reconnecting

```bash
# Later, reconnect to the same session
$ dwtmux my_agent
Attaching to existing tmux session 'my_agent'...
# Attached to existing session, preserving all history
```

### Example 3: Multiple Agents

```bash
# Work with multiple agents in separate tmux sessions
$ deep-worker agent1 &
$ deep-worker agent2 &

$ dwtmux agent1    # Session "agent1"
# Detach: Ctrl+B, D

$ dwtmux agent2    # Session "agent2"
# Detach: Ctrl+B, D

# Switch between sessions:
$ tmux switch-client -t agent1
$ tmux switch-client -t agent2
```

### Example 4: Agent Not Running

```bash
$ dwtmux nonexistent_agent

Error: Agent 'nonexistent_agent' not found

Available agents:
  - my_agent
  - test_agent

Start an agent with: deep-worker nonexistent_agent
```

## Tmux Session Management

### Detaching from Session

To detach from a tmux session without closing it:
- Press `Ctrl+B`, then `D`

### Listing All Sessions

```bash
tmux ls
```

### Killing a Session

```bash
tmux kill-session -t <agent-name>
```

### Tmux Key Bindings (Default)

- `Ctrl+B` - Prefix key (all tmux commands start with this)
- `Ctrl+B, D` - Detach from session
- `Ctrl+B, C` - Create new window
- `Ctrl+B, N` - Next window
- `Ctrl+B, P` - Previous window
- `Ctrl+B, [` - Enter copy mode (scroll history)

## Integration with dwcli

`dwtmux` is essentially a wrapper around `dwcli` that adds tmux session management:

| Command | Description |
|---------|-------------|
| `dwcli <agent>` | Interactive connection (no session persistence) |
| `dwtmux <agent>` | Interactive connection in persistent tmux session |

Both commands provide the same interactive interface, but `dwtmux` allows you to:
- Detach and reconnect without losing context
- Preserve command history and scrollback
- Work with multiple agents in parallel sessions
- Access the session from different terminal windows

## Related Commands

### Deep-Worker Commands

- `deep-worker <name>` - Start a new agent in daemon mode
- `dwcli <name>` - Connect to agent interactively (no tmux)
- `dwcli --list` - List all running agents
- `dwcli --status <name>` - Show agent status
- `dwcli --stop <name>` - Stop an agent

### Tmux Commands

- `tmux ls` - List all tmux sessions
- `tmux attach -t <session>` - Attach to specific session
- `tmux kill-session -t <session>` - Kill specific session
- `tmux kill-server` - Kill all tmux sessions

## Workflow Recommendations

### Development Workflow

```bash
# 1. Start agent in background
deep-worker dev_agent &

# 2. Connect via tmux
dwtmux dev_agent

# 3. Work with agent...
# 4. Detach when needed (Ctrl+B, D)

# 5. Later, reconnect from any terminal
dwtmux dev_agent

# 6. When done, stop agent
dwcli --stop dev_agent
```

### Production Workflow

```bash
# Use systemd or supervisor to manage agent lifecycle
# Use dwtmux for admin/debugging access

# Quick health check
dwtmux prod_agent
# Issue: /status
# Detach: Ctrl+B, D
```

## Troubleshooting

### "tmux is not installed"

```bash
# Ubuntu/Debian
sudo apt-get install tmux

# macOS
brew install tmux

# RHEL/CentOS
sudo yum install tmux
```

### Session Already Exists but Can't Attach

```bash
# List sessions to verify
tmux ls

# Force kill stuck session
tmux kill-session -t <agent-name>

# Recreate
dwtmux <agent-name>
```

### Agent Process Exists but dwcli Can't Connect

```bash
# Check agent status
dwcli --status <agent-name>

# If stale, stop and restart
dwcli --stop <agent-name>
deep-worker <agent-name>
dwtmux <agent-name>
```

## Architecture

### Single Source of Truth

The `dwtmux` script located at `deep_worker/deep_worker_cli/dwtmux` is the **only** copy that should be modified. The global command at `~/.local/bin/dwtmux` is a **symlink** to this repository file.

**Benefits:**
- No need to reinstall or copy after modifications
- All changes are immediately effective
- Easy to track in version control
- Consistent behavior across all users of the repository

**Implications:**
- If you modify the script, changes take effect immediately
- If you delete the repository, the global command will break
- Always use the repository script for development and testing

### Dependencies

- **tmux** - Terminal multiplexer (required)
- **dwcli** - Deep-worker CLI client (from same repository)
- **deep-worker** - Agent launcher (for starting agents)
- **Python 3** - For ProcessRegistry checks

## See Also

- [dwcli Documentation](./README.md) - Interactive client documentation
- [Deep-Worker Architecture](./ARCHITECTURE.md) - System architecture overview
- [Daemon Mode](./DAEMON_MODE.md) - Background agent management
- [tmux Manual](https://man.openbsd.org/tmux) - Official tmux documentation
