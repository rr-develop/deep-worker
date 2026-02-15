# Deep Worker Daemon Mode

## 🎯 Background mode

Deep-worker **always** starts in background (daemon) mode:
- ✅ The terminal is not blocked
- ✅ The process is detached from the terminal session
- ✅ Closing the terminal does not stop the agent
- ✅ All logs go to files, not to the terminal

## 📝 How it works

```bash
# Start the agent
$ cd /my/project
$ deep-worker my-agent
Agent 'my-agent' starting in background (PID will be written to /tmp/deep_worker_my-agent.pid)
Logs: ~/.deep_worker/logs/my-agent.out.log

# The terminal is free right away! You can keep working
$ dwcli --list
Running agents (1):
============================================================
  Agent: my-agent
  PID:   123456
  Working dir: /my/project
  ...
```

## 🔍 Managing agents

### Checking status

```bash
# List all running agents
dwcli --list

# Detailed status
dwcli --status my-agent
```

### Logs

Logs are written to `~/.deep_worker/logs/`:

```bash
# Standard output (stdout)
tail -f ~/.deep_worker/logs/my-agent.out.log

# Errors (stderr)
tail -f ~/.deep_worker/logs/my-agent.err.log

# Both streams at once
tail -f ~/.deep_worker/logs/my-agent.*.log
```

### Stopping an agent

```bash
# Graceful shutdown
dwcli --stop my-agent

# Verify that it stopped
dwcli --status my-agent
```

## 🛠️ Command line options

### --dir: Working directory

**Default**: the current directory (where the command was run from)

```bash
# Use the current directory
cd /my/project
deep-worker my-agent
# Working dir = /my/project

# Specify the directory explicitly
deep-worker my-agent --dir /other/path
# Working dir = /other/path
```

**How it works**:
1. If `--dir` is given, it is used
2. Otherwise `DEEP_WORKER_ORIGINAL_CWD` is used (the current directory at startup)
3. Fallback: `Path.cwd()`

### -sp: Skip Permissions

**Purpose**: Automatically confirm dangerous operations without prompting

```bash
# With permission checks (default)
deep-worker my-agent

# Skip the checks
deep-worker my-agent -sp
deep-worker my-agent --skip-permissions  # the same thing
```

**When to use it**:
- ✅ In CI/CD pipelines (automation)
- ✅ In trusted environments
- ✅ For rapid prototyping
- ❌ **Do NOT use** in production without understanding the risks

## 🔧 Technical details

### Daemonization process

Deep-worker uses the Unix double-fork pattern:

1. **First fork()**:
   - The parent process prints info and exits
   - The child process continues

2. **Detaching from the terminal**:
   - `os.setsid()` - create a new session
   - `os.chdir("/")` - change to the root directory
   - `os.umask(0)` - reset the permission mask

3. **Second fork()**:
   - Guarantees that the process cannot acquire a controlling terminal

4. **Redirecting descriptors**:
   - `stdin` → `/dev/null`
   - `stdout` → `~/.deep_worker/logs/<name>.out.log`
   - `stderr` → `~/.deep_worker/logs/<name>.err.log`

### PID file

Created at `/tmp/deep_worker_<name>.pid`:

```bash
$ cat /tmp/deep_worker_my-agent.pid
123456

$ ps -p 123456
  PID TTY          TIME CMD
123456 ?        00:00:01 python3
```

### RPyC Port

Agent listens on an RPyC port (automatically assigned):

```bash
# Check agent's RPyC port
$ dwcli --status my-agent
Agent Status: my-agent
============================================================
...
  RPyC port: 18861
...
```

## 📊 Usage examples

### Example 1: Project development

```bash
# Terminal 1: Start the agent in the project directory
cd /my/project
deep-worker dev-agent -sp
# The agent started in the background

# Terminal 1: Keep working in the same terminal
git status
vim src/main.py

# Terminal 2: Monitor the logs
tail -f ~/.deep_worker/logs/dev-agent.out.log

# Terminal 3: Manage the agent
dwcli --status dev-agent
dwcli --stop dev-agent
```

### Example 2: Several agents at once

```bash
# Start an agent for each project
cd /project-alpha && deep-worker alpha
cd /project-beta && deep-worker beta -sp
cd /project-gamma && deep-worker gamma --provider openai

# All agents run in parallel
dwcli --list
# Shows 3 agents

# Monitor all logs
tail -f ~/.deep_worker/logs/*.out.log
```

### Example 3: CI/CD usage

```bash
#!/bin/bash
# ci-script.sh

# Start the agent in a CI environment
deep-worker ci-agent -sp --dir /build/workspace

# The agent goes to the background immediately, the script continues
echo "Agent started"

# Run tests/build
./run-tests.sh

# Stop the agent
dwcli --stop ci-agent

# Check the logs
cat ~/.deep_worker/logs/ci-agent.err.log
```

## 🔒 Security

### Process isolation

- ✅ Each agent is a separate process
- ✅ Separate PID, RPyC port, log files
- ✅ Independent working directories
- ✅ Not dependent on the terminal session

### -sp limitations

When using `-sp`:
- ⚠️ The agent can perform dangerous operations without confirmation
- ⚠️ It can modify the file system
- ⚠️ It can run system commands

**Recommendations**:
- Use only in trusted environments
- In production, set up proper sandboxing
- Check the logs for suspicious activity

## 💡 Troubleshooting

### The agent does not start

```bash
# Check the error logs
cat ~/.deep_worker/logs/<name>.err.log

# Check whether the PID file exists
ls -la /tmp/deep_worker_<name>.pid

# Check the RPyC port
dwcli --status <name>
```

### The agent is "stuck"

```bash
# Check the status
dwcli --status <name>

# If it does not respond - kill it manually
cat /tmp/deep_worker_<name>.pid | xargs kill -9

# Clean up the files
rm /tmp/deep_worker_<name>.*
```

### Log output was lost

```bash
# All logs are in this directory
ls -la ~/.deep_worker/logs/

# Find the logs of a specific agent
find ~/.deep_worker/logs/ -name "*my-agent*"
```

---

**Created**: February 2026
**Status**: ✅ Production Ready
