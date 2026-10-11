# Deep Worker `--clean` Option

## Overview

The `--clean` option allows you to start a deep-worker agent with a completely fresh state, removing all previous conversation history and session data while preserving your custom configuration.

## Usage

```bash
deep-worker <agent_name> --clean [other options]
```

### Example

```bash
# Start agent with clean history
deep-worker myagent --dir /path/to/your/project --clean

# With custom provider
deep-worker myagent --provider claude-code-sdk --clean
```

## What Gets Removed

When you use `--clean`, the following data is deleted:

1. **Conversation History**
   - All conversation threads from the agent's thread database
   - All checkpoints and messages for this agent

2. **Session Data**
   - Agent data directory: `~/.deep_worker/{agent_name}/`
   - Session metadata (task counters, status, etc.)

3. **Log Files**
   - All log files: `~/.deep_worker/logs/{agent_name}.*`
   - stdout and stderr logs

## What Gets Preserved

The `--clean` option **does NOT** remove your custom configuration:

1. **User Configuration**
   - `~/.deep_worker/agents/{agent_name}/AGENTS.md` - Your custom agent instructions
   - Agent personality and custom prompts

2. **Custom Skills**
   - `~/.deep_worker/agents/{agent_name}/skills/` - All your custom skills
   - Reusable workflow definitions

3. **Custom Subagents**
   - `~/.deep_worker/agents/{agent_name}/agents/` - Your specialized subagents

## Requirements

- **Agent must NOT be running**: The `--clean` option will fail if the agent is already running
- **Error handling**: Returns clear error message if agent is running

## Use Cases

### 1. Starting Fresh After Testing

```bash
# After development/testing with lots of test queries
deep-worker dev_agent --clean
```

### 2. Clearing Error States

```bash
# Agent had errors, start with clean state
deep-worker problematic_agent --clean
```

### 3. Benchmarking

```bash
# Ensure consistent starting point for performance tests
deep-worker benchmark_agent --clean
```

### 4. Resetting Task Counters

```bash
# Start with fresh counters and metrics
deep-worker analytics_agent --clean
```

## Safety Features

### 1. Running Agent Check

If the agent is currently running, `--clean` will fail with an error:

```
Error: Agent 'myagent' is already running (PID 12345)
Stop it first with: pkill -f 'deep-worker myagent'
```

**Why?** Cleaning while the agent is running could cause data corruption or undefined behavior.

### 2. Graceful Handling

The cleaning process is **idempotent** - it can be run multiple times safely:
- Non-existent files are skipped without errors
- Database operations are atomic
- All operations complete or fail as a unit

### 3. Configuration Preservation

Your custom configuration is **never** deleted:
- `AGENTS.md` is preserved
- Custom skills are preserved
- Custom subagents are preserved

This ensures you don't lose your personalization and customizations.

## Output

The `--clean` option provides detailed feedback:

```
✓ Cleaned history for agent 'myagent':
  - Removed 15 checkpoints from database
  - Removed 42 writes from database
  - Removed data directory: ~/.deep_worker/myagent
  - Removed 2 log file(s)
  - Preserved user config: ~/.deep_worker/agents/myagent/
```

## Technical Details

### Database Cleanup

- Removes all rows from `checkpoints` table where `metadata.agent_name` matches
- Removes all rows from `writes` table for associated thread IDs
- Uses SQLite's JSON extraction for efficient filtering

### File System Cleanup

- Uses `shutil.rmtree()` for directory removal (atomic operation)
- Uses `Path.unlink()` for log file removal
- Operations are non-blocking (async-safe)

### Performance

- Cleanup typically takes < 1 second
- Database operations are optimized with indices
- No impact on other agents running simultaneously

## Comparison with Other Options

| Operation | `--clean` | Delete & Recreate | Manual Cleanup |
|-----------|-----------|-------------------|----------------|
| Removes history | ✓ | ✓ | ✓ |
| Preserves config | ✓ | ✗ | Depends |
| Preserves skills | ✓ | ✗ | Depends |
| Single command | ✓ | ✗ | ✗ |
| Safe (atomic) | ✓ | ✗ | ✗ |
| Prevents errors | ✓ | ✗ | ✗ |

## Troubleshooting

### Agent Won't Start After Clean

**Problem**: Agent fails to start after running with `--clean`

**Solution**: Check logs in `~/.deep_worker/logs/{agent_name}.err.log`

### Database Locked Error

**Problem**: "database is locked" error during cleanup

**Solution**:
1. Stop all agents: `pkill -f deep-worker`
2. Wait 2 seconds
3. Try again with `--clean`

### Partial Cleanup

**Problem**: Some files remain after `--clean`

**Possible causes**:
- Permission issues (check file ownership)
- Files open by another process
- Disk space issues

**Solution**: Check `--clean` output for error messages

## Related Commands

- `deepagents threads list` - View all conversation threads
- `deepagents threads delete <thread_id>` - Delete specific thread
- `pkill -f 'deep-worker <name>'` - Stop running agent

## Best Practices

1. **Stop agent first** - Always stop the agent before using `--clean`
2. **Backup important data** - If you have important conversation history, export it first
3. **Document custom config** - Keep backups of your `AGENTS.md` and skills
4. **Use in development** - Great for development/testing cycles
5. **Avoid in production** - Consider thread deletion instead for production agents

## Examples

### Development Workflow

```bash
# Iteration 1: Test with queries
deep-worker dev_agent

# Stop and clean for next iteration
pkill -f 'deep-worker dev_agent'
deep-worker dev_agent --clean

# Repeat...
```

### CI/CD Pipeline

```yaml
# Ensure clean state for integration tests
- name: Start clean agent
  run: deep-worker test_agent --clean --skip-permissions
```

### Multi-Agent Setup

```bash
# Clean specific agents without affecting others
deep-worker agent1 --clean
deep-worker agent2 --clean

# agent3 keeps its history
deep-worker agent3
```

## See Also

- [Agent Configuration Guide](./AGENT_CONFIG.md)
- [Testing with Clean Agents](../tests/integration_ai/CLEAN_AGENT_USAGE.md)
- Session module source: `deep_worker_cli/sessions.py`
