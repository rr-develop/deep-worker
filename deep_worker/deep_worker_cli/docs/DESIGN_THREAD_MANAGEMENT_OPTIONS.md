# Design: Thread management options for deep-worker

## 📋 Overview

Adding thread management commands to the `deep-worker` CLI:
- `-rm <thread-id>` - removing a thread from the database
- `--history <thread-id>` - viewing the message history of a thread
- `--remove-before <date>` - bulk removal of old threads

All the commands work regardless of the agent's state (running/stopped).

## 🎯 Goals and requirements

### Functional requirements

1. **Removing a thread (`-rm`)**
   - Deletes all the checkpoints for the given thread_id
   - Works even if the agent is running
   - Prints a confirmation: the number of deleted checkpoints
   - A warning if the thread is not found

2. **Thread history (`--history`)**
   - Shows all the messages in a thread (like `dwcli --history`)
   - Formatting: timestamp, role (user/assistant/system), content
   - Works without a running agent (it reads only from the database)
   - Supports pagination (`--limit`, `--offset`)
   - Supports output formats (`--format text|json`)

3. **Removal by date (`--remove-before`)**
   - Deletes all the threads with `updated_at < the_given_date`
   - Supported date formats: ISO 8601, relative ones (7d, 30d, 1y)
   - A dry-run mode (`--dry-run`) for a preview
   - Confirmation before deletion (unless `--yes`)
   - Statistics: how many threads were deleted

### Non-functional requirements

1. **Safety**
   - A running agent's thread cannot be deleted (a warning)
   - Confirmation for bulk operations
   - Atomic transactions (all or nothing)

2. **Performance**
   - Batch deletion for `--remove-before`
   - Indexes on `updated_at` and `thread_id`

3. **Compatibility**
   - Works with AsyncSqliteSaver and InMemorySaver
   - Does not break the existing `--list` functionality

## 📐 Architecture

### Where the code lives

```
deep_worker_cli/
├── cli/
│   └── agent_command.py          # Add the new arguments
├── sessions.py                   # New functions for working with threads
│   ├── remove_thread()           # Removing a thread
│   ├── get_thread_history()      # Getting the history
│   └── remove_threads_before()   # Removal by date
└── process/
    └── registry.py               # Checking for running agents
```

### Architecture diagram

```
┌─────────────────────────────────────────────────────────────┐
│                     deep-worker CLI                         │
│  (agent_command.py)                                         │
└────────────┬────────────────────────────────────────────────┘
             │
             ├─────────────────────────────────────────────────┐
             │                                                 │
             ▼                                                 ▼
┌────────────────────────┐                    ┌─────────────────────────┐
│   Thread Management    │                    │   Process Registry      │
│   (sessions.py)        │                    │   (registry.py)         │
│                        │                    │                         │
│ - remove_thread()      │◄───────────────────│ - lookup()             │
│ - get_thread_history() │   Check if agent   │ - is_agent_running()   │
│ - remove_before()      │   is running       │                         │
└────────┬───────────────┘                    └─────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│               SQLite Database                               │
│      (legacy global DB — removed, see POLICIES §16)         │
│                                                             │
│  checkpoints table:                                         │
│  - thread_id (indexed)                                      │
│  - checkpoint_id                                            │
│  - checkpoint_ns                                            │
│  - parent_checkpoint_id                                     │
│  - type                                                     │
│  - checkpoint (blob)                                        │
│  - metadata (JSON with updated_at)                          │
└─────────────────────────────────────────────────────────────┘
```

## 🔧 Detailed design

### 1. The `-rm <thread-id>` option

#### CLI interface

```bash
# Remove a thread
deep-worker -rm d1:tm

# With confirmation
deep-worker -rm d1:tm --yes

# Without confirmation (interactive mode)
deep-worker -rm d1:tm
# Prompt: "Delete thread 'd1:tm'? This will remove N checkpoints. [y/N]: "
```

#### The argparse arguments

```python
parser.add_argument(
    "-rm", "--remove-thread",
    metavar="THREAD_ID",
    dest="remove_thread",
    help="Remove thread and all its checkpoints from database"
)

parser.add_argument(
    "--yes", "-y",
    action="store_true",
    help="Skip confirmation prompts"
)
```

#### The implementation in sessions.py

```python
async def remove_thread(thread_id: str, force: bool = False) -> dict:
    """Remove thread and all its checkpoints from database.

    Args:
        thread_id: Thread identifier to remove
        force: Skip confirmation and agent running check

    Returns:
        dict with:
        - success: bool
        - checkpoints_deleted: int
        - agent_running: bool
        - message: str

    Raises:
        ValueError: If thread_id is empty
        RuntimeError: If agent is running and not force
    """
    if not thread_id:
        raise ValueError("thread_id cannot be empty")

    # Check if agent is running
    from deep_worker_cli.process.registry import ProcessRegistry
    registry = ProcessRegistry()

    # Extract agent name from thread_id (before :tm or the whole string)
    agent_name = thread_id.split(':')[0]
    agent_info = registry.lookup(agent_name)

    if agent_info and not force:
        raise RuntimeError(
            f"Agent '{agent_name}' is currently running (PID {agent_info.pid}).\n"
            f"Stop the agent first or use --force to delete anyway."
        )

    # Get database connection
    db_path = get_db_path()
    if not db_path.exists():
        return {
            'success': False,
            'checkpoints_deleted': 0,
            'agent_running': False,
            'message': f"Database not found at {db_path}"
        }

    # Delete checkpoints
    import aiosqlite
    async with aiosqlite.connect(str(db_path)) as db:
        # Count checkpoints first
        cursor = await db.execute(
            "SELECT COUNT(*) FROM checkpoints WHERE thread_id = ?",
            (thread_id,)
        )
        count = (await cursor.fetchone())[0]

        if count == 0:
            return {
                'success': False,
                'checkpoints_deleted': 0,
                'agent_running': bool(agent_info),
                'message': f"Thread '{thread_id}' not found in database"
            }

        # Delete all checkpoints for this thread
        await db.execute(
            "DELETE FROM checkpoints WHERE thread_id = ?",
            (thread_id,)
        )
        await db.commit()

        return {
            'success': True,
            'checkpoints_deleted': count,
            'agent_running': bool(agent_info),
            'message': f"Deleted {count} checkpoint(s) for thread '{thread_id}'"
        }
```

#### Handling in agent_command.py

```python
async def remove_thread_command(args: argparse.Namespace) -> int:
    """Handle -rm/--remove-thread command.

    Returns:
        Exit code (0 for success, 1 for error)
    """
    thread_id = args.remove_thread

    try:
        # Try to remove thread
        result = await remove_thread(thread_id, force=args.yes)

        if result['success']:
            print(f"✓ {result['message']}")
            if result['agent_running']:
                print(f"⚠ Warning: Agent is still running. Changes may not take effect immediately.")
            return 0
        else:
            print(f"✗ {result['message']}", file=sys.stderr)
            return 1

    except RuntimeError as e:
        # Agent is running
        print(f"✗ Error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"✗ Error removing thread: {e}", file=sys.stderr)
        return 1

# In main()
if args.remove_thread:
    return asyncio.run(remove_thread_command(args))
```

### 2. The `--history <thread-id>` option

#### CLI interface

```bash
# Show the thread history
deep-worker --history d1

# With a limit
deep-worker --history d1 --limit 10

# With an offset (skip the first N)
deep-worker --history d1 --offset 5 --limit 10

# JSON format
deep-worker --history d1 --format json

# Show only the user's messages
deep-worker --history d1 --role user
```

#### The argparse arguments

```python
parser.add_argument(
    "--history",
    metavar="THREAD_ID",
    dest="show_history",
    help="Show message history for thread (works offline)"
)

# These arguments already exist for --list, just extend their usage:
# --limit, --offset, --format
```

#### The implementation in sessions.py

```python
from typing import Literal

async def get_thread_history(
    thread_id: str,
    limit: int | None = None,
    offset: int = 0,
    role_filter: Literal["user", "assistant", "system"] | None = None
) -> list[dict]:
    """Get message history for a thread.

    Args:
        thread_id: Thread identifier
        limit: Maximum number of messages to return
        offset: Number of messages to skip
        role_filter: Filter by message role

    Returns:
        List of message dicts with:
        - timestamp: str (ISO 8601)
        - role: str (user/assistant/system)
        - content: str
        - checkpoint_id: str

    Raises:
        ValueError: If thread_id is empty
        FileNotFoundError: If database doesn't exist
    """
    if not thread_id:
        raise ValueError("thread_id cannot be empty")

    db_path = get_db_path()
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found at {db_path}")

    import aiosqlite
    import json

    async with aiosqlite.connect(str(db_path)) as db:
        # Get all checkpoints for thread, ordered by checkpoint_id
        query = """
            SELECT checkpoint_id, checkpoint, metadata
            FROM checkpoints
            WHERE thread_id = ?
            ORDER BY checkpoint_id ASC
        """

        if limit:
            query += f" LIMIT {limit} OFFSET {offset}"

        cursor = await db.execute(query, (thread_id,))
        rows = await cursor.fetchall()

        messages = []

        for checkpoint_id, checkpoint_blob, metadata_json in rows:
            # Deserialize checkpoint (pickle)
            import pickle
            checkpoint = pickle.loads(checkpoint_blob)

            # Extract messages from checkpoint
            if 'channel_values' in checkpoint:
                channel_values = checkpoint['channel_values']
                if 'messages' in channel_values:
                    for msg in channel_values['messages']:
                        # Handle both dict and Message objects
                        if isinstance(msg, dict):
                            role = msg.get('type', msg.get('role', 'unknown'))
                            content = msg.get('content', '')
                        else:
                            # LangChain Message object
                            role = msg.type if hasattr(msg, 'type') else 'unknown'
                            content = msg.content if hasattr(msg, 'content') else ''

                        # Filter by role if specified
                        if role_filter and role != role_filter:
                            continue

                        # Get timestamp from metadata
                        metadata = json.loads(metadata_json) if metadata_json else {}
                        timestamp = metadata.get('updated_at', '')

                        messages.append({
                            'timestamp': timestamp,
                            'role': role,
                            'content': content,
                            'checkpoint_id': checkpoint_id
                        })

        return messages
```

#### Output formatting

```python
def format_history_text(messages: list[dict]) -> str:
    """Format message history as text."""
    if not messages:
        return "\nNo messages found in thread.\n"

    lines = ["\n" + "=" * 80 + "\n"]

    for msg in messages:
        # Parse timestamp
        timestamp = msg.get('timestamp', '')
        if timestamp:
            try:
                from datetime import datetime
                dt = datetime.fromisoformat(timestamp).astimezone()
                time_str = dt.strftime("%b %d, %-I:%M%p").lower()
                parts = time_str.split(" ")
                if parts:
                    parts[0] = parts[0].capitalize()
                    time_str = " ".join(parts)
            except (ValueError, TypeError):
                time_str = timestamp
        else:
            time_str = "Unknown time"

        role = msg['role'].upper()
        content = msg['content']

        # Role-specific colors/formatting could be added here
        lines.append(f"[{time_str}] {role}:")
        lines.append(content)
        lines.append("-" * 80)

    lines.append("")
    return "\n".join(lines)


def format_history_json(messages: list[dict], thread_id: str) -> str:
    """Format message history as JSON."""
    import json
    output = {
        'thread_id': thread_id,
        'message_count': len(messages),
        'messages': messages
    }
    return json.dumps(output, ensure_ascii=False, indent=2)
```

#### Handling in agent_command.py

```python
async def show_history_command(args: argparse.Namespace) -> int:
    """Handle --history command.

    Returns:
        Exit code (0 for success, 1 for error)
    """
    thread_id = args.show_history

    try:
        messages = await get_thread_history(
            thread_id=thread_id,
            limit=args.limit if hasattr(args, 'limit') else None,
            offset=args.offset if hasattr(args, 'offset') else 0,
        )

        if args.format == 'json':
            output = format_history_json(messages, thread_id)
        else:
            output = format_history_text(messages)

        print(output)
        return 0

    except FileNotFoundError as e:
        print(f"✗ Error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"✗ Error getting history: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1

# In main()
if args.show_history:
    return asyncio.run(show_history_command(args))
```

### 3. The `--remove-before <date>` option

#### CLI interface

```bash
# Remove threads older than 30 days
deep-worker --remove-before 30d

# Remove threads older than a specific date
deep-worker --remove-before 2026-01-01

# Dry-run (show what would be deleted)
deep-worker --remove-before 30d --dry-run

# Without confirmation
deep-worker --remove-before 30d --yes

# With an agent filter
deep-worker --remove-before 30d --agent d1
```

#### The argparse arguments

```python
parser.add_argument(
    "--remove-before",
    metavar="DATE",
    dest="remove_before",
    help="Remove all threads updated before DATE (formats: ISO 8601, '30d', '1y')"
)

parser.add_argument(
    "--dry-run",
    action="store_true",
    help="Show what would be deleted without actually deleting"
)

# --yes and --agent already exist
```

#### Date parsing

```python
from datetime import datetime, timedelta, UTC
import re

def parse_date(date_str: str) -> datetime:
    """Parse date string into datetime object.

    Supports:
    - ISO 8601: "2026-01-01", "2026-01-01T10:30:00"
    - Relative: "7d" (7 days), "30d" (30 days), "1y" (1 year)

    Args:
        date_str: Date string to parse

    Returns:
        datetime object (timezone-aware, UTC)

    Raises:
        ValueError: If date format is invalid
    """
    # Try relative format first (Nd, Nw, Nm, Ny)
    relative_pattern = r'^(\d+)([dwmy])$'
    match = re.match(relative_pattern, date_str.lower())

    if match:
        amount, unit = match.groups()
        amount = int(amount)

        now = datetime.now(UTC)

        if unit == 'd':  # days
            return now - timedelta(days=amount)
        elif unit == 'w':  # weeks
            return now - timedelta(weeks=amount)
        elif unit == 'm':  # months (approximate)
            return now - timedelta(days=amount * 30)
        elif unit == 'y':  # years (approximate)
            return now - timedelta(days=amount * 365)

    # Try ISO 8601 format
    try:
        # Parse various ISO formats
        dt = datetime.fromisoformat(date_str)

        # Ensure timezone-aware
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC)

        return dt
    except ValueError:
        pass

    raise ValueError(
        f"Invalid date format: '{date_str}'. "
        f"Use ISO 8601 (2026-01-01) or relative (30d, 1y)"
    )
```

#### The implementation in sessions.py

```python
async def remove_threads_before(
    before_date: datetime,
    agent_name: str | None = None,
    dry_run: bool = False
) -> dict:
    """Remove all threads updated before specified date.

    Args:
        before_date: Remove threads with updated_at < this date
        agent_name: Optional filter by agent name
        dry_run: If True, only show what would be deleted

    Returns:
        dict with:
        - success: bool
        - threads_deleted: int
        - checkpoints_deleted: int
        - threads: list[dict] (if dry_run)
        - message: str
    """
    db_path = get_db_path()
    if not db_path.exists():
        return {
            'success': False,
            'threads_deleted': 0,
            'checkpoints_deleted': 0,
            'message': f"Database not found at {db_path}"
        }

    import aiosqlite
    import json

    async with aiosqlite.connect(str(db_path)) as db:
        # Build query to find threads
        base_query = """
            SELECT DISTINCT thread_id, metadata
            FROM checkpoints
            WHERE json_extract(metadata, '$.updated_at') < ?
        """

        params = [before_date.isoformat()]

        if agent_name:
            base_query += " AND json_extract(metadata, '$.agent_name') = ?"
            params.append(agent_name)

        # Get threads to delete
        cursor = await db.execute(base_query, params)
        rows = await cursor.fetchall()

        if not rows:
            return {
                'success': True,
                'threads_deleted': 0,
                'checkpoints_deleted': 0,
                'message': "No threads found matching criteria"
            }

        thread_ids = [row[0] for row in rows]
        threads_info = []

        # Get detailed info for each thread
        for thread_id, metadata_json in rows:
            metadata = json.loads(metadata_json) if metadata_json else {}

            # Count checkpoints for this thread
            count_cursor = await db.execute(
                "SELECT COUNT(*) FROM checkpoints WHERE thread_id = ?",
                (thread_id,)
            )
            checkpoint_count = (await count_cursor.fetchone())[0]

            threads_info.append({
                'thread_id': thread_id,
                'agent_name': metadata.get('agent_name', 'unknown'),
                'updated_at': metadata.get('updated_at', ''),
                'checkpoint_count': checkpoint_count
            })

        if dry_run:
            total_checkpoints = sum(t['checkpoint_count'] for t in threads_info)
            return {
                'success': True,
                'threads_deleted': 0,
                'checkpoints_deleted': 0,
                'threads': threads_info,
                'message': f"Would delete {len(thread_ids)} thread(s) with {total_checkpoints} checkpoint(s)"
            }

        # Delete all checkpoints for these threads (batch operation)
        placeholders = ','.join('?' * len(thread_ids))
        delete_query = f"DELETE FROM checkpoints WHERE thread_id IN ({placeholders})"

        await db.execute(delete_query, thread_ids)

        # Count total checkpoints deleted
        total_checkpoints = sum(t['checkpoint_count'] for t in threads_info)

        await db.commit()

        return {
            'success': True,
            'threads_deleted': len(thread_ids),
            'checkpoints_deleted': total_checkpoints,
            'message': f"Deleted {len(thread_ids)} thread(s) with {total_checkpoints} checkpoint(s)"
        }
```

#### Handling in agent_command.py

```python
async def remove_before_command(args: argparse.Namespace) -> int:
    """Handle --remove-before command.

    Returns:
        Exit code (0 for success, 1 for error)
    """
    try:
        # Parse date
        before_date = parse_date(args.remove_before)

        print(f"Searching for threads updated before {before_date.strftime('%Y-%m-%d %H:%M:%S %Z')}...")

        # Dry run first to show what will be deleted
        result = await remove_threads_before(
            before_date=before_date,
            agent_name=args.agent if hasattr(args, 'agent') else None,
            dry_run=True
        )

        if result['threads_deleted'] == 0 and not result.get('threads'):
            print(result['message'])
            return 0

        # Show threads that will be deleted
        threads = result.get('threads', [])
        print(f"\nFound {len(threads)} thread(s) to delete:\n")

        for thread in threads:
            print(f"  - {thread['thread_id']:<30} "
                  f"(agent: {thread['agent_name']:<15}, "
                  f"updated: {thread['updated_at']}, "
                  f"{thread['checkpoint_count']} checkpoint(s))")

        print(f"\nTotal: {result['checkpoints_deleted']} checkpoint(s) across {len(threads)} thread(s)")

        if args.dry_run:
            print("\n✓ Dry run completed. Use without --dry-run to actually delete.")
            return 0

        # Ask for confirmation unless --yes
        if not args.yes:
            response = input("\nProceed with deletion? [y/N]: ")
            if response.lower() != 'y':
                print("Cancelled.")
                return 0

        # Perform actual deletion
        result = await remove_threads_before(
            before_date=before_date,
            agent_name=args.agent if hasattr(args, 'agent') else None,
            dry_run=False
        )

        if result['success']:
            print(f"\n✓ {result['message']}")
            return 0
        else:
            print(f"\n✗ {result['message']}", file=sys.stderr)
            return 1

    except ValueError as e:
        print(f"✗ Error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"✗ Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1

# In main()
if args.remove_before:
    return asyncio.run(remove_before_command(args))
```

## 🔒 Safety and validation

### Checks before deletion

1. **Checking for a running agent** (for `-rm`):
   ```python
   # Extract the agent name from the thread_id
   agent_name = thread_id.split(':')[0]

   # Check through the ProcessRegistry
   if registry.lookup(agent_name):
       # The agent is running - warn or block
   ```

2. **thread_id validation**:
   - Not empty
   - Exists in the database
   - Matches the format (alphanumeric + :)

3. **Date validation**:
   - A correct format (ISO or relative)
   - Not in the future (a warning, but allowed)

### Transactions

All deletion operations must be atomic:
```python
async with aiosqlite.connect(str(db_path)) as db:
    try:
        # Multiple DELETE operations
        await db.execute(...)
        await db.execute(...)

        # Commit all or nothing
        await db.commit()
    except Exception:
        await db.rollback()
        raise
```

## 📊 Usage examples

### Scenario 1: Removing a specific thread

```bash
# Check the list of threads
$ deep-worker --list
Thread ID                Agent Name           Last Updated
==================================================================
d1                       d1                   Mar 29, 10:51pm
d1:tm                    d1:tm                Mar 29, 10:51pm
old_agent                old_agent            Jan 15, 3:20pm

# Remove the old thread
$ deep-worker -rm old_agent
Delete thread 'old_agent'? This will remove 45 checkpoints. [y/N]: y
✓ Deleted 45 checkpoint(s) for thread 'old_agent'

# Check
$ deep-worker --list
Thread ID                Agent Name           Last Updated
==================================================================
d1                       d1                   Mar 29, 10:51pm
d1:tm                    d1:tm                Mar 29, 10:51pm
```

### Scenario 2: Viewing a thread's history

```bash
# Show the last 10 messages
$ deep-worker --history d1 --limit 10

================================================================================
[Mar 29, 10:48pm] USER:
echo test
--------------------------------------------------------------------------------
[Mar 29, 10:48pm] ASSISTANT:
test
--------------------------------------------------------------------------------
[Mar 29, 10:49pm] USER:
echo fixed
--------------------------------------------------------------------------------
[Mar 29, 10:49pm] ASSISTANT:
fixed
--------------------------------------------------------------------------------

# JSON format
$ deep-worker --history d1 --format json
{
  "thread_id": "d1",
  "message_count": 4,
  "messages": [
    {
      "timestamp": "2026-03-29T20:48:00+00:00",
      "role": "user",
      "content": "echo test",
      "checkpoint_id": "..."
    },
    ...
  ]
}
```

### Scenario 3: Cleaning up old threads

```bash
# Dry-run: see what would be deleted
$ deep-worker --remove-before 30d --dry-run

Searching for threads updated before 2026-02-27 00:00:00 UTC...

Found 12 thread(s) to delete:

  - old_test1               (agent: test_agent    , updated: 2026-02-01T10:30:00, 15 checkpoint(s))
  - old_test2               (agent: test_agent    , updated: 2026-02-10T14:20:00, 23 checkpoint(s))
  - experiment_thread       (agent: experiment    , updated: 2026-01-15T08:45:00, 8 checkpoint(s))
  ...

Total: 234 checkpoint(s) across 12 thread(s)

✓ Dry run completed. Use without --dry-run to actually delete.

# The actual deletion
$ deep-worker --remove-before 30d

[same output as above]

Proceed with deletion? [y/N]: y

✓ Deleted 12 thread(s) with 234 checkpoint(s)
```

### Scenario 4: Removal for a specific agent

```bash
# Remove all the old threads of the test_agent agent
$ deep-worker --remove-before 7d --agent test_agent --yes

Searching for threads updated before 2026-03-22 00:00:00 UTC...

Found 3 thread(s) to delete:

  - test_agent              (agent: test_agent    , updated: 2026-03-20T10:30:00, 45 checkpoint(s))
  - test_agent:tm           (agent: test_agent    , updated: 2026-03-20T10:30:00, 12 checkpoint(s))
  - test_agent_old          (agent: test_agent    , updated: 2026-03-15T08:00:00, 67 checkpoint(s))

Total: 124 checkpoint(s) across 3 thread(s)

✓ Deleted 3 thread(s) with 124 checkpoint(s)
```

## 🧪 Testing

### Unit tests

```python
# tests/unit/test_thread_management.py

import pytest
from deep_worker_cli.sessions import (
    remove_thread,
    get_thread_history,
    remove_threads_before,
    parse_date
)

@pytest.mark.asyncio
async def test_remove_thread():
    """Test thread removal."""
    # Setup: Create test thread in database
    # ...

    result = await remove_thread("test_thread", force=True)

    assert result['success'] == True
    assert result['checkpoints_deleted'] > 0

@pytest.mark.asyncio
async def test_remove_thread_agent_running():
    """Test that removal fails if agent is running."""
    # Setup: Start test agent
    # ...

    with pytest.raises(RuntimeError, match="Agent .* is currently running"):
        await remove_thread("test_agent", force=False)

def test_parse_date_relative():
    """Test relative date parsing."""
    now = datetime.now(UTC)

    result = parse_date("7d")
    assert (now - result).days == 7

    result = parse_date("1y")
    assert (now - result).days == 365

def test_parse_date_iso():
    """Test ISO date parsing."""
    result = parse_date("2026-01-01")
    assert result.year == 2026
    assert result.month == 1
    assert result.day == 1

@pytest.mark.asyncio
async def test_get_thread_history():
    """Test getting thread history."""
    # Setup: Create thread with messages
    # ...

    messages = await get_thread_history("test_thread")

    assert len(messages) > 0
    assert all('role' in m for m in messages)
    assert all('content' in m for m in messages)

@pytest.mark.asyncio
async def test_remove_before_dry_run():
    """Test dry-run doesn't actually delete."""
    # Setup: Create old threads
    # ...

    before = datetime.now(UTC) - timedelta(days=30)
    result = await remove_threads_before(before, dry_run=True)

    assert result['threads_deleted'] == 0  # Dry run doesn't delete
    assert 'threads' in result  # But returns what would be deleted

    # Verify threads still exist
    # ...
```

### Integration tests

```python
# tests/integration_ai/test_thread_management_commands.py

def test_remove_thread_command():
    """Test -rm command."""
    # Create test agent and thread
    # ...

    result = subprocess.run(
        ["deep-worker", "-rm", "test_thread", "--yes"],
        capture_output=True,
        text=True
    )

    assert result.returncode == 0
    assert "Deleted" in result.stdout

    # Verify thread is gone from database
    # ...

def test_history_command():
    """Test --history command."""
    # Create thread with messages
    # ...

    result = subprocess.run(
        ["deep-worker", "--history", "test_thread"],
        capture_output=True,
        text=True
    )

    assert result.returncode == 0
    assert "USER:" in result.stdout
    assert "ASSISTANT:" in result.stdout

def test_remove_before_command():
    """Test --remove-before command."""
    # Create old threads
    # ...

    result = subprocess.run(
        ["deep-worker", "--remove-before", "30d", "--dry-run"],
        capture_output=True,
        text=True
    )

    assert result.returncode == 0
    assert "Would delete" in result.stdout
```

## 📝 Documentation

### Help text

```bash
$ deep-worker --help

Thread Management:
  -rm THREAD_ID, --remove-thread THREAD_ID
                        Remove thread and all its checkpoints from database

  --history THREAD_ID   Show message history for thread (works offline)

  --remove-before DATE  Remove all threads updated before DATE
                        Formats: ISO 8601 (2026-01-01) or relative (30d, 1y)

  --dry-run             Show what would be deleted without actually deleting
                        (use with --remove-before)

  --yes, -y             Skip confirmation prompts

Options for --history:
  --limit N             Limit number of messages
  --offset N            Skip first N messages
  --format {text,json}  Output format (default: text)

Options for --remove-before:
  --agent NAME          Filter by agent name
```

### README examples

Add a section to README.md:

```markdown
## Thread Management

### Remove a thread

```bash
# Remove specific thread
deep-worker -rm d1:tm

# Skip confirmation
deep-worker -rm d1:tm --yes
```

### View thread history

```bash
# Show all messages
deep-worker --history d1

# Show last 10 messages
deep-worker --history d1 --limit 10

# JSON output
deep-worker --history d1 --format json
```

### Clean up old threads

```bash
# Preview (dry-run)
deep-worker --remove-before 30d --dry-run

# Delete threads older than 30 days
deep-worker --remove-before 30d

# Delete for specific agent
deep-worker --remove-before 7d --agent test_agent
```
```

## ⚠️ Potential problems

### 1. Performance with large databases

**Problem**: `--remove-before` may be slow when there are many threads.

**Solution**:
- Batch deletion (already in the design)
- An index on `updated_at`:
  ```sql
  CREATE INDEX IF NOT EXISTS idx_updated_at
  ON checkpoints (json_extract(metadata, '$.updated_at'));
  ```
- A progress bar for long operations

### 2. Concurrent access

**Problem**: The agent writes to the database while we are deleting.

**Solution**:
- SQLite uses file locks (automatically)
- Transactions protect against partial deletes
- A warning if the agent is running

### 3. Message history in different formats

**Problem**: The checkpoint structure may change between LangGraph versions.

**Solution**:
- Try/except during deserialization
- A fallback to the raw content if the structure is unknown
- Logging of parsing errors

### 4. Timezone confusion

**Problem**: Dates in different timezones.

**Solution**:
- Always use UTC internally
- Convert to the local timezone only for display
- State the timezone clearly in the output

## 📦 Affected files

### Changes

```
deep_worker_cli/
├── cli/
│   └── agent_command.py          # +150 lines (the new commands)
├── sessions.py                   # +300 lines (the new functions)
└── tests/
    ├── unit/
    │   └── test_thread_management.py      # NEW FILE
    └── integration_ai/
        └── test_thread_management_commands.py  # NEW FILE
```

### New functions in sessions.py

```python
# Date parsing
def parse_date(date_str: str) -> datetime

# Thread operations
async def remove_thread(thread_id: str, force: bool = False) -> dict
async def get_thread_history(
    thread_id: str,
    limit: int | None = None,
    offset: int = 0,
    role_filter: str | None = None
) -> list[dict]
async def remove_threads_before(
    before_date: datetime,
    agent_name: str | None = None,
    dry_run: bool = False
) -> dict

# Formatting
def format_history_text(messages: list[dict]) -> str
def format_history_json(messages: list[dict], thread_id: str) -> str
```

### New commands in agent_command.py

```python
async def remove_thread_command(args: argparse.Namespace) -> int
async def show_history_command(args: argparse.Namespace) -> int
async def remove_before_command(args: argparse.Namespace) -> int
```

## 🚀 Implementation plan

### Phase 1: `-rm` (1-2 hours)
1. Add the argument to argparse
2. Implement `remove_thread()` in sessions.py
3. Implement `remove_thread_command()` in agent_command.py
4. Unit tests
5. An integration test
6. Manual testing

### Phase 2: `--history` (2-3 hours)
1. Add the argument to argparse
2. Implement `get_thread_history()` in sessions.py
3. Implement the formatting (text/json)
4. Implement `show_history_command()` in agent_command.py
5. Unit tests
6. An integration test
7. Manual testing with different threads

### Phase 3: `--remove-before` (3-4 hours)
1. Add the arguments to argparse
2. Implement `parse_date()`
3. Implement `remove_threads_before()` in sessions.py
4. Implement `remove_before_command()` with a dry-run
5. Unit tests (including date parsing)
6. An integration test
7. Performance testing with a large database
8. Manual testing

### Phase 4: Documentation and finalization (1 hour)
1. Update README.md
2. Update the --help text
3. Add examples to the docstrings
4. Code review
5. Final testing

**Total time**: 7-10 hours

## ✅ Acceptance criteria

### `-rm`
- [ ] Deletes all the checkpoints of a thread
- [ ] Prints the number of deleted checkpoints
- [ ] Warns if the agent is running
- [ ] Works with the --yes flag
- [ ] Returns error code 1 if the thread is not found

### `--history`
- [ ] Shows all the messages in a thread
- [ ] Supports --limit and --offset
- [ ] Supports the text and json formats
- [ ] Works without a running agent
- [ ] Parses timestamps correctly

### `--remove-before`
- [ ] Parses ISO and relative dates
- [ ] Shows a dry-run preview
- [ ] Requires confirmation (unless --yes)
- [ ] Works with the --agent filter
- [ ] Prints deletion statistics
- [ ] Uses batch operations

### General
- [ ] All the tests pass
- [ ] The type hints are correct
- [ ] The documentation is complete
- [ ] The help text is clear
- [ ] The error messages are informative

---

**Document reference**: `docs/DESIGN_THREAD_MANAGEMENT_OPTIONS.md`

**Creation date**: 2026-03-29
**Status**: Ready for Review
