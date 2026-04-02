# Deep Worker `--list` Option Design Document

## Overview

The `--list` (alias `-ls`) option for `deep-worker` command displays a complete list of all conversation threads stored in the sessions database (`~/.deepagents/sessions.db`). This provides users with visibility into all existing conversation threads across all agents.

## Command Syntax

```bash
deep-worker --list [OPTIONS]
deep-worker -ls [OPTIONS]
```

**Note:** The `--list` option is mutually exclusive with running an agent. When `--list` is provided, no agent name is required or accepted.

### Options

- `--list` or `-ls`: Display list of all threads (primary option)
- `--format <FORMAT>`: Output format: `text` (default), `json`
- `--with-counts`: Include message counts (slow operation, disabled by default)
- `--agent <NAME>`: Filter threads by agent name (optional)
- `--sort <FIELD>`: Sort by field: `thread_id` (default), `agent`, `updated`
- `--reverse`: Reverse sort order (default: alphabetical/chronological ascending)

### Examples

```bash
# List all threads (default text format, sorted alphabetically by thread_id)
deep-worker --list

# List threads for specific agent
deep-worker --list --agent myagent

# List sorted by last updated time
deep-worker --list --sort updated

# List with message counts (slower)
deep-worker --list --with-counts

# JSON output for scripting
deep-worker --list --format json
```

## Output Format

### Default (Text) Format

One thread per line, columns separated by spaces, aligned for readability:

**Without `--with-counts` (default, fast):**

```
Thread ID               Agent Name          Last Updated
------------------------------------------------------------------------
my_test_agent           (self)              Mar 29, 5:30pm
my_test_agent_tm        (self)              Mar 29, 5:30pm
test_agent_final        (self)              Mar 28, 2:15pm
0204440a                agent               Mar 16, 8:05pm
02e616b4                agent               Mar 16, 4:46pm
```

**With `--with-counts` (slower):**

```
Thread ID               Agent Name          Messages  Last Updated
--------------------------------------------------------------------------------
my_test_agent           (self)              42        Mar 29, 5:30pm
my_test_agent_tm        (self)              15        Mar 29, 5:30pm
test_agent_final        (self)              8         Mar 28, 2:15pm
0204440a                agent               156       Mar 16, 8:05pm
02e616b4                agent               23        Mar 16, 4:46pm
```

**Column Descriptions:**

1. **Thread ID** (required, first column):
   - Primary identifier for the thread
   - Can be human-readable name (e.g., `my_test_agent`) or hex ID (e.g., `0204440a`)
   - Width: 24 characters (left-aligned)

2. **Agent Name**:
   - Extracted from `metadata.agent_name` in checkpoint
   - Shows `(self)` when `thread_id == agent_name` (new format)
   - Shows actual agent name for legacy hex thread_ids
   - Shows empty string if NULL
   - Width: 20 characters (left-aligned)

3. **Messages** (only with `--with-counts`):
   - Total count of messages in the thread
   - Extracted from latest checkpoint blob (expensive operation)
   - Width: 8 characters (right-aligned)

4. **Last Updated**:
   - Timestamp of most recent checkpoint
   - Format: `MMM DD, H:MMam/pm` (e.g., `Mar 29, 5:30pm`)
   - Width: 20 characters (left-aligned)

### JSON Format

Structured JSON array for programmatic consumption:

**Without `--with-counts` (default):**

```json
{
  "threads": [
    {
      "thread_id": "my_test_agent",
      "agent_name": null,
      "updated_at": "2026-03-29T17:30:15.123456+00:00"
    },
    {
      "thread_id": "my_test_agent_tm",
      "agent_name": null,
      "updated_at": "2026-03-29T17:30:10.987654+00:00"
    },
    {
      "thread_id": "0204440a",
      "agent_name": "agent",
      "updated_at": "2026-03-16T20:05:27.385354+00:00"
    }
  ],
  "total": 3,
  "sort_by": "thread_id",
  "filter_agent": null
}
```

**With `--with-counts`:**

```json
{
  "threads": [
    {
      "thread_id": "my_test_agent",
      "agent_name": null,
      "message_count": 42,
      "updated_at": "2026-03-29T17:30:15.123456+00:00"
    },
    {
      "thread_id": "my_test_agent_tm",
      "agent_name": null,
      "message_count": 15,
      "updated_at": "2026-03-29T17:30:10.987654+00:00"
    },
    {
      "thread_id": "0204440a",
      "agent_name": "agent",
      "message_count": 156,
      "updated_at": "2026-03-16T20:05:27.385354+00:00"
    }
  ],
  "total": 3,
  "sort_by": "thread_id",
  "filter_agent": null
}
```

## Sorting Behavior

### Default Sort: Alphabetical by Thread ID

- Primary sort key: `thread_id` (case-insensitive alphabetical)
- Threads are sorted lexicographically
- Example order: `0204440a`, `my_test_agent`, `my_test_agent_tm`, `test_agent_final`

### Sort Options

1. **`--sort thread_id`** (default):
   - Alphabetical by thread_id
   - Case-insensitive comparison

2. **`--sort agent`**:
   - Sort by agent name from metadata
   - Threads with same agent name: secondary sort by thread_id
   - NULL agent names (new format) appear first

3. **`--sort updated`**:
   - Sort by last updated timestamp
   - Most recent first (descending) by default
   - Use `--reverse` for oldest first

### Reverse Flag

- `--reverse`: Inverts the sort order
- Applies to all sort modes
- Example: `--sort thread_id --reverse` = reverse alphabetical (Z→A)

### Performance Note

By default, `--list` is **fast** because it only queries thread metadata without deserializing checkpoint blobs. Message counting (`--with-counts`) requires deserializing every checkpoint blob and is significantly slower for large databases.

## Technical Implementation

### Database Query

```sql
-- Base query structure
SELECT
    thread_id,
    json_extract(metadata, '$.agent_name') as agent_name,
    MAX(json_extract(metadata, '$.updated_at')) as updated_at
FROM checkpoints
WHERE 1=1  -- Dynamic filter conditions added here
GROUP BY thread_id
ORDER BY thread_id ASC;  -- Sort dynamically based on --sort option
```

### Data Retrieval Process

1. **Connect to Database**:
   - Path: `~/.deepagents/sessions.db`
   - Use `aiosqlite` for async operations
   - Timeout: 30 seconds

2. **Check Table Existence**:
   - Verify `checkpoints` table exists
   - If not: display "No threads found" message and exit cleanly

3. **Query Threads**:
   - No LIMIT clause (fetch ALL threads)
   - Apply optional `--agent` filter in WHERE clause
   - Group by `thread_id` to get unique threads
   - Extract metadata: `agent_name`, `updated_at`

4. **Fetch Message Counts** (only if `--with-counts`):
   - For each thread, retrieve latest checkpoint blob
   - Deserialize using `JsonPlusSerializer`
   - Extract `len(channel_values['messages'])`
   - This is an expensive operation, disabled by default

5. **Sort Results**:
   - Apply sorting in Python after fetching (allows complex sort logic)
   - Sort keys:
     - `thread_id`: `thread.lower()` for case-insensitive
     - `agent`: `(agent or '').lower()` + `thread_id.lower()` secondary
     - `updated`: `updated_at` timestamp

6. **Format Output**:
   - Text: Format with fixed-width columns and padding
   - JSON: Use `json.dumps()` with pretty printing

### Module Location

- **File**: `deep_worker_cli/cli/agent_command.py`
- **Function**: `async def list_threads_command(args: argparse.Namespace) -> int`
- **Helper**: Reuse existing `sessions.py::list_threads()` with modifications

### Argument Parser Changes

```python
# In create_parser():
parser.add_argument(
    "--list", "-ls",
    action="store_true",
    help="List all conversation threads in database",
)

parser.add_argument(
    "--format",
    choices=["text", "json"],
    default="text",
    help="Output format for --list (default: text)",
)

parser.add_argument(
    "--with-counts",
    action="store_true",
    dest="with_counts",
    help="Include message counts (slow, for --list only)",
)

parser.add_argument(
    "--agent",
    metavar="NAME",
    help="Filter by agent name (for --list only)",
)

parser.add_argument(
    "--sort",
    choices=["thread_id", "agent", "updated"],
    default="thread_id",
    help="Sort field for --list (default: thread_id)",
)

parser.add_argument(
    "--reverse",
    action="store_true",
    help="Reverse sort order for --list",
)
```

### Error Handling

1. **Database Not Found**:
   - Message: "No sessions database found. Start an agent first."
   - Exit code: 0 (not an error, just no data)

2. **Database Locked**:
   - Retry with exponential backoff (3 attempts)
   - Message: "Database is locked. Another agent may be starting."
   - Exit code: 1 (temporary error)

3. **Corrupted Checkpoint**:
   - Log warning with thread_id
   - Skip thread and continue with others
   - Display count: "X threads listed (Y failed to load)"

4. **Empty Database**:
   - Message: "No threads found in database."
   - Exit code: 0

### Performance Considerations

1. **Default Mode (without `--with-counts`)**:
   - Fast: Only queries metadata from checkpoints table
   - No blob deserialization required
   - Suitable for regular use

2. **With Counts Mode (`--with-counts`)**:
   - Slow: Requires deserializing every checkpoint blob
   - For large databases (>100 threads), may take several seconds
   - Only use when message counts are actually needed

3. **Memory Usage**:
   - Metadata only: Minimal (~1KB per thread)
   - With counts: ~50KB per thread for checkpoint deserialization
   - Acceptable for typical use cases (<100 threads)

## Integration with Existing Code

### Reuse Existing Functions

- `sessions.py::get_db_path()`: Get database path
- `sessions.py::_table_exists()`: Check table existence
- `sessions.py::list_threads()`: Base query logic (modify to remove LIMIT)
- `sessions.py::_count_messages_from_checkpoint()`: Message counting (only if `--with-counts`)

### New Functions Required

```python
# In sessions.py
async def list_all_threads(
    agent_name: str | None = None,
    sort_by: str = "thread_id",
    reverse: bool = False,
    include_message_count: bool = False,
) -> list[dict]:
    """List ALL threads without pagination.

    Returns list of dicts with:
    - thread_id: str
    - agent_name: str | None
    - updated_at: str (ISO timestamp)
    - message_count: int (only if include_message_count=True)
    """
    pass

# In cli/agent_command.py
async def list_threads_command(args: argparse.Namespace) -> int:
    """Handle --list command."""
    pass

def format_threads_text(threads: list[dict], with_counts: bool) -> str:
    """Format threads as text table."""
    pass

def format_threads_json(threads: list[dict], sort_by: str, filter_agent: str | None) -> str:
    """Format threads as JSON."""
    pass
```

## Exit Codes

- `0`: Success (threads listed or no threads found)
- `1`: Error (database locked, permission denied, etc.)
- `2`: Invalid arguments (conflicting options)

## Validation Rules

1. **Mutually Exclusive Options**:
   - `--list` cannot be used with agent name argument
   - Error: "Cannot specify agent name with --list option"

2. **Format-Specific Options**:
   - `--format`, `--sort`, `--reverse`, `--with-counts` only valid with `--list`
   - Error: "Option --format requires --list"

3. **Agent Filter**:
   - `--agent` only valid with `--list`
   - No validation on agent name (silently returns empty if not found)

## Future Enhancements (Out of Scope)

- Pagination support (`--limit`, `--offset`) - explicitly NOT implemented per requirements
- Filtering by date range (`--since`, `--before`)
- Regular expression filtering (`--filter-regex`)
- Export to CSV format
- Interactive mode for thread deletion
- Display first/last message preview
- Sort by message count (requires `--with-counts`, not implemented initially)

## Testing Checklist

- [ ] Empty database (no checkpoints table)
- [ ] Database with 0 threads
- [ ] Database with 1 thread
- [ ] Database with 100+ threads (performance)
- [ ] Mix of new format (readable names) and old format (hex IDs)
- [ ] Threads with and without agent_name metadata
- [ ] Corrupted checkpoint blobs (with `--with-counts`)
- [ ] All sort modes (thread_id, agent, updated)
- [ ] Reverse sort flag
- [ ] All output formats (text, json)
- [ ] With and without `--with-counts` flag
- [ ] Agent filtering (existing and non-existing agent)
- [ ] Database locked scenario
- [ ] Missing database file
- [ ] Concurrent access during agent operation

## Security Considerations

- Read-only database access (no modifications)
- SQL injection prevented by parameterized queries
- File path validation for database location
- No sensitive data exposure in output (only metadata)

## Backward Compatibility

- Does not affect existing agent functionality
- Does not modify database schema
- Does not require migration of existing data
- Works with both old (hex) and new (readable) thread_id formats

## Documentation Requirements

1. Update `README.md` with `--list` option
2. Add examples to command help text
3. Document in user guide (if exists)
4. Add to CLI reference documentation

## References

- Existing code: `deep_worker_cli/sessions.py::list_threads()`
- Database schema: `~/.deepagents/sessions.db`
- Related option: `--clean` (CLEAN_OPTION.md)
- Command structure: `deep_worker_cli/cli/agent_command.py`
