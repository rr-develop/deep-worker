# Design: Retry on transient ConnectionRefusedError in Execute tool

## Issue

**Issue**: the `execute` tool fails with `Connection refused` (reported via the internal issue tracker).

Symptom: the agent calls the `execute` tool and the command fails with `Connection refused`.
From the logs: `Execute tool fails with Connection refused`.

## Root cause analysis

### How the execute tool runs commands

`CLIShellBackend.execute()` (the CLI's `backends.py:127`) uses `asyncio.create_subprocess_shell()` — a direct local subprocess; it makes no TCP connections.

The subprocess exception is caught by a broad `except Exception` (line 266) and returned as:
```
Error executing command ({type(e).__name__}): {e}
```

### Where ConnectionRefusedError comes from

`asyncio.create_subprocess_shell()` cannot directly raise `ConnectionRefusedError` — it is a socket error. Possible sources:

1. **Sandbox provider** (Daytona/Modal/Runloop) — if the agent is configured for a remote sandbox, the `execute()` call makes an HTTP/TCP call to the provider API, which can return `ConnectionRefusedError` when the service is unavailable. However, the default backend is a local subprocess.

2. **Command inside the subprocess** — if the command (e.g. `curl`, `nc`) makes its own TCP connection, stdout/stderr may contain "Connection refused". The LLM then cites this as a problem with the execute tool.

3. **System failure creating the subprocess** — `os.fork()` or `os.pipe()` can raise `OSError` when resources are exhausted (EMFILE, EAGAIN). Although this is not `ConnectionRefusedError`, it can occasionally occur under system overload.

### Why this is a problem

The current code does not distinguish error types:
- Every `Exception` is turned into a string and returned to the LLM
- There is no retry mechanism for transient failures
- The LLM treats the error as final, even though a retry a second later could succeed

## Fix: Retry once for transient OSError in CLIShellBackend.execute()

### Changes

**File:** the CLI's `backends.py`

**Where:** `CLIShellBackend.execute()` (method, lines ~127-274)

**What we change:**

Wrap the subprocess creation + communicate block (lines ~180-264) in a retry loop:

```
for attempt in (1, 2):
    try:
        proc = await asyncio.create_subprocess_shell(...)
        stdout_bytes, stderr_bytes = await asyncio.wait_for(
            asyncio.shield(proc.communicate()), ...
        )
        ... (formatted result)
        break  # success → exit loop
    except OSError:
        if attempt == 1:
            logger.warning(
                "execute() subprocess creation failed (attempt 1/2), "
                "retrying after 0.5s: %s: %s",
                type(e).__name__, e,
            )
            await asyncio.sleep(0.5)
            continue  # retry once
        raise  # re-raise on second attempt → caught by outer handler
    except (asyncio.TimeoutError, Exception):
        raise  # don't retry on timeout or other exceptions
```

### Why OSError specifically?

- `asyncio.create_subprocess_shell` raises `OSError` when process creation fails (EMFILE, EAGAIN, ENOENT)
- `ConnectionRefusedError` is a subclass of `OSError` (errno.ECONNREFUSED)
- Other `Exception`s (ValueError, etc.) are not transient and retry would not help

### Why a 0.5s delay?

- Enough to recover from a temporary load spike
- Does not block the agent for long
- Well below the standard command timeout (120s)

### What we do NOT change

- The signature of the `execute()` method — the contract is not broken
- The `CLIShellBackend` class — the architecture is unchanged
- `sync_execute`/`async_execute` middleware — only the underlying backend
- Policies: no polling, no new threads, no 777/666 permissions

### Risks

- Retry can mask a persistent error (but 2 attempts is a standard threshold)
- The 0.5s delay is added only on failure; successful calls are unaffected
- An `OSError` on the second attempt propagates to the existing `except Exception` handler — behaviour identical to the current one

## Tests (TDD)

### Unit test (new file)

`tests/unit_tests/backends/test_execute_retry.py`:

```python
# Test 1: first call OSError → retry → success
#   Mock CLIShellBackend.execute() so that the first call to
#   create_subprocess_shell raises OSError, the second succeeds.
#   Assert: the result is an ExecuteResponse with exit_code=0.

# Test 2: both calls OSError → error returned
#   Mock create_subprocess_shell so it raises OSError both times.
#   Assert: the result is an ExecuteResponse with exit_code=1 and "Error executing command".

# Test 3: ValueError (not OSError) → no retry
#   Mock raises ValueError.
#   Assert: create_subprocess_shell is called exactly once.

# Test 4: logger.warning is called on retry
#   Mock create_subprocess_shell so it raises OSError the first time.
#   Assert: logger.warning is called once with the expected message.
#   Assert: asyncio.sleep is called with 0.5.
```
