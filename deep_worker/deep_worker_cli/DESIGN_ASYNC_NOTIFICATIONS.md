# Technical design: Asynchronous notifications through RPyC

**Date:** 2026-03-12
**Version:** 1.0
**Status:** Approved for implementation

---

## Problem

### Symptoms
- In the dwcli TUI, when slow tools are called (sleep 10, web_search), the tool call is **NOT displayed right away**
- The call appears only **AFTER the tool execution finishes** (10+ seconds later)
- All the messages (tool_call + tool_result + response) appear at the same time

### Root cause

**A blocking call chain:**

```
agent.astream() [async]
  ↓
async for chunk in stream:
  ↓
await process_message_chunk() [async]
  ↓
notify_callback(message) [SYNC!]
  ↓
_notify_subscribers(message) [SYNC!]
  ↓
for subscriber in subscribers:
    subscriber.on_message(message) [BLOCKING RPyC CALL!]
```

**The critical point:** `subscriber.on_message(message)` is a **synchronous RPyC call over the network**.

### Detailed analysis of the blocking

**File:** `rpyc_service/agent_service.py`

**Lines 352-356:**
```python
for subscriber in subscribers:
    try:
        subscriber.on_message(message)  # ← a BLOCKING call!
    except Exception as e:
        logger.exception(f"Error notifying subscriber {subscriber}: {e}")
```

**What happens:**
1. The stream loop generates a chunk with a `tool_call`
2. `process_message_chunk()` parses the chunk and calls `notify_callback()`
3. `_notify_subscribers()` synchronously calls `subscriber.on_message()`
4. **RPyC makes a network call** to the client (the dwcli TUI)
5. **The stream loop BLOCKS** until the client's response comes back
6. If the next chunk is the start of a slow tool execution, the loop is stuck for 10+ seconds
7. All the subsequent chunks are **not processed** until the tool finishes

### Comparison with the upstream reference project

**In deepagents (works correctly):**

**File:** the upstream reference project's `textual_adapter.py`

**Lines ~560:**
```python
# Mount tool call message IMMEDIATELY
tool_msg = ToolCallMessage(buffer_name, parsed_args)
await adapter._mount_message(tool_msg)  # ← a DIRECT async call
```

**The key difference:**
- deepagents: a direct `await` call inside the same process
- this project: a synchronous RPyC call over the network, with blocking

---

## Solution

### The principle
**ALL notification calls MUST be asynchronous**, regardless of whether RPyC is used.

The stream loop must NOT block while sending notifications to clients.

### Architecture of the solution

#### Option 1: Thread-based Fire-and-Forget (RECOMMENDED)

Run the RPyC calls in separate threads so that the stream loop can continue immediately.

**Advantages:**
- Minimal code changes
- Does not require changing the RPyC protocol
- Fully non-blocking sending

**Disadvantages:**
- A small overhead for creating threads
- Message delivery order has to be guaranteed

#### Option 2: asyncio.create_task (ALTERNATIVE)

Use asyncio tasks to send notifications in parallel.

**Advantages:**
- Native asyncio integration
- Less overhead than threads

**Disadvantages:**
- Requires changing `_notify_subscribers` to async
- Async has to be propagated through the whole call stack
- Harder with RPyC (which runs in a separate event loop)

---

## Detailed design (Option 1)

### Changes in the files

#### 1. `rpyc_service/agent_service.py`

**Current code (lines 325-357):**
```python
def _notify_subscribers(self, message: dict[str, Any]) -> None:
    """Notify all subscribers about a new message."""
    # ... debug logging ...

    with self._subscribers_lock:
        subscribers = list(self._message_subscribers)

    for subscriber in subscribers:
        try:
            subscriber.on_message(message)  # ← BLOCKS
        except Exception as e:
            logger.exception(f"Error notifying subscriber {subscriber}: {e}")
```

**The NEW code:**
```python
import threading
from queue import Queue
from typing import Any

def _notify_subscribers(self, message: dict[str, Any]) -> None:
    """Notify all subscribers about a new message.

    CRITICAL: This method is called from async stream loop and MUST NOT block.
    RPyC calls to subscribers are executed in separate threads to prevent
    blocking the stream processing.

    Args:
        message: Message dict to send to subscribers
    """
    # DEBUG: Log notification
    import os
    debug_enabled = os.environ.get("DWCLI_DEBUG_CALLBACKS") == "1"
    if debug_enabled:
        import traceback
        content = message.get("content", "")[:50]
        with open("/tmp/dwcli_callback_debug.log", "a") as f:
            f.write(f"\n[AGENT_DEBUG] _notify_subscribers called:\n")
            f.write(f"  message: role={message.get('role')}, content={content}...\n")
            f.write(f"  Stack trace:\n")
            for line in traceback.format_stack()[:-1]:
                f.write(f"    {line.strip()}\n")

    # Get snapshot of current subscribers under lock
    with self._subscribers_lock:
        subscribers = list(self._message_subscribers)

    if debug_enabled:
        with open("/tmp/dwcli_callback_debug.log", "a") as f:
            f.write(f"  subscribers count: {len(subscribers)}\n")

    # Notify each subscriber in a separate thread (fire-and-forget)
    # This prevents blocking the stream loop on RPyC network calls
    for subscriber in subscribers:
        def _notify_single_subscriber(sub=subscriber, msg=message):
            """Thread worker to notify a single subscriber."""
            try:
                sub.on_message(msg)
            except Exception as e:
                logger.exception(f"Error notifying subscriber {sub}: {e}")

        # Start daemon thread for non-blocking notification
        thread = threading.Thread(
            target=_notify_single_subscriber,
            daemon=True,
            name=f"notify-{id(subscriber)}"
        )
        thread.start()

        # IMPORTANT: We do NOT thread.join() here!
        # Stream loop continues immediately without waiting for RPyC to complete
```

**Critical points:**
1. `daemon=True` - the thread terminates when the process terminates
2. We create a separate thread for **EACH** subscriber
3. We do **NOT wait** for the thread to finish (no `thread.join()`)
4. The stream loop continues **immediately**

### Problem: Message ordering

**A potential problem:** If we create a new thread for every notification, the delivery order may be broken.

**Solution:** A thread pool with a queue per subscriber.

#### An improved version with an ordering guarantee:

**The NEW code (with an ordering guarantee):**
```python
import threading
from queue import Queue
from typing import Any, Dict

class AgentService(rpyc.Service):
    def __init__(self, ...):
        # ... existing initialization ...

        # Thread pool for async notifications
        # Each subscriber gets its own queue + worker thread
        self._subscriber_queues: Dict[Any, Queue] = {}
        self._subscriber_threads: Dict[Any, threading.Thread] = {}
        self._subscriber_queue_lock = threading.Lock()

    def _ensure_subscriber_worker(self, subscriber: Any) -> None:
        """Ensure worker thread exists for this subscriber.

        Args:
            subscriber: Subscriber object (RPyC proxy)
        """
        with self._subscriber_queue_lock:
            if subscriber not in self._subscriber_queues:
                # Create queue for this subscriber
                queue = Queue()
                self._subscriber_queues[subscriber] = queue

                # Create worker thread
                def worker():
                    """Worker thread that processes messages for one subscriber."""
                    while True:
                        try:
                            message = queue.get()

                            # None is poison pill - exit thread
                            if message is None:
                                break

                            # Send message via RPyC (blocks this worker thread only)
                            try:
                                subscriber.on_message(message)
                            except Exception as e:
                                logger.exception(f"Error notifying subscriber {subscriber}: {e}")

                            queue.task_done()
                        except Exception as e:
                            logger.exception(f"Worker thread error: {e}")

                thread = threading.Thread(
                    target=worker,
                    daemon=True,
                    name=f"subscriber-worker-{id(subscriber)}"
                )
                thread.start()
                self._subscriber_threads[subscriber] = thread

    def _notify_subscribers(self, message: dict[str, Any]) -> None:
        """Notify all subscribers about a new message.

        CRITICAL: This method is called from async stream loop and MUST NOT block.
        Messages are queued and sent by dedicated worker threads per subscriber.
        This guarantees:
        1. Stream loop never blocks on RPyC calls
        2. Message order is preserved per subscriber
        3. Messages to different subscribers are sent in parallel

        Args:
            message: Message dict to send to subscribers
        """
        # DEBUG: Log notification
        import os
        debug_enabled = os.environ.get("DWCLI_DEBUG_CALLBACKS") == "1"
        if debug_enabled:
            import traceback
            content = message.get("content", "")[:50]
            with open("/tmp/dwcli_callback_debug.log", "a") as f:
                f.write(f"\n[AGENT_DEBUG] _notify_subscribers called:\n")
                f.write(f"  message: type={message.get('type')}, content={content}...\n")
                f.write(f"  Stack trace:\n")
                for line in traceback.format_stack()[:-1]:
                    f.write(f"    {line.strip()}\n")

        # Get snapshot of current subscribers under lock
        with self._subscribers_lock:
            subscribers = list(self._message_subscribers)

        if debug_enabled:
            with open("/tmp/dwcli_callback_debug.log", "a") as f:
                f.write(f"  subscribers count: {len(subscribers)}\n")

        # Queue message for each subscriber (NON-BLOCKING)
        for subscriber in subscribers:
            # Ensure worker thread exists for this subscriber
            self._ensure_subscriber_worker(subscriber)

            # Add message to subscriber's queue
            # This is NON-BLOCKING - queue.put() with unlimited queue
            with self._subscriber_queue_lock:
                queue = self._subscriber_queues.get(subscriber)
                if queue is not None:
                    queue.put(message)

                    if debug_enabled:
                        with open("/tmp/dwcli_callback_debug.log", "a") as f:
                            f.write(f"  Queued message for subscriber {id(subscriber)}, queue size: {queue.qsize()}\n")

        # CRITICAL: Method returns IMMEDIATELY
        # Worker threads handle actual RPyC calls asynchronously

    def unsubscribe(self, subscriber):
        """Unsubscribe from agent messages."""
        with self._subscribers_lock:
            if subscriber in self._message_subscribers:
                self._message_subscribers.remove(subscriber)

        # Clean up worker thread and queue
        with self._subscriber_queue_lock:
            queue = self._subscriber_queues.pop(subscriber, None)
            thread = self._subscriber_threads.pop(subscriber, None)

            if queue is not None:
                # Send poison pill to stop worker
                queue.put(None)

            if thread is not None:
                # Give thread a moment to finish
                thread.join(timeout=1.0)
```

**Architecture:**
```
Stream Loop (async) → _notify_subscribers() [does NOT block!]
                            ↓
                      For each subscriber:
                            ↓
                      queue.put(message) [does NOT block!]
                            ↓
                      Worker Thread (one per subscriber)
                            ↓
                      subscriber.on_message(message) [blocks ONLY the worker thread]
```

**Guarantees:**
1. ✅ The stream loop **NEVER** blocks on RPyC
2. ✅ Message order is **preserved** for each subscriber
3. ✅ Messages to different subscribers are sent **in parallel**
4. ✅ If a subscriber is slow, its queue grows, but the stream does not block

---

### 2. Changes in `stream_processor.py` (NOT required!)

**The current code works correctly:**
```python
async def process_message_chunk(
    message: AIMessage | ToolMessage | HumanMessage,
    metadata: dict[str, Any],
    notify_callback: Callable[[dict[str, Any]], None],  # ← stays SYNC
    tool_call_buffer: Optional[dict[str, Any]] = None,
) -> None:
```

`notify_callback` is called as:
```python
notify_callback({
    "type": "tool_call",
    "name": tool_name,
    "args": args_json,
    "id": call_id,
})
```

After our fix, `notify_callback` (which is `_notify_subscribers`) will be **non-blocking** - it will simply put the message into a queue and return immediately.

---

### 3. Changes in `agent_process.py` (NOT required!)

**Current code:**
```python
async for chunk in self.agent.astream(...):
    # ...
    if notify_callback is not None:
        await process_message_chunk(
            message=message_obj,
            metadata=metadata,
            notify_callback=notify_callback,
            tool_call_buffer=tool_call_buffer,
        )
```

After the fix:
- `await process_message_chunk()` is still async
- Inside it, `notify_callback()` is called (sync, but now non-blocking)
- The stream loop continues immediately

---

## Testing

### Unit tests

**File:** `tests/unit/test_async_notifications.py`

```python
"""Unit tests for asynchronous notification system."""

import threading
import time
from queue import Queue
from unittest.mock import MagicMock, patch

import pytest


def test_notify_subscribers_does_not_block():
    """Verify _notify_subscribers returns immediately even with slow subscribers."""
    from deep_worker_cli.rpyc_service.agent_service import AgentService

    # Create mock subscriber that blocks for 5 seconds
    slow_subscriber = MagicMock()
    def slow_on_message(msg):
        time.sleep(5.0)

    slow_subscriber.on_message = slow_on_message

    # Create service and add subscriber
    service = AgentService(...)
    service._message_subscribers.add(slow_subscriber)

    # Measure time to call _notify_subscribers
    start = time.time()
    service._notify_subscribers({"type": "test", "content": "hello"})
    elapsed = time.time() - start

    # Should return immediately (< 0.1 seconds)
    assert elapsed < 0.1, f"_notify_subscribers blocked for {elapsed}s"

    # Give worker thread time to process
    time.sleep(6.0)

    # Verify message was delivered
    slow_subscriber.on_message.assert_called_once()


def test_message_order_preserved():
    """Verify messages are delivered in order to each subscriber."""
    from deep_worker_cli.rpyc_service.agent_service import AgentService

    # Create subscriber that records message order
    received_messages = []
    subscriber = MagicMock()
    def record_message(msg):
        received_messages.append(msg["index"])
        time.sleep(0.01)  # Small delay to test ordering

    subscriber.on_message = record_message

    # Create service and add subscriber
    service = AgentService(...)
    service._message_subscribers.add(subscriber)

    # Send 10 messages rapidly
    for i in range(10):
        service._notify_subscribers({"type": "test", "index": i})

    # Wait for all to be delivered
    time.sleep(1.0)

    # Verify order
    assert received_messages == list(range(10))


def test_parallel_delivery_to_multiple_subscribers():
    """Verify slow subscriber doesn't delay other subscribers."""
    from deep_worker_cli.rpyc_service.agent_service import AgentService

    # Create fast and slow subscribers
    fast_times = []
    slow_times = []

    fast_subscriber = MagicMock()
    def fast_on_message(msg):
        fast_times.append(time.time())
    fast_subscriber.on_message = fast_on_message

    slow_subscriber = MagicMock()
    def slow_on_message(msg):
        time.sleep(1.0)  # Slow subscriber
        slow_times.append(time.time())
    slow_subscriber.on_message = slow_on_message

    # Create service and add both subscribers
    service = AgentService(...)
    service._message_subscribers.add(fast_subscriber)
    service._message_subscribers.add(slow_subscriber)

    # Send message
    start = time.time()
    service._notify_subscribers({"type": "test", "content": "hello"})

    # Wait for delivery
    time.sleep(2.0)

    # Fast subscriber should receive immediately
    assert len(fast_times) == 1
    assert fast_times[0] - start < 0.1

    # Slow subscriber receives after 1 second, but didn't block fast one
    assert len(slow_times) == 1
    assert slow_times[0] - start >= 1.0
```

### Integration test (existing)

**File:** `tests/integration_ai/test_slow_tool_display.py`

After the fix, this test must **pass**:

```python
@pytest.mark.asyncio
async def test_slow_tool_call_displays_immediately(slow_tool_agent):
    """Tool call message MUST appear in TUI immediately, not after tool completes."""
    # ... existing test code ...

    # Send query requesting sleep tool
    query = (
        "Please call the bash tool to execute 'sleep 10' command, "
        "then say hello to me. Make sure you use the bash tool first."
    )
    session.send_input(query, press_enter=True)

    # CRITICAL: Wait only 1 second, then take screenshot
    time.sleep(1.0)

    # Capture screenshot
    screenshot = session.get_screen_text()

    # Use AI to analyze the screenshot
    result = await verify_screenshot_contains(
        screenshot_text=screenshot,
        original_query=query,
        expected_response_contains="bash",
        model="haiku",
    )

    # Should PASS - tool call visible within 1 second
    assert result.is_valid, "Tool call should appear immediately"
```

---

## Potential problems and solutions

### Problem 1: Queue overflow

**Symptom:** If a subscriber is very slow (or has disconnected), its queue can grow indefinitely.

**Solution:** Add a limit on the queue size:

```python
# In _ensure_subscriber_worker:
queue = Queue(maxsize=1000)  # At most 1000 messages in the queue

# In _notify_subscribers:
try:
    queue.put(message, block=False)  # Non-blocking put
except Full:
    logger.warning(f"Subscriber {id(subscriber)} queue full, dropping message")
```

### Problem 2: A memory leak when a subscriber disconnects

**Symptom:** The worker threads and queues are not cleaned up when a subscriber disconnects.

**Solution:** Already implemented in the `unsubscribe()` method above.

### Problem 3: Errors at shutdown

**Symptom:** When the process terminates, the worker threads may raise errors.

**Solution:** A graceful shutdown in `on_disconnect`:

```python
def on_disconnect(self, conn):
    """Called when a client disconnects."""
    # ... existing cleanup ...

    # Stop all worker threads
    with self._subscriber_queue_lock:
        for subscriber, queue in list(self._subscriber_queues.items()):
            queue.put(None)  # Poison pill

        for thread in self._subscriber_threads.values():
            thread.join(timeout=1.0)

        self._subscriber_queues.clear()
        self._subscriber_threads.clear()
```

---

## Success metrics

After the fix is applied:

1. ✅ The `test_slow_tool_call_displays_immediately` test must **pass**
2. ✅ The tool call must appear in the TUI within **< 1 second**
3. ✅ The stream loop must not block on RPyC calls
4. ✅ Message order must be preserved
5. ✅ There must be no memory or thread leaks

---

## Implementation plan

### Stage 1: The basic fix (1 hour)
1. Change `agent_service.py` - add worker threads
2. Add `_ensure_subscriber_worker()`
3. Change `_notify_subscribers()` to the non-blocking version
4. Change `unsubscribe()` for cleanup

### Stage 2: Testing (1 hour)
1. Write unit tests
2. Run the `test_slow_tool_display.py` integration test
3. Check that there are no regressions in the existing tests

### Stage 3: Optimization (optional)
1. Add a limit on the queue size
2. Add metrics (queue sizes, delays)
3. Graceful shutdown

---

## Alternative approaches (rejected)

### Alternative 1: Make notify_callback async

**Approach:** Change the signature to `async def process_message_chunk(..., notify_callback: Callable[[dict], Awaitable[None]])`

**Rejected because:**
- It requires changes throughout the whole call stack
- RPyC runs in a separate event loop (complex integration)
- The thread-based solution is simpler and more reliable

### Alternative 2: Use asyncio.to_thread()

**Approach:** `await asyncio.to_thread(subscriber.on_message, message)`

**Rejected because:**
- It does not guarantee message order
- It requires async everywhere
- Debugging is harder

### Alternative 3: Batching messages

**Approach:** Collect several messages and send them as a batch.

**Rejected because:**
- It adds latency
- It complicates the code
- It does not solve the blocking problem

---

## Conclusion

The proposed solution:
- ✅ Minimally invasive (a single file)
- ✅ Does not require changing the RPyC protocol
- ✅ Guarantees message order
- ✅ Fully non-blocking
- ✅ Analogous to the deepagents behavior (asynchronous processing)

Once implemented, this project will behave **identically** to deepagents with respect to message streaming, regardless of whether RPyC is used.

---

**Document reference:** `DESIGN_ASYNC_NOTIFICATIONS.md`
