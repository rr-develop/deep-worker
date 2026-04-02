# Research: Asyncio-Native Approach in arpyc

## 📋 Task

Investigate the possibility of integrating an asyncio-native approach (from the arpyc `docs/RPyC_asyncio.txt` notes) into arpyc and verify:

1. Support for cross-process recursion
2. Transparency of async calls (no need to think about the RPC boundary)
3. Performance compared to BgServingThread
4. Execution of callbacks in the event loop

## 🎯 Research Results

### ✅ 1. Recursion WORKS without problems!

**Test:** `test_native_asyncio_recursion()`

```
✅ SUCCESS: Recursion completed in 0.0012s
   - Server B called 3 times
   - Client A called 3 times
   - Total: 6 recursive calls
   - NO DEADLOCK with asyncio-native approach!
```

**Execution flow:**
```
Test → await session.async_call(B.recursive_call, 5, callback)
  B(5) → callback A(4)
    A(4) → B.recursive_call(3)
      B(3) → callback A(2)
        A(2) → B.recursive_call(1)
          B(1) → callback A(0)
            A(0) → "Base case"
```

**Deep recursion (depth=10):**
```
✅ Deep recursion completed in 0.0022s
A calls: 5, B calls: 6, Total: 11
✅ NO DEADLOCK at depth=10 with asyncio-native!
```

### ✅ 2. Callbacks execute in the event loop (Main Thread)!

**Critical difference from BgServingThread:**

| Approach | Callback execution thread |
|--------|---------------------------|
| **BgServingThread** | Separate background thread (BgServingThread) |
| **Asyncio-Native** | **Main thread (event loop)** ✅ |

**Proof from the test:**

```
[Test] Main thread: 133255187216192 (MainThread)
[Test] Callback thread analysis:
  Unique threads: 1
    Thread 133255187216192: MAIN (event loop)

✅ All callbacks in MAIN THREAD (event loop) ✅
```

**This means:**
- ✅ asyncio can be used inside callbacks (via `asyncio.create_task()`)
- ✅ No thread safety problems
- ✅ No delays from switching between threads

### ✅ 3. Transparent async calls

**The API is identical regardless of whether the call is local or over RPC:**

```python
async with RpycAsyncSessionNative("127.0.0.1", port) as session:
    # Just await - no need to think about the RPC boundary
    result1 = await session.async_call(session.root.add, 10, 20)
    result2 = await session.async_call(session.root.multiply, 5, 7)

    # Callbacks work transparently
    result3 = await session.async_call(session.root.process, local_callback, 5)
```

**Transparency is achieved through:**
1. `rpyc.async_()` - immediate return of an AsyncResult
2. `await_rpyc()` - conversion into an asyncio.Future
3. `loop.add_reader()` - serving incoming RPC in the event loop

### 🚀 4. Performance - **156x FASTER!**

**Benchmark results:**

```
================================================================================
COMPARISON RESULTS:
================================================================================

Depth      BgThread (ms)        Native (ms)          Speedup
--------------------------------------------------------------------------------
3          301.15               2.59                 116.37x
5          301.36               1.27                 237.91x
10         302.09               1.94                 155.62x

================================================================================
CONCLUSIONS:
================================================================================
Average latency:
  BgServingThread: 301.53ms
  Asyncio-Native:  1.93ms
  Speedup:         156.08x ✅

Thread usage:
  BgServingThread: Main + BgServing (2 threads)
  Asyncio-Native:  Main only (1 thread)

Callback location:
  BgServingThread: Separate BgServing thread
  Asyncio-Native:  Main event loop thread

Polling:
  BgServingThread: sleep(0.1) every cycle  ← DELAY!
  Asyncio-Native:  No polling, event-driven  ← NO DELAYS!
```

**Why such a difference?**

1. **BgServingThread** has `sleep_interval=0.1` (100ms delay!)
   ```python
   while not self._stop_event.is_set():
       conn.serve(serve_interval=0.0)
       time.sleep(sleep_interval=0.1)  # ← DELAY!
   ```

2. **Asyncio-Native** uses an event-driven approach:
   ```python
   loop.add_reader(conn.fileno(), on_rpyc_readable)
   # Called IMMEDIATELY when data is ready!
   ```

## 📐 Architecture of the asyncio-native approach

### Operating principle

```
┌─────────────────────────────────────────────────────────────┐
│ MainThread (asyncio event loop)                             │
│                                                              │
│  1. await session.async_call(...)                           │
│     └─> rpyc.async_(fn)(*args)  ← Immediate return          │
│         └─> AsyncResult                                     │
│             └─> await_rpyc() → asyncio.Future               │
│                                                              │
│  2. loop.add_reader(conn.fileno(), on_rpyc_readable)        │
│     └─> Called when the RPyC socket is readable             │
│         └─> conn.poll(0)  ← Processes all transactions      │
│             └─> Incoming callbacks execute HERE!            │
│                                                              │
│  3. AsyncResult.add_callback(on_done)                       │
│     └─> loop.call_soon_threadsafe(future.set_result)        │
│         └─> await future returns the result                 │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Key components

1. **`loop.add_reader(conn.fileno(), on_rpyc_readable)`**
   - Registers a callback for the RPyC socket
   - The event loop invokes the callback when the socket is ready
   - WITHOUT polling delays!

2. **`conn.poll(0)` inside the callback**
   - `poll(0)` = process one transaction WITHOUT blocking
   - Returns `True` if something was processed, `False` if there was nothing
   - The loop `while conn.poll(0): pass` processes all available transactions

3. **`await_rpyc(AsyncResult) → asyncio.Future`**
   - Converts an RPyC AsyncResult into an asyncio.Future
   - Uses `AsyncResult.add_callback()` for notification
   - Uses `loop.call_soon_threadsafe()` for safe result delivery

## 🔧 Implementation

### Files

1. **`arpyc/session_native.py`**
   - New `RpycAsyncSessionNative` class
   - Uses `loop.add_reader()` instead of `BgServingThread`
   - API identical to `RpycAsyncSession`

2. **`arpyc/__init__.py`**
   - Added export of `RpycAsyncSessionNative`
   - Documentation about the two approaches
   - Both approaches are available

3. **Tests:**
   - `test_arpyc_native_recursion.py` - recursion, threads, transparent calls
   - `test_arpyc_comparison.py` - performance comparison

### Usage

```python
from arpyc import RpycAsyncSessionNative

async def main():
    # Connection
    async with RpycAsyncSessionNative("localhost", 18861) as session:
        # Transparent async calls
        result = await session.async_call(session.root.method, arg1, arg2)

        # With callbacks (for recursion)
        def callback(value, server_func):
            # Executes in the event loop (MainThread)!
            if value > 0:
                return server_func(value - 1, callback)
            return "done"

        result = await session.async_call(
            session.root.recursive_method,
            depth=5,
            callback=callback
        )
```

## 📊 Comparison of approaches

| Characteristic | BgServingThread | Asyncio-Native |
|---------------|-----------------|----------------|
| **Latency** | ~301ms (avg) | ~2ms (avg) ✅ |
| **Speedup** | Baseline | **156x faster** ✅ |
| **Threads** | 2 (Main + BgServing) | 1 (Main only) ✅ |
| **Callbacks thread** | BgServingThread | **Event loop (Main)** ✅ |
| **Polling delays** | sleep(0.1) every cycle | **No delays** (event-driven) ✅ |
| **Recursion** | ✅ Supported | ✅ **Supported** |
| **Platform** | All platforms | Unix-like (Windows limitations) |
| **Complexity** | Higher (2 threads) | Lower (1 thread) ✅ |

## ✅ Conclusions

### 1. The asyncio-native approach is FULLY WORKABLE

- ✅ Recursion works without deadlock
- ✅ Callbacks in the event loop (asyncio can be used)
- ✅ Transparent async calls
- ✅ **156x faster** than BgServingThread!

### 2. BgServingThread problems eliminated

**Problem:** *"I really dislike these delays in arpyc: `time.sleep(sleep_interval=0.1)`"*

**Solution:** ✅ In asyncio-native there are NO polling delays!
- Event-driven: `loop.add_reader()` is invoked ONLY when data is ready
- `conn.poll(0)` processes WITHOUT blocking
- Result: **156x speedup**!

**Problem:** *"It is not very clear why a separate connection-serving thread was made there"*

**Solution:** ✅ In asyncio-native there is NO separate thread!
- Everything runs in the event loop (MainThread)
- Callbacks execute in the same thread as asyncio
- `asyncio.create_task()` can safely be used inside callbacks

### 3. Transparency of async calls

**Goal:** *"Of particular interest are ways of transparently calling through async, where the calling side does not need to think at all about whether the specified call happens in the current process or in the process on the other side of the RPyC connection"*

**Result:** ✅ ACHIEVED!

```python
# Just await - no need to think about the RPC boundary!
result = await session.async_call(session.root.any_method, *args)

# Works with callbacks for recursion
result = await session.async_call(session.root.recursive, callback=local_fn)
```

### 4. Usage recommendations

**Use `RpycAsyncSessionNative` when:**
- ✅ Maximum performance is needed
- ✅ Callbacks in the event loop are needed
- ✅ Low latency matters
- ✅ Running on a Unix-like system (Linux, macOS)

**Use `RpycAsyncSession` (BgServingThread) when:**
- ⚠️ Windows support is required (add_reader is limited)
- ⚠️ Existing code already uses BgServingThread

**For new code:** `RpycAsyncSessionNative` is recommended (156x faster!)

## 🧪 Testing

### Running the tests

```bash
# Run from the CLI package directory

# Recursion tests
.venv/bin/python tests/integration_ai/test_arpyc_native_recursion.py

# Performance comparison
.venv/bin/python tests/integration_ai/test_arpyc_comparison.py

# Both tests together
.venv/bin/python tests/integration_ai/test_arpyc_native_recursion.py && \
.venv/bin/python tests/integration_ai/test_arpyc_comparison.py
```

### Test results

```
✅ test_native_asyncio_recursion: PASSED (0.0012s, depth=5)
✅ test_native_deep_recursion: PASSED (0.0022s, depth=10)
✅ test_bgservingthread_performance: PASSED (301ms avg)
✅ test_native_asyncio_performance: PASSED (2ms avg)
✅ Speedup: 156.08x
```

## 📚 Sources

1. **Document:** the arpyc `docs/RPyC_asyncio.txt` notes
   - Approach based on `loop.add_reader()`
   - The `conn.poll(0)` pattern
   - AsyncResult → asyncio.Future conversion

2. **RPyC documentation:**
   - https://rpyc.readthedocs.io/en/latest/api/core_protocol.html

3. **Python asyncio:**
   - https://docs.python.org/3/library/asyncio-eventloop.html
   - https://docs.python.org/3/library/asyncio-queue.html

## 🎉 Summary

The asyncio-native approach has been **SUCCESSFULLY INTEGRATED** into arpyc!

**Key achievements:**
- ✅ Recursion works (depth=10 tested)
- ✅ Callbacks in the event loop
- ✅ Transparent async calls
- ✅ **156x speedup** vs BgServingThread
- ✅ NO polling delays
- ✅ Optional (the old approach is preserved)

**Recommendation:** Use `RpycAsyncSessionNative` by default for all new projects.
