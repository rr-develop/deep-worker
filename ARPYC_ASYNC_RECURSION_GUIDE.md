# Guide: Asynchronous Recursion over RPyC with arpyc

## 📋 Question

**How do you implement recursion over RPyC when both the client and the server functions are asynchronous (async/await)?**

## 🎯 Short answer

**Server-side exposed_ function:**
- ✅ A SYNC wrapper around the async logic
- ✅ Uses `asyncio.run()` to run the async function
- ✅ Works perfectly (RPyC calls it in a separate thread)

**Client-side callback function:**
- ⚠️ **PROBLEM:** With asyncio-native arpyc the callback executes in the event loop
- ❌ Cannot use `asyncio.run()` (already inside an event loop)
- ❌ Cannot use `await` (not an async function)

**Solutions:**
1. ✅ **RECOMMENDED:** Sync callbacks (without async logic)
2. ✅ `nest_asyncio` (allows nested event loops)
3. ✅ BgServingThread (callbacks in a separate thread)

---

## 🔬 Detailed analysis

### Architecture of a recursive call

```
┌─────────────────────────────────────────────────────────────────┐
│ SERVER (separate process)                                       │
│                                                                 │
│  def exposed_recursive_call(depth, callback):  ← SYNC wrapper   │
│      return asyncio.run(                       ← Creates a loop │
│          _async_logic(depth, callback)                          │
│      )                                                          │
│                                                                 │
│  async def _async_logic(depth, callback):      ← ASYNC logic    │
│      await asyncio.sleep(0.001)  ← Async operations             │
│      if depth > 0:                                              │
│          result = callback(...)  ← Call into the client         │
│          return result                                          │
│                                                                 │
└──────────────────────┬──────────────────────────────────────────┘
                       │
                       │ RPC call (callback)
                       │
┌──────────────────────▼──────────────────────────────────────────┐
│ CLIENT (asyncio-native arpyc)                                   │
│                                                                 │
│  def callback(depth, server_func):  ← SYNC wrapper              │
│      # ⚠️ PROBLEM: Executes in the event loop!                  │
│      # ❌ CANNOT: asyncio.run() → RuntimeError                  │
│      # ❌ CANNOT: await → SyntaxError (not async def)           │
│                                                                 │
│      # ✅ SOLUTION 1: Sync logic (WITHOUT async)                │
│      return sync_logic(depth, server_func)                      │
│                                                                 │
│      # ✅ SOLUTION 2: nest_asyncio                              │
│      # import nest_asyncio; nest_asyncio.apply()                │
│      # return asyncio.run(async_logic(...))                     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## ✅ SOLUTION 1: Sync callbacks (RECOMMENDED)

### Description

- The server uses async logic (via asyncio.run inside the exposed method)
- The client uses a simple sync function as the callback
- Best performance (asyncio-native arpyc)
- No dependencies

### Server code

```python
class AsyncRecursiveService(rpyc.Service):
    async def _async_recursive_logic(self, depth, callback):
        """ASYNC logic with await."""
        log = f"Server processing depth={depth}"
        print(log)

        # await can be used here!
        await asyncio.sleep(0.001)
        await some_async_operation()

        if depth > 0:
            # Call the callback (it is sync, but that is fine)
            result = callback(depth - 1, self.exposed_recursive_call)
            return f"Server: {result}"
        return "Server: done"

    def exposed_recursive_call(self, depth, callback):
        """SYNC wrapper for RPyC (an exposed method must be sync)."""
        # asyncio.run() works - the exposed method is called in a separate thread
        return asyncio.run(self._async_recursive_logic(depth, callback))
```

### Client code

```python
async def main():
    async with RpycAsyncSessionNative("localhost", 18861) as session:

        # ✅ Simple SYNC callback (without async logic)
        def client_callback(depth, server_func):
            """SYNC callback - simple, fast, works."""
            print(f"Client processing depth={depth}")

            # Ordinary sync logic (WITHOUT await)
            if depth > 0:
                result = server_func(depth - 1, client_callback)
                return f"Client: {result}"
            return "Client: done"

        # Call
        result = await session.async_call(
            session.root.recursive_call,
            depth=5,
            callback=client_callback
        )
```

### Test results

```
✅ SUCCESS: Completed in 0.0053s
A calls: 3, B calls: 3

✅ SOLUTION 2 WORKS PERFECTLY!
   - Server: async logic ✅
   - Client: sync callback ✅
   - No dependencies ✅
   - Simple and clean ✅
```

### Advantages

- ✅ **Simplicity:** Minimal code, easy to understand
- ✅ **Performance:** Asyncio-native (156x faster than BgServingThread)
- ✅ **No dependencies:** nest_asyncio is not needed
- ✅ **Reliability:** No problems with nested event loops

### Disadvantages

- ⚠️ The client callback cannot use `await`
- ⚠️ If async logic is needed in the callback, use the other solutions

### When to use

- ✅ **Always use this solution by default!**
- ✅ When the client callback is simple enough (no I/O)
- ✅ When maximum performance is needed

---

## ✅ SOLUTION 2: nest_asyncio

### Description

- Uses the `nest_asyncio` library for nested event loops
- Allows `asyncio.run()` inside an already running event loop
- Both the server and the client can have async logic

### Installation

```bash
pip install nest-asyncio
```

### Client code

```python
import nest_asyncio

# Apply the patch ONCE at the start of the program
nest_asyncio.apply()

async def main():
    async with RpycAsyncSessionNative("localhost", 18861) as session:

        # ASYNC client logic
        async def _async_client_logic(depth, server_func):
            """ASYNC logic - await can be used!"""
            print(f"Client async processing depth={depth}")

            # We can use await!
            await asyncio.sleep(0.001)
            await some_async_operation()

            if depth > 0:
                result = server_func(depth - 1, client_callback)
                return f"Client: {result}"
            return "Client: done"

        # SYNC wrapper
        def client_callback(depth, server_func):
            """SYNC wrapper - called by RPyC."""
            # With nest_asyncio, asyncio.run() works even inside an event loop!
            return asyncio.run(_async_client_logic(depth, server_func))

        result = await session.async_call(
            session.root.recursive_call,
            depth=5,
            callback=client_callback
        )
```

### Test results

```
✅ SUCCESS: Completed in 0.0094s
A calls: 3, B calls: 3

✅ SOLUTION 1 WORKS!
   - Server: async logic ✅
   - Client: async logic ✅
   - nest_asyncio: enables nested event loops ✅
```

⚠️ **Warning:** Context warnings are visible (a side effect of nest_asyncio):
```
RuntimeWarning: cannot enter context: <_contextvars.Context object> is already entered
```

### Advantages

- ✅ Both the server and the client can have async logic
- ✅ `await` can be used in callbacks
- ✅ Good performance (asyncio-native)

### Disadvantages

- ⚠️ Requires an extra dependency (`nest-asyncio`)
- ⚠️ Context warnings (not critical, but unpleasant)
- ⚠️ Patches asyncio globally

### When to use

- ✅ When the callback MUST perform async operations (I/O, await)
- ✅ When maximum performance is needed (asyncio-native)
- ⚠️ Only if Solution 1 does not fit

---

## ✅ SOLUTION 3: BgServingThread

### Description

- Uses the old arpyc approach with BgServingThread
- Callbacks execute in a separate thread (not in the event loop)
- In a separate thread `asyncio.run()` works without problems

### Client code

```python
from arpyc import RpycAsyncSession  # Old approach with BgServingThread

async def main():
    # RpycAsyncSession instead of RpycAsyncSessionNative
    async with RpycAsyncSession("localhost", 18861) as session:

        # ASYNC client logic
        async def _async_client_logic(depth, server_func):
            """ASYNC logic."""
            await asyncio.sleep(0.001)

            if depth > 0:
                result = server_func(depth - 1, client_callback)
                return f"Client: {result}"
            return "Client: done"

        # SYNC wrapper
        def client_callback(depth, server_func):
            """SYNC wrapper - executes in the BgServingThread."""
            # There is no event loop in the BgServingThread, so asyncio.run() works!
            return asyncio.run(_async_client_logic(depth, server_func))

        result = await session.async_call(
            session.root.recursive_call,
            depth=5,
            callback=client_callback
        )
```

### Test results

```
✅ SUCCESS: Completed in 0.3077s  ← Slower!
A calls: 3, B calls: 3

✅ SOLUTION 3 WORKS!
   - Server: async logic ✅
   - Client: async logic ✅
   - BgServingThread: separate thread allows asyncio.run() ✅
   - Trade-off: ~156x slower than asyncio-native ⚠️
```

### Advantages

- ✅ Both the server and the client can have async logic
- ✅ `await` can be used in callbacks
- ✅ No warnings or context problems
- ✅ Does not require nest_asyncio

### Disadvantages

- ❌ **About 156x slower!** (0.3077s vs 0.0053s)
- ❌ Polling delays (sleep_interval=0.1)
- ❌ Callbacks in a separate thread (not in the event loop)

### When to use

- ⚠️ When the callback MUST perform async operations
- ⚠️ When nest_asyncio does not fit (conflicts, warnings)
- ❌ **NOT recommended** for new projects (performance)

---

## 📊 Comparison of the solutions

| Criterion | Solution 1 (Sync) | Solution 2 (nest_asyncio) | Solution 3 (BgThread) |
|----------|------------------|--------------------------|----------------------|
| **Performance** | ✅ 0.0053s | ✅ 0.0094s | ❌ 0.3077s (156x slower) |
| **Server async logic** | ✅ Yes | ✅ Yes | ✅ Yes |
| **Client async logic** | ❌ No | ✅ Yes | ✅ Yes |
| **Dependencies** | ✅ None | ⚠️ nest-asyncio | ✅ None (built in) |
| **Warnings** | ✅ None | ⚠️ Context warnings | ✅ None |
| **Simplicity** | ✅✅✅ Very simple | ✅✅ Simple | ✅ Simple |
| **Thread usage** | ✅ 1 thread | ✅ 1 thread | ⚠️ 2 threads |
| **Callback location** | ✅ Event loop | ✅ Event loop | ⚠️ BgServingThread |

---

## 🎯 Recommendations

### Use Solution 1 (Sync callbacks) when:

- ✅ **Always use it by default!**
- ✅ The client callback is simple enough
- ✅ Maximum performance is needed
- ✅ You want a minimum of dependencies

### Use Solution 2 (nest_asyncio) when:

- ⚠️ The callback MUST perform async operations (I/O, await)
- ✅ Good performance is needed (asyncio-native)
- ⚠️ You are ready to add a dependency
- ⚠️ Context warnings are not critical

### Use Solution 3 (BgServingThread) when:

- ⚠️ The callback MUST perform async operations
- ❌ nest_asyncio does not fit (conflicts)
- ❌ Performance is not critical
- ❌ **NOT recommended for new projects**

---

## 💡 Best practices

### ✅ DO (Do it this way):

```python
# Server: async logic via asyncio.run()
class Service(rpyc.Service):
    async def _async_logic(self, depth, callback):
        await asyncio.sleep(0.001)  # Async work
        if depth > 0:
            return callback(depth - 1, self.exposed_method)
        return "done"

    def exposed_method(self, depth, callback):
        return asyncio.run(self._async_logic(depth, callback))

# Client: a simple sync callback
async with RpycAsyncSessionNative(...) as session:
    def callback(depth, server_func):
        # Simple sync logic
        if depth > 0:
            return server_func(depth - 1, callback)
        return "done"

    result = await session.async_call(session.root.method, 5, callback)
```

### ❌ DON'T (Do not do it this way):

```python
# ❌ DO NOT make an async callback without a wrapper
async def callback(depth, server_func):  # ← async def!
    # RPyC expects a SYNC function!
    # This will cause the error: coroutine was never awaited
    ...

# ❌ DO NOT use asyncio.run() without nest_asyncio in asyncio-native
def callback(depth, server_func):
    # Inside an event loop (asyncio-native) this does not work!
    return asyncio.run(async_logic(...))  # ← RuntimeError!
```

---

## 📁 Code examples

### Full example (Solution 1 - Recommended)

See the file: `test_arpyc_async_recursive_SOLUTION.py::test_solution2_sync_callbacks()`

### Full example (Solution 2 - nest_asyncio)

See the file: `test_arpyc_async_recursive_SOLUTION.py::test_solution1_nest_asyncio()`

### Full example (Solution 3 - BgServingThread)

See the file: `test_arpyc_async_recursive_SOLUTION.py::test_solution3_bgservingthread()`

---

## 🔍 Conclusion

**Question:** How does recursion work when the functions are asynchronous?

**Answer:**

1. **Server-side exposed function:**
   - Must be SYNC (an RPyC requirement)
   - Wraps the async logic via `asyncio.run()`
   - ✅ Works perfectly (called in a separate thread)

2. **Client-side callback function:**
   - Must be SYNC (it is passed into RPyC)
   - ⚠️ **Problem:** In asyncio-native it executes in the event loop
   - ✅ **Solution 1:** Keep the callback simple (sync logic) - RECOMMENDED
   - ✅ **Solution 2:** Use nest_asyncio for async logic
   - ⚠️ **Solution 3:** Use BgServingThread (slow)

**Recommendation:** Use **Solution 1** (sync callbacks) by default for the best performance and simplicity!
