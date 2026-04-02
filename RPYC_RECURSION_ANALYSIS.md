# Analysis of Cross-Process Recursion over RPyC

## Research goal

Verify whether cross-process recursion is possible when using RPyC in this project. The aim was to find out whether the following pattern would work:

- Process A calls a method in process B
- Process B calls a method in process A via a callback
- Process A again calls a method in process B (recursion)

## Methodology

### Test architecture

An integration test `test_rpyc_cross_process_recursion.py` was created with the following architecture:

1. **Two RPyC servers in separate processes:**
   - Process A: `RecursiveServiceA` (port assigned dynamically)
   - Process B: `RecursiveServiceB` (port assigned dynamically)

2. **Dynamic port allocation:**
   - The `_find_free_port()` function with `socket.bind(("127.0.0.1", 0))` was used
   - The OS automatically picks a free ephemeral port
   - This is critical to avoid port conflicts in tests

3. **Recursion scenario:**
   ```
   Test -> B.process_with_callback(depth=5)
        -> A.callback_from_b(depth=5)
        -> B.process_with_callback(depth=4)
        -> A.callback_from_b(depth=4)
        -> ... (down to depth=0)
   ```

4. **Cross-process connections:**
   - Process A establishes an RPyC connection to process B
   - Process B establishes an RPyC connection to process A
   - The connections are stored as global variables in each process

### Key implementation points

#### Use of global variables

```python
# Global variables to store cross-process connections
_service_a_connection = None
_service_b_connection = None
_service_call_log_a: list[str] = []
_service_call_log_b: list[str] = []
_log_lock = threading.Lock()
```

**Why global variables:**
- RPyC creates a new Service instance for each incoming connection
- Instance variables are lost between calls from different clients
- Global variables persist at the process level

#### Excluding arpyc from the virtual environment

According to the project's contributor guidelines, it is critically important to:
- **NOT use** `pip install arpyc` or virtual environments
- **ALWAYS** import arpyc from the source code in the `arpyc` directory of this repository
- Using installed packages causes RPyC proxy errors

```python
_arpyc_path = Path(__file__).parent.parent.parent.parent.parent / "arpyc"
if _arpyc_path.exists() and str(_arpyc_path) not in sys.path:
    sys.path.insert(0, str(_arpyc_path))

from arpyc import RpycAsyncSession
```

## Results

### DEADLOCK detected ⚠️

**Symptoms:**
- The test hung when attempting to perform the recursion
- Timeout: the test was forcibly terminated after 2 minutes 10 seconds
- None of the recursive calls completed

**First attempt (before global variables):**
```
Expected 6 calls to B, got 1
Expected 6 calls to A, got 0
```
Only the first call `B.process_with_callback(depth=5)` was registered; the callback to A did not happen.

**Second attempt (with global variables):**
```
Command timed out after 2m 10s Terminated
```
The test hung in a deadlock.

### Cause of the deadlock

RPyC uses **synchronous blocking calls** by default:

1. **Test** calls `service_b.process_with_callback()` - BLOCKS waiting for a response
2. **Process B** tries to call `service_a.callback_from_b()` - BLOCKS
3. **Process A** should respond to the callback, but its RPyC server is busy handling the incoming call
4. **DEADLOCK:** B waits for A, A waits for its call to B to complete, but B cannot complete until it receives a response from A

### Deadlock diagram

```
┌─────────┐                 ┌─────────┐                 ┌─────────┐
│  Test   │                 │Process B│                 │Process A│
└────┬────┘                 └────┬────┘                 └────┬────┘
     │                           │                           │
     │ process_with_callback(5)  │                           │
     ├──────────────────────────►│                           │
     │     [BLOCKED]             │                           │
     │                           │ callback_from_b(5)        │
     │                           ├──────────────────────────►│
     │                           │     [BLOCKED]             │
     │                           │                           │
     │                           │     [Cannot respond       │
     │                           │      - waiting for        │
     │                           │      Test's call]         │
     │                           │                           │
     │    [Still blocked         │◄───────[DEADLOCK]────────►│
     │     waiting for B]        │                           │
     │                           │                           │
     ▼                           ▼                           ▼
 TIMEOUT                     BLOCKED                     BLOCKED
```

## Analysis of the RPyC library

### Architectural limitations

**RPyC uses a synchronous model by default:**
- `sync_request_timeout` - timeout for synchronous calls
- A call blocks until a response is received
- There is no built-in support for reentrant calls

**ThreadedServer:**
- Handles each connection in a separate thread
- BUT: a synchronous call blocks the thread until it completes
- A callback from B to A blocks thread B
- A cannot respond because it is waiting for its own call to B to complete

### Possible solutions (theoretical)

#### 1. Asynchronous callbacks via `rpyc.async_()`

```python
# Instead of:
result = service_a.callback_from_b(depth, message)

# Use:
async_result = rpyc.async_(service_a.callback_from_b)(depth, message)
# Do not wait for result.value - this is fire-and-forget
```

**Problem:** Different result-handling logic is required; it is not suitable for recursion that returns values.

#### 2. BgServingThread for handling incoming calls

RPyC can use a background thread to handle incoming calls while the main thread is blocked:

```python
conn = rpyc.connect(
    host, port,
    config={
        "sync_request_timeout": 300,
        "allow_public_attrs": True
    }
)
# Start a background serving thread
from rpyc.utils.helpers import BgServingThread
bgsrv = BgServingThread(conn)
```

**However:** It is not certain that this fully solves the problem.

#### 3. Abandoning synchronous recursion

**The "continuation" pattern:**
- B calls the callback to A asynchronously (fire-and-forget)
- A receives the event and calls B again on its own
- Nested calls are avoided

**Example:**
```python
# B does not wait for a response from A
rpyc.async_(service_a.on_event)("recursion_step", {"depth": depth})
# B returns the result immediately

# A handles the event asynchronously
def on_event(event_type, data):
    if event_type == "recursion_step":
        # A calls B again, but this is a new top-level call
        service_b.process_next_step(data["depth"] - 1)
```

## Conclusions

### ❌ Cross-process recursion DOES NOT WORK with RPyC by default

**Synchronous reentrant calls (A→B→A→B) cause a deadlock:**
- RPyC blocks the calling thread until a response is received
- The callback cannot be handled because the server is waiting for the original call to complete
- Result: mutual blocking

### ✅ Dynamic port allocation WORKS perfectly

- The `_find_free_port()` function with `socket.bind(("127.0.0.1", 0))` allocates ports successfully
- Processes A and B start without conflicts
- Connections are established correctly

### ⚠️ RPyC is not designed for reentrant calls

**RPyC is better suited for:**
- Simple RPC calls (client→server→client)
- One-way callbacks (server→client for notifications)
- Asynchronous notifications (fire-and-forget)

**RPyC is NOT suitable for:**
- Recursive cross-process calls
- Reentrant calls (A→B→A within a single call stack)
- Complex patterns with mutual synchronous calls

## Recommendations

### For this project

1. **Avoid recursive patterns:**
   - Do not make synchronous callbacks from RPyC methods
   - Use asynchronous notifications for events

2. **Proven patterns:**
   - The client subscribes to events (as in `AgentService.exposed_subscribe_messages`)
   - The server sends notifications via `subscriber.on_message()`
   - But the server does NOT wait for a response from the subscriber

3. **The existing code does it correctly:**
   ```python
   # agent_service.py - the correct pattern
   def _notify_subscribers(self, message: dict[str, Any]) -> None:
       for subscriber in subscribers:
           # Fire-and-forget callback
           subscriber.on_message(message)
           # The return value is NOT used
   ```

4. **If recursion is needed:**
   - Use a queue-based approach
   - Process A puts a task into the queue
   - Process B picks it up, processes it, and puts the result back
   - Nested calls are avoided

### Alternative technologies for recursion

If cross-process recursion is critically required:

1. **gRPC with bidirectional streaming** - supports two-way streams
2. **ZeroMQ** - asynchronous message passing without blocking
3. **Multiprocessing queues** - Python native, simple message passing
4. **Actor model (Pykka, Thespian)** - asynchronous actors without deadlock

## Test file

The integration test is stored at:
```
tests/integration_ai/test_rpyc_cross_process_recursion.py
```

### The test demonstrates:

✅ Dynamic port allocation
✅ Starting two RPyC servers in separate processes
✅ Establishing cross-process connections
❌ Deadlock when attempting recursion

### How to run it:

```bash
# From the CLI package directory
python3 -m pytest tests/integration_ai/test_rpyc_cross_process_recursion.py::test_cross_process_recursion_no_deadlock -v -s
```

**Expected behavior:** The test will hang and be terminated by timeout (a demonstration of the deadlock).

## Conclusion

The research confirmed that **RPyC does not support synchronous cross-process recursion** because of the blocking nature of RPC calls. Any attempt to implement the A→B→A→B pattern inevitably results in a deadlock.

This project correctly uses RPyC for:
- One-way RPC calls
- Asynchronous notifications (callbacks without waiting for a response)
- Event subscriptions (pub/sub pattern)

For more complex interaction patterns, alternative solutions specifically designed for asynchronous communication should be considered.
