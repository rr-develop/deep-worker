# Developer Examples - Fire-and-Forget Pattern

This document provides code examples for developers working with the fire-and-forget architecture in deep-worker-web.

## Table of Contents

1. [Backend: Implementing Fire-and-Forget RPC Calls](#backend-fire-and-forget)
2. [Frontend: Handling WebSocket Responses](#frontend-websocket)
3. [Error Handling Patterns](#error-handling)
4. [Testing Fire-and-Forget Code](#testing)

---

## Backend: Implementing Fire-and-Forget RPC Calls

### Basic Pattern

```python
from rpyc.utils.helpers import fire_and_forget_async

async def my_rpc_method(self, agent_name: str) -> dict[str, Any]:
    """Example RPC call using fire-and-forget pattern."""

    # Connect if needed
    if not await self.is_connected(agent_name):
        if not await self.connect_to_agent(agent_name):
            return {"success": False, "error": "Failed to connect"}

    import uuid
    request_id = str(uuid.uuid4())
    root = self._roots[agent_name]

    # Define success callback
    async def on_success(response: dict[str, Any]) -> None:
        """Handle successful RPC response."""
        logger.info(f"RPC success for {agent_name}")
        await self._broadcast_message({
            "type": "my_response",
            "agent_name": agent_name,
            "request_id": request_id,
            "success": True,
            "data": response,
        })

    # Define error callback
    async def on_error(exc: BaseException) -> None:
        """Handle RPC errors."""
        logger.error(f"RPC error for {agent_name}: {exc}")
        await self._broadcast_message({
            "type": "my_error",
            "agent_name": agent_name,
            "request_id": request_id,
            "success": False,
            "error": str(exc),
            "error_type": type(exc).__name__,
        })

    # Fire and forget with timeout
    fire_and_forget_async(
        root.my_rpc_method(args),
        timeout=10.0,  # 10 second timeout
        success_callback=on_success,
        error_callback=on_error,
        name=f"my_rpc_{agent_name}",
    )

    # Return immediately
    return {
        "success": True,
        "message": "Request sent (response will arrive via WebSocket)",
        "request_id": request_id,
    }
```

### Choosing Timeouts

```python
# Quick status queries - 5 seconds
fire_and_forget_async(
    root.get_status(),
    timeout=5.0,
    ...
)

# Message processing - 60 seconds (can be slow)
fire_and_forget_async(
    root.invoke_agent(message),
    timeout=60.0,
    ...
)

# Large data retrieval - 15 seconds
fire_and_forget_async(
    root.get_history(limit=100),
    timeout=15.0,
    ...
)

# Cleanup operations - 2 seconds (short timeout)
fire_and_forget_async(
    root.unsubscribe_messages(callback),
    timeout=2.0,
    ...
)
```

### Error Handling in Callbacks

```python
async def on_error(exc: BaseException) -> None:
    """Comprehensive error handling."""

    # Classify error
    error_type = type(exc).__name__
    error_msg = str(exc)

    # User-friendly messages
    if error_type == "TimeoutError":
        user_msg = "Agent not responding (timeout)"
    elif error_type in ("ConnectionError", "EOFError"):
        user_msg = "Connection to agent lost"
    elif error_type == "RemoteError":
        user_msg = f"Agent error: {error_msg}"
    else:
        user_msg = f"Unexpected error: {error_msg}"

    # Log technical details
    logger.error(f"RPC error: {error_type} - {error_msg}")

    # Broadcast to UI
    await self._broadcast_message({
        "type": "operation_error",
        "agent_name": agent_name,
        "error": user_msg,
        "error_type": error_type,
        "timestamp": datetime.now().isoformat(),
    })
```

---

## Frontend: Handling WebSocket Responses

### Basic WebSocket Handler

```javascript
// Handle WebSocket messages
function handleWebSocketMessage(data) {
    console.log('WebSocket message:', data);

    switch (data.type) {
        case 'agent_response':
            handleAgentResponse(data);
            break;
        case 'agent_error':
            handleAgentError(data);
            break;
        case 'history_response':
            handleHistoryResponse(data);
            break;
        case 'history_error':
            handleHistoryError(data);
            break;
        default:
            console.warn('Unknown message type:', data.type);
    }
}
```

### Handling Agent Responses

```javascript
function handleAgentResponse(data) {
    const { agent_name, response } = data;

    // Only update if this is the selected agent
    if (agent_name !== selectedAgent) {
        return;
    }

    // Show success notification
    showNotification('Message processed successfully', 'success');

    // Update UI with response
    displayAgentResponse(response);

    // Reload history to show updated conversation
    loadAgentHistory(agent_name);
}
```

### Handling Errors with User Feedback

```javascript
function handleAgentError(data) {
    const { agent_name, error, error_type } = data;

    // Create user-friendly error message
    let message = `Agent error: ${error}`;

    if (error_type === 'TimeoutError') {
        message = 'Agent not responding (timeout)';
    } else if (error_type === 'ConnectionError' || error_type === 'EOFError') {
        message = 'Connection to agent lost';
    }

    // Show error notification
    showNotification(message, 'error');

    // Log for debugging
    console.error(`Agent ${agent_name} error:`, error);

    // Optionally reload history (might have partial results)
    if (agent_name === selectedAgent) {
        setTimeout(() => loadAgentHistory(agent_name), 1000);
    }
}
```

### Request Correlation Pattern

```javascript
// Track pending requests
const pendingRequests = new Map();

// Send request with correlation ID
async function requestHistory(agentName) {
    const requestId = generateUUID();

    // Store pending request
    const promise = new Promise((resolve, reject) => {
        pendingRequests.set(requestId, {
            resolve,
            reject,
            timeout: setTimeout(() => {
                pendingRequests.delete(requestId);
                reject(new Error('Request timed out'));
            }, 20000), // 20s frontend timeout
        });
    });

    // Send request
    await fetch(`/api/agents/${agentName}/history?request_id=${requestId}`);

    return promise;
}

// Handle response
function handleHistoryResponse(data) {
    const { request_id, history } = data;

    const pending = pendingRequests.get(request_id);
    if (pending) {
        clearTimeout(pending.timeout);
        pending.resolve(history);
        pendingRequests.delete(request_id);
    }
}

// Usage
try {
    const history = await requestHistory('my-agent');
    displayHistory(history);
} catch (error) {
    showNotification('Failed to load history: ' + error.message, 'error');
}
```

### Notification System

```javascript
function showNotification(message, type = 'info') {
    // Create notification element
    const notification = document.createElement('div');
    notification.className = `notification notification-${type}`;
    notification.textContent = message;

    // Get or create container
    let container = document.getElementById('notification-container');
    if (!container) {
        container = document.createElement('div');
        container.id = 'notification-container';
        container.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            z-index: 1000;
            display: flex;
            flex-direction: column;
            gap: 10px;
        `;
        document.body.appendChild(container);
    }

    container.appendChild(notification);

    // Style based on type
    notification.style.cssText = `
        padding: 15px 20px;
        border-radius: 4px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.2);
        max-width: 400px;
        animation: slideIn 0.3s ease-out;
        font-size: 14px;
    `;

    if (type === 'success') {
        notification.style.backgroundColor = '#4CAF50';
        notification.style.color = 'white';
    } else if (type === 'error') {
        notification.style.backgroundColor = '#f44336';
        notification.style.color = 'white';
    } else {
        notification.style.backgroundColor = '#2196F3';
        notification.style.color = 'white';
    }

    // Auto-remove after 5 seconds
    setTimeout(() => {
        notification.style.animation = 'slideOut 0.3s ease-in';
        setTimeout(() => notification.remove(), 300);
    }, 5000);
}
```

---

## Error Handling Patterns

### Backend Error Classification

```python
async def classify_and_handle_error(exc: BaseException) -> dict[str, str]:
    """Classify errors and return user-friendly messages."""

    error_map = {
        "TimeoutError": {
            "user_msg": "Operation timed out - agent may be unresponsive",
            "severity": "warning",
        },
        "ConnectionError": {
            "user_msg": "Connection lost - agent may have crashed",
            "severity": "error",
        },
        "EOFError": {
            "user_msg": "Agent process terminated unexpectedly",
            "severity": "error",
        },
        "RemoteError": {
            "user_msg": f"Agent reported error: {str(exc)}",
            "severity": "error",
        },
        "AttributeError": {
            "user_msg": "Feature not supported by this agent version",
            "severity": "warning",
        },
    }

    error_type = type(exc).__name__
    error_info = error_map.get(error_type, {
        "user_msg": f"Unexpected error: {str(exc)}",
        "severity": "error",
    })

    return {
        "error_type": error_type,
        "user_message": error_info["user_msg"],
        "technical_message": str(exc),
        "severity": error_info["severity"],
    }
```

### Frontend Error Display

```javascript
function displayError(errorData) {
    const { agent_name, error, error_type, severity } = errorData;

    // Create error notification with retry option
    const notification = createNotification({
        type: severity || 'error',
        title: `Agent Error: ${agent_name}`,
        message: error,
        actions: [
            {
                label: 'Retry',
                onclick: () => retryLastOperation(agent_name)
            },
            {
                label: 'Dismiss',
                onclick: () => notification.remove()
            }
        ],
        persistent: true, // Requires user action to dismiss
    });

    // Add to notification area
    document.getElementById('notification-area').appendChild(notification);
}
```

---

## Testing Fire-and-Forget Code

### Unit Test Pattern

```python
import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock

@pytest.mark.asyncio
async def test_fire_and_forget_returns_immediately():
    """Test that fire-and-forget returns instantly."""
    import time

    service = AgentService(mock_registry)
    await service.initialize()

    # Mock slow RPC call (10 seconds)
    mock_root = AsyncMock()
    async def slow_rpc(*args, **kwargs):
        await asyncio.sleep(10)
        return {"success": True}

    mock_root.my_method = slow_rpc
    service._roots["test-agent"] = mock_root
    service._connections["test-agent"] = MagicMock()

    # Call should return immediately
    start = time.time()
    result = await service.my_fire_and_forget_method("test-agent")
    elapsed = time.time() - start

    # Verify instant return
    assert elapsed < 1.0, f"Took {elapsed}s, expected < 1s"
    assert result["success"] is True
    assert "WebSocket" in result["message"]

@pytest.mark.asyncio
async def test_fire_and_forget_broadcasts_success():
    """Test that success callback broadcasts result."""
    service = AgentService(mock_registry)
    await service.initialize()

    # Track broadcasts
    messages = []
    async def capture(msg):
        messages.append(msg)

    service.register_message_callback(capture)

    # Mock successful RPC
    mock_root = AsyncMock()
    mock_root.my_method = AsyncMock(return_value={"data": "success"})
    service._roots["test-agent"] = mock_root
    service._connections["test-agent"] = MagicMock()

    # Call method
    await service.my_fire_and_forget_method("test-agent")

    # Wait for callback
    await asyncio.sleep(0.5)

    # Verify broadcast
    assert len(messages) > 0
    response = next((m for m in messages if m["type"] == "my_response"), None)
    assert response is not None
    assert response["success"] is True

@pytest.mark.asyncio
async def test_fire_and_forget_handles_timeout():
    """Test that timeout fires correctly."""
    service = AgentService(mock_registry)
    await service.initialize()

    # Track broadcasts
    messages = []
    async def capture(msg):
        messages.append(msg)

    service.register_message_callback(capture)

    # Mock hanging RPC
    mock_root = AsyncMock()
    async def hang_forever(*args, **kwargs):
        await asyncio.sleep(9999)

    mock_root.my_method = hang_forever
    service._roots["test-agent"] = mock_root
    service._connections["test-agent"] = MagicMock()

    # Call method (with 2s timeout for testing)
    await service.my_fire_and_forget_method("test-agent", timeout=2.0)

    # Wait for timeout
    await asyncio.sleep(2.5)

    # Verify timeout error broadcast
    error = next((m for m in messages if m["type"] == "my_error"), None)
    assert error is not None
    assert "TimeoutError" in error["error_type"]
```

### Integration Test Pattern

```python
@pytest.mark.integration
@pytest.mark.asyncio
async def test_real_agent_fire_and_forget():
    """Test fire-and-forget with real agent process."""
    # Start real agent
    agent_proc = start_test_agent()

    try:
        service = AgentService()
        await service.initialize()
        await service.connect_to_agent("test-agent")

        # Track messages
        messages = []
        async def capture(msg):
            messages.append(msg)

        service.register_message_callback(capture)

        # Send message
        result = await service.send_message("test-agent", "hello")
        assert result["success"] is True

        # Wait for response (should be quick)
        await asyncio.sleep(5)

        # Verify we got response
        response = next((m for m in messages if m["type"] == "agent_response"), None)
        assert response is not None

    finally:
        agent_proc.terminate()
        agent_proc.wait()
```

---

## Best Practices

### 1. Always Set Timeouts

```python
# Good
fire_and_forget_async(
    rpc_call(),
    timeout=10.0,  # Explicit timeout
    ...
)

# Bad - may hang forever
fire_and_forget_async(
    rpc_call(),  # No timeout!
    ...
)
```

### 2. Always Provide Error Callbacks

```python
# Good
fire_and_forget_async(
    rpc_call(),
    error_callback=handle_error,  # Handle errors
    ...
)

# Bad - errors only logged to stderr
fire_and_forget_async(
    rpc_call(),  # No error handling!
    ...
)
```

### 3. Broadcast All Results

```python
async def on_success(result):
    """Always broadcast results to WebSocket."""
    await self._broadcast_message({
        "type": "operation_success",
        "data": result,
    })

async def on_error(exc):
    """Always broadcast errors to WebSocket."""
    await self._broadcast_message({
        "type": "operation_error",
        "error": str(exc),
    })
```

### 4. Use Request IDs for Correlation

```python
import uuid

request_id = str(uuid.uuid4())

# Include in broadcast
await self._broadcast_message({
    "type": "response",
    "request_id": request_id,  # Frontend can correlate
    "data": result,
})
```

### 5. Handle Callback Exceptions

```python
async def on_success(result):
    try:
        # Your callback logic
        await process_result(result)
    except Exception as e:
        # Log but don't crash
        logger.error(f"Error in success callback: {e}")
```

---

## Troubleshooting

### Fire-and-Forget Not Working

**Symptom:** UI still blocks

**Check:**
1. Using `await` on fire_and_forget result? Don't!
2. Using correct import: `from rpyc.utils.helpers import fire_and_forget_async`
3. Running in async context? (must have event loop)

### Callbacks Not Firing

**Symptom:** No WebSocket messages received

**Check:**
1. WebSocket connected? (check browser console)
2. Callback registered? (`service.register_message_callback()`)
3. Task being garbage collected? (keep reference if needed)

### Timeouts Not Working

**Symptom:** Operations hang forever despite timeout

**Check:**
1. Using AsyncioServer? (ThreadedServer won't work!)
2. Connection has asyncio serving enabled?
3. Timeout value correct? (not None)

---

**For more details, see:**
- `docs/FIRE_AND_FORGET_MIGRATION_DESIGN.md`
- `docs/guide_fire_and_forget.md` in the rpyc_async source checkout
