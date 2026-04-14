# Problem analysis: Missing history and last response

**Date:** 2026-04-06
**Status:** 🔴 CRITICAL BUG - History is not displayed after the fire-and-forget migration

---

## Problem

After migrating to the fire-and-forget pattern:
- ❌ The agent's last response is NOT displayed in the UI
- ❌ The full dialog is NOT available ("Full Dialog" button)
- ✅ In the reference web application everything works correctly

---

## Root Cause

### Backend Issue: API router incompatibility with fire-and-forget

**File:** `app/routers/agents.py`
**Function:** `get_history()` (lines 145-199)

```python
@router.get("/{agent_name}/history", response_model=list[MessageInfo])
async def get_history(...) -> list[MessageInfo]:
    # ...
    response = await service.get_agent_history(agent_name, limit, offset)  # Line 176

    if not response.get("success"):
        raise HTTPException(...)

    # PROBLEM: Tries to get history from the immediate response
    history = response.get("history", [])  # Line 185 - RECEIVES []
    messages = []

    for msg in history:  # The loop does not execute - history is empty!
        messages.append(MessageInfo(...))

    return messages  # Returns an EMPTY list!
```

**What happens:**

1. **Before the migration** (blocking await):
   ```python
   service.get_agent_history() returned:
   {
       "success": true,
       "history": [...]  # ✅ The real history
   }
   ```

2. **After the migration** (fire-and-forget):
   ```python
   service.get_agent_history() returns:
   {
       "success": true,
       "message": "History request sent (response will arrive via WebSocket)",
       "request_id": "uuid-1234"
       # ❌ THERE IS NO "history" FIELD!
   }
   ```

3. **Result:**
   - `response.get("history", [])` → `[]` (an empty list)
   - The router returns `[]` to the client
   - The frontend receives an empty array and shows "No response yet"

---

## Frontend Issue: Expecting a synchronous response

**File:** `static/js/main.js`
**Function:** `loadAgentHistory()` (lines 166-208)

```javascript
async function loadAgentHistory(name) {
    try {
        const response = await fetch(`/api/agents/${name}/history`);
        if (!response.ok) {
            console.error('Failed to load agent history');
            return;
        }

        const history = await response.json();  // ❌ Receives []
        console.log(`Loaded ${history.length} messages for ${name}`);  // "Loaded 0 messages"

        // Find last message
        let lastMessage = null;
        for (let i = history.length - 1; i >= 0; i--) {  // The loop does NOT execute!
            // ...
        }

        // Update UI
        if (lastMessage && lastMessage.content) {
            responseContent.textContent = lastMessage.content;
        } else {
            responseContent.textContent = 'No response yet';  // ❌ Always shows this
        }
    } catch (error) {
        console.error('Error loading agent history:', error);
    }
}
```

**Problems:**

1. The frontend expects to receive the history array immediately from the REST endpoint
2. After the migration the endpoint returns an empty array
3. The WebSocket handler `handleHistoryResponse` was added, but is NEVER USED
4. The UI always shows "No response yet"

---

## Mismatch with the design

The design document (`FIRE_AND_FORGET_MIGRATION_DESIGN.md`) describes:

```python
# The backend was supposed to:
async def get_agent_history(...):
    # Fire and forget
    fire_and_forget_async(...)

    return {
        "success": True,
        "message": "History request sent (response will arrive via WebSocket)",
        "request_id": req_id,
    }
```

```javascript
// The frontend was supposed to:
function handleHistoryResponse(data) {
    const { request_id, history } = data;

    const pending = pendingRequests.get(request_id);
    if (pending) {
        pending.resolve(history);
        pendingRequests.delete(request_id);
    }
}
```

**But it was NOT implemented that way!**

---

## Precise diagnosis

### 1. Backend Service (agent_service.py)

✅ **CORRECTLY IMPLEMENTED:**
```python
async def get_agent_history(...) -> dict[str, Any]:
    # ...
    fire_and_forget_async(
        root.get_history(limit, offset),
        timeout=15.0,
        success_callback=on_success,  # ✅ Broadcast via WebSocket
        error_callback=on_error,
        name=f"get_history_{agent_name}",
    )

    return {
        "success": True,
        "message": "History request sent (response will arrive via WebSocket)",
        "request_id": req_id,  # ✅ Returns request_id
    }
```

### 2. Backend Router (agents.py)

❌ **NOT UPDATED for fire-and-forget:**
```python
@router.get("/{agent_name}/history", response_model=list[MessageInfo])
async def get_history(...) -> list[MessageInfo]:
    response = await service.get_agent_history(agent_name, limit, offset)

    # ❌ ERROR: Expects a synchronous response with the history
    history = response.get("history", [])  # Receives []

    # ❌ ERROR: Returns an empty list instead of request_id
    return messages  # []
```

**It should be:**
```python
@router.get("/{agent_name}/history")
async def get_history(...) -> dict[str, Any]:
    response = await service.get_agent_history(agent_name, limit, offset)

    # ✅ Return the immediate response with request_id
    return response  # {"success": true, "request_id": "...", "message": "..."}
```

### 3. Frontend (main.js)

❌ **NOT UPDATED for fire-and-forget:**
```javascript
async function loadAgentHistory(name) {
    const response = await fetch(`/api/agents/${name}/history`);
    const history = await response.json();  // ❌ Expects an array

    // ❌ Handles it as a synchronous response
    for (let i = history.length - 1; i >= 0; i--) {
        // ...
    }
}
```

**It should be:**
```javascript
async function loadAgentHistory(name) {
    const requestId = generateUUID();

    // Store pending request
    const promise = new Promise((resolve, reject) => {
        pendingRequests.set(requestId, { resolve, reject, ... });
    });

    // Send request
    await fetch(`/api/agents/${name}/history?request_id=${requestId}`);

    // Wait for WebSocket response
    const history = await promise;

    // Update UI
    displayHistory(history);
}
```

### 4. WebSocket Handlers

✅ **ADDED but NOT USED:**
```javascript
// Handler exists (line 336-349)
function handleAgentResponse(data) {
    // ...
    loadAgentHistory(agent_name);  // ✅ Triggers a reload
}

// ❌ BUT handleHistoryResponse DOES NOT EXIST!
// It should be:
function handleHistoryResponse(data) {
    const { request_id, history } = data;
    // Correlate with pending request
    // Update UI with history
}
```

---

## Why it works in the reference web application

The reference web application uses a **SYNCHRONOUS** approach:

```python
# Reference web application - sessions.py (line 468)
dialog_text = await session.get_history_formatted_async()
```

- Uses `await` and blocks until the result is received
- Does NOT use fire-and-forget
- Returns the data directly in the REST response
- The frontend receives the data immediately

**In deep_worker_web we migrated to fire-and-forget, but did NOT update:**
1. The router endpoints (they still expect synchronous data)
2. The frontend code (it still expects data from REST rather than WebSocket)

---

## Affected components

### Backend

1. ❌ `app/routers/agents.py::get_history()` (lines 145-199)
   - Must return `{"success": true, "request_id": "..."}` instead of `list[MessageInfo]`
   - Must change `response_model` from `list[MessageInfo]` to `dict[str, Any]`

2. ❌ `app/routers/agents.py::get_status()` (lines 262-311)
   - The same problem - returns a fallback instead of request_id

### Frontend

1. ❌ `static/js/main.js::loadAgentHistory()` (lines 166-208)
   - Must use request correlation
   - Must wait for the WebSocket response

2. ❌ `static/js/main.js` - `handleHistoryResponse()` IS MISSING
   - A handler for the `history_response` WebSocket message must be added

3. ❌ `static/js/main.js::handleWebSocketMessage()` (lines 288-311)
   - Does NOT handle `history_response`
   - Does NOT handle `detailed_status_response`

4. ❌ `static/dialog.html` - if it exists
   - Must be updated for fire-and-forget

---

## Comparison: What works vs What does NOT work

### ✅ Works

1. **send_message** (invoke_agent)
   - ✅ Backend: fire-and-forget implemented
   - ✅ Frontend: `handleAgentResponse()` handles the WebSocket
   - ✅ UI: Shows notifications

2. **subscribe_messages**
   - ✅ Backend: fire-and-forget
   - ✅ The WebSocket callback works

3. **get_status (connection)**
   - ✅ Backend: fire-and-forget
   - ✅ Broadcast works

### ❌ Does NOT work

1. **get_history**
   - ❌ The router returns `[]` instead of request_id
   - ❌ The frontend expects a synchronous array
   - ❌ The WebSocket handler is NOT IMPLEMENTED
   - ❌ Request correlation is NOT IMPLEMENTED

2. **get_status (detailed)**
   - ❌ The router returns a fallback
   - ❌ The frontend does NOT handle the WebSocket response

3. **Full Dialog**
   - ❌ Does NOT work because of the get_history problem

---

## Solution

### Option 1: Roll get_history back to synchronous (FAST)

**Pros:**
- Minimal changes
- Works like the reference web application
- The history is displayed immediately

**Cons:**
- The UI may block while loading a large history
- Does not match the fire-and-forget architecture
- Contradicts the goal of the migration

### Option 2: Complete the fire-and-forget migration (THE RIGHT WAY)

**Changes:**

1. **Backend Router:**
   ```python
   @router.get("/{agent_name}/history")
   async def get_history(...) -> dict[str, Any]:
       response = await service.get_agent_history(agent_name, limit, offset)
       return response  # {"success": true, "request_id": "..."}
   ```

2. **Frontend - Request Correlation:**
   ```javascript
   const pendingRequests = new Map();

   async function loadAgentHistory(name) {
       const requestId = generateUUID();

       const promise = new Promise((resolve, reject) => {
           pendingRequests.set(requestId, {
               resolve, reject,
               timeout: setTimeout(() => {
                   reject(new Error('Timeout'));
                   pendingRequests.delete(requestId);
               }, 20000),
           });
       });

       await fetch(`/api/agents/${name}/history?request_id=${requestId}`);

       const history = await promise;
       displayHistory(history);
   }
   ```

3. **Frontend - WebSocket Handler:**
   ```javascript
   function handleHistoryResponse(data) {
       const { request_id, history } = data;

       const pending = pendingRequests.get(request_id);
       if (pending) {
           clearTimeout(pending.timeout);
           pending.resolve(history);
           pendingRequests.delete(request_id);
       }
   }

   // Add to handleWebSocketMessage switch
   case 'history_response':
       handleHistoryResponse(data);
       break;
   ```

**Pros:**
- Completes the fire-and-forget migration
- The UI never blocks
- Matches the design
- Scalable

**Cons:**
- More changes
- Requires testing

### Option 3: Hybrid approach (COMPROMISE)

- `get_history` - roll back to synchronous (UI needs it immediately)
- `invoke_agent` - keep fire-and-forget (it can be slow)

---

## Recommendation

**I recommend Option 2** - complete the fire-and-forget migration properly:

1. It matches the original goal of the migration
2. It eliminates potential UI blocking
3. The code will be consistent
4. The design and examples are already written in DEVELOPER_EXAMPLES.md

**Why NOT Option 1:**
- The history can be large (100+ messages)
- It contradicts the goal of the migration (eliminating blocking)
- It creates an inconsistency in the architecture

---

## Priority

🔴 **CRITICAL** - Blocks core functionality (viewing agent responses)

---

## Next steps

1. ✅ Problem analysis performed
2. ⏳ Waiting for the user's decision (which option to choose)
3. ⏳ Implementation of the chosen option
4. ⏳ Testing with a real agent
5. ⏳ Documentation update
6. ⏳ Commit the fix

---

**File:** `docs/ANALYSIS_HISTORY_ISSUE.md`
