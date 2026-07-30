// Deep-Worker-Web - Main JavaScript

// State
let ws = null;
let selectedAgent = null;
let agents = {};
let selectedAgents = new Set(); // Multi-select for batch operations
// request_id → pending entry. Two coexisting shapes discriminated by
// ``type``:
//   * { type: 'last_response' | 'status' | 'history' | …,
//       agentName, resolve, reject, timeout }
//       Used by request/response promise correlation (``loadLastResponse``
//       etc.) — a WS event with the matching request_id resolves the
//       promise.
//   * { type: 'button', btn, originalText, op, agentName, startedAt }
//       Used by the lifecycle button tracker (``trackButton`` /
//       ``finalizeRequest``) so stop/restart/batch buttons can be
//       re-enabled on the terminal WS event with the matching request_id.
// Request IDs are UUIDs (client-side) or server-issued uuid4 hex —
// both are globally unique, so the two use-cases coexist safely in one
// Map.
let pendingRequests = new Map();
let currentApprovalId = null;
let autoApproveEnabled = false;

// Generate UUID for request correlation
function generateUUID() {
    return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function(c) {
        const r = Math.random() * 16 | 0;
        const v = c == 'x' ? r : (r & 0x3 | 0x8);
        return v.toString(16);
    });
}

// Initialize on page load
document.addEventListener('DOMContentLoaded', () => {
    console.log('Deep-Worker-Web starting...');
    loadAgents();
    connectWebSocket();
    setupEventListeners();
});

// Load agents from API
async function loadAgents() {
    try {
        const response = await fetch('/api/agents');
        if (!response.ok) throw new Error('Failed to load agents');
        
        const agentList = await response.json();
        console.log(`Loaded ${agentList.length} agents`);
        
        // Store in state
        agentList.forEach(agent => {
            agents[agent.name] = agent;
        });
        
        renderAgentList();
    } catch (error) {
        console.error('Error loading agents:', error);
        showError('Failed to load agents. Make sure agents are running.');
    }
}

// Render agent list in UI
function renderAgentList() {
    const container = document.getElementById('agent-list');
    
    if (Object.keys(agents).length === 0) {
        container.innerHTML = '<p class="loading">No agents running</p>';
        return;
    }
    
    container.innerHTML = '';
    
    for (const [name, agent] of Object.entries(agents)) {
        const item = document.createElement('div');
        item.className = 'agent-item';
        if (selectedAgent === name) {
            item.classList.add('selected');
        }
        
        const statusClass = getStatusDotClass(agent.status);
        const textClass = getStatusTextClass(agent.status);
        const statusLabel = getStatusDisplayLabel(agent);
        const checked = selectedAgents.has(name) ? 'checked' : '';
        item.innerHTML = `
            <div class="agent-item-row">
                <input type="checkbox" class="agent-checkbox" data-agent="${name}" ${checked}
                       onclick="event.stopPropagation(); toggleAgentSelection('${name}', this.checked)">
                <div class="agent-item-info" onclick="selectAgent('${name}')">
                    <div class="agent-item-name">
                        <span class="status-indicator ${statusClass}"></span>
                        <strong>${name}</strong>
                        <span class="${textClass}">[${statusLabel}]</span>
                    </div>
                    <div class="agent-item-status">${agent.provider} / ${agent.model}</div>
                </div>
            </div>
        `;
        
        container.appendChild(item);
    }
    updateBatchToolbar();
}

function getStatusDisplayLabel(agent) {
    if (!agent || !agent.status) return 'unknown';
    const status = agent.status;
    const metadata = agent.status_metadata;
    const substatus = metadata && typeof metadata.substatus === 'string'
        ? metadata.substatus.trim()
        : '';
    if (status === 'idle' && substatus) {
        return `idle(${substatus})`;
    }
    return status;
}

// Get CSS class for status indicator dot (matches the web UI)
//
// IMPORTANT: the 'interrupted' check MUST come before any
// status.includes('waiting') check. Otherwise — depending on string
// content — interrupted could leak into the waiting branch. As a guard,
// keep 'interrupted' as an exact-match early-return.
function getStatusDotClass(status) {
    if (status === 'ready' || status === 'idle') return 'status-ready';
    if (status === 'working' || status === 'running') return 'status-working';
    if (status === 'error') return 'status-error';
    if (status === 'interrupted') return 'status-interrupted';
    if (status.includes('waiting')) return 'status-waiting';
    return '';
}

// Get CSS class for status text (matches the web UI)
function getStatusTextClass(status) {
    if (status === 'ready' || status === 'idle') return 'status-text-ready';
    if (status === 'working' || status === 'running') return 'status-text-working';
    if (status === 'error') return 'status-text-error';
    if (status === 'interrupted') return 'status-text-interrupted';
    if (status.includes('waiting')) return 'status-text-waiting_input';
    return 'status-text-unknown';
}

// Decide whether a stop/restart *failure* event means the agent is GONE
// (dead / not in the registry) rather than a transient failure against a live
// agent. When true, the failed-op handler should remove the card — otherwise
// a phantom card for an already-dead agent lingers and "Stop" appears to do
// nothing (observed in a past incident). Backend signals this via a LookupError raised
// from request_stop ("Agent '<name>' not found"). We match on error_type AND
// the message text so a backend wording/type tweak can't silently regress.
// Pure function (no DOM/globals) so it is unit-testable in node.
function stopFailureMeansAgentGone(data) {
    if (!data || typeof data !== 'object') return false;
    const errorType = String(data.error_type || '');
    const error = String(data.error || '');
    if (errorType === 'LookupError') return true;
    if (/\bnot found\b/i.test(error)) return true;
    if (/\bnot running\b/i.test(error)) return true;
    return false;
}

// Select an agent
function selectAgent(name) {
    console.log(`Selecting agent: ${name}`);
    selectedAgent = name;

    // Update UI
    renderAgentList();

    const agent = agents[name];
    if (!agent) return;

    // Show selected agent panel
    document.getElementById('no-agent-selected').style.display = 'none';
    document.getElementById('agent-selected').style.display = 'flex';

    // Update agent info
    document.getElementById('selected-agent-name').textContent = name;
    document.getElementById('agent-provider').textContent = agent.provider;
    document.getElementById('agent-model').textContent = agent.model;

    // Per-agent uptime: push started_at into the DOM once, then start
    // a single local clock-tick that re-renders every second. NO
    // network — the value is already in memory.
    setAgentUptimeStartedAt(agent.started_at);
    startAgentUptimeTicker();

    // Update Stop TaskManager button state AND TM control row
    // visibility so the row appears immediately based on the info
    // we already have in ``agents[name]`` (updated by
    // handleAgentStatus / initial snapshot), without waiting for
    // the next status event.
    updateStopTaskManagerButton(agent);
    updateTmControls(agent);

    // Reflect the agent's stored error state in the persistent panel:
    // show it if this agent is in error, clear it otherwise. This makes
    // the error survive an agent re-select / page reload.
    if (agent.status === 'error' && agent.error_message) {
        showLastError(agent.error_message);
    } else {
        dismissLastError();
    }

    // Load last assistant response
    loadLastResponse(name);
}

// Setup event listeners
function setupEventListeners() {
    // Send message button
    document.getElementById('send-btn').onclick = sendMessage;
    
    // Enter key in textarea
    document.getElementById('message-input').addEventListener('keydown', (e) => {
        if (e.ctrlKey && e.key === 'Enter') {
            sendMessage();
        }
    });
    
    // Full dialog button + verbosity dropdown
    document.getElementById('full-dialog-btn').onclick = openFullDialog;
    restoreFullDialogVerbosity();

    // Interrupt button — cancels the current agent invocation
    // (process keeps running, ready for next user message)
    const interruptBtn = document.getElementById('interrupt-btn');
    if (interruptBtn) {
        interruptBtn.onclick = interruptAgent;
    }

    // Stop TaskManager button
    document.getElementById('stop-taskmanager-btn').onclick = stopTaskManager;
}

// Interrupt currently running agent invocation.
// The agent process keeps running; only the in-flight task is cancelled.
async function interruptAgent() {
    if (!selectedAgent) return;

    const btn = document.getElementById('interrupt-btn');
    const original = btn ? btn.textContent : null;
    if (btn) {
        btn.disabled = true;
        btn.textContent = '⏸ Interrupting…';
    }

    try {
        const response = await fetch(`/api/agents/${selectedAgent}/interrupt`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
        });

        if (!response.ok) {
            const detail = await response.text();
            throw new Error(`HTTP ${response.status}: ${detail}`);
        }

        // The actual confirmation arrives via WebSocket (agent_interrupted /
        // agent_interrupt_result). We just acknowledge the dispatch here.
        showNotification(`Interrupt sent to ${selectedAgent}`, 'success');
    } catch (error) {
        console.error('Error interrupting agent:', error);
        showNotification(`Failed to interrupt: ${error.message}`, 'error');
    } finally {
        if (btn) {
            btn.disabled = false;
            btn.textContent = original || '⏸ Interrupt';
        }
    }
}

// Send message to agent
async function sendMessage() {
    if (!selectedAgent) {
        alert('Please select an agent first');
        return;
    }
    
    const messageInput = document.getElementById('message-input');
    const message = messageInput.value.trim();
    
    if (!message) {
        alert('Please enter a message');
        return;
    }
    
    console.log(`Sending message to ${selectedAgent}: ${message}`);
    
    try {
        const response = await fetch(`/api/agents/${selectedAgent}/message`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ message }),
        });
        
        if (!response.ok) throw new Error('Failed to send message');
        
        // Clear input
        messageInput.value = '';
        
        // Show success
        console.log('Message sent successfully');
    } catch (error) {
        console.error('Error sending message:', error);
        alert('Failed to send message. Please try again.');
    }
}

// Copy last response to clipboard (matches the web UI copyLastResponse)
async function copyLastResponse(event) {
    const content = document.getElementById('last-response-content');
    const text = content.textContent;

    if (!text || text === 'No response yet') {
        return;
    }

    try {
        await navigator.clipboard.writeText(text);

        const btn = event.target;
        const originalText = btn.textContent;
        btn.textContent = '✅ Copied!';
        btn.style.background = '#2a5a2a';

        setTimeout(() => {
            btn.textContent = originalText;
            btn.style.background = '';
        }, 2000);
    } catch (err) {
        console.error('Copy error:', err);
    }
}

// ---- Last Error panel ---------------------------------------------------
// The error panel is SEPARATE from the Last Response panel. It is
// persistent (never auto-dismissed), copyable, and shows the full
// agent error text. It never overwrites the model's last response.
// In a past incident, errors used to be a transient toast that
// vanished and could not be copied.

// Show (or update) the persistent error panel with the given text.
function showLastError(errorText) {
    const panel = document.getElementById('last-error-panel');
    const content = document.getElementById('last-error-content');
    if (!panel || !content) return;
    content.textContent = errorText || '';
    panel.style.display = '';
}

// Hide the error panel (manual dismiss only).
function dismissLastError() {
    const panel = document.getElementById('last-error-panel');
    const content = document.getElementById('last-error-content');
    if (panel) panel.style.display = 'none';
    if (content) content.textContent = '';
}

// Copy the current error text to the clipboard.
async function copyLastError(event) {
    const content = document.getElementById('last-error-content');
    const text = content ? content.textContent : '';
    if (!text) return;
    try {
        await navigator.clipboard.writeText(text);
        const btn = event.target;
        const originalText = btn.textContent;
        btn.textContent = '✅ Copied!';
        btn.style.background = '#2a5a2a';
        setTimeout(() => {
            btn.textContent = originalText;
            btn.style.background = '';
        }, 2000);
    } catch (err) {
        console.error('Copy error:', err);
    }
}

// localStorage key for the persisted verbosity choice (per design §3.6 UI).
const FULL_DIALOG_VERBOSITY_KEY = 'deep_worker_web.dialog_verbosity';

// Allowed Full Dialog verbosity values. 'system_prompt' and 'tools' are
// introspection modes (§9 of DESIGN_FULL_DIALOG_FINAL_RESPONSE_PAIRS.md):
// they show the invariant startup system prompt and the provider-dependent
// tool list instead of conversation turns.
const FULL_DIALOG_VERBOSITY_VALUES = ['final', 'text', 'full', 'system_prompt', 'tools'];

// Read verbosity from the dropdown; fall back to 'final'.
function getFullDialogVerbosity() {
    const sel = document.getElementById('full-dialog-verbosity');
    const v = sel ? sel.value : 'final';
    return FULL_DIALOG_VERBOSITY_VALUES.includes(v) ? v : 'final';
}

// Restore persisted verbosity choice into the dropdown on page load.
function restoreFullDialogVerbosity() {
    const sel = document.getElementById('full-dialog-verbosity');
    if (!sel) return;
    try {
        const saved = localStorage.getItem(FULL_DIALOG_VERBOSITY_KEY);
        if (saved && FULL_DIALOG_VERBOSITY_VALUES.includes(saved)) {
            sel.value = saved;
        }
    } catch (e) {
        // localStorage unavailable — ignore
    }
    sel.addEventListener('change', () => {
        try {
            localStorage.setItem(FULL_DIALOG_VERBOSITY_KEY, sel.value);
        } catch (e) { /* ignore */ }
    });
}

// Open full dialog in new tab. Verbosity comes from the dropdown.
function openFullDialog() {
    if (!selectedAgent) return;

    const verbosity = getFullDialogVerbosity();
    const url = `/api/agents/${selectedAgent}/dialog-text?verbosity=${encodeURIComponent(verbosity)}`;
    window.open(url, '_blank');
}

// Load agent history using fire-and-forget with request correlation
async function loadAgentHistory(name) {
    const requestId = generateUUID();

    try {
        // Create promise for this request
        const historyPromise = new Promise((resolve, reject) => {
            pendingRequests.set(requestId, {
                type: 'history',
                agentName: name,
                resolve,
                reject,
                timeout: setTimeout(() => {
                    pendingRequests.delete(requestId);
                    reject(new Error('History request timed out'));
                }, 20000), // 20s frontend timeout
            });
        });

        // Send fire-and-forget request
        const response = await fetch(`/api/agents/${name}/history?request_id=${requestId}`);
        if (!response.ok) {
            pendingRequests.delete(requestId);
            console.error('Failed to request agent history');
            return;
        }

        const result = await response.json();
        console.log(`History request sent for ${name}, request_id: ${requestId}`);

        // Wait for WebSocket response
        const history = await historyPromise;

        // Process and display history
        displayHistory(history, name);

    } catch (error) {
        console.error('Error loading agent history:', error);
        showNotification(`Failed to load history: ${error.message}`, 'error');
    }
}

// Display history (extracted from loadAgentHistory)
function displayHistory(history, name) {
    console.log(`Processing ${history.length} messages for ${name}`);

    // Find last assistant message (final response)
    let lastMessage = null;
    for (let i = history.length - 1; i >= 0; i--) {
        const msg = history[i];
        if (msg.type === 'AIMessage' && msg.content && msg.content.trim().length > 0) {
            lastMessage = msg;
            break;
        }
    }

    // Update last response display
    const responseContent = document.getElementById('last-response-content');
    if (lastMessage && lastMessage.content) {
        responseContent.textContent = lastMessage.content;
        console.log(`Updated last response: ${lastMessage.content.substring(0, 100)}...`);
    } else {
        responseContent.textContent = 'No response yet';
        console.log('No substantial messages found in history');
    }
}

// Load last assistant response using fire-and-forget with request correlation
async function loadLastResponse(name) {
    const requestId = generateUUID();

    try {
        const responsePromise = new Promise((resolve, reject) => {
            pendingRequests.set(requestId, {
                type: 'last_response',
                agentName: name,
                resolve,
                reject,
                timeout: setTimeout(() => {
                    pendingRequests.delete(requestId);
                    reject(new Error('Last response request timed out'));
                }, 10000),
            });
        });

        const response = await fetch(`/api/agents/${name}/last-response?request_id=${requestId}`);
        if (!response.ok) {
            pendingRequests.delete(requestId);
            console.error('Failed to request last response');
            return;
        }

        console.log(`Last response request sent for ${name}, request_id: ${requestId}`);

        const lastResponse = await responsePromise;

        // Update UI
        const responseContent = document.getElementById('last-response-content');
        if (lastResponse && lastResponse.content) {
            responseContent.textContent = lastResponse.content;
            console.log(`Updated last response: ${lastResponse.content.substring(0, 100)}...`);
        } else {
            responseContent.textContent = 'No response yet';
        }
    } catch (error) {
        console.error('Error loading last response:', error);
    }
}

// Handle last response result (fire-and-forget success callback)
function handleLastResponseResult(data) {
    const { request_id, agent_name, last_response } = data;
    console.log(`Last response result for ${agent_name}, request_id: ${request_id}`);

    const pending = pendingRequests.get(request_id);
    if (pending && pending.type !== 'button') {
        clearTimeout(pending.timeout);
        pending.resolve(last_response);
        pendingRequests.delete(request_id);
    } else if (agent_name === selectedAgent) {
        // No pending request — direct UI update
        const responseContent = document.getElementById('last-response-content');
        if (last_response && last_response.content) {
            responseContent.textContent = last_response.content;
        }
    }
}

// Handle last response error (fire-and-forget error callback)
function handleLastResponseError(data) {
    const { request_id, agent_name, error } = data;
    console.error(`Last response error for ${agent_name}:`, error);

    const pending = pendingRequests.get(request_id);
    if (pending && pending.type !== 'button') {
        clearTimeout(pending.timeout);
        pending.reject(new Error(error));
        pendingRequests.delete(request_id);
    }
}

// Handle approval request from agent
function handleApprovalRequest(data) {
    const { agent_name, approval_id, actions } = data;
    console.log(`Approval request for ${agent_name}:`, actions);

    if (agent_name !== selectedAgent) return;

    // If auto-approve is enabled, send approve immediately
    if (autoApproveEnabled) {
        sendApprovalDecision(agent_name, approval_id, 'approve');
        return;
    }

    // Show approval panel
    currentApprovalId = approval_id;
    const panel = document.getElementById('approval-panel');
    const details = document.getElementById('approval-details');

    // Format action details
    let html = '';
    for (const action of actions) {
        const name = action.name || 'unknown';
        const args = action.args || {};
        const desc = action.description || '';
        html += `<strong>${name}</strong>`;
        if (desc) html += `\n${desc}`;
        else html += `\n${JSON.stringify(args, null, 2)}`;
        html += '\n';
    }
    details.textContent = html.trim();

    panel.style.display = 'block';
}

// Handle approval result
function handleApprovalResult(data) {
    const { agent_name, approval_id, decision } = data;
    console.log(`Approval result for ${agent_name}: ${decision}`);

    // Hide panel
    const panel = document.getElementById('approval-panel');
    if (panel) panel.style.display = 'none';
    currentApprovalId = null;
}

// Send approval decision to backend
async function sendApprovalDecision(agentName, approvalId, decision) {
    try {
        const response = await fetch(`/api/agents/${agentName}/approve`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ approval_id: approvalId, decision }),
        });
        if (!response.ok) console.error('Failed to send approval');
    } catch (error) {
        console.error('Error sending approval:', error);
    }
}

// Approve button handler
function approveAction() {
    if (!selectedAgent || !currentApprovalId) return;
    sendApprovalDecision(selectedAgent, currentApprovalId, 'approve');
    document.getElementById('approval-panel').style.display = 'none';
    currentApprovalId = null;
}

// Reject button handler
function rejectAction() {
    if (!selectedAgent || !currentApprovalId) return;
    sendApprovalDecision(selectedAgent, currentApprovalId, 'reject');
    document.getElementById('approval-panel').style.display = 'none';
    currentApprovalId = null;
}

// Auto-approve all button handler
function autoApproveAll() {
    if (!selectedAgent || !currentApprovalId) return;
    autoApproveEnabled = true;
    sendApprovalDecision(selectedAgent, currentApprovalId, 'auto_approve_all');
    document.getElementById('approval-panel').style.display = 'none';
    currentApprovalId = null;
    showNotification('Auto-approve enabled for this session', 'success');
}

// Update Stop TaskManager button state based on agent status
function updateStopTaskManagerButton(agent) {
    const stopBtn = document.getElementById('stop-taskmanager-btn');
    if (!stopBtn) return;

    // Enable button only if TaskManager is active
    const taskManagerActive = agent.task_manager_active === true;
    stopBtn.disabled = !taskManagerActive;

    // Update button text to show state
    if (taskManagerActive) {
        stopBtn.textContent = '⏹️ Stop TaskManager';
        stopBtn.title = 'Stop the running TaskManager';
    } else {
        stopBtn.textContent = '⏹️ Stop TaskManager';
        stopBtn.title = 'TaskManager is not active';
    }
}

// Stop TaskManager (fire-and-forget, result arrives via WebSocket).
async function stopTaskManager() {
    if (!selectedAgent) return;

    if (!confirm('Stop TaskManager for this agent?')) return;

    const btn = document.getElementById('stop-taskmanager-btn');
    try {
        const response = await fetch(`/api/agents/${selectedAgent}/stop_task_manager`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ force: false }),
        });

        if (!response.ok) {
            showNotification(`Failed to stop TaskManager (HTTP ${response.status})`, 'error');
            return;
        }
        const data = await response.json();
        trackButton(data.request_id, btn, 'task_manager_stop', selectedAgent, '⏳ Stopping TM…');
        showNotification('TaskManager stop requested', 'info');
    } catch (error) {
        showNotification('Error stopping TaskManager: ' + error.message, 'error');
    }
}

// Toggle TaskManager enabled/disabled
async function toggleTaskManager(enabled) {
    if (!selectedAgent) return;
    // Update the local cache IMMEDIATELY so the choice sticks when the
    // user switches to another agent and back. ``selectAgent`` /
    // ``updateTmControls`` read ``agents[name].task_manager_disabled``;
    // if we only POST and rely on a later status broadcast, a switch-
    // away-and-back in between re-checks the box from the stale cache.
    // (The backend also broadcasts agent_status with the new value, and
    // the agent is the source of truth — but the local write makes the
    // UX correct instantly and independent of broadcast timing.)
    // User report 2026-05-20: disabling TM did not stick across
    // re-select. Disabling must be permanent until the user re-enables.
    const agent = agents[selectedAgent];
    if (agent) {
        agent.task_manager_disabled = !enabled;
        agent.has_task_manager = true;
    }
    try {
        await fetch(`/api/agents/${selectedAgent}/task-manager/toggle`, {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({enabled}),
        });
        showNotification(`TaskManager ${enabled ? 'enabled' : 'disabled'}`, 'success');
    } catch (error) {
        console.error('Error toggling TaskManager:', error);
    }
}

// Reset TaskManager to initial state
async function resetTaskManager() {
    if (!selectedAgent) return;
    if (!confirm('Reset TaskManager? This clears all evaluation history.')) return;
    try {
        const response = await fetch(`/api/agents/${selectedAgent}/task-manager/reset`, {
            method: 'POST',
        });
        if (response.ok) {
            showNotification('TaskManager reset to initial state', 'success');
        }
    } catch (error) {
        console.error('Error resetting TaskManager:', error);
    }
}

// WebSocket connection
function connectWebSocket() {
    // Use current host for WebSocket connection
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws/agents`;
    console.log(`Connecting to WebSocket: ${wsUrl}`);
    ws = new WebSocket(wsUrl);
    
    ws.onopen = () => {
        console.log('WebSocket connected');
        // Start ping interval
        setInterval(() => {
            if (ws.readyState === WebSocket.OPEN) {
                ws.send(JSON.stringify({ type: 'ping' }));
            }
        }, 30000);
    };
    
    ws.onmessage = (event) => {
        try {
            const data = JSON.parse(event.data);
            handleWebSocketMessage(data);
        } catch (error) {
            console.error('Error parsing WebSocket message:', error);
        }
    };
    
    ws.onerror = (error) => {
        console.error('WebSocket error:', error);
    };
    
    ws.onclose = () => {
        console.log('WebSocket closed, reconnecting...');
        setTimeout(connectWebSocket, 3000);
    };
}

// -------- Fire-and-forget lifecycle: per-request UI tracker --------------
//
// Every button click against a fire-and-forget endpoint gets a request_id
// back. We register it in the shared ``pendingRequests`` Map (declared at
// the top of this file) so the matching WS terminal event
// (_completed/_failed) can re-enable the button. Entries here use
// ``type: 'button'`` to distinguish them from promise-correlation entries
// produced by ``loadLastResponse`` / history / status flows.

function trackButton(requestId, btn, op, agentName, pendingText) {
    if (!requestId || !btn) return;
    const originalText = btn.textContent;
    pendingRequests.set(requestId, {
        type: 'button',
        btn,
        originalText,
        op,
        agentName,
        startedAt: Date.now(),
    });
    btn.disabled = true;
    if (pendingText) btn.textContent = pendingText;
}

function finalizeRequest(requestId) {
    const entry = pendingRequests.get(requestId);
    if (!entry || entry.type !== 'button') return;
    pendingRequests.delete(requestId);
    try {
        entry.btn.disabled = false;
        if (entry.originalText) entry.btn.textContent = entry.originalText;
    } catch (_) { /* button gone from DOM — fine */ }
}

function labelForOp(op) {
    const m = {
        agent_stop: 'Stop',
        agent_restart: 'Restart',
        agent_spawn: 'Start',
        batch: 'Batch',
        task_manager_stop: 'Stop TaskManager',
        agent_summarize: 'Summarize context',
    };
    return m[op] || op;
}

// Handle WebSocket messages
function handleWebSocketMessage(data) {
    console.log('WebSocket message:', data);

    // Update uptime from any message that includes it
    if (data.uptime) {
        updateUptimeDisplay(data.uptime);
    }

    switch (data.type) {
        case 'agent_message':
            handleAgentMessage(data);
            break;
        case 'agent_response':
            handleAgentResponse(data);
            break;
        case 'agent_error':
            handleAgentError(data);
            break;
        case 'agent_summarized':
            handleAgentSummarized(data);
            break;
        case 'history_response':
            handleHistoryResponse(data);
            break;
        case 'history_error':
            handleHistoryError(data);
            break;
        case 'agent_metadata':
            handleAgentMetadata(data);
            break;
        case 'approval_request':
            handleApprovalRequest(data);
            break;
        case 'approval_result':
            handleApprovalResult(data);
            break;
        case 'last_response_result':
            handleLastResponseResult(data);
            break;
        case 'last_response_error':
            handleLastResponseError(data);
            break;
        case 'detailed_status_response':
            handleDetailedStatusResponse(data);
            break;
        case 'detailed_status_error':
            handleDetailedStatusError(data);
            break;
        case 'agent_status':
            handleAgentStatus(data);
            break;
        case 'agent_discovered':
            handleAgentDiscovered(data);
            break;
        case 'agent_started_at_changed':
            handleAgentStartedAtChanged(data);
            break;
        case 'agent_terminated':
            handleAgentTerminated(data);
            break;
        case 'agent_interrupted':
            handleAgentInterrupted(data);
            break;
        case 'agent_interrupt_result':
            handleAgentInterruptResult(data);
            break;
        case 'channel_quarantined':
            // A client RPyC channel to the agent was flooded and quarantined.
            // This is a WARNING ONLY — the agent keeps working and its status
            // is unchanged. Show a non-blocking warning; do NOT touch the
            // agent's status badge or the error panel. Observed in a past incident.
            handleChannelQuarantined(data);
            break;

        // ---- Fire-and-forget lifecycle events ----
        case 'agent_stop_started':
        case 'agent_restart_started':
        case 'agent_spawn_started':
        case 'batch_started':
        case 'agent_shutdown_requested':
            // Informational — UI already knows (we just clicked). Log only.
            console.log(`lifecycle ${data.type}: ${data.agent_name || ''} (${data.request_id || ''})`);
            break;
        case 'agent_stop_escalated':
            showNotification(
                `Escalated to SIGKILL: ${data.agent_name} (grace expired)`,
                'info',
            );
            break;
        case 'batch_progress':
            // Throttled progress; log only — summary shows final state.
            if (data.payload) {
                console.log(
                    `batch ${data.payload.action}: ${data.payload.completed}/${data.payload.total}`,
                );
            }
            break;
        case 'agent_stop_completed':
        case 'agent_restart_completed':
        case 'agent_spawn_completed':
            finalizeRequest(data.request_id);
            showNotification(
                `${labelForOp(data.type.replace(/_completed$/, ''))} ok: ${data.agent_name || ''}`,
                'success',
            );
            break;
        case 'batch_completed':
            finalizeRequest(data.request_id);
            {
                const p = data.payload || {};
                showNotification(
                    `Batch ${p.action || ''}: ${p.completed || 0}/${p.total || 0}` +
                    (p.failed ? `, ${p.failed} failed` : ''),
                    p.failed ? 'error' : 'success',
                );
            }
            break;
        case 'agent_stop_failed':
        case 'agent_restart_failed':
        case 'agent_spawn_failed':
        case 'batch_failed':
            finalizeRequest(data.request_id);
            // If a stop/restart failed because the agent is GONE (not found /
            // not running), the card is a phantom — remove it so the UI is
            // consistent and "Stop" visibly resolves it (observed in a past incident).
            if (
                (data.type === 'agent_stop_failed' || data.type === 'agent_restart_failed') &&
                data.agent_name &&
                stopFailureMeansAgentGone(data)
            ) {
                handleAgentTerminated({ agent_name: data.agent_name });
                showNotification(
                    `${data.agent_name} was already gone — removed from the list.`,
                    'info',
                );
                break;
            }
            showNotification(
                `${labelForOp(data.type.replace(/_failed$/, ''))} failed: ${data.agent_name || ''} — ${data.error || 'unknown error'} (${data.error_type || ''})`,
                'error',
            );
            break;

        // ---- TaskManager stop (previously silent) ----
        case 'task_manager_stopped':
            showNotification(`TaskManager stopped: ${data.agent_name || ''}`, 'success');
            break;
        case 'task_manager_stop_error':
            showNotification(
                `TaskManager stop failed: ${data.agent_name || ''} — ${data.error || 'unknown'}`,
                'error',
            );
            break;

        case 'pong':
            // Ping response, ignore
            break;
        default:
            console.warn('Unknown message type:', data.type);
    }
}

// Handle agent message
function handleAgentMessage(data) {
    const { agent_name, message } = data;

    // Only update if this is the selected agent (as per design decision Q2)
    if (agent_name !== selectedAgent) {
        return;
    }

    // ========================================================================
    // No-Silent-API-Errors Policy (POLICIES.md §9): a streamed error event
    // (type === 'error') carries a real provider/API error. It MUST land in
    // the PERSISTENT error panel (copyable, never auto-dismissed) — NOT the
    // 5-second self-dismissing toast, and NOT silently ignored. This covers
    // recoverable provider/API errors where the agent stays IDLE (so there is
    // no status='error' snapshot to drive the panel), e.g. a non-terminal grok
    // ACP error. Fatal errors already reach the panel via the status snapshot.
    // ========================================================================
    if (message && message.type === 'error') {
        const errText = (typeof message.content === 'string' && message.content)
            ? message.content
            : 'Provider API error';
        showLastError(errText);
        // Do not fall through to Last-Response handling for an error event.
        return;
    }

    // ========================================================================
    // Update Last Response for ASSISTANT TEXT messages only
    // ========================================================================
    //
    // INCLUDE (show in Last Response):
    //   - type: "assistant" (final response in non-streaming mode)
    //   - type: "assistant_chunk" (progressive response in streaming mode)
    //
    // EXCLUDE (do NOT show in Last Response):
    //   - type: "tool_call" (even though role="assistant", this is a tool invocation)
    //   - type: "tool_result" (role="tool", this is tool output)
    //   - type: "user" (user messages)
    //   - type: "system" (system messages)
    //   - type: "error" (errors)
    //   - type: "token_update" (metadata)
    //
    // This ensures Last Response shows ONLY the agent's textual response,
    // not tool invocations or their outputs.
    //
    const isAssistantText = (
        message.role === 'assistant' &&                        // Must be from assistant
        (message.type === 'assistant' ||                       // Final response OR
         message.type === 'assistant_chunk') &&                // Streaming chunk
        message.content &&                                     // Must have content
        typeof message.content === 'string' &&                 // Content must be string
        message.content.trim().length > 0                      // Not empty/whitespace
    );

    if (isAssistantText) {
        const responseContent = document.getElementById('last-response-content');

        // Update with the latest content
        // In streaming mode, chunks arrive with accumulated content
        // In non-streaming mode, we get the final complete response
        responseContent.textContent = message.content;

        // Log for debugging
        console.log(`[Last Response] Updated (${message.type}): ${message.content.length} chars`);
    }
}

// Handle agent response (fire-and-forget callback)
function handleAgentResponse(data) {
    const { agent_name, response } = data;
    console.log(`Agent ${agent_name} responded:`, response);

    // This handler is for the fire-and-forget callback from send_message()
    // It indicates that the agent FINISHED processing, but the actual response
    // should have already arrived via the subscription (handleAgentMessage).
    //
    // We keep this handler ONLY for:
    // 1. Success notification
    // 2. History reload
    //
    // We DO NOT update Last Response here - that's done via handleAgentMessage()
    // from the real-time WebSocket subscription.

    // Only update if this is the selected agent
    if (agent_name !== selectedAgent) {
        return;
    }

    // Display success notification
    showNotification(`Message processed successfully`, 'success');

    // NOTE: Last Response update removed - handled by handleAgentMessage()
    // The response should have already been displayed via WebSocket subscription

    // Reload last response
    loadLastResponse(agent_name);
}

// Handle agent error (fire-and-forget error callback)
function handleAgentError(data) {
    const { agent_name, error, error_type } = data;
    console.error(`Agent ${agent_name} error:`, error);

    // Short, friendly summary for the transient toast.
    let summary = `Agent error: ${error}`;
    if (error_type === 'TimeoutError') {
        summary = `Agent not responding (timeout)`;
    } else if (error_type === 'ConnectionError' || error_type === 'EOFError') {
        summary = `Connection to agent lost`;
    }
    showNotification(summary, 'error');
    if (data.request_id) finalizeRequest(data.request_id);

    // For the selected agent, surface the FULL error text persistently
    // in the dedicated error panel — copyable, does not auto-dismiss,
    // and does NOT overwrite the model's last response.
    if (agent_name === selectedAgent) {
        const fullText = error_type ? `${error_type}: ${error}` : String(error);
        showLastError(fullText);
        // Intentionally NOT reloading last-response here: the error must
        // not replace the previous successful model response.
    }
}

function handleAgentSummarized(data) {
    finalizeRequest(data.request_id);
    if (data.summarized) {
        showNotification(`Context summarized for '${data.agent_name}'`, 'success');
    } else {
        showNotification(`No context could be compacted for '${data.agent_name}'`, 'info');
    }
}

// Handle history response (fire-and-forget success callback)
function handleHistoryResponse(data) {
    const { request_id, agent_name, history } = data;
    console.log(`History response for ${agent_name}, request_id: ${request_id}, ${history.length} messages`);

    // Find pending request
    const pending = pendingRequests.get(request_id);
    if (pending && pending.type !== 'button') {
        clearTimeout(pending.timeout);
        pending.resolve(history);
        pendingRequests.delete(request_id);
    } else {
        // No pending request - might be auto-refresh
        if (agent_name === selectedAgent) {
            displayHistory(history, agent_name);
        }
    }
}

// Handle history error (fire-and-forget error callback)
function handleHistoryError(data) {
    const { request_id, agent_name, error } = data;
    console.error(`History error for ${agent_name}:`, error);

    // Find pending request
    const pending = pendingRequests.get(request_id);
    if (pending && pending.type !== 'button') {
        clearTimeout(pending.timeout);
        pending.reject(new Error(error));
        pendingRequests.delete(request_id);
    }

    showNotification(`Failed to load history: ${error}`, 'error');
}

// Handle detailed status response (fire-and-forget success callback)
function handleDetailedStatusResponse(data) {
    const { request_id, agent_name, status } = data;
    console.log(`Detailed status response for ${agent_name}, request_id: ${request_id}`);

    // Find pending request
    const pending = pendingRequests.get(request_id);
    if (pending && pending.type !== 'button') {
        clearTimeout(pending.timeout);
        pending.resolve(status);
        pendingRequests.delete(request_id);
    }

    // Update UI if this is the selected agent
    if (agent_name === selectedAgent && status) {
        document.getElementById('agent-provider').textContent = status.provider;
        document.getElementById('agent-model').textContent = status.model;

        // Update TaskManager button
        const agent = agents[agent_name];
        if (agent) {
            agent.task_manager_active = status.task_manager_active;
            updateStopTaskManagerButton(agent);
        }
    }
}

// Handle detailed status error (fire-and-forget error callback)
function handleDetailedStatusError(data) {
    const { request_id, agent_name, error } = data;
    console.error(`Detailed status error for ${agent_name}:`, error);

    // Find pending request
    const pending = pendingRequests.get(request_id);
    if (pending && pending.type !== 'button') {
        clearTimeout(pending.timeout);
        pending.reject(new Error(error));
        pendingRequests.delete(request_id);
    }

    showNotification(`Failed to get status: ${error}`, 'error');
}

// Handle agent status change
// Handle agent metadata update (provider/model fetched after connect)
function handleAgentMetadata(data) {
    const { agent_name, provider, model, status_metadata } = data;
    if (agents[agent_name]) {
        if (provider) agents[agent_name].provider = provider;
        if (model) agents[agent_name].model = model;
        if (status_metadata && typeof status_metadata === 'object') {
            agents[agent_name].status_metadata = status_metadata;
        } else if ('status_metadata' in data) {
            agents[agent_name].status_metadata = null;
        }

        if (agent_name === selectedAgent) {
            document.getElementById('agent-provider').textContent = agents[agent_name].provider;
            document.getElementById('agent-model').textContent = agents[agent_name].model;
        }
    }
}

function handleAgentStatus(data) {
    const { agent_name, status, task_manager_active, task_manager_disabled, error_message, status_metadata } = data;

    if (agents[agent_name]) {
        agents[agent_name].status = status;
        if (status_metadata && typeof status_metadata === 'object') {
            agents[agent_name].status_metadata = status_metadata;
        } else {
            agents[agent_name].status_metadata = null;
        }
        // Keep the last known error text on the agent record so a page
        // reload / agent re-select can re-show it.
        if (status === 'error' && error_message) {
            agents[agent_name].error_message = error_message;
        } else if (status !== 'error') {
            // Recovered → drop the stored error.
            agents[agent_name].error_message = null;
        }
        if (task_manager_active !== undefined) {
            agents[agent_name].task_manager_active = task_manager_active;
            agents[agent_name].has_task_manager = true;
        }
        if (task_manager_disabled !== undefined) {
            agents[agent_name].task_manager_disabled = task_manager_disabled;
        }

        // Update UI if this is the selected agent
        if (agent_name === selectedAgent) {
            updateStopTaskManagerButton(agents[agent_name]);
            updateTmControls(agents[agent_name]);
            // Persistent error panel mirrors the agent's error state.
            if (status === 'error' && error_message) {
                showLastError(error_message);
            } else if (status !== 'error') {
                dismissLastError();
            }
        }

        renderAgentList();
    }
}

// Update TaskManager controls visibility and state
function updateTmControls(agent) {
    const toggle = document.getElementById('tm-toggle-label');
    const resetBtn = document.getElementById('reset-tm-btn');
    const stopBtn = document.getElementById('stop-taskmanager-btn');
    if (!toggle || !resetBtn || !stopBtn) return;

    if (agent.has_task_manager) {
        toggle.style.display = 'flex';
        resetBtn.style.display = '';
        stopBtn.style.display = '';
        const checkbox = document.getElementById('tm-enabled-checkbox');
        if (checkbox) {
            checkbox.checked = !agent.task_manager_disabled;
        }
    } else {
        toggle.style.display = 'none';
        resetBtn.style.display = 'none';
        stopBtn.style.display = 'none';
    }
}

// Handle agent discovered
function handleAgentDiscovered(data) {
    const agent = data.agent;
    console.log('Agent discovered:', agent.name);

    agents[agent.name] = agent;
    renderAgentList();

    // Covers the ungraceful-restart path: reconcile briefly sees the
    // agent as GONE, then rediscovers it with a fresh started_at.
    // Push the new value into the uptime panel and reset the ticker.
    if (agent.name === selectedAgent) {
        setAgentUptimeStartedAt(agent.started_at);
        startAgentUptimeTicker();
    }
}

// Handle graceful restart of an already-known agent: only started_at
// changed (pid may or may not change; that's a separate concern
// handled by reconcile). Reset the uptime ticker for the panel.
function handleAgentStartedAtChanged(data) {
    const { agent_name, started_at } = data;
    if (!agent_name || !started_at) return;
    if (agents[agent_name]) {
        agents[agent_name].started_at = started_at;
    }
    if (agent_name === selectedAgent) {
        setAgentUptimeStartedAt(started_at);
        startAgentUptimeTicker();
    }
}

// Handle agent terminated
function handleAgentTerminated(data) {
    const { agent_name } = data;
    console.log('Agent terminated:', agent_name);

    delete agents[agent_name];

    // Drop from batch multi-select too — otherwise the checkbox
    // for a dead agent stays ticked in memory, and the batch
    // toolbar keeps counting it as "selected" even though the
    // row is gone from the list. Re-render reads ``selectedAgents``
    // via ``has(name)``; cleaning it keeps the Set honest.
    if (selectedAgents && selectedAgents.has(agent_name)) {
        selectedAgents.delete(agent_name);
    }

    // Deselect if it was the selected agent
    if (selectedAgent === agent_name) {
        selectedAgent = null;
        document.getElementById('no-agent-selected').style.display = 'flex';
        document.getElementById('agent-selected').style.display = 'none';
        // Stop the per-agent uptime ticker — nothing to tick against
        // anymore. (Single-instance guarantee also enforced at
        // startAgentUptimeTicker.)
        stopAgentUptimeTicker();
    }

    renderAgentList();
}

// Handle agent_interrupted broadcast (sent by agent itself when its
// current invocation is cancelled — also fires when interrupt comes from
// another client like dwcli).
function handleAgentInterrupted(data) {
    const agentName = data.agent_name || (data.message && data.message.agent_name);
    console.log('[Interrupt] agent_interrupted received:', data);

    // Optimistically reflect the new state in the local agent map so the
    // UI shows the agent as interrupted (amber) until the next status
    // poll confirms it. INTERRUPTED is distinct from WAITING_INPUT:
    // - INTERRUPTED = user paused this turn; awaiting next user message
    //   (often a correction). Rendered in amber.
    // - WAITING_INPUT = inside an in-progress turn, waiting for a tool to
    //   return. Rendered in blue.
    const target = agentName || selectedAgent;
    if (target && agents[target]) {
        agents[target].status = 'interrupted';
        agents[target].interrupted = true;
        renderAgentList();
    }

    if (target === selectedAgent) {
        showNotification(
            `Agent ${target} interrupted — awaiting your input`,
            'info',
        );
    }
}

// Handle agent_interrupt_result — service-level acknowledgment that the
// interrupt RPC dispatched (success or failure on transport).
function handleAgentInterruptResult(data) {
    const { agent_name, success, message, error } = data;
    if (success) {
        console.log(`[Interrupt] dispatched for ${agent_name}: ${message || 'ok'}`);
    } else {
        console.warn(`[Interrupt] failed for ${agent_name}: ${error || 'unknown'}`);
        if (agent_name === selectedAgent) {
            showNotification(
                `Interrupt failed: ${error || 'unknown error'}`,
                'error',
            );
        }
    }
}

// Channel-quarantine warning. A client's RPyC channel to the agent was
// flooded and quarantined. The agent KEEPS WORKING and its status is
// unchanged — so we show a non-blocking warning toast only, and deliberately
// do NOT touch the agent's status badge or the "Last Error" panel.
// Observed in a past incident.
function handleChannelQuarantined(data) {
    const { agent_name, peer, connid } = data;
    console.warn(
        `[channel_quarantined] agent=${agent_name} connid=${connid} peer=${peer} — ` +
        `a client flooded an RPyC channel; it was quarantined. Agent keeps working.`,
    );
    // Only surface a toast for the agent the user is looking at, to avoid
    // noise from background agents.
    if (agent_name === selectedAgent) {
        showNotification(
            `Warning: a client connection to ${agent_name} was flooded and ` +
            `quarantined. The agent keeps working — status unchanged.`,
            'warning',
        );
    }
}

// Show error message
function showError(message) {
    const container = document.getElementById('agent-list');
    container.innerHTML = `<p class="loading" style="color: red;">${message}</p>`;
}

// Show notification to user
function showNotification(message, type = 'info') {
    // Create notification element
    const notification = document.createElement('div');
    notification.className = `notification notification-${type}`;
    notification.textContent = message;

    // Add to page (create container if needed)
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

    // Style notification
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
    } else if (type === 'warning') {
        notification.style.backgroundColor = '#ff9800';
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

// ==============================================================================
// SERVER RESTART FUNCTIONALITY
// ==============================================================================

async function restartServer() {
    if (!confirm('Are you sure you want to restart deep-worker-web server?\n\nAll agent connections will be re-established after restart.')) {
        return;
    }

    try {
        // Send restart request
        await fetch('/api/server/restart', {
            method: 'POST',
            signal: AbortSignal.timeout(3000)  // Don't wait forever
        }).catch(() => {
            // Ignore errors - server might die before responding
        });

        showNotification('Server restart initiated...', 'info');

        // Wait for server to restart and reconnect
        setTimeout(() => {
            showNotification('Reconnecting to server...', 'info');
            connectWebSocket();  // Reconnect WebSocket
        }, 3000);

    } catch (error) {
        console.error('Error restarting server:', error);
        showNotification('Error initiating server restart', 'error');
    }
}

// ==============================================================================
// UPTIME DISPLAY
// ==============================================================================

function updateUptimeDisplay(uptimeData) {
    const uptimeEl = document.getElementById('server-uptime');
    if (uptimeEl && uptimeData && uptimeData.uptime_formatted) {
        uptimeEl.textContent = `Uptime: ${uptimeData.uptime_formatted}`;
    }
}

// ==============================================================================
// AGENT UPTIME DISPLAY (per-agent, event-driven; NO polling)
// ==============================================================================
//
// The agent's ``started_at`` is delivered event-driven:
//   * initial GET /api/agents snapshot,
//   * WS ``agent_discovered`` (new or ungracefully-restarted agent),
//   * WS ``agent_started_at_changed`` (graceful restart of a known agent).
//
// The 1-second setInterval below is NOT a polling loop — it makes NO
// network requests. It is a pure clock-tick that re-renders the already-
// received ``started_at`` relative to ``Date.now()``. See
// docs/DESIGN_AGENT_UPTIME_UI.md.

let _agentUptimeTickerId = null;

// Pure function — safely tested in node.
function formatAgentUptime(startedAtIso, nowMs) {
    if (!startedAtIso) return '-';
    const startedMs = Date.parse(startedAtIso);
    if (!Number.isFinite(startedMs)) return '-';
    // Clamp negative delta (clock skew: started_at in the future).
    let deltaSec = Math.max(0, Math.floor((nowMs - startedMs) / 1000));

    const days = Math.floor(deltaSec / 86400);
    deltaSec -= days * 86400;
    const hours = Math.floor(deltaSec / 3600);
    deltaSec -= hours * 3600;
    const mins = Math.floor(deltaSec / 60);
    const secs = deltaSec - mins * 60;

    const pad = (n) => String(n).padStart(2, '0');

    if (days > 0) return `${days}d ${pad(hours)}:${pad(mins)}:${pad(secs)}`;
    if (hours > 0) return `${pad(hours)}:${pad(mins)}:${pad(secs)}`;
    if (mins > 0) return `${pad(mins)}:${pad(secs)}`;
    return `${secs}s`;
}

function refreshAgentUptimeDisplay() {
    const el = document.getElementById('agent-uptime');
    if (!el) return;
    const startedAtIso = el.getAttribute('data-started-at') || '';
    el.textContent = formatAgentUptime(startedAtIso, Date.now());
}

function startAgentUptimeTicker() {
    // Single-instance guarantee — always stop the previous ticker before
    // starting a new one. Prevents leaked intervals when the user quickly
    // switches selectedAgent or after multiple ``agent_started_at_changed``
    // events for the same panel.
    stopAgentUptimeTicker();
    refreshAgentUptimeDisplay();
    _agentUptimeTickerId = setInterval(refreshAgentUptimeDisplay, 1000);
}

function stopAgentUptimeTicker() {
    if (_agentUptimeTickerId !== null) {
        clearInterval(_agentUptimeTickerId);
        _agentUptimeTickerId = null;
    }
}

function setAgentUptimeStartedAt(startedAtIso) {
    const el = document.getElementById('agent-uptime');
    if (!el) return;
    el.setAttribute('data-started-at', startedAtIso || '');
    refreshAgentUptimeDisplay();
}

// Note: node-test export is bundled with the trailing module.exports
// block near the end of this file (a single exports object avoids
// clobbering earlier assignments).

// ==============================================================================
// AGENT SELECTION & BATCH OPERATIONS
// ==============================================================================

function toggleAgentSelection(name, checked) {
    if (checked) {
        selectedAgents.add(name);
    } else {
        selectedAgents.delete(name);
    }
    updateBatchToolbar();
    // Update "select all" checkbox state
    const selectAllEl = document.getElementById('select-all-checkbox');
    if (selectAllEl) {
        const totalAgents = Object.keys(agents).length;
        selectAllEl.checked = selectedAgents.size === totalAgents && totalAgents > 0;
        selectAllEl.indeterminate = selectedAgents.size > 0 && selectedAgents.size < totalAgents;
    }
}

function toggleSelectAll(checked) {
    if (checked) {
        Object.keys(agents).forEach(name => selectedAgents.add(name));
    } else {
        selectedAgents.clear();
    }
    renderAgentList();
}

function updateBatchToolbar() {
    const toolbar = document.getElementById('batch-toolbar');
    const countEl = document.getElementById('selection-count');
    if (!toolbar || !countEl) return;

    if (selectedAgents.size > 0) {
        toolbar.style.display = 'flex';
        countEl.textContent = `${selectedAgents.size} selected`;
    } else {
        toolbar.style.display = 'none';
    }
}

async function batchStop() {
    if (selectedAgents.size === 0) return;
    const names = Array.from(selectedAgents);
    const btn = document.getElementById('batch-stop-btn');

    try {
        const response = await fetch('/api/agents/batch', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ action: 'stop', agent_names: names }),
        });
        if (!response.ok) {
            const err = await response.json().catch(() => ({}));
            showNotification(`Batch stop rejected: ${err.detail || response.status}`, 'error');
            return;
        }
        const data = await response.json();
        // Track the aggregate batch id — batch_completed will release us.
        trackButton(data.request_id, btn, 'batch', names.join(','), '⏳ Stopping…');
        showNotification(`Stop requested for ${names.length} agent(s)`, 'info');
        selectedAgents.clear();
        renderAgentList();
    } catch (error) {
        showNotification('Error stopping agents: ' + error.message, 'error');
    }
}

async function batchRestart() {
    if (selectedAgents.size === 0) return;
    const names = Array.from(selectedAgents);
    const btn = document.getElementById('batch-restart-btn');

    try {
        const response = await fetch('/api/agents/batch', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ action: 'restart', agent_names: names }),
        });
        if (!response.ok) {
            const err = await response.json().catch(() => ({}));
            showNotification(`Batch restart rejected: ${err.detail || response.status}`, 'error');
            return;
        }
        const data = await response.json();
        trackButton(data.request_id, btn, 'batch', names.join(','), '⏳ Restarting…');
        showNotification(`Restart requested for ${names.length} agent(s)`, 'info');
        selectedAgents.clear();
        renderAgentList();
    } catch (error) {
        showNotification('Error restarting agents: ' + error.message, 'error');
    }
}

async function stopCurrentAgent() {
    if (!selectedAgent) return;
    const btn = document.getElementById('stop-agent-btn');
    try {
        const response = await fetch(`/api/agents/${selectedAgent}/stop`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({}),
        });
        if (!response.ok) {
            const err = await response.json().catch(() => ({}));
            showNotification(`Stop rejected: ${err.detail || response.status}`, 'error');
            return;
        }
        const data = await response.json();
        trackButton(data.request_id, btn, 'agent_stop', selectedAgent, '⏳ Stopping…');
        showNotification(`Stop requested for '${selectedAgent}'`, 'info');
    } catch (error) {
        showNotification('Error stopping agent: ' + error.message, 'error');
    }
}

async function restartCurrentAgent() {
    if (!selectedAgent) return;
    const btn = document.getElementById('restart-agent-btn');
    try {
        const response = await fetch(`/api/agents/${selectedAgent}/restart`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({}),
        });
        if (!response.ok) {
            const err = await response.json().catch(() => ({}));
            showNotification(`Restart rejected: ${err.detail || response.status}`, 'error');
            return;
        }
        const data = await response.json();
        trackButton(data.request_id, btn, 'agent_restart', selectedAgent, '⏳ Restarting…');
        showNotification(`Restart requested for '${selectedAgent}'`, 'info');
    } catch (error) {
        showNotification('Error restarting agent: ' + error.message, 'error');
    }
}

async function summarizeCurrentAgent() {
    if (!selectedAgent) return;
    const btn = document.getElementById('summarize-agent-btn');
    try {
        const response = await fetch(`/api/agents/${selectedAgent}/summarize`, {
            method: 'POST',
        });
        if (!response.ok) {
            const err = await response.json().catch(() => ({}));
            showNotification(`Summarization rejected: ${err.detail || response.status}`, 'error');
            return;
        }
        const data = await response.json();
        trackButton(data.request_id, btn, 'agent_summarize', selectedAgent, 'Summarizing…');
        showNotification(`Summarization requested for '${selectedAgent}' (${data.request_id})`, 'info');
    } catch (error) {
        showNotification('Error requesting summarization: ' + error.message, 'error');
    }
}

// ---------------------------------------------------------------------------
// Test hook (no-op in the browser). Lets node-based unit tests import pure
// decision functions from this file without a DOM or a module bundler. The
// guard checks for CommonJS ``module`` which exists in node but not in the
// browser, so this line is inert when the page loads main.js as a <script>.
// ---------------------------------------------------------------------------
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        stopFailureMeansAgentGone,
        formatAgentUptime,
        handleAgentSummarized,
    };
}
