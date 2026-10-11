// Deep-Worker-Web - Main JavaScript

// State
let ws = null;
// Exposed on ``globalThis`` (window in browser, global in node) so that
// node-side unit tests can pre-seed the selected agent / ``agents``
// before driving the broadcast handlers.
//
// A related internal incident analysis (not included here), frontend section: the previous
// form was
//
//     let selectedAgent = (globalThis.selectedAgent = null);
//
// which copies the value ONCE at load. A ``let`` at module scope does
// not create a ``window`` property, so a later bare ``selectedAgent =
// name`` in ``selectAgent()`` updated only the module binding while
// ``globalThis.selectedAgent`` stayed ``null`` forever. Every handler
// that guards on ``globalThis.selectedAgent`` (handleAgentMessage,
// handleAgentContextChanged, _applyTokenUpdateIfFirstSeen) therefore
// early-returned on every event in a real browser — while node tests
// that pre-seed ``globalThis.selectedAgent`` directly stayed green,
// because they never exercise ``selectAgent()``.
//
// The fix: there is NO module-local binding any more. The selected
// agent lives solely as a ``globalThis`` property, and every read and
// write in this file goes through ``globalThis.selectedAgent``, so the
// two spellings cannot diverge because there is only one slot.
//
// A ``var`` alias was rejected deliberately: in a browser classic script
// ``var`` binds to the global object and would work, but under the
// node-side CommonJS test harness it is module-scoped and silently
// diverges again — i.e. the tests would keep passing while production
// broke, which is the exact failure mode this incident is about.
//
// ``agents`` below is safe as a plain alias: it is only ever MUTATED
// (``agents[name] = …``), never reassigned, so the object identity that
// was published on globalThis stays correct.
globalThis.selectedAgent = null;
let agents = (typeof globalThis !== 'undefined') ? (globalThis.agents = {}) : {};
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
        if (globalThis.selectedAgent === name) {
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

// Get CSS class for status indicator dot
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

// Get CSS class for status text
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
// nothing. Backend signals this via a LookupError raised
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
    globalThis.selectedAgent = name;

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

    // Context status bar — paint from in-memory AgentInfo mirror.
    _paintContextSpans(agent);

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

    // The MCP and Stats panels belong to the previously selected agent, so
    // drop their content and re-fetch only if the user actually opens them.
    // Selecting an agent must not trigger a stats recompute or an MCP read.
    mcpTabLoadedFor = null;
    statsTabLoadedFor = null;
    const mcpHost = document.getElementById('mcp-attached-list');
    if (mcpHost) mcpHost.replaceChildren();
    const statsHost = document.getElementById('stats-tab-content');
    if (statsHost) statsHost.replaceChildren();

    const activeTab = getActiveAgentTab();
    if (activeTab === 'mcp') loadMcpTab(name);
    if (activeTab === 'stats') loadStatsTab(name);
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
    restoreFullDialogMode();

    // Interrupt button — cancels the current agent invocation
    // (process keeps running, ready for next user message)
    const interruptBtn = document.getElementById('interrupt-btn');
    if (interruptBtn) {
        interruptBtn.onclick = interruptAgent;
    }

    // Stop TaskManager button
    document.getElementById('stop-taskmanager-btn').onclick = stopTaskManager;

    // Agent panel tabs and their lazily-loaded content.
    setupAgentTabs();
    setupMcpTab();
    setupStatsTab();
}

// Interrupt currently running agent invocation.
// The agent process keeps running; only the in-flight task is cancelled.
async function interruptAgent() {
    if (!globalThis.selectedAgent) return;

    const btn = document.getElementById('interrupt-btn');
    const original = btn ? btn.textContent : null;
    if (btn) {
        btn.disabled = true;
        btn.textContent = '⏸ Interrupting…';
    }

    try {
        const response = await fetch(`/api/agents/${globalThis.selectedAgent}/interrupt`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
        });

        if (!response.ok) {
            const detail = await response.text();
            throw new Error(`HTTP ${response.status}: ${detail}`);
        }

        // The actual confirmation arrives via WebSocket (agent_interrupted /
        // agent_interrupt_result). We just acknowledge the dispatch here.
        showNotification(`Interrupt sent to ${globalThis.selectedAgent}`, 'success');
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
    if (!globalThis.selectedAgent) {
        alert('Please select an agent first');
        return;
    }
    
    const messageInput = document.getElementById('message-input');
    const message = messageInput.value.trim();
    
    if (!message) {
        alert('Please enter a message');
        return;
    }
    
    console.log(`Sending message to ${globalThis.selectedAgent}: ${message}`);
    
    try {
        const response = await fetch(`/api/agents/${globalThis.selectedAgent}/message`, {
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

// Copy last response to clipboardasync function copyLastResponse(event) {
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
// Errors used to be a transient toast that
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

// localStorage key for the persisted Full Dialog ACTION mode
// (DESIGN_FULL_DIALOG_ACTION_MODE.md §3.2). Global, not per agent: survives
// agent switches and browser restarts.
const FULL_DIALOG_MODE_KEY = 'deep_worker_web.dialog_mode';

// What the 📜 button does:
//   tab      — open the static snapshot in a new tab (original behaviour);
//   copy     — fetch the snapshot and put it on the clipboard, render nothing;
//   download — save the snapshot as a .txt file, render nothing.
// copy/download exist for screen-reader users: rendering a huge plain-text
// tab makes Chrome build a full accessibility tree and serve the reader on
// its UI thread — the whole browser freezes (measured 57 s for 150 KB).
const FULL_DIALOG_MODE_VALUES = ['tab', 'copy', 'download'];

function getFullDialogMode() {
    const sel = document.getElementById('full-dialog-mode');
    const v = sel ? sel.value : 'tab';
    return FULL_DIALOG_MODE_VALUES.includes(v) ? v : 'tab';
}

function restoreFullDialogMode() {
    const sel = document.getElementById('full-dialog-mode');
    if (!sel) return;
    try {
        const saved = localStorage.getItem(FULL_DIALOG_MODE_KEY);
        if (saved && FULL_DIALOG_MODE_VALUES.includes(saved)) {
            sel.value = saved;
        }
    } catch (e) {
        // localStorage unavailable — ignore
    }
    sel.addEventListener('change', () => {
        try {
            localStorage.setItem(FULL_DIALOG_MODE_KEY, sel.value);
        } catch (e) { /* ignore */ }
    });
}

function fullDialogUrl(agent, verbosity, download) {
    const base = `/api/agents/${encodeURIComponent(agent)}/dialog-text?verbosity=${encodeURIComponent(verbosity)}`;
    return download ? `${base}&download=1` : base;
}

// Fetch the snapshot and place it on the clipboard. Nothing is rendered:
// the text lives only in a JS string and the clipboard. There is
// deliberately NO hidden-textarea / execCommand fallback — that would put
// the text back into the DOM and recreate the accessibility-tree freeze.
async function copyFullDialogToClipboard(agent, verbosity) {
    if (!navigator.clipboard || !navigator.clipboard.writeText) {
        showNotification('Clipboard API unavailable in this context — use "Download .txt"', 'error');
        return;
    }
    const resp = await fetch(fullDialogUrl(agent, verbosity, false), { cache: 'no-store' });
    const text = await resp.text();
    if (!resp.ok) {
        let reason = text;
        try { reason = JSON.parse(text).detail || text; } catch (e) { /* plain text */ }
        showNotification(`Full Dialog: HTTP ${resp.status} — ${reason}`, 'error');
        return;
    }
    await navigator.clipboard.writeText(text);
    showNotification(`Copied ${text.length.toLocaleString()} characters (${agent}, ${verbosity})`, 'success');
}

// Save the snapshot as a file. Same-origin <a download> never opens a tab and
// never renders; the server names the file (Content-Disposition). If the
// server answers 503/504/500 the error body is what gets saved — deliberate,
// the file then states the reason (design §6).
function downloadFullDialog(agent, verbosity) {
    const a = document.createElement('a');
    a.href = fullDialogUrl(agent, verbosity, true);
    a.download = `dialog_${agent}_${verbosity}.txt`;
    a.style.display = 'none';
    document.body.appendChild(a);
    a.click();
    a.remove();
    showNotification(`Downloading Full Dialog (${agent}, ${verbosity})`, 'info');
}

// The 📜 button. Verbosity and mode come from the two dropdowns.
//
// tab mode: the tab is a STATIC, CLICK-TIME SNAPSHOT (text/plain, no script,
// no streaming, Cache-Control: no-store) — it never updates itself.
// 'noopener' puts it in its own browsing-context group (separate renderer
// process), so a heavy dialog can never stall this page and vice versa.
// See docs/DESIGN_FULL_DIALOG_STATIC_SNAPSHOT.md.
function openFullDialog() {
    if (!globalThis.selectedAgent) return;

    const verbosity = getFullDialogVerbosity();
    const mode = getFullDialogMode();
    const agent = globalThis.selectedAgent;

    if (mode === 'copy') {
        const btn = document.getElementById('full-dialog-btn');
        if (btn) btn.disabled = true;
        copyFullDialogToClipboard(agent, verbosity)
            .catch((e) => showNotification(`Copy failed: ${e && e.message ? e.message : e}`, 'error'))
            .finally(() => { if (btn) btn.disabled = false; });
        return;
    }
    if (mode === 'download') {
        downloadFullDialog(agent, verbosity);
        return;
    }
    const url = fullDialogUrl(agent, verbosity, false);
    window.open(url, '_blank', 'noopener,noreferrer');
}

// Open the token/cache stats page for the selected agent (§6.2).
// The page fetches the snapshot itself on load; no polling.
function openAgentStats() {
    if (!globalThis.selectedAgent) return;
    const url = `/static/stats.html?agent=${encodeURIComponent(globalThis.selectedAgent)}`;
    window.open(url, '_blank', 'noopener,noreferrer');
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
    } else if (agent_name === globalThis.selectedAgent) {
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

    if (agent_name !== globalThis.selectedAgent) return;

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
    if (!globalThis.selectedAgent || !currentApprovalId) return;
    sendApprovalDecision(globalThis.selectedAgent, currentApprovalId, 'approve');
    document.getElementById('approval-panel').style.display = 'none';
    currentApprovalId = null;
}

// Reject button handler
function rejectAction() {
    if (!globalThis.selectedAgent || !currentApprovalId) return;
    sendApprovalDecision(globalThis.selectedAgent, currentApprovalId, 'reject');
    document.getElementById('approval-panel').style.display = 'none';
    currentApprovalId = null;
}

// Auto-approve all button handler
function autoApproveAll() {
    if (!globalThis.selectedAgent || !currentApprovalId) return;
    autoApproveEnabled = true;
    sendApprovalDecision(globalThis.selectedAgent, currentApprovalId, 'auto_approve_all');
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
        stopBtn.textContent = '⏹️ Stop TM';
        stopBtn.title = 'Stop the running TM (TaskManager)';
    } else {
        stopBtn.textContent = '⏹️ Stop TM';
        stopBtn.title = 'TM (TaskManager) is not active';
    }
}

// Stop TaskManager (fire-and-forget, result arrives via WebSocket).
async function stopTaskManager() {
    if (!globalThis.selectedAgent) return;

    if (!confirm('Stop TM (TaskManager) for this agent?')) return;

    const btn = document.getElementById('stop-taskmanager-btn');
    try {
        const response = await fetch(`/api/agents/${globalThis.selectedAgent}/stop_task_manager`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ force: false }),
        });

        if (!response.ok) {
            showNotification(`Failed to stop TaskManager (HTTP ${response.status})`, 'error');
            return;
        }
        const data = await response.json();
        trackButton(data.request_id, btn, 'task_manager_stop', globalThis.selectedAgent, '⏳ Stopping TM…');
        showNotification('TaskManager stop requested', 'info');
    } catch (error) {
        showNotification('Error stopping TaskManager: ' + error.message, 'error');
    }
}

// Toggle TaskManager enabled/disabled
async function toggleTaskManager(enabled) {
    if (!globalThis.selectedAgent) return;
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
    const agent = agents[globalThis.selectedAgent];
    if (agent) {
        agent.task_manager_disabled = !enabled;
        agent.has_task_manager = true;
    }
    try {
        await fetch(`/api/agents/${globalThis.selectedAgent}/task-manager/toggle`, {
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
    if (!globalThis.selectedAgent) return;
    if (!confirm('Reset TaskManager? This clears all evaluation history.')) return;
    try {
        const response = await fetch(`/api/agents/${globalThis.selectedAgent}/task-manager/reset`, {
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
        task_manager_stop: 'Stop TM',
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
        case 'agent_context_changed':
            // Auto-refresh contract: the backend fires this on every
            // status poll where the 6 context fields (agent_*/tm_*)
            // OR the operator-set ``context_limit`` changed
            // (diff-check in agent_service._maybe_broadcast_context_changed).
            // Without this case the operator has to refresh the page
            // to see any context update — regression from commit
            // 3bdf351 where the case was omitted, no test exercised
            // the switch itself, and the AST-guard did not exist yet.
            // See tests/unit/test_ws_router_context_changed.js and
            // tests/unit/test_ws_router_covers_backend_broadcasts.py.
            handleAgentContextChanged(data);
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
            // agent's status badge or the error panel.
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
        case 'agent_shutdown_rpc_failed': {
            // Backend attempted an in-process shutdown RPC and it failed;
            // it falls back to SIGTERM automatically, but the operator
            // should see WHY the graceful path did not work.  Warning
            // (not error) — the stop still proceeds.
            const p = data.payload || {};
            const reason = p.reason || 'unknown';
            const detail = p.detail ? ` (${p.detail})` : '';
            showNotification(
                `Graceful shutdown RPC failed for ${data.agent_name || ''}: ${reason}${detail} — falling back to SIGTERM`,
                'warning',
            );
            break;
        }
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
            // consistent and "Stop" visibly resolves it.
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
    // Single source of truth: the selected agent lives only as a
    // globalThis property (see the declaration at the top of this file).
    const sel = globalThis.selectedAgent;

    // Only update if this is the selected agent (as per design decision Q2)
    if (agent_name !== sel) {
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

    // Best-effort token_update propagation to the context status bar.
    // Only fires before a snapshot is known (post-snapshot rule: M7).
    _applyTokenUpdateIfFirstSeen(message);
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
    if (agent_name !== globalThis.selectedAgent) {
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
    if (agent_name === globalThis.selectedAgent) {
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
        if (agent_name === globalThis.selectedAgent) {
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
    if (agent_name === globalThis.selectedAgent && status) {
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

        if (agent_name === globalThis.selectedAgent) {
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
        if (agent_name === globalThis.selectedAgent) {
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
    if (agent.name === globalThis.selectedAgent) {
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
    if (agent_name === globalThis.selectedAgent) {
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
    if (globalThis.selectedAgent === agent_name) {
        globalThis.selectedAgent = null;
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
    const target = agentName || globalThis.selectedAgent;
    if (target && agents[target]) {
        agents[target].status = 'interrupted';
        agents[target].interrupted = true;
        renderAgentList();
    }

    if (target === globalThis.selectedAgent) {
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
        if (agent_name === globalThis.selectedAgent) {
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
function handleChannelQuarantined(data) {
    const { agent_name, peer, connid } = data;
    console.warn(
        `[channel_quarantined] agent=${agent_name} connid=${connid} peer=${peer} — ` +
        `a client flooded an RPyC channel; it was quarantined. Agent keeps working.`,
    );
    // Only surface a toast for the agent the user is looking at, to avoid
    // noise from background agents.
    if (agent_name === globalThis.selectedAgent) {
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
        // Announce toasts to screen readers (NVDA/JAWS): copy/download
        // results are ONLY reported through toasts. Polite = queued, does
        // not interrupt the current utterance.
        container.setAttribute('role', 'status');
        container.setAttribute('aria-live', 'polite');
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

// =====================================================================
// Context status bar (DESIGN_AGENT_CONTEXT_STATUS_BAR.md)
// Pure formatting helper exposed for both browser and node tests.
// =====================================================================
function _fmtK(n) {
    // "1500" -> "1.5k", "42100" -> "42.1k", "999" -> "999".
    if (n == null) return null;
    if (typeof n !== 'number' || !Number.isFinite(n)) return null;
    if (n < 1000) return String(n);
    const k = n / 1000;
    if (Number.isInteger(k)) return `${k}k`;
    // 1 decimal place, trim trailing zero when second decimal is zero.
    const s = k.toFixed(1);
    return s.endsWith('.0') ? `${Math.round(k)}k` : `${s}k`;
}

function formatAgentContext(used, size, semantics) {
    // No data at all → "-".
    if (used == null && size == null && semantics == null) return '-';

    // Snapshot semantics: provider reported BOTH used AND size. The UI
    // uses "X/Y" so the user can see how close they are to the limit.
    // Snapshot-without-size (used != null, size == null) is an anomaly
    // (provider declared a snapshot but didn't carry the size); render
    // "X/?" instead of a bare number so the user sees something is
    // wrong with the snapshot rather than mistaking it for a normal
    // tokens-only reading.
    if (semantics === 'context_snapshot') {
        const uStr = _fmtK(used == null ? 0 : used);
        const sStr = size == null ? '?' : _fmtK(size);
        return `${uStr}/${sStr}`;
    }
    // Tokens semantics (or null semantics): the provider returned token
    // counts but no size (no context_snapshot). Numbers come from the
    // provider's tokenizer — they are EXACT, not estimated. Render
    // just the count, no qualifier.
    //
    // An "(est.)" qualifier is reserved for a future path where the
    // system estimates token usage WITHOUT provider data (e.g. char/4
    // heuristics or a local tokenizer fallback). No such path exists
    // in the code today; flagging a provider-reported count as an
    // estimate would mislead the operator about its provenance.
    const uStr = _fmtK(used);
    if (uStr == null) return '-';
    return `${uStr} tokens`;
}

// Render the ``Last turn`` per-turn token total.  This is DELIBERATELY
// separate from ``formatAgentContext`` so the two numbers can never be
// confused with each other or with the operator-set ``-l`` limit.  The
// per-turn total is the sum of prompt + completion tokens the provider
// billed for the most recent response; it is NOT a cumulative context
// gauge and MUST NOT be compared with ``context_limit``.
// rg1_worker 2026-08-22 (Context: 2265.4k vs Limit: 400k regression).
function formatLastTurnTokens(total) {
    const s = _fmtK(total);
    if (s == null) return null;
    return `${s} tokens`;
}

// Render the operator-set context limit (-l flag). Pure formatter —
// isolated for unit testing in node. The shape is "400k (-l)" so the
// operator can see at a glance:
//   - the SIZE of the cap (so they know what the guard is holding);
//   - the SOURCE (the -l flag) — distinguishes it from a provider's
//     default window that happens to match (rg1_worker 2026-08-18 — the
//     original silent failure).
function formatContextLimit(limit) {
    // Missing (-l not set) → "-" — the UI must not invent a value
    // where the operator did not set one.
    if (limit == null) return '-';
    // Defensive: NaN / Infinity / negative are not valid -l values
    // (parse_context_limit normalises to a positive int). Render the
    // missing-marker rather than "null (-l)" or "NaN (-l)".
    const fmt = _fmtK(limit);
    if (fmt == null) return '-';
    return `${fmt} (-l)`;
}

// Render both the agent-context and (when applicable) the TM-context
// spans from the per-agent in-memory state. Treats `data-` attributes on
// the span as the source of truth so a manual refresh races correctly
// with broadcast-driven updates.
function _paintContextSpans(agentInfo) {
    const agentEl = document.getElementById('agent-context');
    const tmWrap = document.getElementById('agent-tm-context-wrap');
    const tmEl = document.getElementById('agent-tm-context');
    const limitEl = document.getElementById('agent-context-limit');
    const lastTurnWrap = document.getElementById('agent-last-turn-wrap');
    const lastTurnEl = document.getElementById('agent-last-turn');
    if (!agentEl) return;
    agentEl.textContent = formatAgentContext(
        agentInfo.agent_context_used,
        agentInfo.agent_context_size,
        agentInfo.agent_context_semantics,
    );
    if (limitEl) {
        limitEl.textContent = formatContextLimit(agentInfo.context_limit);
    }
    // ``Last turn`` row — visible only when the backend has an actual
    // per-turn number to show.  Keeping the wrap hidden when the number
    // is missing avoids a dangling ``Last turn: -`` that would look like
    // a broken field to the operator.
    // Suppress the row entirely when it would restate the ``Context``
    // number.  Under tokens semantics ``Context`` shows ``prompt_tokens``
    // and ``Last turn`` shows ``total_tokens``; a provider that reports no
    // completion accounting makes the two collide, and painting the same
    // value on two rows reads as a broken field.  Under snapshot semantics
    // ``Context`` renders as ``used/size``, so an equal per-turn total is
    // NOT a duplicate and the row stays.
    if (lastTurnWrap) {
        const isTokensSemantics = agentInfo.agent_context_semantics === 'tokens';
        const duplicatesContext = (
            isTokensSemantics
            && agentInfo.agent_last_turn_tokens != null
            && agentInfo.agent_last_turn_tokens === agentInfo.agent_context_used
        );
        const lastStr = duplicatesContext
            ? null
            : formatLastTurnTokens(agentInfo.agent_last_turn_tokens);
        if (lastStr != null) {
            lastTurnWrap.style.display = '';
            if (lastTurnEl) lastTurnEl.textContent = lastStr;
        } else {
            lastTurnWrap.style.display = 'none';
            if (lastTurnEl) lastTurnEl.textContent = '';
        }
    }
    if (tmWrap) {
        const hasTm = agentInfo.has_task_manager === true;
        tmWrap.style.display = hasTm ? '' : 'none';
        if (hasTm && tmEl) {
            tmEl.textContent = formatAgentContext(
                agentInfo.tm_context_used,
                agentInfo.tm_context_size,
                agentInfo.tm_context_semantics,
            );
        } else if (tmEl) {
            tmEl.textContent = '';
        }
    }
}

// Receives the backend ``agent_context_changed`` broadcast and mirrors
// the 6 fields into the in-memory agent_info, then re-paints the
// spans. Isolated as a separate export so the node test can drive
// snapshot + token_update sequences directly. Reads the selected agent
// and ``agents`` from ``globalThis`` so node-side tests can pre-seed
// them; in the browser ``globalThis === window`` (no-op).
function handleAgentContextChanged(data) {
    if (!data || data.agent_name !== globalThis.selectedAgent) return;
    const agentInfo = globalThis.agents[globalThis.selectedAgent];
    if (!agentInfo) return;
    agentInfo.agent_context_used = data.agent_context_used;
    agentInfo.agent_context_size = data.agent_context_size;
    agentInfo.agent_context_semantics = data.agent_context_semantics;
    // ``*_last_turn_tokens`` is a NEW field (rg1_worker 2026-08-22); older
    // backends won't send it, so keep the previous value when the key is
    // absent instead of clobbering it to ``undefined``.
    if ('agent_last_turn_tokens' in data) {
        agentInfo.agent_last_turn_tokens = data.agent_last_turn_tokens;
    }
    agentInfo.tm_context_used = data.tm_context_used;
    agentInfo.tm_context_size = data.tm_context_size;
    agentInfo.tm_context_semantics = data.tm_context_semantics;
    if ('tm_last_turn_tokens' in data) {
        agentInfo.tm_last_turn_tokens = data.tm_last_turn_tokens;
    }
    // Mirror the operator-set -l value so the limit span re-paints when
    // the agent restarts with a new limit. The backend diff-check
    // already collapses broadcasts when the value is unchanged.
    if ('context_limit' in data) {
        agentInfo.context_limit = data.context_limit;
    }
    _paintContextSpans(agentInfo);
}

// Inside ``handleAgentMessage`` the agent may push a ``token_update``
// carrying ``total_tokens`` (no kind, no semantics). For the FIRST
// message — before any snapshot has been seen — we use it as a
// best-effort estimate by storing it as
// ``agents[selected].agent_context_used = total_tokens`` and
// ``semantics = 'tokens'``. Once a snapshot is known, we MUST NOT
// overwrite it. TM NEVER receives a token_update (the TM uses
// ``manager_agent.ainvoke`` directly without a notify-callback).
function _applyTokenUpdateIfFirstSeen(message) {
    if (!message || message.type !== 'token_update') return;
    const agentInfo = globalThis.agents[globalThis.selectedAgent];
    if (!agentInfo) return;
    const totalTokens = message.total_tokens;
    if (typeof totalTokens !== 'number' || !Number.isFinite(totalTokens)) return;
    // Snapshot already known — leave it alone (M7).
    if (agentInfo.agent_context_semantics != null) return;
    agentInfo.agent_context_used = totalTokens;
    agentInfo.agent_context_semantics = 'tokens';
    _paintContextSpans(agentInfo);
}

function startAgentUptimeTicker() {
    // Single-instance guarantee — always stop the previous ticker before
    // starting a new one. Prevents leaked intervals when the user quickly
    // switches the selected agent or after multiple ``agent_started_at_changed``
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
    if (!globalThis.selectedAgent) return;
    const btn = document.getElementById('stop-agent-btn');
    try {
        const response = await fetch(`/api/agents/${globalThis.selectedAgent}/stop`, {
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
        trackButton(data.request_id, btn, 'agent_stop', globalThis.selectedAgent, '⏳ Stopping…');
        showNotification(`Stop requested for '${globalThis.selectedAgent}'`, 'info');
    } catch (error) {
        showNotification('Error stopping agent: ' + error.message, 'error');
    }
}

async function restartCurrentAgent() {
    if (!globalThis.selectedAgent) return;
    const btn = document.getElementById('restart-agent-btn');
    try {
        const response = await fetch(`/api/agents/${globalThis.selectedAgent}/restart`, {
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
        trackButton(data.request_id, btn, 'agent_restart', globalThis.selectedAgent, '⏳ Restarting…');
        showNotification(`Restart requested for '${globalThis.selectedAgent}'`, 'info');
    } catch (error) {
        showNotification('Error restarting agent: ' + error.message, 'error');
    }
}

async function summarizeCurrentAgent() {
    if (!globalThis.selectedAgent) return;
    const btn = document.getElementById('summarize-agent-btn');
    try {
        const response = await fetch(`/api/agents/${globalThis.selectedAgent}/summarize`, {
            method: 'POST',
        });
        if (!response.ok) {
            const err = await response.json().catch(() => ({}));
            showNotification(`Summarization rejected: ${err.detail || response.status}`, 'error');
            return;
        }
        const data = await response.json();
        trackButton(data.request_id, btn, 'agent_summarize', globalThis.selectedAgent, 'Summarizing…');
        showNotification(`Summarization requested for '${globalThis.selectedAgent}' (${data.request_id})`, 'info');
    } catch (error) {
        showNotification('Error requesting summarization: ' + error.message, 'error');
    }
}

// ===========================================================================
// Agent panel tabs (Dialog / MCP / Stats)
//
// No framework here: the tab switch follows the two patterns already used in
// this file — show/hide sibling panels, and localStorage for persistence.
// The inactive panel gets the ``hidden`` attribute rather than a display
// class so it leaves the accessibility tree entirely; a large hidden-but-
// present DOM is a measured browser-freeze risk under a screen reader.
// ===========================================================================

const AGENT_TAB_KEY = 'deepworker.agentTab';
const AGENT_TABS = ['dialog', 'mcp', 'stats'];

// Panels are populated lazily, once per selected agent, so switching agents
// does not fetch anything for tabs the user never opens.
let mcpTabLoadedFor = null;
let statsTabLoadedFor = null;

function getActiveAgentTab() {
    try {
        const stored = localStorage.getItem(AGENT_TAB_KEY);
        if (AGENT_TABS.includes(stored)) return stored;
    } catch (e) { /* private mode — fall through to the default */ }
    return 'dialog';
}

function switchAgentTab(tab) {
    if (!AGENT_TABS.includes(tab)) tab = 'dialog';

    AGENT_TABS.forEach(name => {
        const panel = document.getElementById(`tab-${name}`);
        const btn = document.getElementById(`tab-btn-${name}`);
        const active = name === tab;
        if (panel) panel.hidden = !active;
        if (btn) {
            btn.setAttribute('aria-selected', active ? 'true' : 'false');
            btn.classList.toggle('active', active);
        }
    });

    try { localStorage.setItem(AGENT_TAB_KEY, tab); } catch (e) { /* ignore */ }

    // Fetch on first activation only. Never on agent select, never on a timer.
    if (tab === 'mcp' && globalThis.selectedAgent && mcpTabLoadedFor !== globalThis.selectedAgent) {
        loadMcpTab(globalThis.selectedAgent);
    }
    if (tab === 'stats' && globalThis.selectedAgent && statsTabLoadedFor !== globalThis.selectedAgent) {
        loadStatsTab(globalThis.selectedAgent);
    }
}

function setupAgentTabs() {
    const bar = document.querySelector('.agent-tabs');
    if (!bar) return;
    const tabs = Array.from(bar.querySelectorAll('.agent-tab'));

    tabs.forEach(btn => {
        btn.onclick = () => switchAgentTab(btn.dataset.tab);
        // Arrow-key navigation is expected of a real tablist.
        btn.addEventListener('keydown', (e) => {
            const idx = tabs.indexOf(btn);
            let next = null;
            if (e.key === 'ArrowRight') next = tabs[(idx + 1) % tabs.length];
            if (e.key === 'ArrowLeft') next = tabs[(idx - 1 + tabs.length) % tabs.length];
            if (next) {
                e.preventDefault();
                next.focus();
                switchAgentTab(next.dataset.tab);
            }
        });
    });

    switchAgentTab(getActiveAgentTab());
}

// ===========================================================================
// MCP tab
// ===========================================================================

function mcpEl(tag, cls, text) {
    const node = document.createElement(tag);
    if (cls) node.className = cls;
    // textContent only — server names, commands and URLs are user-supplied.
    if (text !== undefined && text !== null) node.textContent = String(text);
    return node;
}

function mcpSpecSummary(spec) {
    if (!spec) return '';
    if (spec.url) return spec.url;
    const args = Array.isArray(spec.args) && spec.args.length ? ' ' + spec.args.join(' ') : '';
    return (spec.command || '') + args;
}

function renderMcpAttached(body, liveByServer) {
    const host = document.getElementById('mcp-attached-list');
    if (!host) return;
    host.replaceChildren();

    (body.layer_errors || []).forEach(err => {
        host.appendChild(mcpEl('div', 'mcp-error',
            `Layer "${err.layer}" could not be read: ${err.message}`));
    });

    const servers = body.servers || [];
    if (!servers.length) {
        host.appendChild(mcpEl('em', null, 'No MCP servers configured for this agent.'));
        return;
    }

    servers.forEach(srv => {
        const row = mcpEl('div', 'mcp-row');

        const attached = liveByServer && Object.prototype.hasOwnProperty.call(liveByServer, srv.name);
        const state = mcpEl('span', 'mcp-state', srv.disabled ? '✖' : (attached ? '✔' : '○'));
        state.title = srv.disabled ? 'Disabled'
            : (attached ? 'Attached and running' : 'Configured — applies after restart');
        row.appendChild(state);

        row.appendChild(mcpEl('span', 'mcp-name', srv.name));
        row.appendChild(mcpEl('span', 'mcp-transport', (srv.spec && srv.spec.type) || 'stdio'));
        row.appendChild(mcpEl('span', 'mcp-target', mcpSpecSummary(srv.spec)));

        const badges = mcpEl('span', 'mcp-badges');
        badges.appendChild(mcpEl('span', 'mcp-badge', srv.layer));
        if (srv.weight === 'heavy') {
            const b = mcpEl('span', 'mcp-badge mcp-badge-warn', 'heavy');
            b.title = srv.weight_reason || 'May be slow to start.';
            badges.appendChild(b);
        }
        if (srv.spec && srv.spec.auto_approve) {
            const b = mcpEl('span', 'mcp-badge mcp-badge-warn', 'auto-approve');
            b.title = 'Tools from this server skip approval prompts.';
            badges.appendChild(b);
        }
        if (srv.env_status === 'unresolved') {
            const b = mcpEl('span', 'mcp-badge mcp-badge-warn', 'env?');
            b.title = 'Unset variable(s): ' + (srv.unresolved_vars || []).join(', ') +
                '. If also unset for the agent, it will fail to start.';
            badges.appendChild(b);
        }
        // Tool counts come only from what the agent actually reports.
        // A configured-but-not-attached server shows no count at all —
        // never a 0, which would read as "this server has no tools".
        if (attached) {
            badges.appendChild(mcpEl('span', 'mcp-badge',
                `${liveByServer[srv.name].length} tools`));
        }
        row.appendChild(badges);

        const actions = mcpEl('span', 'mcp-actions');
        if (srv.editable) {
            const toggle = mcpEl('button', 'btn btn-secondary btn-sm',
                srv.disabled ? 'Enable' : 'Disable');
            toggle.onclick = () => setMcpEnabled(srv.name, !!srv.disabled);
            actions.appendChild(toggle);

            const del = mcpEl('button', 'btn btn-danger btn-sm', '✕');
            del.title = 'Remove from this agent';
            del.onclick = () => removeMcpServer(srv.name);
            actions.appendChild(del);
        } else {
            const note = mcpEl('span', 'mcp-readonly', `from ${srv.layer}`);
            note.title = 'Defined in a shared layer; edit it there or override it here.';
            actions.appendChild(note);
        }
        row.appendChild(actions);

        host.appendChild(row);
    });
}

async function loadMcpTab(agentName) {
    mcpTabLoadedFor = agentName;
    const host = document.getElementById('mcp-attached-list');
    if (host) host.replaceChildren(mcpEl('em', null, 'Loading…'));

    try {
        const resp = await fetch(`/api/agents/${encodeURIComponent(agentName)}/mcp`);
        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
        const body = await resp.json();

        // Live tools are best-effort: a stopped agent must not break the tab.
        let liveByServer = null;
        try {
            const liveResp = await fetch(
                `/api/agents/${encodeURIComponent(agentName)}/mcp/live`);
            if (liveResp.ok) {
                const live = await liveResp.json();
                if (live.success) liveByServer = live.by_server || {};
            }
        } catch (e) { /* agent not running — leave counts absent */ }

        renderMcpAttached(body, liveByServer);

        const hint = document.getElementById('mcp-restart-hint');
        if (hint) {
            const pending = (body.servers || []).some(s =>
                !s.disabled && liveByServer &&
                !Object.prototype.hasOwnProperty.call(liveByServer, s.name));
            hint.hidden = !pending;
        }
    } catch (error) {
        if (host) {
            host.replaceChildren(mcpEl('div', 'mcp-error',
                'Could not load MCP config: ' + error.message));
        }
    }
}

async function loadMcpCatalog(rescan) {
    const host = document.getElementById('mcp-catalog-list');
    if (host) host.replaceChildren(mcpEl('em', null, rescan ? 'Scanning…' : 'Loading…'));

    try {
        const resp = rescan
            ? await fetch('/api/mcp/catalog/scan', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({}),
            })
            : await fetch('/api/mcp/catalog');
        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
        const body = await resp.json();
        renderMcpCatalog(body.servers || []);
    } catch (error) {
        if (host) {
            host.replaceChildren(mcpEl('div', 'mcp-error',
                'Could not load the catalog: ' + error.message));
        }
    }
}

function renderMcpCatalog(servers) {
    const host = document.getElementById('mcp-catalog-list');
    if (!host) return;
    host.replaceChildren();

    if (!servers.length) {
        host.appendChild(mcpEl('em', null,
            'Nothing found yet. Press Scan to look for configured servers, or Add one manually.'));
        return;
    }

    servers.forEach(entry => {
        const row = mcpEl('div', 'mcp-row');
        row.appendChild(mcpEl('span', 'mcp-name', entry.label || entry.id));
        row.appendChild(mcpEl('span', 'mcp-transport', (entry.spec && entry.spec.type) || 'stdio'));
        row.appendChild(mcpEl('span', 'mcp-target', mcpSpecSummary(entry.spec)));

        const badges = mcpEl('span', 'mcp-badges');
        if (entry.weight && entry.weight !== 'unknown') {
            const b = mcpEl('span',
                'mcp-badge' + (entry.weight === 'heavy' ? ' mcp-badge-warn' : ''),
                entry.weight);
            if (entry.weight_reason) b.title = entry.weight_reason;
            badges.appendChild(b);
        }
        if (entry.pinned) badges.appendChild(mcpEl('span', 'mcp-badge', 'manual'));
        if (entry.stale) {
            const b = mcpEl('span', 'mcp-badge mcp-badge-warn', 'stale');
            b.title = 'Not seen in the latest scan. Kept in case it comes back.';
            badges.appendChild(b);
        }
        // Only ever shown after an explicit Test filled tools_hint.
        if (Array.isArray(entry.tools_hint)) {
            badges.appendChild(mcpEl('span', 'mcp-badge', `${entry.tools_hint.length} tools`));
        }
        row.appendChild(badges);

        const actions = mcpEl('span', 'mcp-actions');
        const attach = mcpEl('button', 'btn btn-primary btn-sm', 'Attach');
        attach.onclick = () => attachCatalogEntry(entry);
        actions.appendChild(attach);
        row.appendChild(actions);

        host.appendChild(row);
    });
}

async function attachCatalogEntry(entry) {
    if (!globalThis.selectedAgent) return;
    await saveMcpServer(entry.id, entry.spec);
}

async function saveMcpServer(name, spec) {
    try {
        const resp = await fetch(
            `/api/agents/${encodeURIComponent(globalThis.selectedAgent)}/mcp/${encodeURIComponent(name)}`,
            {
                method: 'PUT',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ spec }),
            });
        const body = await resp.json();
        if (!resp.ok) {
            showNotification('Could not attach: ' + (body.detail || resp.status), 'error');
            return;
        }
        (body.warnings || []).forEach(w => showNotification(w, 'warning'));
        showNotification(
            `'${name}' saved. Restart the agent to apply.`, 'success');
        loadMcpTab(globalThis.selectedAgent);
    } catch (error) {
        showNotification('Could not attach: ' + error.message, 'error');
    }
}

async function removeMcpServer(name) {
    if (!confirm(`Remove '${name}' from this agent?`)) return;
    try {
        const resp = await fetch(
            `/api/agents/${encodeURIComponent(globalThis.selectedAgent)}/mcp/${encodeURIComponent(name)}`,
            { method: 'DELETE' });
        if (!resp.ok) {
            showNotification(`Could not remove '${name}'`, 'error');
            return;
        }
        showNotification(`'${name}' removed. Restart the agent to apply.`, 'success');
        loadMcpTab(globalThis.selectedAgent);
    } catch (error) {
        showNotification('Could not remove: ' + error.message, 'error');
    }
}

async function setMcpEnabled(name, enabled) {
    try {
        const resp = await fetch(
            `/api/agents/${encodeURIComponent(globalThis.selectedAgent)}/mcp/${encodeURIComponent(name)}/enable`,
            {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ enabled }),
            });
        if (!resp.ok) {
            showNotification(`Could not update '${name}'`, 'error');
            return;
        }
        loadMcpTab(globalThis.selectedAgent);
    } catch (error) {
        showNotification('Could not update: ' + error.message, 'error');
    }
}

function parseKeyValueList(raw) {
    const out = {};
    (raw || '').split(',').forEach(pair => {
        const trimmed = pair.trim();
        if (!trimmed) return;
        const eq = trimmed.indexOf('=');
        if (eq <= 0) return;
        out[trimmed.slice(0, eq).trim()] = trimmed.slice(eq + 1).trim();
    });
    return out;
}

function readMcpForm() {
    const type = document.getElementById('mcp-f-type').value;
    const name = (document.getElementById('mcp-f-name').value || '').trim();
    const spec = { type };

    if (type === 'http') {
        spec.url = (document.getElementById('mcp-f-url').value || '').trim();
        const headers = parseKeyValueList(document.getElementById('mcp-f-headers').value);
        if (Object.keys(headers).length) spec.headers = headers;
    } else {
        spec.command = (document.getElementById('mcp-f-command').value || '').trim();
        const args = (document.getElementById('mcp-f-args').value || '').trim();
        if (args) spec.args = args.split(/\s+/);
        const cwd = (document.getElementById('mcp-f-cwd').value || '').trim();
        if (cwd) spec.cwd = cwd;
        const env = parseKeyValueList(document.getElementById('mcp-f-env').value);
        if (Object.keys(env).length) spec.env = env;
    }

    if (document.getElementById('mcp-f-auto-approve').checked) spec.auto_approve = true;
    return { name, spec };
}

function updateMcpFormVisibility() {
    const http = document.getElementById('mcp-f-type').value === 'http';
    [['mcp-l-url', 'mcp-f-url'], ['mcp-l-headers', 'mcp-f-headers']].forEach(([l, f]) => {
        document.getElementById(l).hidden = !http;
        document.getElementById(f).hidden = !http;
    });
    [['mcp-l-command', 'mcp-f-command'], ['mcp-l-args', 'mcp-f-args'],
     ['mcp-l-cwd', 'mcp-f-cwd'], ['mcp-l-env', 'mcp-f-env']].forEach(([l, f]) => {
        document.getElementById(l).hidden = http;
        document.getElementById(f).hidden = http;
    });
}

async function testMcpForm() {
    const { spec } = readMcpForm();
    const feedback = document.getElementById('mcp-form-feedback');
    feedback.textContent = 'Connecting…';
    try {
        const resp = await fetch(
            `/api/agents/${encodeURIComponent(globalThis.selectedAgent)}/mcp/test`,
            {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ spec }),
            });
        const body = await resp.json();
        if (!resp.ok) {
            feedback.textContent = 'Failed: ' + (body.detail || resp.status);
            return;
        }
        feedback.textContent = body.success
            ? `OK — ${body.tool_count} tool(s): ${body.tool_names.slice(0, 8).join(', ')}` +
              (body.tool_names.length > 8 ? ', …' : '')
            : 'Failed: ' + body.error;
    } catch (error) {
        feedback.textContent = 'Failed: ' + error.message;
    }
}

async function saveMcpFromForm() {
    const { name, spec } = readMcpForm();
    if (!name) {
        document.getElementById('mcp-form-feedback').textContent = 'A name is required.';
        return;
    }
    await saveMcpServer(name, spec);
}

async function saveMcpToCatalog() {
    const { name, spec } = readMcpForm();
    const feedback = document.getElementById('mcp-form-feedback');
    if (!name) {
        feedback.textContent = 'A name is required.';
        return;
    }
    try {
        const resp = await fetch(`/api/mcp/catalog/${encodeURIComponent(name)}`, {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ spec }),
        });
        const body = await resp.json();
        if (!resp.ok) {
            feedback.textContent = 'Failed: ' + (body.detail || resp.status);
            return;
        }
        feedback.textContent = 'Saved to the catalog.';
        loadMcpCatalog(false);
    } catch (error) {
        feedback.textContent = 'Failed: ' + error.message;
    }
}

function setupMcpTab() {
    const on = (id, handler) => {
        const node = document.getElementById(id);
        if (node) node.onclick = handler;
    };
    on('mcp-refresh-btn', () => globalThis.selectedAgent && loadMcpTab(globalThis.selectedAgent));
    on('mcp-scan-btn', () => loadMcpCatalog(true));
    on('mcp-add-toggle-btn', () => {
        const form = document.getElementById('mcp-add-form');
        if (form) {
            form.hidden = !form.hidden;
            if (!form.hidden) updateMcpFormVisibility();
        }
    });
    on('mcp-test-btn', testMcpForm);
    on('mcp-save-btn', saveMcpFromForm);
    on('mcp-save-catalog-btn', saveMcpToCatalog);

    const type = document.getElementById('mcp-f-type');
    if (type) type.onchange = updateMcpFormVisibility;
}

// ===========================================================================
// Stats tab — on-request only (no timer, no interval, no recompute)
// ===========================================================================

async function loadStatsTab(agentName) {
    statsTabLoadedFor = agentName;
    const host = document.getElementById('stats-tab-content');
    if (!host) return;
    host.replaceChildren(mcpEl('em', null, 'Loading…'));

    try {
        const resp = await fetch(`/api/agents/${encodeURIComponent(agentName)}/stats`);
        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
        const body = await resp.json();
        renderStatsTab(body);
    } catch (error) {
        host.replaceChildren(mcpEl('div', 'mcp-error',
            'Could not load stats: ' + error.message));
    }
}

function renderStatsTab(data) {
    const host = document.getElementById('stats-tab-content');
    if (!host) return;
    host.replaceChildren();

    const stats = (data && data.stats) || data || {};
    const rows = [
        ['Input tokens', stats.input_tokens],
        ['Output tokens', stats.output_tokens],
        ['Cache read', stats.cache_read_tokens],
        ['Cache write', stats.cache_creation_tokens],
        ['Total tokens', stats.total_tokens],
        ['Requests', stats.requests],
    ];

    const table = mcpEl('div', 'stats-grid');
    rows.forEach(([label, value]) => {
        if (value === undefined || value === null) return;
        table.appendChild(mcpEl('span', 'stats-label', label));
        table.appendChild(mcpEl('span', 'stats-value', value));
    });

    if (!table.childElementCount) {
        host.appendChild(mcpEl('em', null, 'No statistics available for this agent.'));
        return;
    }
    host.appendChild(table);
}

function setupStatsTab() {
    const btn = document.getElementById('stats-refresh-btn');
    if (btn) btn.onclick = () => globalThis.selectedAgent && loadStatsTab(globalThis.selectedAgent);
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
        formatAgentContext,
        formatContextLimit,
        formatLastTurnTokens,
        handleAgentContextChanged,
        handleAgentMessage,
        handleAgentSummarized,
        // Exposed so node-side tests can drive the WS router directly
        // and assert dispatch — prevents the ``no case for a broadcast''
        // regression class (test_ws_router_context_changed.js).
        handleWebSocketMessage,
        // Exposed so a node test can drive the ONE code path a user
        // click runs. A related internal incident: every
        // node test used to pre-seed ``globalThis.selectedAgent`` by hand
        // and none ever called this function, so the whole suite stayed
        // green while the single line that publishes the selection was
        // broken in production (test_select_agent_publishes_global.js).
        selectAgent,
        mcpSpecSummary,
        parseKeyValueList,
        // Exposed for node-side DOM tests of the Context / Last turn
        // duplicate-suppression gate (test_last_turn_hidden_when_duplicate.js).
        _paintContextSpans,
    };
}
