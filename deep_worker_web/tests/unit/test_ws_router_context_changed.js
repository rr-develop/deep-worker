// Regression: handleWebSocketMessage MUST route ``agent_context_changed``
// broadcasts to ``handleAgentContextChanged``. Without the case in the
// switch, the broadcast lands in the ``default`` branch as
// "Unknown message type" and the status bar never auto-updates — the
// operator has to refresh the page to see the new context number.
//
// Regression from commit 3bdf351 ("feat: agent context status bar"):
// the backend emits ``agent_context_changed`` correctly and the handler
// is written, but the switch in ``handleWebSocketMessage`` (main.js:970)
// never dispatches to it. No test exercised the actual switch, so the
// missing wire slipped through. THIS test drives the switch directly.
//
// Run with: node tests/unit/test_ws_router_context_changed.js

const path = require('path');
const assert = require('assert');

// Minimal DOM stub — mirrors test_isolation_preserves_snapshot.js so a
// full painter run can execute without a real browser.
function makeStubElement() {
    let _textContent = null;
    return {
        get textContent() { return _textContent; },
        set textContent(v) { _textContent = v; },
        style: {},
        setAttribute() {},
        getAttribute() { return null; },
        get classList() {
            return {
                add() {}, remove() {}, toggle() {}, contains() { return false; },
            };
        },
        appendChild() {},
        removeChild() {},
        replaceChildren() {},
        addEventListener() {},
        removeEventListener() {},
    };
}

const stubById = {};
function getStubElement(id) {
    if (!stubById[id]) stubById[id] = makeStubElement();
    return stubById[id];
}

global.document = {
    addEventListener: () => {},
    getElementById: (id) => getStubElement(id),
    querySelector: () => null,
    querySelectorAll: () => [],
    createElement: () => makeStubElement(),
};
global.window = {};
global.WebSocket = class {};
global.fetch = async () => ({ ok: false, json: async () => [] });

const mainPath = path.resolve(__dirname, '..', '..', 'static', 'js', 'main.js');
const main = require(mainPath);

console.log('test_ws_router_context_changed — switch dispatches agent_context_changed');

// Test hook contract: the switch dispatcher must be reachable from node
// so we can drive it without a live WebSocket.
if (typeof main.handleWebSocketMessage !== 'function') {
    throw new Error(
        'main.handleWebSocketMessage is not exported — impl must add it to '
        + 'the CommonJS test-hook block so this router path is testable.',
    );
}

const AGENT = 'ctx_router_agent';
globalThis.agents = {
    [AGENT]: {
        name: AGENT,
        has_task_manager: false,
        // Pre-seed with distinct values so we can prove the router ran.
        agent_context_used: 1,
        agent_context_size: 2,
        agent_context_semantics: null,
        tm_context_used: null, tm_context_size: null, tm_context_semantics: null,
        context_limit: null,
    },
};
globalThis.selectedAgent = AGENT;

// Fire the backend broadcast payload verbatim (agent_service.py:676-686).
main.handleWebSocketMessage({
    type: 'agent_context_changed',
    agent_name: AGENT,
    agent_context_used: 123000,
    agent_context_size: 400000,
    agent_context_semantics: 'context_snapshot',
    tm_context_used: null,
    tm_context_size: null,
    tm_context_semantics: null,
    context_limit: 400000,
});

// Side-effect assertion — the router must have delegated to
// handleAgentContextChanged, which updates the in-memory mirror AND
// paints ``agent-context``. Both are load-bearing: the mirror keeps
// the next status poll consistent, the paint is what the operator
// actually sees.
const info = globalThis.agents[AGENT];
assert.strictEqual(
    info.agent_context_used, 123000,
    'router did not update agent_context_used — the switch case for '
    + 'agent_context_changed is missing',
);
assert.strictEqual(
    info.agent_context_size, 400000,
    'router did not update agent_context_size',
);
assert.strictEqual(
    info.agent_context_semantics, 'context_snapshot',
    'router did not update agent_context_semantics',
);
assert.strictEqual(
    info.context_limit, 400000,
    'router did not mirror context_limit — the -l value cannot re-paint '
    + 'without a page refresh',
);
const painted = stubById['agent-context'] && stubById['agent-context'].textContent;
assert.strictEqual(
    painted, '123k/400k',
    `router did not repaint the agent-context span; got ${JSON.stringify(painted)}`,
);
console.log('  ok  router dispatched → mirror updated, span painted:', painted);

// Guard: an unknown type must not touch the mirror. Prevents an
// over-broad fix (e.g. catch-all) from silently accepting typos.
const before = info.agent_context_used;
main.handleWebSocketMessage({
    type: 'definitely_not_a_real_event',
    agent_name: AGENT,
});
assert.strictEqual(
    info.agent_context_used, before,
    'unknown type must NOT alter the mirror; the fix must be narrow',
);
console.log('  ok  unknown type is a no-op (narrow fix)');

console.log('PASS: test_ws_router_context_changed');
