// Node-side unit test for the ``token_update does NOT overwrite a known
// snapshot`` invariant.
//
// Design doc: docs/DESIGN_AGENT_CONTEXT_STATUS_BAR.md (test_isolation_preserves_snapshot).
//
// Run with: node tests/unit/test_isolation_preserves_snapshot.js
//
// The test stubs the DOM and verifies the live ``handleAgentMessage`` /
// ``handleAgentContextChanged`` paths in main.js do NOT let a noisy
// ``token_update`` event overwrite a previously-rendered snapshot.
//
// RED-first: the export ``handleAgentContextChanged`` (or whichever
// internal API the impl chooses) doesn't exist yet. The require() will
// fail to destructure the name → TypeError on first call.

const path = require('path');
const assert = require('assert');

// DOM stub: track textContent on the elements we touch.
function makeStubElement() {
    let _textContent = null;
    return {
        get textContent() { return _textContent; },
        set textContent(v) { _textContent = v; },
        style: {},
        setAttribute(k, v) { /* noop */ },
        getAttribute(k) { return null; },
        get classList() { return { add() {}, remove() {}, toggle() {}, contains() { return false; } }; },
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

console.log('test_isolation_preserves_snapshot — UI e2e semantics');

const AGENT = 'iso_agent';

// Pre-seed an AgentInfo with snapshot semantics already on the in-memory
// mirror — simulates a prior agent_context_changed event that painted
// the status bar with a provider-supplied snapshot.
globalThis.agents = {
    [AGENT]: {
        name: AGENT,
        has_task_manager: false,
        agent_context_used: 42000,
        agent_context_size: 200000,
        agent_context_semantics: 'context_snapshot',
        tm_context_used: null,
        tm_context_size: null,
        tm_context_semantics: null,
    },
};
globalThis.selectedAgent = AGENT;

// 1) Drive the snapshot path: handler should paint '42k/200k'.
if (typeof main.handleAgentContextChanged !== 'function') {
    throw new Error(
        'main.handleAgentContextChanged is not exported — impl must add it.'
    );
}

main.handleAgentContextChanged({
    type: 'agent_context_changed',
    agent_name: AGENT,
    agent_context_used: 42000,
    agent_context_size: 200000,
    agent_context_semantics: 'context_snapshot',
    tm_context_used: null, tm_context_size: null, tm_context_semantics: null,
});

const afterSnapshot = stubById['agent-context'] && stubById['agent-context'].textContent;
assert.strictEqual(
    afterSnapshot, '42k/200k',
    `after snapshot: expected "42k/200k", got ${JSON.stringify(afterSnapshot)}`
);
console.log('  ok  snapshot paints:', afterSnapshot);

// M6 — TM-wrap MUST stay hidden when has_task_manager is false
// (the gate lives in `_paintContextSpans`, not in the broadcast itself).
assert.strictEqual(
    stubById['agent-tm-context-wrap'].style.display, 'none',
    'tm-wrap MUST be hidden when has_task_manager is false (M6)'
);
console.log('  ok  tm-wrap hidden when has_task_manager is false');

// 2) Drive a token_update event that arrives AFTER the snapshot.
//    The handler MUST NOT overwrite the snapshot — only allowed when
//    agent_context_semantics is still null (best-effort for first
//    message). Here semantics is 'context_snapshot', so it MUST stay.
if (typeof main.handleAgentMessage !== 'function') {
    throw new Error(
        'main.handleAgentMessage is not exported — impl must add it (or ' +
        'expose the token_update branch as a separate function).'
    );
}

main.handleAgentMessage({
    agent_name: AGENT,
    message: {
        type: 'token_update',
        total_tokens: 5000,
        input_tokens: 4000,
        output_tokens: 1000,
    },
});

const afterTokenUpdate = stubById['agent-context'] && stubById['agent-context'].textContent;
assert.strictEqual(
    afterTokenUpdate, '42k/200k',
    `after token_update: snapshot must be preserved. Got ${JSON.stringify(afterTokenUpdate)} — the impl let token_update overwrite a snapshot.`
);
console.log('  ok  token_update did NOT overwrite snapshot:', afterTokenUpdate);

// M8 — token_update MUST NOT touch tm_context_* fields. The agent
// type is 'agent' here; TM context lives in tm_context_* and stays
// null regardless of token_update events.
assert.strictEqual(
    globalThis.agents[AGENT].tm_context_used, null,
    'tm_context_used MUST stay null after token_update (M8)'
);
assert.strictEqual(
    globalThis.agents[AGENT].tm_context_size, null,
    'tm_context_size MUST stay null after token_update (M8)'
);
assert.strictEqual(
    globalThis.agents[AGENT].tm_context_semantics, null,
    'tm_context_semantics MUST stay null after token_update (M8)'
);
console.log('  ok  tm_context_* fields untouched after token_update');

// M8 second path: even WITHOUT a prior snapshot (semantics=null),
// the token_update handler MUST NOT touch tm_context_* — TM never
// receives a notify-callback, so the status bar's TM-context span
// stays empty until an explicit agent_context_changed event arrives.
// Without this, an M8 mutation could pass the snapshot-path test
// because M7 early-return would never trigger the bad code.
globalThis.agents[AGENT] = {
    name: AGENT,
    has_task_manager: true,
    agent_context_used: null,
    agent_context_size: null,
    agent_context_semantics: null,
    tm_context_used: null,
    tm_context_size: null,
    tm_context_semantics: null,
};
main.handleAgentMessage({
    agent_name: AGENT,
    message: {
        type: 'token_update',
        total_tokens: 5000,
        input_tokens: 4000,
        output_tokens: 1000,
    },
});
assert.strictEqual(
    globalThis.agents[AGENT].tm_context_used, null,
    'tm_context_used MUST stay null even without prior snapshot (M8 path)'
);
assert.strictEqual(
    globalThis.agents[AGENT].tm_context_size, null,
    'tm_context_size MUST stay null even without prior snapshot (M8 path)'
);
assert.strictEqual(
    globalThis.agents[AGENT].tm_context_semantics, null,
    'tm_context_semantics MUST stay null even without prior snapshot (M8 path)'
);
console.log('  ok  tm_context_* untouched when no snapshot exists');

console.log('test_isolation_preserves_snapshot passed.');