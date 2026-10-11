// Regression: agent_context_changed carrying the NEW ``agent_last_turn_tokens``
// field must (a) update the mirror, (b) show the ``Last turn`` wrap
// element, (c) paint the formatted value.
//
// Companion of test_ws_router_context_changed.js — that one asserts the
// switch dispatches; THIS one asserts the ``Last turn`` UI actually
// appears when the backend surfaces the per-turn total.
//
// Run with: node tests/unit/test_last_turn_paints_from_broadcast.js

const path = require('path');
const assert = require('assert');

function makeStubElement() {
    let _textContent = null;
    return {
        get textContent() { return _textContent; },
        set textContent(v) { _textContent = v; },
        style: {},
        setAttribute() {}, getAttribute() { return null; },
        get classList() {
            return {
                add() {}, remove() {}, toggle() {}, contains() { return false; },
            };
        },
        appendChild() {}, removeChild() {}, replaceChildren() {},
        addEventListener() {}, removeEventListener() {},
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

console.log('test_last_turn_paints_from_broadcast');

const AGENT = 'last_turn_agent';
globalThis.agents = {
    [AGENT]: {
        name: AGENT,
        has_task_manager: false,
        agent_context_used: null,
        agent_context_size: null,
        agent_context_semantics: null,
        agent_last_turn_tokens: null,
        tm_context_used: null, tm_context_size: null, tm_context_semantics: null,
        tm_last_turn_tokens: null,
        context_limit: null,
    },
};
globalThis.selectedAgent = AGENT;

// 1) Full hybrid payload — snapshot AND per-turn total present.
main.handleAgentContextChanged({
    type: 'agent_context_changed',
    agent_name: AGENT,
    agent_context_used: 390_000,
    agent_context_size: 1_000_000,
    agent_context_semantics: 'context_snapshot',
    agent_last_turn_tokens: 2_265_400,
    tm_context_used: null,
    tm_context_size: null,
    tm_context_semantics: null,
    tm_last_turn_tokens: null,
    context_limit: 400_000,
});
const info = globalThis.agents[AGENT];
assert.strictEqual(info.agent_last_turn_tokens, 2_265_400, 'mirror updated');
// The Context field renders the SNAPSHOT — NEVER the per-turn total.
assert.strictEqual(
    stubById['agent-context'].textContent, '390k/1000k',
    'Context must show the observed snapshot, not the per-turn total',
);
// Last turn wrap is visible and populated.
assert.strictEqual(
    stubById['agent-last-turn-wrap'].style.display, '',
    'last-turn wrap must be visible when a per-turn total is present',
);
assert.strictEqual(
    stubById['agent-last-turn'].textContent, '2265.4k tokens',
    'last-turn span must render the per-turn total',
);
console.log('  ok  hybrid payload → snapshot in Context, per-turn in Last turn');

// 2) Snapshot-only payload (no per-turn total yet, e.g. after a summarize
//    or on a provider that only emits usage_update).  The wrap must
//    HIDE so the operator does not see a stale ``Last turn: -``.
main.handleAgentContextChanged({
    type: 'agent_context_changed',
    agent_name: AGENT,
    agent_context_used: 100_000,
    agent_context_size: 400_000,
    agent_context_semantics: 'context_snapshot',
    agent_last_turn_tokens: null,
    tm_context_used: null, tm_context_size: null, tm_context_semantics: null,
    tm_last_turn_tokens: null,
    context_limit: 400_000,
});
assert.strictEqual(
    stubById['agent-last-turn-wrap'].style.display, 'none',
    'last-turn wrap must hide when the per-turn total is missing',
);
assert.strictEqual(
    stubById['agent-context'].textContent, '100k/400k',
    'Context re-paints as the new snapshot',
);
console.log('  ok  snapshot-only payload → Last turn hidden');

console.log('PASS: test_last_turn_paints_from_broadcast');
