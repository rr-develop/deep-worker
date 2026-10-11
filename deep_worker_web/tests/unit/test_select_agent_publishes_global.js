// Regression for a "context never refreshes" defect — the default-suite pin.
//
// ``selectAgent()`` is the ONE function a user click runs, and its first
// statement is what publishes the selection that every broadcast handler
// guards on:
//
//     handleAgentContextChanged / handleAgentMessage /
//     _applyTokenUpdateIfFirstSeen  →  if (data.agent_name !==
//                                          globalThis.selectedAgent) return;
//
// The defect: main.js declared
//
//     let selectedAgent = (globalThis.selectedAgent = null);
//
// which copies the value ONCE at load. A ``let`` at module scope creates
// no property on the global object, so ``selectAgent()``'s bare
// ``selectedAgent = name`` wrote the module binding while
// ``globalThis.selectedAgent`` stayed ``null`` for the page's entire
// lifetime. Every context frame was discarded on the handler's first
// line and the status bar never moved in a real browser.
//
// WHY THE WHOLE NODE SUITE STAYED GREEN THROUGH IT: every existing test
// assigned ``globalThis.selectedAgent = AGENT`` by hand and NOT ONE ever
// called ``selectAgent()``. They constructed the single state a browser
// can never reach, jumping over the broken line. The Playwright guard in
// tests/integration/ covers this in a real DOM, but it is marked ``ui``
// and pytest.ini's addopts (``-m "not ui and not integration"``) skip it
// in the run people actually execute.
//
// So THIS test deliberately:
//   1. does NOT pre-assign globalThis.selectedAgent — it starts unset,
//   2. calls selectAgent() for real,
//   3. asserts the selection is observable at globalThis.selectedAgent.
//
// Run with: node tests/unit/test_select_agent_publishes_global.js

const path = require('path');
const assert = require('assert');

function makeStubElement() {
    let _textContent = null;
    return {
        get textContent() { return _textContent; },
        set textContent(v) { _textContent = v; },
        innerHTML: '',
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

console.log('test_select_agent_publishes_global — a click publishes the selection');

// Test hook contract.
if (typeof main.selectAgent !== 'function') {
    throw new Error(
        'main.selectAgent is not exported — impl must add it to the CommonJS '
        + 'test-hook block. Without it no default-suite test can drive the one '
        + 'code path a user click runs (a related internal incident analysis, not included here).',
    );
}

const AGENT = 'select_agent_pin';

// ── case 1: the bare defect, from a genuinely unset starting state ────
// NOTE: no ``globalThis.selectedAgent = AGENT`` anywhere above. That
// hand-seeding is exactly what hid the bug. Start from null, the way a
// freshly loaded page does.
globalThis.selectedAgent = null;
globalThis.agents = {};

// renderAgentList() and the rest of selectAgent() touch a lot of DOM;
// the stub is thin on purpose, and the assignment we care about is the
// FIRST statement of the function, so a later throw does not affect what
// is being pinned. Any throw is swallowed deliberately.
try {
    main.selectAgent(AGENT);
} catch (e) {
    // ignore — DOM-stub shortfall, not the subject of this test
}

assert.strictEqual(
    globalThis.selectedAgent, AGENT,
    'selectAgent() did not publish the selection to globalThis.selectedAgent — '
    + 'every broadcast handler guards on that property and will early-return, '
    + 'so the UI silently stops updating (the 2026-08-22 defect). A module-local '
    + '`let`/`var` binding reproduces exactly this.',
);
console.log('  ok  selectAgent() publishes to globalThis.selectedAgent');

// ── case 2: the selection is what the handlers actually consume ───────
// Proves the pin is load-bearing end-to-end inside node: after a real
// selectAgent() call (and NOT a hand-seeded global), a context broadcast
// for that agent must pass the guard and reach the mirror.
globalThis.agents = {
    [AGENT]: {
        name: AGENT,
        has_task_manager: false,
        agent_context_used: 1,
        agent_context_size: 2,
        agent_context_semantics: null,
        tm_context_used: null, tm_context_size: null, tm_context_semantics: null,
        context_limit: null,
    },
};
globalThis.selectedAgent = null;
try {
    main.selectAgent(AGENT);
} catch (e) { /* ignore DOM-stub shortfall */ }

main.handleAgentContextChanged({
    agent_name: AGENT,
    agent_context_used: 21277,
    agent_context_size: 400000,
    agent_context_semantics: 'context_snapshot',
    tm_context_used: null,
    tm_context_size: null,
    tm_context_semantics: null,
    context_limit: null,
});

assert.strictEqual(
    globalThis.agents[AGENT].agent_context_used, 21277,
    'a context broadcast did not reach the mirror after a REAL selectAgent() '
    + 'call — the handler guard rejected it, which is the operator-visible '
    + '"Context never refreshes" symptom.',
);
console.log('  ok  a context broadcast passes the guard after a real selectAgent()');

console.log('PASS: test_select_agent_publishes_global');
