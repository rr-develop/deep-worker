// Regression: the ``Last turn`` row must NOT render a visual duplicate of
// the ``Context`` row.
//
// Background (owner, 2026-08-22).  ``Context`` now sources from
// ``prompt_tokens`` and ``Last turn`` from ``total_tokens``.  For codex the
// two genuinely differ (505.8k vs 507.3k — completion_tokens=1498) and both
// rows carry information.  But a provider that reports
// ``total_tokens == prompt_tokens`` (no completion accounting, or a
// tokens-only adapter where the normalizer sets total = prompt + 0) would
// paint the SAME number twice:
//
//     Context:   150k tokens
//     Last turn: 150k tokens
//
// which reads as a broken/duplicated field.  In that case the ``Last turn``
// wrap must be hidden — it adds no information.
//
// The gate lives in ``_paintContextSpans``, so this test drives that
// function through a DOM stub rather than testing a pure formatter.
//
// Run with: node tests/unit/test_last_turn_hidden_when_duplicate.js

const path = require('path');
const assert = require('assert');

const mainPath = path.resolve(__dirname, '..', '..', 'static', 'js', 'main.js');

// Element stubs keyed by id, so assertions can read back style/text.
function makeEl() {
    return { style: {}, textContent: '', setAttribute() {}, appendChild() {}, removeChild() {} };
}

const els = {
    'agent-context': makeEl(),
    'agent-context-limit': makeEl(),
    'agent-last-turn-wrap': makeEl(),
    'agent-last-turn': makeEl(),
    'agent-tm-context-wrap': makeEl(),
    'agent-tm-context': makeEl(),
};

global.document = {
    addEventListener: () => {},
    getElementById: (id) => els[id] || null,
    querySelector: () => null,
    querySelectorAll: () => [],
    createElement: () => makeEl(),
};
global.window = {};
global.WebSocket = class {};
global.fetch = async () => ({ ok: false, json: async () => [] });

const { _paintContextSpans } = require(mainPath);

assert.strictEqual(
    typeof _paintContextSpans, 'function',
    '_paintContextSpans must be exported for node-side DOM tests',
);

console.log('test_last_turn_hidden_when_duplicate — no duplicated row');

// 1) Distinct numbers (the live codex case): BOTH rows visible.
_paintContextSpans({
    agent_context_used: 505_788,
    agent_context_size: null,
    agent_context_semantics: 'tokens',
    agent_last_turn_tokens: 507_286,
    context_limit: 250_000,
    has_task_manager: false,
});
assert.strictEqual(
    els['agent-context'].textContent, '505.8k tokens',
    'Context must render prompt_tokens',
);
assert.notStrictEqual(
    els['agent-last-turn-wrap'].style.display, 'none',
    'Last turn must stay VISIBLE when it differs from Context — the codex ' +
    'case (505.8k vs 507.3k) carries two distinct facts',
);
assert.strictEqual(
    els['agent-last-turn'].textContent, '507.3k tokens',
    'Last turn must render total_tokens',
);
console.log('  ok  distinct values keep both rows');

// 2) Identical numbers: Last turn hidden (would be a visual duplicate).
_paintContextSpans({
    agent_context_used: 150_000,
    agent_context_size: null,
    agent_context_semantics: 'tokens',
    agent_last_turn_tokens: 150_000,
    context_limit: 250_000,
    has_task_manager: false,
});
assert.strictEqual(
    els['agent-context'].textContent, '150k tokens',
    'Context still renders when Last turn is suppressed',
);
assert.strictEqual(
    els['agent-last-turn-wrap'].style.display, 'none',
    'Last turn must be HIDDEN when it equals Context — rendering the same ' +
    'number on two rows reads as a broken field',
);
assert.strictEqual(
    els['agent-last-turn'].textContent, '',
    'hidden Last turn must also clear its text so a later repaint cannot ' +
    'flash a stale value',
);
console.log('  ok  identical values hide the Last turn row');

// 3) Snapshot semantics (Context is "X/Y", Last turn is a raw total):
//    numbers are not comparable, so the row must remain visible even if
//    the used-part happens to equal the total.
_paintContextSpans({
    agent_context_used: 150_000,
    agent_context_size: 400_000,
    agent_context_semantics: 'context_snapshot',
    agent_last_turn_tokens: 150_000,
    context_limit: 400_000,
    has_task_manager: false,
});
assert.notStrictEqual(
    els['agent-last-turn-wrap'].style.display, 'none',
    'under snapshot semantics Context renders as "used/size", so a Last ' +
    'turn equal to the used-part is NOT a visual duplicate and must show',
);
console.log('  ok  snapshot semantics keep the row');

// 4) Missing total → hidden (pre-existing behaviour must not regress).
_paintContextSpans({
    agent_context_used: 150_000,
    agent_context_size: null,
    agent_context_semantics: 'tokens',
    agent_last_turn_tokens: null,
    context_limit: 250_000,
    has_task_manager: false,
});
assert.strictEqual(
    els['agent-last-turn-wrap'].style.display, 'none',
    'missing per-turn total must keep the wrap hidden (no "Last turn: -")',
);
console.log('  ok  missing total stays hidden');

console.log('ALL PASS');
