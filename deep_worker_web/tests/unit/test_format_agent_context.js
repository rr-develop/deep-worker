// Node-side unit test for the pure ``formatAgentContext(used, size, semantics)``
// exported from static/js/main.js. Design doc:
// docs/DESIGN_AGENT_CONTEXT_STATUS_BAR.md.
//
// Run with: node tests/unit/test_format_agent_context.js
//
// All 10 cases from the review. RED-first contract: formatAgentContext
// MUST NOT exist yet in main.js, so the require() below will fail with
// ``TypeError: ... is not a function`` or destructuring ``undefined``.
// That is the expected RED state.

const path = require('path');
const assert = require('assert');

// Minimal DOM/globals stubs so that top-level references in main.js
// don't blow up during require (they only run at DOMContentLoaded,
// which we don't fire).
global.document = {
    addEventListener: () => {},
    getElementById: () => null,
};
global.window = {};
global.WebSocket = class {};
global.fetch = async () => ({ ok: false, json: async () => [] });

const { formatAgentContext } = require(
    path.resolve(__dirname, '..', '..', 'static', 'js', 'main.js')
);

function check(label, used, size, semantics, expected) {
    const got = formatAgentContext(used, size, semantics);
    assert.strictEqual(
        got, expected,
        `${label}: formatAgentContext(${used}, ${size}, ${JSON.stringify(semantics)}) => ${JSON.stringify(got)}, expected ${JSON.stringify(expected)}`
    );
    console.log(`  ok  ${label}: ${got}`);
}

console.log('formatAgentContext — pure function tests');

// Empty / unknown semantics → "-"
check('all null', null, null, null, '-');

// Provider snapshot, both fields populated.
check('snapshot used+size', 42100, 200000, 'context_snapshot', '42.1k/200k');

// Provider tokens only — exact count from the provider's tokenizer,
// NOT an estimate. No "(est.)" qualifier.
check('tokens used only', 1500, null, 'tokens', '1.5k tokens');

// Snapshot edge: zero used, real size.
check('zero used', 0, 1000, 'context_snapshot', '0/1k');

// Snapshot edge: missing used but size known → 0/size.
check('null used', null, 1000, 'context_snapshot', '0/1k');

// Anomaly: snapshot declared but size missing → "Xk/?" (NOT "est.").
check('snapshot no size (anomaly)', 800, null, 'context_snapshot', '800/?');

// Provider tokens without size — still exact from the provider's
// tokenizer. No "(est.)" qualifier.
check('tokens no size', 800, null, 'tokens', '800 tokens');

// Zero-zero edge for snapshot.
check('zero/zero snapshot', 0, 0, 'context_snapshot', '0/0');

// Zero-zero edge for tokens — exact count of zero from the provider.
check('zero/zero tokens', 0, 0, 'tokens', '0 tokens');

// Null semantics + null size + non-null used → fall through to tokens
// path. Used is provider-reported (came from the recorder) — exact.
check('null semantics', 1000, null, null, '1k tokens');

console.log('All formatAgentContext tests passed.');