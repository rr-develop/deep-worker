// Regression: the ``Last turn`` UI row must be filled ONLY from the
// per-turn total, and the ``Context`` field must NEVER surface a
// per-turn total as its value.
//
// Root cause pinned by the ACP hybrid-record fix (rg1_worker
// 2026-08-22): the ``Context: 2265.4k tokens`` regression happened
// because ``formatAgentContext`` fell back to a tokens-only rendering
// when the provider dropped the snapshot.  These tests keep both
// invariants pinned at the frontend layer, independent of the backend
// hybrid emit.
//
// Run with: node tests/unit/test_last_turn_tokens_ui.js

const path = require('path');
const assert = require('assert');

const mainPath = path.resolve(__dirname, '..', '..', 'static', 'js', 'main.js');
// Minimal DOM stub (browser features touched by main.js load-time).
global.document = {
    addEventListener: () => {},
    getElementById: () => null,
    querySelector: () => null,
    querySelectorAll: () => [],
    createElement: () => ({
        style: {}, setAttribute() {}, appendChild() {}, removeChild() {},
    }),
};
global.window = {};
global.WebSocket = class {};
global.fetch = async () => ({ ok: false, json: async () => [] });

const {
    formatAgentContext, formatLastTurnTokens,
} = require(mainPath);

console.log('test_last_turn_tokens_ui — Context vs Last turn separation');

// 1) formatLastTurnTokens: null → null (UI hides the wrap), value → "…k tokens".
assert.strictEqual(
    formatLastTurnTokens(null), null,
    'null total must return null so the wrap stays hidden',
);
assert.strictEqual(
    formatLastTurnTokens(undefined), null,
    'undefined total must return null',
);
assert.strictEqual(
    formatLastTurnTokens(0), '0 tokens',
    '0 must render as "0 tokens" (still a valid observation)',
);
assert.strictEqual(
    formatLastTurnTokens(2_265_400), '2265.4k tokens',
    'per-turn total must format as "Nk tokens"',
);
assert.strictEqual(
    formatLastTurnTokens(400_000), '400k tokens',
    'round thousands drop the decimal',
);
console.log('  ok  formatLastTurnTokens shape');

// 2) formatAgentContext with a hybrid record (both used+size AND a
//    per-turn total on the way) must render the SNAPSHOT view — the
//    per-turn total is NOT this function's business.
assert.strictEqual(
    formatAgentContext(390_000, 1_000_000, 'context_snapshot'),
    '390k/1000k',
    'hybrid snapshot must render as "used/size", not the per-turn total',
);
console.log('  ok  Context field uses snapshot even when a per-turn total is set elsewhere');

// 3) Tokens-only semantics (older providers with no snapshot at all)
//    keeps its historical rendering — this is the minimax fallback and
//    must not regress.
assert.strictEqual(
    formatAgentContext(42_100, null, 'tokens'), '42.1k tokens',
    'tokens-only providers keep the historical "Nk tokens" rendering',
);
console.log('  ok  tokens-only fallback preserved (minimax)');

console.log('PASS: test_last_turn_tokens_ui');
