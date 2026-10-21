// Node-side unit test for ``formatContextLimit(limit)`` — the
// operator-set -l flag display in the web UI status bar.
//
// Pinned: the operator MUST be able to see what limit the guard is
// holding. Without this, the UI cannot distinguish "guard holds 400k"
// from "guard silently allowed the provider default" — the very bug
// agent_a hit on 2026-08-18.
//
// Run from this directory with:
//   node tests/unit/test_format_context_limit.js

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

const { formatContextLimit } = require(
    path.resolve(__dirname, '..', '..', 'static', 'js', 'main.js')
);

let passed = 0;
let failed = 0;

function expect(label, actual, expected) {
    if (actual === expected) {
        passed += 1;
    } else {
        failed += 1;
        console.error(`FAIL: ${label}`);
        console.error(`  expected: ${JSON.stringify(expected)}`);
        console.error(`  actual:   ${JSON.stringify(actual)}`);
    }
}

// No -l set → "-" (operator used the default).
expect("null limit → '-'", formatContextLimit(null), '-');
expect("undefined limit → '-'", formatContextLimit(undefined), '-');

// Exact "k" formatting when the value divides cleanly.
expect("400000 → '400k (-l)'", formatContextLimit(400000), '400k (-l)');
expect("1000 → '1k (-l)'", formatContextLimit(1000), '1k (-l)');

// Sub-thousand values render as-is, no "k" suffix.
expect("500 → '500 (-l)'", formatContextLimit(500), '500 (-l)');

// Non-clean division uses one decimal place.
expect("42100 → '42.1k (-l)'", formatContextLimit(42100), '42.1k (-l)');

// Edge: 0 is technically valid (-l 0 means "block everything").
// We render "0 (-l)" so the operator can see they explicitly set it.
expect("0 → '0 (-l)'", formatContextLimit(0), '0 (-l)');

// NaN / non-finite → '-' (defensive — render missing-marker so the
// operator does not see "NaN (-l)" and think the guard is somehow
// holding a non-numeric limit).
expect("NaN → '-'", formatContextLimit(NaN), '-');
expect("Infinity → '-'", formatContextLimit(Infinity), '-');

// The (-l) suffix MUST be present so the operator knows the value
// comes from the operator-set flag (not the provider default).
const result = formatContextLimit(400000);
if (!result.includes('(-l)')) {
    failed += 1;
    console.error(`FAIL: '(-l)' suffix missing in formatted output: ${result}`);
} else {
    passed += 1;
}

console.log(`\n${passed} passed, ${failed} failed`);
if (failed > 0) process.exit(1);
