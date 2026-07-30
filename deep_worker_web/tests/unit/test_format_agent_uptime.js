// Node-side unit test for the pure ``formatAgentUptime(iso, nowMs)``
// exported from static/js/main.js. Design doc:
// docs/DESIGN_AGENT_UPTIME_UI.md.
//
// Run with: node tests/unit/test_format_agent_uptime.js
//
// The browser side of main.js references DOM globals in other places;
// the node ``require`` still works because ``module.exports`` is set at
// top-level execution BEFORE any DOM access — but to be safe we install
// minimal DOM stubs first.

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

const { formatAgentUptime } = require(
    path.resolve(__dirname, '..', '..', 'static', 'js', 'main.js')
);

function check(label, iso, nowMs, expected) {
    const got = formatAgentUptime(iso, nowMs);
    assert.strictEqual(
        got, expected,
        `${label}: formatAgentUptime(${JSON.stringify(iso)}, ${nowMs}) => ${JSON.stringify(got)}, expected ${JSON.stringify(expected)}`
    );
    console.log(`  ok  ${label}: ${got}`);
}

const T0 = Date.parse('2026-07-04T12:00:00Z');
const iso = new Date(T0).toISOString();

console.log('formatAgentUptime — pure function tests');

// 0 seconds -> "0s"
check('exactly zero', iso, T0, '0s');

// 5 seconds -> "5s"
check('5 seconds', iso, T0 + 5_000, '5s');

// 42 seconds -> "42s"
check('42 seconds', iso, T0 + 42_000, '42s');

// 65 seconds -> "01:05"
check('65 seconds', iso, T0 + 65_000, '01:05');

// 3665 seconds = 1h 1m 5s -> "01:01:05"
check('3665 seconds', iso, T0 + 3665_000, '01:01:05');

// 90000 seconds = 1d 1h 0m 0s -> "1d 01:00:00"
check('90000 seconds', iso, T0 + 90_000_000, '1d 01:00:00');

// Clock skew: now is BEFORE started_at → clamp to "0s", never negative.
check('clock skew (now before started)', iso, T0 - 60_000, '0s');

// Missing / falsy iso -> "-"
check('null iso', null, T0, '-');
check('empty string iso', '', T0, '-');

// Invalid iso -> "-"
check('invalid iso', 'not-a-date', T0, '-');

console.log('All formatAgentUptime tests passed.');
