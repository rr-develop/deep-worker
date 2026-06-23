# Design: Fix parse_manager_response rejecting parenthesized qualifiers

**Status: ✅ IMPLEMENTED — commit pending**

## Problem

`parse_manager_response()` in `prompts.py` does a naive match:

```python
status_text = line_clean.replace('STATUS:', '').strip().lower()
if status_text in ('done', 'in_progress', 'unable'):
```

When the manager AI returns `STATUS: done (escalated correctly)` or
`STATUS: done (qualifier)`, the extracted `status_text` is
`"done (escalated correctly)"` — which is NOT in the tuple → raises
`ValueError`.

This was observed in production: the merger's test-runner reports
`done (escalated correctly)` and `parse_manager_response` rejects it.

## Fix

Extract only the **first word** of the status field before any space,
parenthesis, or bracket. Everything after the first word is transparently
ignored (the qualifier is informational, not used by calling code).

**Change in `prompts.py:186`:**

```python
# Before:
status_text = line_clean.replace('STATUS:', '').strip().lower()
if status_text in ('done', 'in_progress', 'unable'):

# After:
status_text = line_clean.replace('STATUS:', '', 1).strip().lower().split()[0]
if status_text in ('done', 'in_progress', 'unable'):
```

### Regression guard (v2)

A bare `STATUS:` line (nothing after the colon) produces an empty string
after `.strip().lower()`, and `"".split()` returns `[]` → `IndexError`.
Callers expect `ValueError`. Fix: check for empty token before `.split()`.

```python
status_text_raw = line_clean.replace('STATUS:', '', 1).strip().lower()
if not status_text_raw:
    raise ValueError(
        f"Empty STATUS value after 'STATUS:'. "
        f"Full response: {result_text[:200]}"
    )
status_text = status_text_raw.split()[0]
```

This accepts:
- `STATUS: done` → `"done"` ✓ (unchanged)
- `STATUS: done (escalated correctly)` → `"done"` ✓ (new)
- `STATUS: in_progress` → `"in_progress"` ✓ (unchanged)
- `STATUS: unable` → `"unable"` ✓ (unchanged)
- `STATUS: invalid` → `"invalid"` → raises ValueError ✓ (unchanged)
- `STATUS:` (empty) → raises ValueError, never IndexError ✓ (regression guard, new)

## Test changes

None of the existing tests should be affected. Add three new test cases:
1. `test_parse_done_with_parenthesized_qualifier` — verifies
   `STATUS: done (escalated correctly)` parses as `done`.
2. `test_parse_in_progress_with_qualifier` — verifies
   `STATUS: in_progress (needs more work)` parses as `in_progress`.
3. `test_parse_bare_status_raises_valueerror` — verifies
   bare `STATUS:` raises `ValueError`, not `IndexError`.

No architecture/contract/policy change — only one line of parsing logic
plus an empty-token guard before `.split()`.
