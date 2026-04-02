# Integration Test Skip Permissions Policy

## ⚠️ CRITICAL REQUIREMENT ⚠️

**ALL integration tests MUST use `skip_permissions=True` by default!**

This policy is MANDATORY and applies to ALL test fixtures that create agents.

---

## Why This Is Critical

### The Problem

Integration tests invoke tools (read, write, execute, etc.) during normal operation. Without `skip_permissions=True`, agents will:

1. **Pause execution** waiting for user approval
2. **Hang forever** in CI/CD environments (no human to approve)
3. **Block the entire test suite** from completing
4. **Cause timeouts** and false test failures

### The Solution

**Default to `skip_permissions=True` in ALL fixtures:**

```python
# ✅ CORRECT - Default behavior
def clean_agent(..., skip_permissions: bool = True):
    ...

def agent_fixture(..., skip_permissions: bool = True):
    ...
```

---

## Implementation Status

### ✅ Fixed Fixtures

| Fixture | File | Default | Status |
|---------|------|---------|--------|
| `clean_agent` | `clean_agent_fixture.py` | `skip_permissions=True` | ✅ FIXED |
| `agent_fixture` | `agent_fixture.py` | `skip_permissions=True` | ✅ ALREADY CORRECT |

### Code Changes Made

#### 1. `clean_agent_fixture.py`

**Added parameter:**
```python
def _setup_clean_agent(
    name: str,
    working_dir: str = "/tmp",
    provider: str | None = None,
    skip_permissions: bool = True  # ← NEW parameter with safe default
) -> subprocess.Popen:
```

**Added comprehensive warning comments:**
```python
# ============================================================================
# CRITICAL: skip_permissions MUST be True by default for all integration tests!
# ============================================================================
# WHY: Integration tests invoke tools (read, write, execute, etc.) and would
# hang forever waiting for approval prompts that can't be provided in CI/CD.
#
# ONLY set skip_permissions=False for tests that specifically test the
# approval/interrupt workflow (HITL - human-in-the-loop).
#
# ⚠️  DO NOT CHANGE THIS DEFAULT WITHOUT EXPLICIT APPROVAL! ⚠️
# Changing this will break 95% of integration tests.
# ============================================================================
```

**Dynamic command building:**
```python
cmd = ["deep-worker", name, "--dir", working_dir]
if skip_permissions:
    cmd.append("--skip-permissions")  # ← Now conditional but defaults to True
if provider:
    cmd.extend(["--provider", provider])
```

#### 2. `agent_fixture.py`

**Already correct, but added warnings:**
- Enhanced docstrings with critical warnings
- Added inline comments on parameter defaults
- Clarified when to use `skip_permissions=False`

### Documentation Updates

| File | Status |
|------|--------|
| `README.md` | ✅ Updated with skip_permissions policy |
| `CLEAN_AGENT_USAGE.md` | ✅ Added critical warnings |
| `SKIP_PERMISSIONS_POLICY.md` | ✅ NEW (this file) |

---

## When to Use `skip_permissions=False`

**ONLY for tests that specifically test the approval workflow:**

### Valid Use Cases (Rare!)

1. **HITL workflow tests** - Testing human-in-the-loop approval mechanism
2. **Interrupt handling tests** - Testing agent behavior when tools are denied
3. **Approval menu tests** - Testing UI for permission prompts

### Example (Hypothetical Future Test)

```python
def test_approval_workflow_deny_tool(clean_agent):
    """Test that agent handles denied tool permission correctly."""
    # This is the ONLY valid reason to use skip_permissions=False
    agent_process = clean_agent(
        "test_approval_workflow",
        skip_permissions=False  # ← Explicitly disabled for this test
    )

    # Test would need to programmatically respond to approval prompts
    # (e.g., via mock, or automation framework)
    ...
```

### ⚠️ Requirements for `skip_permissions=False` Tests

If you create a test with `skip_permissions=False`:

1. **Mark the test clearly:**
   ```python
   @pytest.mark.approval_test  # Custom marker
   @pytest.mark.skip(reason="Requires manual approval - not suitable for CI/CD")
   def test_approval_workflow(...):
   ```

2. **Document WHY it's needed** in the test docstring

3. **Ensure it doesn't run in CI/CD** (use `@pytest.mark.skip` or similar)

4. **Provide automation** if possible (mock approval responses)

---

## Current Test Audit

### Tests Using Fixtures (All Use Default skip_permissions=True ✅)

| Test File | Fixture | Skip Permissions | Status |
|-----------|---------|------------------|--------|
| `test_agent_basic_example.py` | `agent_fixture` | ✅ True (default) | CORRECT |
| `test_claude_code_integration.py` | `agent_fixture` | ✅ True (explicit) | CORRECT |
| `test_interactive_real.py` | `agent_fixture` | ✅ True (explicit) | CORRECT |
| `test_agent_tool_call_duplication_deepseek.py` | `clean_agent` | ✅ True (default) | CORRECT |
| `test_agent_tool_call_duplication_claude_code.py` | `clean_agent` | ✅ True (default) | CORRECT |
| `test_websearch_delegation_deepseek.py` | `clean_agent` | ✅ True (default) | CORRECT |
| `test_agent_multiple_queries.py` | `clean_agent` | ✅ True (default) | CORRECT |
| `test_terminal_emulator.py` | subprocess (manual) | ✅ True (`--skip-permissions`) | CORRECT |
| `test_agent_process_lifecycle.py` | subprocess (manual) | ✅ True (`--skip-permissions`) | CORRECT |
| `test_slow_tool_display.py` | subprocess (manual) | ✅ True (`--skip-permissions`) | CORRECT |
| `test_taskmanager_tui.py` | subprocess (manual) | ✅ True (`--skip-permissions`) | CORRECT |
| `test_rpyc_integration.py` | `AgentProcess` | ✅ True (explicit) | CORRECT |
| `test_interactive_rpyc.py` | `AgentProcess` | ✅ True (explicit) | CORRECT |

**Result:** ✅ **100% compliance - All tests use skip_permissions=True**

### Tests WITHOUT Approval Testing (Expected: 100%)

Currently: **0 tests test the approval workflow** (none use `skip_permissions=False`)

This is expected and correct for the current test suite.

---

## Enforcement

### Code Review Checklist

When reviewing integration tests:

- [ ] Does the test create an agent?
- [ ] Does it use `skip_permissions=True` (default)?
- [ ] If `skip_permissions=False`, is it justified?
- [ ] If `skip_permissions=False`, is the test marked to skip in CI/CD?
- [ ] Are there clear comments explaining the permission setting?

### CI/CD Checks

**Future improvement:** Add a linter/checker that flags:
- Tests using `skip_permissions=False` without explicit justification
- Tests that might hang on approval prompts

---

## Migration Guide

### If You're Adding a New Test

```python
# ✅ CORRECT - Use fixture with defaults
def test_my_new_feature(clean_agent):
    agent = clean_agent("test_my_feature")  # skip_permissions=True by default
    ...
```

### If You're Updating an Existing Test

**No changes needed!** All tests already comply with the policy.

### If You MUST Test Approval Workflow

```python
# 1. Mark clearly
@pytest.mark.approval_test
@pytest.mark.skip(reason="Requires manual approval")
def test_approval_workflow_acceptance(clean_agent):
    """Test user accepting tool permission.

    This test REQUIRES manual interaction and cannot run in CI/CD.
    """
    # 2. Explicitly disable skip_permissions
    agent = clean_agent(
        "test_approval",
        skip_permissions=False  # ← MUST justify in comments
    )

    # 3. Automate approval response somehow
    # (via mock, test automation framework, etc.)
    ...
```

---

## Summary

### Key Points

1. ✅ **Default:** `skip_permissions=True` in ALL fixtures
2. ⚠️ **Critical:** Changing this breaks 95% of tests
3. 🚫 **Forbidden:** Using `skip_permissions=False` without explicit justification
4. ✅ **Status:** 100% compliance across all current tests

### Responsibility

**Everyone who modifies integration test fixtures MUST:**

1. Preserve `skip_permissions=True` as the default
2. Add warnings if changing behavior
3. Document any exceptions clearly
4. Ensure CI/CD tests don't hang

---

## References

- **Fixtures:** `clean_agent_fixture.py`, `agent_fixture.py`
- **Documentation:** `README.md`, `CLEAN_AGENT_USAGE.md`
- **Discussion:** See commit history for rationale

---

**Document Version:** 1.0
**Last Updated:** 2026-03-16
**Status:** ✅ ENFORCED
