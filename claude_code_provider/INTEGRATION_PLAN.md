# Claude Code Provider Integration Plan

## Goal
Integrate ClaudeCodeChatModel as a new AI provider in the CLI with support for simple model names (sonnet, haiku, opus).

## Principles
1. **TDD approach** - tests first, then code
2. **Do not break the defaults** - the default provider remains "deepseek"
3. **Fix existing bugs** - create_model() must use the config
4. **Simple model names** - no hardcoded versions

## Stages

### Stage 1: Fixing critical bugs (TDD)

**Problem**: `create_model()` does not use `config.provider` and `config.model`

**Steps**:
1. ✅ Write tests for `create_model()` with the `provider` and `model_name` parameters
   - Test: a call with `provider="deepseek"` and `model_name="deepseek-chat"`
   - Test: a call with `provider="openai"` and `model_name="gpt-4o"`
   - Test: a call without parameters (auto-detection) - must work as before
   - Test: backward compatibility with `model_name_override`

2. ✅ Change the signature of `create_model()` in `config.py`:
   ```python
   def create_model(
       provider: str | None = None,
       model_name: str | None = None,
       model_name_override: str | None = None
   ) -> BaseChatModel:
   ```

3. ✅ Update the logic inside `create_model()`:
   - If `provider` and `model_name` are set explicitly → use them
   - If not → fall back to auto-detection (the current behavior)
   - Support `model_name_override` for backward compatibility

4. ✅ Update the call in `agent_process.py` line 173:
   ```python
   model = create_model(
       provider=self.config.provider,
       model_name=self.config.model
   )
   ```

5. ✅ Run the tests - verify that the defaults are not broken

**Success criterion**: All tests pass, the defaults work as before, but now provider/model can be set explicitly.

---

### Stage 2: Adding Claude Code to agent_config.py (TDD)

**Steps**:
1. ✅ Write tests for `AgentConfig` with `provider="claude-code"`:
   - Test: creation with `provider="claude-code"` without an API key → success
   - Test: creation with `provider="claude-code"` and `model="sonnet"` → success
   - Test: creation with `provider="claude-code"` and `model=None` → success
   - Test: creation with an invalid provider → error

2. ✅ Change `agent_config.py`:
   - Line 52: add `"claude-code"` to `valid_providers`
   - Lines 58-80: add a branch for `claude-code` (does not require an API key)
   - Line 31: change `model: str` to `model: str | None = None`
   - Lines 47-49: allow `model=None`

3. ✅ Run the tests

**Success criterion**: `AgentConfig` accepts `provider="claude-code"` without errors.

---

### Stage 3: Adding Claude Code to config.py (TDD)

**Steps**:
1. ✅ Write tests for `create_model()` with `provider="claude-code"`:
   - Test: creation with `provider="claude-code"`, `model_name="sonnet"`
   - Test: creation with `provider="claude-code"`, `model_name="haiku"`
   - Test: creation with `provider="claude-code"`, `model_name="opus"`
   - Test: creation with `provider="claude-code"`, `model_name=None` (default: sonnet)
   - Test: creation with the full name `claude-sonnet-4-5-20250929`
   - Mock `ClaudeCodeChatModel` to avoid dependencies

2. ✅ Add a branch to `create_model()` (after line 1258):
   ```python
   elif provider == "claude-code":
       # Import from source
       # Map simple names (sonnet/haiku/opus)
       # Create ClaudeCodeChatModel
   ```

3. ✅ Run the tests

**Success criterion**: `create_model(provider="claude-code", model_name="sonnet")` returns a `ClaudeCodeChatModel`.

---

### Stage 4: Adding to the CLI (agent_command.py)

**Steps**:
1. ✅ Change `agent_command.py`:
   - Lines 42-47: add `"claude-code"` to `choices`
   - Lines 36-40: change the default for `--model` to `None`

2. ✅ Write an integration test:
   - Test: argument parsing with `--provider claude-code`
   - Test: argument parsing with `--provider claude-code --model haiku`

3. ✅ Run the tests

**Success criterion**: The CLI accepts `--provider claude-code` without errors.

---

### Stage 5: Integration testing

**Steps**:
1. ✅ Create an end-to-end test:
   - Mock `ClaudeSDKClient`
   - Create an `AgentConfig` with `provider="claude-code"`
   - Call `create_model()`
   - Verify that a `ClaudeCodeChatModel` is returned

2. ✅ Test with the real provider (if Claude Code is installed):
   - Create an agent with `provider="claude-code"`
   - Send a simple request
   - Check the response

3. ✅ Verify that the defaults are not broken:
   - Run without flags → must use deepseek
   - Run with `--provider deepseek` → must work
   - Run with `--provider openai` → must work (if a key is present)

**Success criterion**: All providers work, the defaults are not broken.

---

### Stage 6: Final check

**Steps**:
1. ✅ Run all the CLI unit tests
2. ✅ Run all the claude_code_provider tests (44 tests)
3. ✅ Check pyright (strict mode)
4. ✅ Create a test agent:
   ```bash
   deep-worker test-claude --provider claude-code --model sonnet
   ```
5. ✅ Connect through the CLI client and send a request
6. ✅ Check the logs for errors

**Success criterion**: All tests green, the agent starts and works.

---

## Files to change

### Files to be modified (4):
1. `process/agent_process.py` (1 line)
2. `config.py` (~50 lines)
3. `agent_config_pkg/agent_config.py` (~30 lines)
4. `cli/agent_command.py` (2 lines)

### New test files:
1. `tests/test_config_create_model.py` (new)
2. `tests/test_agent_config_claude_code.py` (new)
3. `tests/test_cli_claude_code.py` (new)

---

## Checking the defaults

After each stage, verify:

```bash
# 1. Default provider (deepseek)
deep-worker test-agent
# Should use deepseek (if DEEPSEEK_API_KEY is present)

# 2. Auto-detection works
unset DEEPSEEK_API_KEY
export OPENAI_API_KEY=sk-test
deep-worker test-agent
# Should use openai

# 3. An explicit provider works
deep-worker test-agent --provider deepseek
# Should use deepseek

# 4. Claude Code works
deep-worker test-agent --provider claude-code
# Should use claude-code without errors
```

---

## Risks and mitigation

| Risk | Probability | Mitigation |
|------|-------------|-----------|
| Breaking the defaults | Medium | TDD + a check after each stage |
| Conflict with existing code | Low | Backward compatibility via `model_name_override` |
| Claude SDK not installed | High | Graceful fallback with a clear error message |
| Model versions change | High | Use simple names (sonnet/haiku/opus) |

---

## Time estimate

- Stage 1: ~30 minutes (bug fixes + tests)
- Stage 2: ~20 minutes (agent_config.py + tests)
- Stage 3: ~30 minutes (config.py + tests)
- Stage 4: ~10 minutes (CLI + tests)
- Stage 5: ~20 minutes (integration tests)
- Stage 6: ~10 minutes (final check)

**Total: ~2 hours**

---

## Success criteria for the whole project

✅ All tests pass (unit + integration)
✅ Pyright strict: 0 errors
✅ Default provider = deepseek (unchanged)
✅ The `--provider` and `--model` flags work
✅ The Claude Code provider works with simple model names
✅ No hardcoded model versions
✅ Backward compatibility preserved
✅ Documentation updated
