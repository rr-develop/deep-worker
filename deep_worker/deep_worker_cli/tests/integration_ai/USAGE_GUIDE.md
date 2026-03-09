# Usage Guide: AI-Powered Screenshot Validation

## Quick Start

### For Async Tests
```python
from tests.integration_ai.check_reply import verify_screenshot_contains

result = await verify_screenshot_contains(
    screenshot_text=terminal_screen,
    original_query="What is 2+2?",
    expected_response_contains="4",
    model="haiku",  # or "sonnet", "opus"
)

if not result.is_valid:
    pytest.fail(f"Validation failed: {result.comment}")
```

### For Sync Tests
```python
from tests.integration_ai.check_reply import verify_screenshot_contains_sync

result = verify_screenshot_contains_sync(
    screenshot_text=terminal_screen,
    original_query="What is 2+2?",
    expected_response_contains="4",
    model="haiku",
)

assert result.is_valid, f"Validation failed: {result.comment}"
```

---

## Function Reference

### `verify_screenshot_contains()`

**Signature**:
```python
async def verify_screenshot_contains(
    screenshot_text: str,
    original_query: str,
    expected_response_contains: Optional[str] = None,
    model: Literal["haiku", "sonnet", "opus"] = "haiku",
) -> CheckReplyResult
```

**Parameters**:
- `screenshot_text` (str): Raw text from terminal emulator's `get_screen_text()`
- `original_query` (str): The user's query (e.g., "What is 2+2?")
- `expected_response_contains` (Optional[str]): Expected content in response (e.g., "4")
  - If None, just checks for query presence
- `model` (str): Claude model to use
  - "haiku" (default): Fastest, cheapest (~$0.0003 per validation)
  - "sonnet": Balanced
  - "opus": Best quality, most expensive

**Returns**: `CheckReplyResult`
```python
class CheckReplyResult:
    is_valid: bool                    # True if validation passed
    comment: str                      # Explanation of result
    confidence: Optional[float]       # 0.0-1.0 confidence score
    model_used: str                   # Model that performed validation
    validation_time_ms: Optional[float]  # Processing time
    cost_usd: Optional[float]         # Estimated cost
```

**Raises**:
- `ValidationError`: If validation infrastructure fails
- `ImportError`: If claude_sdk is not available

**Example**:
```python
@pytest.mark.asyncio
async def test_agent_response():
    session = DwcliTerminalSession(agent_name="test_agent")
    session.start()

    session.send_input("Solve: 15 + 7", press_enter=True)
    time.sleep(2)  # Wait for response

    screenshot = session.get_screen_text()

    result = await verify_screenshot_contains(
        screenshot_text=screenshot,
        original_query="Solve: 15 + 7",
        expected_response_contains="22",
        model="haiku",
    )

    assert result.is_valid, f"Response validation failed: {result.comment}"
    session.close()
```

---

### `verify_screenshot_contains_sync()`

**Signature**:
```python
def verify_screenshot_contains_sync(
    screenshot_text: str,
    original_query: str,
    expected_response_contains: Optional[str] = None,
    model: Literal["haiku", "sonnet", "opus"] = "haiku",
) -> CheckReplyResult
```

**Note**: Same parameters as async version, but returns synchronously.

**Example** (for sync/pytest tests):
```python
def test_agent_response():
    session = DwcliTerminalSession(agent_name="test_agent")
    session.start()

    session.send_input("Hello!", press_enter=True)
    time.sleep(1)

    screenshot = session.get_screen_text()

    result = verify_screenshot_contains_sync(
        screenshot_text=screenshot,
        original_query="Hello!",
        model="haiku",
    )

    assert result.is_valid
    session.close()
```

---

## Common Patterns

### Pattern 1: Simple Query Validation
```python
# Just check if query is present (no response check)
result = await verify_screenshot_contains(
    screenshot_text=screen,
    original_query="Show me the weather",
    model="haiku",
)
assert result.is_valid, "Query not visible on screen"
```

### Pattern 2: Query + Response Validation
```python
# Check both query and response
result = await verify_screenshot_contains(
    screenshot_text=screen,
    original_query="What is the capital of France?",
    expected_response_contains="Paris",
    model="haiku",
)
assert result.is_valid
```

### Pattern 3: Multiple Choice Response
```python
# Accept any of multiple possible responses
result = await verify_screenshot_contains(
    screenshot_text=screen,
    original_query="2+2=?",
    expected_response_contains="4",  # Also accepts "2+2 = 4", "The answer is 4", etc.
    model="haiku",
)
```

### Pattern 4: Cost-Conscious Testing
```python
# Use Haiku for fastest, cheapest validation
for query, response in test_cases:
    result = await verify_screenshot_contains(
        screenshot_text=screenshot,
        original_query=query,
        expected_response_contains=response,
        model="haiku",  # $0.0003 per validation
    )
    assert result.is_valid

# Expensive tests might use Sonnet or Opus
```

### Pattern 5: With Error Details
```python
result = await verify_screenshot_contains(
    screenshot_text=screen,
    original_query="Hello",
    expected_response_contains="Hi",
    model="haiku",
)

if not result.is_valid:
    logger.error(f"Validation failed: {result.comment}")
    logger.error(f"Cost: ${result.cost_usd:.4f}")
    logger.error(f"Time: {result.validation_time_ms:.0f}ms")
    logger.error(f"Model: {result.model_used}")
    pytest.fail("Screenshot validation failed")
```

---

## Cost Analysis

### Per-Validation Cost (Claude Haiku)
```
Typical validation:
  - Input tokens:  ~400 tokens
  - Output tokens: ~100 tokens
  - Cost: $0.0003 USD

For context:
  - 1000 validations = $0.30
  - Daily validations (100) = $0.03
  - Monthly validations (3000) = $0.90
```

### Model Comparison
| Model | Speed | Cost | Best For |
|-------|-------|------|----------|
| Haiku | Fast | Cheapest (~$0.0003) | Standard tests |
| Sonnet | Medium | Medium | Complex analysis |
| Opus | Slow | Expensive | Premium quality |

---

## Troubleshooting

### "ImportError: claude_sdk not available"
```python
# Try importing with fallback
try:
    from tests.integration_ai.check_reply import verify_screenshot_contains
except ImportError:
    # Use simple string matching as fallback
    assert query in screen_text
    logger.warning("AI validation unavailable, using basic validation")
```

### "ValidationError: Failed to parse response"
- AI response format was unexpected
- This is a system/infrastructure error, not a test failure
- Retry the validation or check claude_sdk connection

### Validation timeout (>30 seconds)
- claude_sdk processing can be slow
- Network connection issue
- This is normal for AI validation
- Solution: Retry or use a simpler validation

### High costs
- Using Opus instead of Haiku
- Change to: `model="haiku"`
- Cost reduces from $0.05+ to $0.0003 per validation

---

## Best Practices

### ✅ DO
1. **Use Haiku by default** - sufficient for text analysis, cheapest
2. **Specify expected response** - helps AI focus validation
3. **Handle ValidationError separately** - infrastructure issues vs test failures
4. **Log results for debugging** - cost, time, model, confidence
5. **Use in async contexts** - faster than sync wrapper
6. **Batch validations when possible** - reduces total cost

### ❌ DON'T
1. **Don't use for every assertion** - only for complex validations
2. **Don't rely on exact matching** - use with variable responses
3. **Don't forget to handle ImportError** - claude_sdk may be unavailable
4. **Don't use Opus for simple checks** - waste of money
5. **Don't validate same screenshot twice** - cache if needed
6. **Don't mix with naive string matching** - be consistent

---

## Integration with test_deepagents_query_does_not_disappear

This validation function was specifically designed for the improved test:

```python
@pytest.mark.asyncio
async def test_deepagents_query_does_not_disappear():
    session = DeepAgentsTerminalSession(...)
    session.start()

    # Send query
    query = "What is 2+2?"
    session.send_input(query, press_enter=True)

    # Wait for response
    time.sleep(2)
    screen = session.get_screen_text()

    # Use AI validation
    result = await verify_screenshot_contains(
        screenshot_text=screen,
        original_query=query,
        expected_response_contains="4",
        model="haiku",
    )

    assert result.is_valid
    session.close()
```

---

## Advanced Usage

### Custom Validation with Context
```python
result = await verify_screenshot_contains(
    screenshot_text=screen,
    original_query="Calculate 25 * 4",
    expected_response_contains="100",
    model="haiku",
)

# Result contains detailed information
print(f"Valid: {result.is_valid}")
print(f"Comment: {result.comment}")
print(f"Confidence: {result.confidence}")
print(f"Time: {result.validation_time_ms}ms")
print(f"Cost: ${result.cost_usd:.6f}")
```

### Batch Validation
```python
from tests.integration_ai.check_reply import batch_check_replies

results = await batch_check_replies(
    query_reply_pairs=[
        ("What is 2+2?", "4"),
        ("What is 3*3?", "9"),
        ("What is 5+5?", "10"),
    ],
    model="haiku",
    fail_fast=False,  # Continue even if some fail
)

for result in results:
    print(f"Valid: {result.is_valid}, Cost: ${result.cost_usd:.4f}")
```

---

## FAQ

**Q: How much does each validation cost?**
A: ~$0.0003 with Haiku. Cheaper than a penny for ~3000 validations.

**Q: How long does validation take?**
A: 4-5 seconds typically (most is API latency).

**Q: Can I use this offline?**
A: No, requires Claude API. Fallback to simple string matching if offline.

**Q: What if the response is different than expected?**
A: AI is smart enough to recognize equivalent responses (e.g., "4", "four", "2+2=4").

**Q: Should I use this for every test?**
A: No. Use for:
- Non-deterministic outputs
- Complex visual validation
- Natural language responses
Skip for simple exact matches or deterministic output.

**Q: Can I customize the validation prompt?**
A: Not directly, but you can:
1. Check if claude_sdk allows custom prompts
2. Fork the function and modify
3. Use different models (Sonnet/Opus for more control)

---

## Support

For issues or questions:
1. Check the test implementation: `test_terminal_emulator.py`
2. Review documentation: `TEST_IMPROVEMENT_DETAILS.md`
3. Check Claude SDK docs: claude_sdk/src/
4. Look at examples in this file

---

**Last Updated**: 2026-02-11
**Status**: Production Ready ✅
