# CHECK_REPLY Guide: AI-Powered Response Validation

Guide for using `check_reply()` to validate agent responses with AI meta-queries.

## Why Meta-Queries?

AI agent responses are non-deterministic. Traditional string matching is fragile:

```python
# FRAGILE: Breaks with variations
assert "4" in response  # Fails for "four", "2+2=4", "The answer is 4"
assert response == "4"  # Fails for "4.", " 4", "4\n"
```

Meta-queries use AI to validate AI:

```python
# ROBUST: Handles any correct format
result = await check_reply("What is 2+2?", response)
assert result.is_valid
```

## Basic Usage

```python
from tests.integration_ai import check_reply

result = await check_reply(
    query_text="What is the capital of France?",
    reply_text="Paris is the capital of France.",
    model="haiku"  # Default, cheapest
)

if result.is_valid:
    print("✓ Response is valid")
else:
    print(f"✗ Invalid: {result.comment}")
```

## API Reference

### check_reply()

```python
async def check_reply(
    query_text: str,              # Original query
    reply_text: str,              # Reply to validate
    model: Literal["haiku", "sonnet", "opus"] = "haiku",
    strict_mode: bool = False,    # More demanding validation
    additional_context: Optional[str] = None,  # Extra context
) -> CheckReplyResult
```

**Returns:** `CheckReplyResult` with:
- `is_valid: bool` - Whether reply is valid
- `comment: str` - Explanation if invalid
- `confidence: float` - Confidence (0.0-1.0)
- `model_used: str` - Model used
- `validation_time_ms: float` - Time taken
- `cost_usd: float` - Estimated cost

## Model Selection

### Haiku (Default)

```python
result = await check_reply(query, reply, model="haiku")
```

- **Cost:** ~$0.001 per validation
- **Speed:** Fast (~200-500ms)
- **Use for:** Simple validations, arithmetic, facts

### Sonnet

```python
result = await check_reply(query, reply, model="sonnet")
```

- **Cost:** ~$0.003 per validation
- **Speed:** Medium (~500-1000ms)
- **Use for:** Complex reasoning, nuanced answers

### Opus

```python
result = await check_reply(query, reply, model="opus")
```

- **Cost:** ~$0.015 per validation
- **Speed:** Slower (~1000-2000ms)
- **Use for:** Critical validation, very complex reasoning

## Strict Mode

```python
result = await check_reply(
    query,
    reply,
    model="sonnet",
    strict_mode=True  # More demanding
)
```

Strict mode makes validation more precise:
- Requires exact answers for factual questions
- Less tolerance for irrelevant information
- Better for testing precise functionality

## Additional Context

```python
result = await check_reply(
    query="Calculate the area",
    reply="25",
    additional_context="Area of a 5x5 square should be 25"
)
```

Use when:
- Reply needs domain-specific interpretation
- Validation requires background knowledge
- Query is ambiguous

## Common Patterns

### Pattern 1: Simple Assertion

```python
result = await check_reply(query, reply)
assert result.is_valid, f"Invalid: {result.comment}"
```

### Pattern 2: Using validate_test_reply()

```python
from tests.integration_ai import validate_test_reply

# Automatically raises AssertionError if invalid
await validate_test_reply(query, reply, test_name="test_math")
```

### Pattern 3: Batch Validation

```python
from tests.integration_ai import batch_check_replies

pairs = [
    ("What is 2+2?", "4"),
    ("What is 3+3?", "6"),
]

results = await batch_check_replies(pairs, model="haiku")
assert all(r.is_valid for r in results)
```

### Pattern 4: Conditional Model Selection

```python
# Try haiku first, fallback to sonnet
result = await check_reply(query, reply, model="haiku")

if not result.is_valid:
    logger.warning("Haiku validation failed, trying sonnet")
    result = await check_reply(query, reply, model="sonnet")

assert result.is_valid
```

## Injection Prevention

Meta-queries use structured prompts to prevent text injection:

```python
# SAFE - structured sections
"""
=== ORIGINAL QUERY ===
{query}

=== REPLY TO VALIDATE ===
{reply}

=== INSTRUCTIONS ===
Validate if reply answers query.
"""
```

The clear section boundaries prevent the reply from injecting instructions.

## Cost Optimization

### 1. Use Haiku by Default

```python
# ✓ GOOD - haiku for simple validation
await check_reply(query, reply, model="haiku")

# ✗ BAD - opus unnecessarily expensive
await check_reply("What is 2+2?", reply, model="opus")
```

### 2. Batch Validations

```python
# ✓ GOOD - validate multiple at once
results = await batch_check_replies(pairs)

# ✗ BAD - individual validations in loop
for query, reply in pairs:
    await check_reply(query, reply)  # Higher overhead
```

### 3. Cache Results

```python
# Cache validation results for repeated queries
validation_cache = {}

key = (query, reply)
if key not in validation_cache:
    result = await check_reply(query, reply)
    validation_cache[key] = result
```

## Troubleshooting

### Validation Always Fails

**Problem:** Valid responses marked as invalid

**Solutions:**

1. Check response format:
```python
print(f"Reply: '{reply}'")  # Check for hidden chars
```

2. Add context:
```python
result = await check_reply(
    query, reply,
    additional_context="Numeric answer expected"
)
```

3. Try better model:
```python
result = await check_reply(query, reply, model="sonnet")
```

### ValidationError Raised

**Problem:** Infrastructure failure, not invalid response

**Causes:**
- claude_sdk not available
- Malformed response from Claude
- Network issues

**Solution:**
```python
try:
    result = await check_reply(query, reply)
except ValidationError as e:
    logger.error(f"Validation infrastructure failed: {e}")
    # This is not a test failure, it's an infra issue
    pytest.skip("Validation infrastructure unavailable")
```

### High Costs

**Problem:** Tests too expensive

**Solutions:**

1. Use haiku for simple validations
2. Reduce test count
3. Mock validations in unit tests:
```python
# Unit test with mock
@patch('tests.integration_ai.check_reply')
async def test_logic(mock_check):
    mock_check.return_value = CheckReplyResult(
        is_valid=True, comment="", model_used="haiku"
    )
    # Test logic without actual validation
```

## Best Practices

### 1. Clear Queries

```python
# ✓ GOOD - specific query
"Calculate the area of a 5x5 square"

# ✗ BAD - ambiguous query
"What's the area?"
```

### 2. Reasonable Expectations

```python
# ✓ GOOD - validate correctness, not format
result = await check_reply("What is 2+2?", "The answer is 4")
# Accepts various correct formats

# ✗ BAD - overly strict
assert reply == "4"  # Rejects "The answer is 4"
```

### 3. Error Messages

```python
# ✓ GOOD - informative failure
assert result.is_valid, (
    f"Agent failed validation\n"
    f"Query: {query}\n"
    f"Reply: {reply}\n"
    f"Reason: {result.comment}"
)

# ✗ BAD - unclear failure
assert result.is_valid
```

### 4. Log Costs

```python
result = await check_reply(query, reply)
logger.info(
    f"Validation: {result.is_valid}, "
    f"cost: ${result.cost_usd:.4f}, "
    f"time: {result.validation_time_ms:.0f}ms"
)
```

## Examples

### Example 1: Factual Question

```python
result = await check_reply(
    query_text="Who wrote Romeo and Juliet?",
    reply_text="William Shakespeare wrote Romeo and Juliet.",
    model="haiku"
)
assert result.is_valid
```

### Example 2: Calculation

```python
result = await check_reply(
    query_text="What is 15% of 200?",
    reply_text="30",
    model="haiku"
)
assert result.is_valid
```

### Example 3: Complex Reasoning

```python
result = await check_reply(
    query_text="Explain why the sky is blue",
    reply_text="The sky appears blue due to Rayleigh scattering...",
    model="sonnet",  # Need better model for complex validation
    strict_mode=True
)
assert result.is_valid
```

### Example 4: With Context

```python
result = await check_reply(
    query_text="Is it safe?",
    reply_text="Yes, the operation is safe",
    additional_context="Query asks about safety of surgical procedure",
    model="sonnet"
)
assert result.is_valid
```

## Related Documentation

- [README.md](./README.md) - Infrastructure overview
- [models.py](./models.py) - Pydantic models
- [check_reply.py](./check_reply.py) - Implementation

## License

Same as deep_worker_cli project.
