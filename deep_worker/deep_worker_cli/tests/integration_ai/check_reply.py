"""AI-powered response validation using claude_sdk meta-queries.

This module provides functions to validate agent responses by asking another
AI (meta-query) whether the response correctly answers the original query.

This approach is more robust than string matching for non-deterministic AI responses.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Literal, Optional, cast

# Add claude_sdk to path
CLAUDE_SDK_PATH = Path(__file__).parent.parent.parent.parent.parent / "claude_sdk" / "src"
if CLAUDE_SDK_PATH.exists() and str(CLAUDE_SDK_PATH) not in sys.path:
    sys.path.insert(0, str(CLAUDE_SDK_PATH))

try:
    from claude_agent_sdk import query  # type: ignore[import-untyped]
    from claude_agent_sdk.types import (  # type: ignore[import-untyped]
        AssistantMessage,
        ClaudeAgentOptions,
        TextBlock,
    )

    CLAUDE_SDK_AVAILABLE = True
except ImportError:
    CLAUDE_SDK_AVAILABLE = False
    logging.warning(
        "claude_sdk not available. AI validation will not work. "
        f"Expected path: {CLAUDE_SDK_PATH}"
    )

from .models import CheckReplyResult, MetaQueryRequest, ValidationError

logger = logging.getLogger(__name__)


# Model cost estimates (USD per 1M input tokens / 1M output tokens)
MODEL_COSTS = {
    "haiku": (0.25, 1.25),  # $0.25 input, $1.25 output per 1M tokens
    "sonnet": (3.00, 15.00),  # $3.00 input, $15.00 output per 1M tokens
    "opus": (15.00, 75.00),  # $15.00 input, $75.00 output per 1M tokens
}


def estimate_cost(
    model: str, prompt_length: int, response_length: int
) -> float:
    """Estimate validation cost in USD.

    Args:
        model: Model name (haiku/sonnet/opus)
        prompt_length: Length of prompt in characters
        response_length: Length of response in characters

    Returns:
        Estimated cost in USD
    """
    if model not in MODEL_COSTS:
        return 0.0

    # Rough estimate: 4 chars = 1 token
    prompt_tokens = prompt_length / 4
    response_tokens = response_length / 4

    input_cost_per_1m, output_cost_per_1m = MODEL_COSTS[model]

    input_cost = (prompt_tokens / 1_000_000) * input_cost_per_1m
    output_cost = (response_tokens / 1_000_000) * output_cost_per_1m

    return input_cost + output_cost


async def check_reply(
    query_text: str,
    reply_text: str,
    model: Literal["haiku", "sonnet", "opus"] = "haiku",
    strict_mode: bool = False,
    additional_context: Optional[str] = None,
) -> CheckReplyResult:
    """Validate whether a reply correctly answers a query using AI meta-query.

    This function sends a structured prompt to Claude asking it to validate
    whether the reply correctly answers the original query.

    Args:
        query_text: The original query sent to the agent
        reply_text: The agent's reply to validate
        model: Claude model to use (haiku=fast/cheap, sonnet=balanced, opus=best)
        strict_mode: Whether to use strict validation criteria
        additional_context: Optional context to help validation

    Returns:
        CheckReplyResult with validation outcome

    Raises:
        ValidationError: If validation infrastructure fails
        ImportError: If claude_sdk is not available

    Example:
        >>> result = await check_reply("What is 2+2?", "The answer is 4")
        >>> assert result.is_valid
        >>> print(result)  # ✓ Valid (model: haiku)

        >>> result = await check_reply("What is 2+2?", "I don't know")
        >>> assert not result.is_valid
        >>> print(result.comment)  # "Reply does not answer the question"
    """
    if not CLAUDE_SDK_AVAILABLE:
        raise ImportError(
            f"claude_sdk not available. Expected at: {CLAUDE_SDK_PATH}. "
            "Cannot perform AI validation without claude_sdk."
        )

    # Build meta-query request
    request = MetaQueryRequest(
        original_query=query_text,
        reply_text=reply_text,
        model=model,
        strict_mode=strict_mode,
        additional_context=additional_context,
    )

    prompt = request.build_prompt()
    start_time = time.time()

    logger.debug(f"Validating reply using {model} model")
    logger.debug(f"Query: {query_text[:100]}...")
    logger.debug(f"Reply: {reply_text[:100]}...")

    try:
        # Configure claude_sdk options
        options = ClaudeAgentOptions(
            model=model,
            permission_mode="bypassPermissions",  # No user approvals for validation
        )

        # Execute meta-query
        response_text = ""
        async for message in query(prompt=prompt, options=options):
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock):
                        response_text += block.text

        # Calculate timing and cost
        elapsed_ms = (time.time() - start_time) * 1000
        estimated_cost = estimate_cost(model, len(prompt), len(response_text))

        logger.debug(f"Validation response ({elapsed_ms:.0f}ms): {response_text[:200]}...")

        # Parse JSON response
        try:
            # Try to extract JSON from response (Claude might add text around it)
            response_text = response_text.strip()

            # Find JSON object in response
            json_start = response_text.find("{")
            json_end = response_text.rfind("}") + 1

            if json_start == -1 or json_end == 0:
                raise ValueError("No JSON object found in response")

            json_text = response_text[json_start:json_end]
            result_data = json.loads(json_text)

            # Validate required fields
            if "is_valid" not in result_data:
                raise ValueError("Response missing 'is_valid' field")

            is_valid = bool(result_data["is_valid"])
            comment = str(result_data.get("comment", ""))
            confidence = result_data.get("confidence")

            # Convert confidence to float if present
            if confidence is not None:
                try:
                    confidence = float(confidence)
                except (ValueError, TypeError):
                    logger.warning(f"Invalid confidence value: {confidence}")
                    confidence = None

            result = CheckReplyResult(
                is_valid=is_valid,
                comment=comment,
                confidence=confidence,
                model_used=model,
                validation_time_ms=elapsed_ms,
                cost_usd=estimated_cost,
            )

            logger.info(
                f"Validation result: {result} (cost: ${estimated_cost:.4f}, time: {elapsed_ms:.0f}ms)"
            )

            return result

        except (json.JSONDecodeError, ValueError, KeyError) as e:
            logger.error(f"Failed to parse validation response: {e}")
            logger.error(f"Raw response: {response_text}")

            raise ValidationError(
                message=f"Failed to parse validation response: {e}",
                original_query=query_text,
                reply_text=reply_text,
                raw_response=response_text,
            ) from e

    except Exception as e:
        if isinstance(e, ValidationError):
            raise

        logger.error(f"Validation query failed: {e}")
        raise ValidationError(
            message=f"Validation query failed: {e}",
            original_query=query_text,
            reply_text=reply_text,
        ) from e


async def validate_test_reply(
    query_text: str,
    reply_text: str,
    test_name: str = "reply_validation",
    model: Literal["haiku", "sonnet", "opus"] = "haiku",
    strict_mode: bool = False,
) -> bool:
    """Convenience function for tests that raises AssertionError if invalid.

    This is a wrapper around check_reply() that automatically fails the test
    if validation fails.

    Args:
        query_text: Original query
        reply_text: Reply to validate
        test_name: Name of test (for error messages)
        model: Claude model to use
        strict_mode: Whether to use strict validation

    Returns:
        True if valid (always, since invalid raises)

    Raises:
        AssertionError: If reply is invalid (causes test to fail)
        ValidationError: If validation infrastructure fails

    Example:
        >>> await validate_test_reply("What is 2+2?", response)
        # Test passes if response is valid, fails with clear message if not
    """
    result = await check_reply(
        query_text=query_text,
        reply_text=reply_text,
        model=model,
        strict_mode=strict_mode,
    )

    if not result.is_valid:
        error_msg = f"[{test_name}] Reply validation failed: {result.comment}"
        logger.error(error_msg)
        logger.error(f"Query: {query_text}")
        logger.error(f"Reply: {reply_text}")
        raise AssertionError(error_msg)

    logger.info(f"[{test_name}] Reply validated successfully")
    return True


async def batch_check_replies(
    query_reply_pairs: list[tuple[str, str]],
    model: Literal["haiku", "sonnet", "opus"] = "haiku",
    fail_fast: bool = False,
) -> list[CheckReplyResult]:
    """Validate multiple query-reply pairs in batch.

    Args:
        query_reply_pairs: List of (query, reply) tuples
        model: Claude model to use for all validations
        fail_fast: If True, stop on first invalid reply

    Returns:
        List of CheckReplyResult objects

    Raises:
        AssertionError: If fail_fast=True and any reply is invalid
        ValidationError: If validation infrastructure fails

    Example:
        >>> pairs = [
        ...     ("What is 2+2?", "4"),
        ...     ("What is 3+3?", "6"),
        ... ]
        >>> results = await batch_check_replies(pairs)
        >>> assert all(r.is_valid for r in results)
    """
    results: list[CheckReplyResult] = []

    for i, (query, reply) in enumerate(query_reply_pairs, 1):
        logger.info(f"Validating pair {i}/{len(query_reply_pairs)}")

        result = await check_reply(query, reply, model=model)
        results.append(result)

        if fail_fast and not result.is_valid:
            raise AssertionError(
                f"Validation failed for pair {i}: {result.comment}"
            )

    # Summary statistics
    valid_count = sum(1 for r in results if r.is_valid)
    total_cost = sum(r.cost_usd or 0.0 for r in results)
    total_time = sum(r.validation_time_ms or 0.0 for r in results)

    logger.info(
        f"Batch validation complete: {valid_count}/{len(results)} valid, "
        f"${total_cost:.4f} cost, {total_time:.0f}ms total"
    )

    return results


def check_reply_sync(
    query_text: str,
    reply_text: str,
    model: Literal["haiku", "sonnet", "opus"] = "haiku",
) -> CheckReplyResult:
    """Synchronous wrapper for check_reply (for non-async tests).

    Args:
        query_text: Original query
        reply_text: Reply to validate
        model: Claude model to use

    Returns:
        CheckReplyResult

    Note:
        This creates an event loop if needed. Prefer async version when possible.
    """
    import asyncio

    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    return loop.run_until_complete(
        check_reply(query_text, reply_text, model=model)
    )


async def verify_screenshot_contains(
    screenshot_text: str,
    original_query: str,
    expected_response_contains: Optional[str] = None,
    model: Literal["haiku", "sonnet", "opus"] = "haiku",
) -> CheckReplyResult:
    """Verify that a screenshot contains both query and response using AI analysis.

    This function uses an AI agent to analyze a terminal screenshot and verify:
    1. The original query is visible on the screen
    2. A response to that query is present (if expected_response_contains is provided)

    Args:
        screenshot_text: Raw text from terminal emulator screen
        original_query: The query that was sent (e.g., "What is 2+2?")
        expected_response_contains: Optional expected content in response (e.g., "4")
        model: Claude model to use for analysis (haiku=cheapest, opus=best)

    Returns:
        CheckReplyResult with validation outcome

    Raises:
        ValidationError: If validation infrastructure fails
        ImportError: If claude_sdk is not available

    Example:
        >>> screenshot = session.get_screen_text()
        >>> result = await verify_screenshot_contains(
        ...     screenshot,
        ...     original_query="What is 2+2?",
        ...     expected_response_contains="4"
        ... )
        >>> assert result.is_valid, f"Screenshot invalid: {result.comment}"
    """
    if not CLAUDE_SDK_AVAILABLE:
        raise ImportError(
            f"claude_sdk not available. Expected at: {CLAUDE_SDK_PATH}. "
            "Cannot perform screenshot analysis without claude_sdk."
        )

    # Build prompt for screenshot analysis
    sections = [
        "You are an expert at analyzing terminal screenshots.",
        "Your task is to verify whether a screenshot contains specific text.",
        "",
        "=== SCREENSHOT TEXT ===",
        screenshot_text,
        "",
        "=== VERIFICATION TASK ===",
        f"1. Check if this screenshot contains the query: '{original_query}'",
    ]

    if expected_response_contains:
        sections.extend(
            [
                f"2. Check if the screenshot contains a response mentioning or containing: '{expected_response_contains}'",
                "",
                "Both the query and response must be present for validation to pass.",
            ]
        )
    else:
        sections.extend(
            [
                "2. Check if there is any visible response or output on the screen",
                "",
                "The query must be present for validation to pass.",
            ]
        )

    sections.extend(
        [
            "",
            "=== INSTRUCTIONS ===",
            "Analyze the screenshot carefully.",
            "The text may be split across multiple lines due to terminal wrapping.",
            "It may include terminal UI elements, boxes, colors, etc.",
            "",
            "Look for the query text (or parts of it) anywhere on the screen.",
            "Look for a response that directly or indirectly answers the query.",
            "",
            "Respond ONLY with valid JSON in this exact format:",
            "{",
            '  "is_valid": true or false,',
            '  "comment": "explanation of what was found or missing",',
            '  "query_found": true or false,',
            '  "response_found": true or false',
            "}",
            "",
            "Rules:",
            "- is_valid: true ONLY if both query AND response are found",
            "- query_found: true if the query text (or significant part of it) is visible",
            "- response_found: true if a response is visible",
            f"- comment: briefly explain what you found (e.g., 'Query visible, response shows {expected_response_contains}')",
        ]
    )

    prompt = "\n".join(sections)
    start_time = time.time()

    logger.debug(f"Analyzing screenshot for query: '{original_query}'")
    logger.debug(f"Expected response contains: {expected_response_contains}")

    try:
        # Configure claude_sdk options
        options = ClaudeAgentOptions(
            model=model,
            permission_mode="bypassPermissions",  # No user approvals for validation
        )

        # Execute screenshot analysis query
        response_text = ""
        async for message in query(prompt=prompt, options=options):
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock):
                        response_text += block.text

        # Calculate timing and cost
        elapsed_ms = (time.time() - start_time) * 1000
        estimated_cost = estimate_cost(model, len(prompt), len(response_text))

        logger.debug(
            f"Screenshot analysis response ({elapsed_ms:.0f}ms): {response_text[:200]}..."
        )

        # Parse JSON response
        try:
            response_text = response_text.strip()

            # Find JSON object in response
            json_start = response_text.find("{")
            json_end = response_text.rfind("}") + 1

            if json_start == -1 or json_end == 0:
                raise ValueError("No JSON object found in response")

            json_text = response_text[json_start:json_end]
            result_data = json.loads(json_text)

            # Validate required fields
            if "is_valid" not in result_data:
                raise ValueError("Response missing 'is_valid' field")

            is_valid = bool(result_data["is_valid"])
            comment = str(result_data.get("comment", ""))

            result = CheckReplyResult(
                is_valid=is_valid,
                comment=comment,
                confidence=None,
                model_used=model,
                validation_time_ms=elapsed_ms,
                cost_usd=estimated_cost,
            )

            logger.info(
                f"Screenshot validation: {result} (cost: ${estimated_cost:.4f}, time: {elapsed_ms:.0f}ms)"
            )

            return result

        except (json.JSONDecodeError, ValueError, KeyError) as e:
            logger.error(f"Failed to parse screenshot analysis response: {e}")
            logger.error(f"Raw response: {response_text}")

            raise ValidationError(
                message=f"Failed to parse screenshot analysis response: {e}",
                original_query=original_query,
                raw_response=response_text,
            ) from e

    except Exception as e:
        if isinstance(e, ValidationError):
            raise

        logger.error(f"Screenshot analysis query failed: {e}")
        raise ValidationError(
            message=f"Screenshot analysis query failed: {e}",
            original_query=original_query,
        ) from e


def verify_screenshot_contains_sync(
    screenshot_text: str,
    original_query: str,
    expected_response_contains: Optional[str] = None,
    model: Literal["haiku", "sonnet", "opus"] = "haiku",
) -> CheckReplyResult:
    """Synchronous wrapper for verify_screenshot_contains (for non-async tests).

    Args:
        screenshot_text: Raw text from terminal emulator screen
        original_query: The query that was sent
        expected_response_contains: Optional expected content in response
        model: Claude model to use

    Returns:
        CheckReplyResult

    Note:
        This creates an event loop if needed. Prefer async version when possible.
    """
    import asyncio

    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    return loop.run_until_complete(
        verify_screenshot_contains(
            screenshot_text=screenshot_text,
            original_query=original_query,
            expected_response_contains=expected_response_contains,
            model=model,
        )
    )
