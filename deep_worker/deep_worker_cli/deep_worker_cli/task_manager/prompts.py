"""TaskManager prompts - exact copies from ai_lib/processes/task_manager.py.

CRITICAL: These prompts MUST be identical to ai_lib to ensure consistent behavior.
"""

from pathlib import Path
from typing import Literal, Optional


# System prompt for TaskManager agent
TASK_MANAGER_SYSTEM_PROMPT = """You are an expert at evaluating task completion and providing actionable feedback.

IF THE USER ASKS NOT TO DO SOMETHING, FOR EXAMPLE, NOT TO CHANGE CODE, NOT TO RUN OR NOT TO STOP SCRIPTS, THEN TASK MANAGER HAS NO RIGHT TO REQUEST FROM ITS SUBORDINATE TO DO ANYTHING THAT CONTRADICTS THIS REQUIREMENT. THESE REQUIREMENTS APPLY EQUALLY TO TASK MANAGER ITSELF AND TO ITS SUBORDINATE."""


def format_evaluation_prompt(
    initial_request: str,
    last_response: str,
    session_file_path: Optional[str],
    project_path: Optional[str]
) -> str:
    """
    Format the initial evaluation prompt for the manager.

    This is sent on the first evaluation of a new task.
    EXACT COPY from ai_lib/processes/task_manager.py:_format_evaluation_prompt

    Args:
        initial_request: The user's original request
        last_response: The subordinate's final response
        session_file_path: Path to subordinate's session file
        project_path: Project directory path

    Returns:
        Formatted prompt string
    """
    project_context = f"\nProject path: {Path(project_path).resolve()}" if project_path else ""
    session_file_info = f"\nSubordinate session file: {session_file_path}" if session_file_path else ""

    prompt = f"""You are a Task Manager AI. Your role is to monitor a subordinate AI agent and ensure it fully completes user tasks.

Your responsibilities:
- Evaluate if the subordinate has FULLY completed the user's initial request
- Provide specific, actionable corrections when work is incomplete
- Identify impossible tasks with concrete proof{project_context}{session_file_info}

EVALUATION CRITERIA:

1. STATUS 'done' - Return this ONLY if:
   - Every aspect of the initial request has been addressed
   - The solution is complete and functional
   - Nothing important has been missed

2. STATUS 'in_progress' - Return this if:
   - Some parts of the initial request are not yet implemented
   - Implementation is incomplete
   - Additional work is required to fully satisfy the request
   - You must provide a corrective prompt to guide the subordinate

3. STATUS 'unable' - Return this ONLY if you can PROVE impossibility through:
   - Contradictory requirements (e.g., "make it both sync and async simultaneously")
   - Technical impossibility that cannot be easily resolved (missing critical tools/permissions with no easy fix)
   - Clear looping pattern (5+ similar corrections without progress)

IMPORTANT:
- Simple claims of impossibility are NOT sufficient for 'unable' status
- The subordinate must provide convincing proof with specific technical reasons
- Focus on the INITIAL request, not intermediate corrections
- Be honest: partial completion means 'in_progress', not 'done'

OUTPUT FORMAT (must be strictly followed):
STATUS: <done|in_progress|unable>
REASONING: <brief explanation of your evaluation>
MESSAGE: <corrective prompt for in_progress, proof for unable, or NONE for done>

Generated messages and corrective requests should be formulated in the same language as the initial user request (but statuses should not be translated).

Examples:

STATUS: done
REASONING: The requested function was fully implemented with correct functionality
MESSAGE: NONE

STATUS: in_progress
REASONING: Initial request required function AND tests, but only function was created
MESSAGE: Please create comprehensive tests for the function as specified in the initial request

STATUS: unable
REASONING: Request contains contradictory requirements - function cannot be both synchronous and asynchronous in Python.
MESSAGE: The initial request is contradictory: a Python function must be either synchronous (def) or asynchronous (async def), but cannot be both simultaneously. Please clarify which pattern you need.


========== INITIAL USER REQUEST ==========
```
{initial_request}
```
========== END OF INITIAL USER REQUEST ==========


========== SUBORDINATE'S FINAL RESPONSE ==========
```
{last_response}
```
========== END OF SUBORDINATE'S FINAL RESPONSE ==========

Now evaluate the task and return the required response."""

    return prompt


def format_followup_prompt(last_response: str) -> str:
    """
    Format a follow-up evaluation prompt for the manager.

    This is sent after a correction to re-evaluate progress.
    The initial request is already in the manager's context.
    EXACT COPY from ai_lib/processes/task_manager.py:_format_followup_prompt

    Args:
        last_response: The subordinate's latest final response

    Returns:
        Formatted prompt string
    """
    prompt = f"""
========== SUBORDINATE'S RESPONSE AFTER CORRECTION ==========
```
{last_response}
```
========== END OF SUBORDINATE'S RESPONSE AFTER CORRECTION ==========

Re-evaluate if the initial user request is now fully completed.
Return your evaluation in the same format as before.

IF THE USER ASKS NOT TO DO SOMETHING, FOR EXAMPLE, NOT TO CHANGE CODE, NOT TO RUN OR NOT TO STOP SCRIPTS, THEN TASK MANAGER HAS NO RIGHT TO REQUEST FROM ITS SUBORDINATE TO DO ANYTHING THAT CONTRADICTS THIS REQUIREMENT. THESE REQUIREMENTS APPLY EQUALLY TO TASK MANAGER ITSELF AND TO ITS SUBORDINATE."""

    return prompt


def parse_manager_response(result_text: str) -> tuple[Literal['done', 'in_progress', 'unable'], Optional[str]]:
    """
    Parse the manager AI response into status and message.

    EXACT COPY from ai_lib/processes/task_manager.py:_parse_manager_response

    Args:
        result_text: The raw text response from the manager AI

    Returns:
        Tuple of (status, message)

    Raises:
        ValueError: If result is empty, missing STATUS, or has invalid STATUS value
    """
    # NO FALLBACK! Validate input first
    if not result_text or not result_text.strip():
        raise ValueError("Empty evaluation result from manager AI - this indicates a critical error in task evaluation")

    lines = result_text.strip().split('\n')

    status = None
    reasoning = None
    message = None

    # Parse the structured output
    for line in lines:
        line = line.strip()
        # Remove markdown formatting (**, ##, etc.) before checking
        line_clean = line.replace('**', '').replace('##', '').strip()

        if line_clean.startswith('STATUS:'):
            status_text = line_clean.replace('STATUS:', '').strip().lower()
            if status_text in ('done', 'in_progress', 'unable'):
                status = status_text
            else:
                raise ValueError(
                    f"Invalid STATUS value: '{status_text}'. "
                    f"Must be one of: done, in_progress, unable. "
                    f"Full response: {result_text[:200]}"
                )
        elif line_clean.startswith('REASONING:'):
            reasoning = line_clean.replace('REASONING:', '').strip()
        elif line_clean.startswith('MESSAGE:'):
            msg_text = line_clean.replace('MESSAGE:', '').strip()
            if msg_text and msg_text.upper() != 'NONE':
                message = msg_text

    # If MESSAGE is multiline, capture the rest
    if 'MESSAGE:' in result_text:
        message_start = result_text.find('MESSAGE:') + len('MESSAGE:')
        message_content = result_text[message_start:].strip()
        if message_content and message_content.upper() != 'NONE':
            message = message_content

    # NO FALLBACK! Status MUST be present
    if status is None:
        raise ValueError(
            f"Missing STATUS field in manager AI response. "
            f"Expected format: 'STATUS: <done|in_progress|unable>'. "
            f"Full response: {result_text[:200]}"
        )

    # Ensure message is None for 'done' status
    if status == 'done':
        message = None

    # Ensure message exists for 'in_progress' and 'unable'
    # This is ACCEPTABLE - providing helpful default message for valid statuses
    if status in ('in_progress', 'unable') and not message:
        if status == 'in_progress':
            message = "Additional work is needed to complete the initial request."
        else:
            message = "Task cannot be completed due to identified constraints."

    return status, message  # type: ignore
