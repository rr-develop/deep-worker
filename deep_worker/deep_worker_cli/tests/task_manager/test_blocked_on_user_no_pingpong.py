"""Regression tests: TaskManager must not ping-pong "wait for the user".

Observed defect: the subordinate answers "I am waiting for the user's
confirmation", the manager replies "wait for the user's confirmation",
the subordinate repeats itself, and the loop never converges.

Root cause was purely prompt-level: the evaluation criteria had no branch
for "blocked awaiting user input". Waiting is not 'done', and 'unable'
demanded proof of impossibility, so the model was pushed into
'in_progress' — and every 'in_progress' MESSAGE becomes another
subordinate turn (see task_manager.evaluate: only 'in_progress' returns a
prompt).

These tests pin the BLOCKED-ON-USER rule into both the initial and the
follow-up evaluation prompts, and pin the machinery that makes the
"stay silent" branch actually terminate the loop.
"""

import inspect

from deep_worker_cli.task_manager import task_manager as task_manager_module
from deep_worker_cli.task_manager.prompts import (
    TASK_MANAGER_SYSTEM_PROMPT,
    format_evaluation_prompt,
    format_followup_prompt,
    parse_manager_response,
)


def _evaluation_prompt() -> str:
    return format_evaluation_prompt(
        initial_request="Do the thing",
        last_response="I am waiting for the user's confirmation",
        session_file_path=None,
        project_path=None,
    )


class TestBlockedOnUserRuleInPrompts:
    """The rule must be present, and present in BOTH prompts."""

    def test_evaluation_prompt_has_blocked_on_user_rule(self):
        prompt = _evaluation_prompt()

        assert "BLOCKED-ON-USER RULE" in prompt

    def test_followup_prompt_has_blocked_on_user_rule(self):
        # The follow-up prompt is the one that actually loops, so the rule
        # must be restated there and not only in the first evaluation.
        prompt = format_followup_prompt(last_response="Still waiting for the user")

        assert "BLOCKED-ON-USER RULE" in prompt

    def test_rule_offers_answer_instead_of_user_branch(self):
        prompt = _evaluation_prompt()

        assert "ANSWER INSTEAD OF THE USER" in prompt
        # Branch (a) must stay bounded to answers already determined by the
        # request / explicit instructions / project policies.
        assert "unambiguous" in prompt

    def test_rule_offers_stay_silent_branch(self):
        prompt = _evaluation_prompt()

        assert "STAY SILENT" in prompt
        # The silent branch must be expressed through the existing
        # terminating status, with no message for the subordinate.
        assert "STATUS: unable" in prompt
        assert "MESSAGE: NONE" in prompt

    def test_stay_silent_branch_waives_impossibility_proof(self):
        # Without this carve-out the model refuses 'unable' (it cannot
        # prove impossibility) and falls back to 'in_progress' — the very
        # behaviour that creates the ping-pong.
        prompt = _evaluation_prompt()

        assert "does NOT require proof of impossibility" in prompt

    def test_rule_forbids_wait_for_the_user_replies(self):
        prompt = _evaluation_prompt()

        assert "STRICTLY FORBIDDEN" in prompt
        assert "wait for the user" in prompt
        assert "infinite" in prompt

    def test_rule_does_not_shield_unfinished_work(self):
        # Residual edge raised on review: the subordinate could claim it is
        # "waiting for the user" while non-user-gated work is still
        # unfinished, and the override wording alone could push the model
        # into silence too early.
        prompt = _evaluation_prompt()

        assert "does NOT require the user" in prompt
        assert "ordinary incomplete" in prompt
        assert "not a" in prompt and "shield" in prompt

    def test_rule_overrides_other_criteria(self):
        # Ordering matters: the generic 'unable' criteria demand proof, so
        # the rule has to declare precedence explicitly.
        prompt = _evaluation_prompt()

        assert "overrides other criteria" in prompt


class TestCoreResponsibilitiesPreserved:
    """The fix must not weaken the manager's main duties."""

    def test_incomplete_work_is_still_in_progress(self):
        prompt = _evaluation_prompt()

        assert "STATUS 'in_progress'" in prompt
        assert "Additional work is required to fully satisfy the request" in prompt

    def test_done_still_requires_full_completion(self):
        prompt = _evaluation_prompt()

        assert "STATUS 'done'" in prompt
        assert "Every aspect of the initial request has been addressed" in prompt
        assert "partial completion means 'in_progress', not 'done'" in prompt

    def test_impossibility_criteria_still_present(self):
        prompt = _evaluation_prompt()

        assert "Contradictory requirements" in prompt
        assert "Simple claims of impossibility are NOT sufficient" in prompt

    def test_user_prohibitions_still_binding(self):
        # This guard rail lives in the system prompt and in the follow-up
        # prompt; the new rule must not displace it.
        assert "IF THE USER ASKS NOT TO DO SOMETHING" in TASK_MANAGER_SYSTEM_PROMPT
        assert "IF THE USER ASKS NOT TO DO SOMETHING" in format_followup_prompt(
            last_response="anything"
        )


class TestSilentBranchTerminatesTheLoop:
    """The prompt rule only helps if 'unable' really stops the exchange."""

    def test_unable_without_message_parses(self):
        # Branch (b) emits "MESSAGE: NONE" under 'unable'. The parser must
        # accept it rather than raising.
        status, _message = parse_manager_response(
            "STATUS: unable\n"
            "REASONING: subordinate is blocked awaiting user input: which branch to use\n"
            "MESSAGE: NONE"
        )

        assert status == "unable"

    def test_unable_sends_nothing_to_subordinate(self):
        # Only 'in_progress' may produce a subordinate turn. Guard the
        # branch that makes silence possible.
        source = inspect.getsource(
            task_manager_module.TaskManager.on_subordinate_response
        )

        unable_branch = source.split("elif status == 'unable':", 1)[1]
        unable_branch = unable_branch.split("elif status == 'in_progress':", 1)[0]

        assert "return None" in unable_branch
        assert "TaskManager:>" not in unable_branch
