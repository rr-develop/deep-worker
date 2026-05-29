"""DeepSeek chat model with reasoning_content round-trip.

``deepseek-v4-flash`` (and other DeepSeek thinking models) return a
``reasoning_content`` field alongside ``content``. The DeepSeek API REQUIRES
that ``reasoning_content`` be passed BACK on every subsequent turn, or it
rejects the request:

    400 - The `reasoning_content` in the thinking mode must be passed back to
          the API.

``langchain_deepseek.ChatDeepSeek`` (1.0.1) only handles ONE direction: it
reads ``reasoning_content`` from API responses into
``AIMessage.additional_kwargs``. Its ``_get_request_payload`` does NOT put it
back into the outgoing assistant messages — so on a multi-step turn (tool call
-> re-invoke) the field is dropped and the API returns the 400, crashing the
agent to ERROR (incident observed 2026-05-22).

``ReasoningChatDeepSeek`` closes that loop: it re-injects each assistant
message's stored ``reasoning_content`` into the matching outgoing assistant
payload entry. The input messages and ``payload["messages"]`` are in 1:1
positional order (verified — system/tool/assistant all map straight through),
so we walk both together and copy the field where present.

DO NOT drop this subclass back to plain ChatDeepSeek until the upstream
library re-sends reasoning_content itself.
"""

from __future__ import annotations

from typing import Any, cast

from langchain_core.language_models import LanguageModelInput
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.prompt_values import PromptValue
from langchain_deepseek import ChatDeepSeek


def _as_messages(input_: LanguageModelInput) -> list[BaseMessage] | None:
    """Best-effort extraction of the BaseMessage list from a model input.

    ``_get_request_payload`` receives whatever was passed to the model: a
    PromptValue, a list of messages, or (rarely) a string. We only need the
    message list to recover reasoning_content; if we can't get one cleanly we
    return None and skip re-injection (degrades to stock ChatDeepSeek behavior
    — no crash).
    """
    if isinstance(input_, PromptValue):
        return list(input_.to_messages())
    if isinstance(input_, list):
        # Keep only genuine BaseMessage items (the comprehension also narrows
        # the type for the checker — a raw ``list(input_)`` is
        # list[MessageLikeRepresentation]).
        msgs = [m for m in input_ if isinstance(m, BaseMessage)]
        if len(msgs) == len(input_):
            return msgs
    return None


class ReasoningChatDeepSeek(ChatDeepSeek):
    """ChatDeepSeek that re-sends ``reasoning_content`` on outgoing assistant
    messages so DeepSeek's thinking-mode contract is satisfied."""

    def _get_request_payload(
        self,
        input_: LanguageModelInput,
        *,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        payload = super()._get_request_payload(input_, stop=stop, **kwargs)

        source = _as_messages(input_)
        if source is None:
            return payload

        out_messages = payload.get("messages")
        # Positional 1:1 correlation between source messages and the converted
        # payload messages. Guard on equal length; if anything diverges (a
        # future library change), skip re-injection rather than mis-attach.
        if not isinstance(out_messages, list) or len(out_messages) != len(source):
            return payload

        for src, out in zip(source, out_messages):
            if not isinstance(src, AIMessage):
                continue
            if not isinstance(out, dict) or out.get("role") != "assistant":
                continue
            out_dict = cast("dict[str, Any]", out)
            reasoning: Any = src.additional_kwargs.get("reasoning_content")
            if reasoning:
                out_dict["reasoning_content"] = reasoning

        return payload
