"""Deep Agents come with planning, filesystem, and subagents."""

from collections.abc import Callable, Sequence
from typing import Any, cast

from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware, InterruptOnConfig, TodoListMiddleware
from langchain.agents.middleware.types import AgentMiddleware
from langchain.agents.structured_output import ResponseFormat
from langchain.chat_models import init_chat_model
from langchain_anthropic import ChatAnthropic
from langchain_anthropic.middleware import AnthropicPromptCachingMiddleware
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage
from langchain_core.tools import BaseTool
from langgraph.cache.base import BaseCache
from langgraph.graph.state import CompiledStateGraph
from langgraph.store.base import BaseStore
from langgraph.types import Checkpointer

from deepagents.backends import StateBackend
from deepagents.backends.protocol import BackendFactory, BackendProtocol
from deepagents.middleware.bounded_tool_call import BoundedToolCallMiddleware
from deepagents.middleware.filesystem import FilesystemMiddleware
from deepagents.middleware.memory import MemoryMiddleware
from deepagents.middleware.message_timestamps import MessageTimestampMiddleware
from deepagents.middleware.patch_tool_calls import PatchToolCallsMiddleware
from deepagents.middleware.skills import SkillsMiddleware
from deepagents.middleware.subagents import (
    GENERAL_PURPOSE_SUBAGENT,
    CompiledSubAgent,
    SubAgent,
    SubAgentMiddleware,
)
from deepagents.middleware.summarization import (
    SummarizationMiddleware,
    _apply_keep_override,
    _compute_summarization_defaults,
)

BASE_AGENT_PROMPT = "In order to complete the objective that the user asks of you, you have access to a number of standard tools."


def get_default_model() -> ChatAnthropic:
    """Get the default model for deep agents.

    Returns:
        `ChatAnthropic` instance configured with Claude Sonnet 4.5.
    """
    return ChatAnthropic(
        model_name="claude-sonnet-4-5-20250929",
        max_tokens=20000,
    )


def _bind_tools_for_middleware(
    model: BaseChatModel,
    tools: Sequence[BaseTool | Callable | dict[str, Any]] | None,
) -> BaseChatModel:
    """Return the model variant that should be handed to middleware.

    The 2026-06-13 incident — SummarizationMiddleware unbound-model capture
    (see a related internal incident analysis (not included here)).

    LangChain's ``bind_tools`` contract returns a NEW model instance —
    the raw ``model`` we received from the caller carries no tool
    inventory. ``SummarizationMiddleware`` keeps the reference it gets
    at construction time (``self.model = model``) and calls
    ``self.model.ainvoke(prompt)`` in ``_acreate_summary``. If we give
    it the raw, pre-bind model, that summary call goes through an
    instance whose tool metadata is empty — and for providers that
    share a singleton client across instances
    (``ClaudeCodeChatModel._client_manager``) the empty rebind
    silently propagates ``tools_metadata=[]`` into the next
    ``ProxyMCPServer`` rebuild on session rotation. The Claude CLI then
    has zero ``mcp__deepworker__*`` tools.

    This helper performs the bind once at agent-build time so all three
    ``SummarizationMiddleware`` sites in ``create_deep_agent`` see the
    same bound instance the turn-node uses. The model passed to
    ``create_agent(model, tools=...)`` is left untouched — LangGraph
    binds it internally for the turn-node — so the bind is purely
    additive: middleware sees a bound model, the rest of the wiring is
    unchanged.

    Args:
        model: The raw chat model the caller handed to ``create_deep_agent``.
        tools: The tool list the caller will pass to ``create_agent``;
            ``None`` or empty means there is nothing to bind, return as-is.

    Returns:
        The bind_tools-result when tools are present and ``model`` is
        bindable, otherwise ``model`` unchanged. Best-effort: any failure
        falls back to the raw model so a model without ``bind_tools`` (or
        a binding that raises for some unexpected tool shape) cannot
        break agent construction.
    """
    if not tools:
        return model
    bind = getattr(model, "bind_tools", None)
    if bind is None:
        return model
    try:
        return cast(BaseChatModel, bind(list(tools)))
    except Exception:  # noqa: BLE001
        # Never break agent construction over the middleware-binding path.
        # The middleware will still receive `model` and the only loss is
        # the bug fix — the agent runs exactly as it did before incident
        # 2026-06-13 was patched.
        return model


def create_deep_agent(
    model: str | BaseChatModel | None = None,
    tools: Sequence[BaseTool | Callable | dict[str, Any]] | None = None,
    *,
    system_prompt: str | SystemMessage | None = None,
    middleware: Sequence[AgentMiddleware] = (),
    subagents: list[SubAgent | CompiledSubAgent] | None = None,
    skills: list[str] | None = None,
    memory: list[str] | None = None,
    response_format: ResponseFormat | None = None,
    context_schema: type[Any] | None = None,
    checkpointer: Checkpointer | None = None,
    store: BaseStore | None = None,
    backend: BackendProtocol | BackendFactory | None = None,
    interrupt_on: dict[str, bool | InterruptOnConfig] | None = None,
    debug: bool = False,
    name: str | None = None,
    cache: BaseCache | None = None,
    summarization_history_prefix: str | None = None,
    summarization_on_summarized: Callable[[str], None] | None = None,
    summarization_token_counter: Callable[[Any], int] | None = None,
    context_limit: int | None = None,
    summarization_hot_cache_ttl: dict[str, int] | None = None,
    summarization_keep: tuple[str, int | float] | None = None,
    summarization_provider: str | None = None,
) -> CompiledStateGraph:
    """Create a deep agent.

    !!! warning "Deep agents require a LLM that supports tool calling!"

    By default, this agent has access to the following tools:

    - `write_todos`: manage a todo list
    - `ls`, `read_file`, `write_file`, `edit_file`, `glob`, `grep`: file operations
    - `execute`: run shell commands
    - `task`: call subagents

    The `execute` tool allows running shell commands if the backend implements `SandboxBackendProtocol`.
    For non-sandbox backends, the `execute` tool will return an error message.

    Args:
        model: The model to use.

            Defaults to `claude-sonnet-4-5-20250929`.

            Use the `provider:model` format (e.g., `openai:gpt-5`) to quickly switch between models.
        tools: The tools the agent should have access to.

            In addition to custom tools you provide, deep agents include built-in tools for planning,
            file management, and subagent spawning.
        system_prompt: Custom system instructions to prepend before the base deep agent
            prompt.

            If a string, it's concatenated with the base prompt.
        middleware: Additional middleware to apply after the standard middleware stack
            (`TodoListMiddleware`, `FilesystemMiddleware`, `SubAgentMiddleware`,
            `SummarizationMiddleware`, `AnthropicPromptCachingMiddleware`,
            `PatchToolCallsMiddleware`).
        subagents: The subagents to use.

            Each subagent should be a `dict` with the following keys:

            - `name`
            - `description` (used by the main agent to decide whether to call the sub agent)
            - `prompt` (used as the system prompt in the subagent)
            - (optional) `tools`
            - (optional) `model` (either a `LanguageModelLike` instance or `dict` settings)
            - (optional) `middleware` (list of `AgentMiddleware`)
        skills: Optional list of skill source paths (e.g., `["/skills/user/", "/skills/project/"]`).

            Paths must be specified using POSIX conventions (forward slashes) and are relative
            to the backend's root. When using `StateBackend` (default), provide skill files via
            `invoke(files={...})`. With `FilesystemBackend`, skills are loaded from disk relative
            to the backend's `root_dir`. Later sources override earlier ones for skills with the
            same name (last one wins).
        memory: Optional list of memory file paths (`AGENTS.md` files) to load
            (e.g., `["/memory/AGENTS.md"]`).

            Display names are automatically derived from paths.

            Memory is loaded at agent startup and added into the system prompt.
        response_format: A structured output response format to use for the agent.
        context_schema: The schema of the deep agent.
        checkpointer: Optional `Checkpointer` for persisting agent state between runs.
        store: Optional store for persistent storage (required if backend uses `StoreBackend`).
        backend: Optional backend for file storage and execution.

            Pass either a `Backend` instance or a callable factory like `lambda rt: StateBackend(rt)`.
            For execution support, use a backend that implements `SandboxBackendProtocol`.
        interrupt_on: Mapping of tool names to interrupt configs.

            Pass to pause agent execution at specified tool calls for human approval or modification.

            Example: `interrupt_on={"edit_file": True}` pauses before every edit.
        debug: Whether to enable debug mode. Passed through to `create_agent`.
        name: The name of the agent. Passed through to `create_agent`.
        cache: The cache to use for the agent. Passed through to `create_agent`.

    Returns:
        A configured deep agent.
    """
    if model is None:
        model = get_default_model()
    elif isinstance(model, str):
        model = init_chat_model(model)

    # Compute summarization defaults based on model profile
    summarization_defaults = _compute_summarization_defaults(
        model, context_limit=context_limit
    )
    if summarization_keep is not None:
        summarization_defaults = _apply_keep_override(
            summarization_defaults, summarization_keep
        )

    # Per-caller override for where SummarizationMiddleware offloads
    # evicted conversation history. When None, the middleware's own
    # default (`/conversation_history`) is used; callers that wire up
    # a real on-disk directory (e.g. deep_worker_cli) pass an absolute
    # path here so the path quoted in the synthetic summary message
    # corresponds to an actual, human-inspectable file.
    def _summarization_extra_kwargs() -> dict[str, Any]:
        kwargs: dict[str, Any] = {}
        if summarization_history_prefix is not None:
            kwargs["history_path_prefix"] = summarization_history_prefix
        if summarization_on_summarized is not None:
            kwargs["on_summarized"] = summarization_on_summarized
        # Provider-reported context floor (2026-08-19): the
        # caller may hand in a ``token_counter`` that prefers the live
        # number the provider observed over the chars/4 heuristic. It is
        # applied to EVERY SummarizationMiddleware site (main, general-
        # purpose subagent, custom subagents) so the trigger compares the
        # same number the provider is actually holding.
        if summarization_token_counter is not None:
            kwargs["token_counter"] = summarization_token_counter
        if summarization_hot_cache_ttl:
            kwargs["hot_cache_ttl_overrides"] = dict(summarization_hot_cache_ttl)
        return kwargs


    backend = backend if backend is not None else (lambda rt: StateBackend(rt))

    # ─────────────────────────────────────────────────────────────────────
    # Tool selection must survive delegation.
    #
    # A subagent is a SEPARATE graph with its OWN middleware stack and its
    # OWN FilesystemMiddleware — so a restriction installed on the main
    # agent does not reach it. Concretely: ``--tools '*,task'`` denies
    # every tool but ``task``, the model spawns a subagent, and that
    # subagent gets ``ls``/``read_file``/``write_file``/``execute`` back
    # from its own filesystem middleware and runs them. Verified before
    # this fix: the denied tool's body executed inside the subagent while
    # the main graph refused the identical call.
    #
    # That is a straight violation of "nothing beyond the tools named",
    # and it is the delegation path — the one a restricted agent is most
    # likely to reach for once its own tools are gone.
    #
    # The selection middleware is identified structurally (it declares
    # ``applies_to_subagents``) rather than by importing it: ``deepagents``
    # must not depend on the CLI layer, which sits above it. Any
    # middleware that opts in this way is propagated verbatim — the same
    # instance, so the selection and its logging stay singular.
    # ─────────────────────────────────────────────────────────────────────
    inherited_middleware: list[AgentMiddleware[Any, Any, Any]] = [
        mw for mw in (middleware or ())
        if getattr(mw, "applies_to_subagents", False)
    ]

    # ─────────────────────────────────────────────────────────────────────
    # The 2026-06-13 incident — SummarizationMiddleware unbound-model capture.
    #
    # LangChain's ``bind_tools`` contract returns a NEW model instance; the
    # original ``model`` parameter we received is left without any tool
    # metadata. ``SummarizationMiddleware`` stores ``self.model = model``
    # at construction time and calls ``self.model.ainvoke(prompt)`` from
    # ``_acreate_summary``. If we pass the raw, pre-bind reference here
    # the summary call goes through an instance with empty tools metadata
    # — and for providers that share a singleton client (e.g.
    # ``ClaudeCodeChatModel._client_manager``) that empty rebind silently
    # propagates ``tools_metadata=[]`` into the next session-rotation,
    # stripping every ``mcp__deepworker__*`` tool from the Claude CLI
    # subprocess. The symptom was ``[MCP_DIAG] proxy_tool_count`` going
    # 23 → 0 after the first summarization.
    #
    # Fix: compute the bound model ONCE here and hand it to every
    # ``SummarizationMiddleware`` site. The turn-node model passed to
    # ``create_agent`` is left untouched — LangGraph still calls
    # ``bind_tools`` on it internally; we only change WHAT the middleware
    # sees. See
    # a related internal incident analysis (not included here).
    # ─────────────────────────────────────────────────────────────────────
    summarization_model = _bind_tools_for_middleware(model, tools)
    if summarization_provider:
        setattr(model, "_deep_worker_provider", summarization_provider)
        setattr(summarization_model, "_deep_worker_provider", summarization_provider)

    # Build general-purpose subagent with default middleware stack
    # Heterogeneous middleware stack — entries specialize StateT differently
    # (TodoListMiddleware → PlanningState, others → AgentState). Python
    # generics are invariant, so `list[AgentMiddleware]` (which ty resolves
    # to `list[AgentMiddleware[AgentState[Any], ...]]`) rejects the
    # PlanningState variant. Use `list[AgentMiddleware[Any, Any, Any]]` —
    # the element contract is still AgentMiddleware, only the state/context
    # binding is erased at the container level.
    gp_middleware: list[AgentMiddleware[Any, Any, Any]] = [
        TodoListMiddleware(),
        FilesystemMiddleware(backend=backend),
        MessageTimestampMiddleware(),
        SummarizationMiddleware(
            model=summarization_model,
            backend=backend,
            trigger=summarization_defaults["trigger"],
            keep=summarization_defaults["keep"],
            trim_tokens_to_summarize=None,
            truncate_args_settings=summarization_defaults["truncate_args_settings"],
            **_summarization_extra_kwargs(),
        ),
        AnthropicPromptCachingMiddleware(unsupported_model_behavior="ignore"),
        PatchToolCallsMiddleware(),
    ]
    if skills is not None:
        gp_middleware.append(SkillsMiddleware(backend=backend, sources=skills))
    # Selection goes after the tool-CONTRIBUTING middleware (filesystem,
    # skills) so its ``wrap_model_call`` sees everything they added, and
    # AFTER HumanInTheLoop so that its cut runs first.
    #
    # The two hooks order OPPOSITELY, which is why the placement looks
    # backwards: ``wrap_tool_call`` wrappers nest outer-to-inner in list
    # order, but ``after_model`` GRAPH NODES are chained in REVERSE list
    # order (verified on the compiled graph: model → HITL.after_model →
    # ToolSelection.after_model). HITL does its work in such a node, so a
    # selection placed BEFORE it would have its cut run too late: the user
    # is asked to authorise a tool the agent is not allowed to have, and
    # only after they say yes does the barrier refuse it. Observed — one
    # interrupt raised for a denied `ls`. A forbidden call must be refused
    # outright, never negotiated.
    if interrupt_on is not None:
        gp_middleware.append(HumanInTheLoopMiddleware(interrupt_on=interrupt_on))
    gp_middleware.extend(inherited_middleware)

    general_purpose_spec: SubAgent = cast(SubAgent, {
        **GENERAL_PURPOSE_SUBAGENT,
        "model": model,
        "tools": tools or [],
        "middleware": gp_middleware,
    })

    # Process user-provided subagents to fill in defaults for model, tools, and middleware
    processed_subagents: list[SubAgent | CompiledSubAgent] = []
    for spec in subagents or []:
        if "runnable" in spec:
            # CompiledSubAgent — an already-compiled graph whose middleware
            # stack was fixed when the caller built it. A selection cannot
            # be pushed into it, and its tools are outside our reach.
            #
            # When a restriction is in force this is fail-CLOSED, not a
            # warning. Building the agent anyway would produce one that
            # ANNOUNCES a limit ("only these tools") while shipping a
            # documented hole in it — and the whole point of the feature is
            # a measurement you can trust. A log line does not fix that: it
            # lands in a file nobody reads while the experiment quietly
            # produces wrong numbers. Refusing to build is the only outcome
            # that cannot mislead.
            #
            # The CLI layer rejects compiled subagents earlier
            # (``_merge_common_tools_into_subagents``), so no deep-worker
            # agent reaches this branch; the guard is for the public API.
            if inherited_middleware:
                name = spec.get("name", "<unnamed>")
                raise ValueError(
                    f"compiled subagent {name!r} cannot be constrained by a "
                    f"tool selection: its graph was compiled elsewhere and "
                    f"its middleware stack can no longer be changed. "
                    f"Refusing to build an agent that claims a restriction "
                    f"it cannot enforce. Build {name!r} from a SubAgent "
                    f"spec (name/description/system_prompt) so the "
                    f"selection applies to it, or drop the tool selection."
                )
            processed_subagents.append(spec)
        else:
            # SubAgent - fill in defaults and prepend base middleware
            subagent_model = spec.get("model", model)
            if isinstance(subagent_model, str):
                subagent_model = init_chat_model(subagent_model)

            # Build middleware: base stack + skills (if specified) + user's middleware
            subagent_summarization_defaults = _compute_summarization_defaults(
                subagent_model, context_limit=context_limit
            )
            if summarization_keep is not None:
                subagent_summarization_defaults = _apply_keep_override(
                    subagent_summarization_defaults, summarization_keep
                )
            # Same incident 2026-06-13 fix as the gp_middleware site above:
            # hand the bind_tools-result to SummarizationMiddleware so the
            # summary call carries the subagent's tool inventory. Per-
            # subagent tools default to the parent's ``tools`` (see the
            # ``processed_spec`` below), so we bind against the same list
            # ``create_agent`` will use for the subagent's turn-node.
            subagent_tools_for_bind = spec.get("tools", tools or [])
            subagent_summarization_model = _bind_tools_for_middleware(
                subagent_model, subagent_tools_for_bind
            )
            # See explanation above the gp_middleware declaration.
            subagent_middleware: list[AgentMiddleware[Any, Any, Any]] = [
                TodoListMiddleware(),
                FilesystemMiddleware(backend=backend),
                MessageTimestampMiddleware(),
                SummarizationMiddleware(
                    model=subagent_summarization_model,
                    backend=backend,
                    trigger=subagent_summarization_defaults["trigger"],
                    keep=subagent_summarization_defaults["keep"],
                    trim_tokens_to_summarize=None,
                    truncate_args_settings=subagent_summarization_defaults["truncate_args_settings"],
                    **_summarization_extra_kwargs(),
                ),
                AnthropicPromptCachingMiddleware(unsupported_model_behavior="ignore"),
                PatchToolCallsMiddleware(),
            ]
            subagent_skills = spec.get("skills")
            if subagent_skills:
                subagent_middleware.append(SkillsMiddleware(backend=backend, sources=subagent_skills))
            # Last, for both reasons that decide the position (see the gp
            # stack): after any spec middleware that CONTRIBUTES tools, so
            # ``wrap_model_call`` sees them; and after a spec-provided
            # HumanInTheLoopMiddleware, so the selection's ``after_model``
            # node — chained in reverse list order — cuts denied calls
            # before HITL can raise an approval prompt for them.
            subagent_middleware.extend(spec.get("middleware", []))
            subagent_middleware.extend(inherited_middleware)

            processed_spec: SubAgent = cast(SubAgent, {
                **spec,
                "model": subagent_model,
                "tools": spec.get("tools", tools or []),
                "middleware": subagent_middleware,
            })
            processed_subagents.append(processed_spec)

    # Combine GP with processed user-provided subagents
    all_subagents: list[SubAgent | CompiledSubAgent] = [general_purpose_spec, *processed_subagents]

    # Build main agent middleware stack
    # See explanation above the gp_middleware declaration.
    deepagent_middleware: list[AgentMiddleware[Any, Any, Any]] = [
        # OUTERMOST tool-call wrapper (first = outermost): the single source
        # of truth for the per-tool-call timeout (POLICIES.md §7.1). Its bound
        # covers EVERY tool of EVERY provider and every inner tool-call
        # middleware. No other place may wait on a tool. See
        # docs/DESIGN_UNIFIED_BOUNDED_TOOL_WAIT.md.
        BoundedToolCallMiddleware(),
        TodoListMiddleware(),
    ]
    if memory is not None:
        deepagent_middleware.append(MemoryMiddleware(backend=backend, sources=memory))
    if skills is not None:
        deepagent_middleware.append(SkillsMiddleware(backend=backend, sources=skills))
    deepagent_middleware.extend(
        [
            FilesystemMiddleware(backend=backend),
            MessageTimestampMiddleware(),
            SubAgentMiddleware(
                backend=backend,
                subagents=all_subagents,
            ),
            SummarizationMiddleware(
                # The 2026-06-13 incident: use the bind_tools-result, NOT the
                # raw ``model``. See the gp_middleware site above and
                # a related internal incident analysis (not included here).
                model=summarization_model,
                backend=backend,
                trigger=summarization_defaults["trigger"],
                keep=summarization_defaults["keep"],
                trim_tokens_to_summarize=None,
                truncate_args_settings=summarization_defaults["truncate_args_settings"],
                **_summarization_extra_kwargs(),
            ),
            AnthropicPromptCachingMiddleware(unsupported_model_behavior="ignore"),
            PatchToolCallsMiddleware(),
        ]
    )
    # HITL first so that caller middleware ends up AFTER it: ``after_model``
    # nodes are chained in reverse list order, so a ToolSelectionMiddleware
    # passed in here must sit last for its cut to run before HITL can ask
    # the user to approve a tool the agent is not allowed to have.
    if interrupt_on is not None:
        deepagent_middleware.append(HumanInTheLoopMiddleware(interrupt_on=interrupt_on))
    if middleware:
        deepagent_middleware.extend(middleware)

    # Combine system_prompt with BASE_AGENT_PROMPT
    if system_prompt is None:
        final_system_prompt: str | SystemMessage = BASE_AGENT_PROMPT
    elif isinstance(system_prompt, SystemMessage):
        # SystemMessage: append BASE_AGENT_PROMPT to content_blocks
        new_content = [
            *system_prompt.content_blocks,
            {"type": "text", "text": f"\n\n{BASE_AGENT_PROMPT}"},
        ]
        final_system_prompt = SystemMessage(content_blocks=cast(Any, new_content))
    else:
        # String: simple concatenation
        final_system_prompt = system_prompt + "\n\n" + BASE_AGENT_PROMPT

    return create_agent(
        model,
        system_prompt=final_system_prompt,
        tools=tools,
        middleware=deepagent_middleware,
        response_format=response_format,
        context_schema=context_schema,
        checkpointer=checkpointer,
        store=store,
        debug=debug,
        name=name,
        cache=cache,
    )
    # NOTE: No recursion_limit. Agents must run without any step limits.
    # recursion_limit is STRICTLY FORBIDDEN — causes GraphRecursionError
    # on long-running tasks with many tool calls.
