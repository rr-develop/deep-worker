from __future__ import annotations

from typing import Any

from langchain_core.tools import tool


@tool
async def history_memory(
    query: str,
    top_k: int = 5,
    agent_name: str | None = None,
    runtime: Any = None,
    working_dir: str | None = None,
) -> str:
    """Recall indexed conversation history.

    Pass agent_name to query another agent's memory. top_k must be between
    1 and 20.
    """
    from deep_worker_cli.semantic_memory import client as memory_client
    from deep_worker_cli.semantic_memory.service.proto import semantic_memory_pb2 as pb

    validation_error = memory_client.clamp_top_k(top_k)
    if validation_error is not None:
        return validation_error
    summaries_dir, agent_label = memory_client.resolve_history_target(
        runtime, agent_name
    )
    client = memory_client.SemanticMemoryClient()
    try:
        status = await client.index_status(pb.INDEX_KIND_HISTORY, summaries_dir)
        if not status.exists or not status.complete:
            tenants = await client.list_tenants(pb.INDEX_KIND_HISTORY)
            return memory_client.render_no_history_index(
                agent_label, tenants, explicit=agent_name is not None
            )
        return await client.search_history(query, summaries_dir, top_k)
    except Exception as exc:
        return f"Error: History memory service failed: {exc}"
