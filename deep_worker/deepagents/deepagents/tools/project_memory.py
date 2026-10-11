from __future__ import annotations

from typing import Any

from langchain_core.tools import tool


@tool
async def project_memory(
    query: str,
    top_k: int = 5,
    repo_path: str | None = None,
    runtime: Any = None,
    working_dir: str | None = None,
) -> str:
    """Recall relevant knowledge from an indexed repository.

    Pass repo_path to query a repository other than the current working
    directory. top_k must be between 1 and 20.
    """
    from deep_worker_cli.semantic_memory import client as memory_client
    from deep_worker_cli.semantic_memory.service.proto import semantic_memory_pb2 as pb

    validation_error = memory_client.clamp_top_k(top_k)
    if validation_error is not None:
        return validation_error
    resolved_dir = memory_client.resolve_project_path(runtime, working_dir, repo_path)
    client = memory_client.SemanticMemoryClient()
    try:
        status = await client.index_status(pb.INDEX_KIND_PROJECT, resolved_dir)
        if not status.exists or not status.complete:
            tenants = await client.list_tenants(pb.INDEX_KIND_PROJECT)
            return memory_client.render_no_project_index(
                resolved_dir, tenants, explicit=repo_path is not None
            )
        return await client.search_project(query, resolved_dir, top_k)
    except Exception as exc:
        return f"Error: Project memory service failed: {exc}"
