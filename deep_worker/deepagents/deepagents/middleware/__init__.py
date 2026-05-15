"""Middleware for the agent.

NOTE: ProjectMemoryMiddleware is intentionally NOT exported from this
__init__.py. It is imported lazily by agent.py where it is needed, and
pulling it here would trigger its heavy dependencies (llama_index, torch,
lancedb) on every agent creation regardless of whether a semantic index
exists.
"""

from deepagents.middleware.filesystem import FilesystemMiddleware
from deepagents.middleware.memory import MemoryMiddleware
from deepagents.middleware.skills import SkillsMiddleware
from deepagents.middleware.subagents import CompiledSubAgent, SubAgent, SubAgentMiddleware
from deepagents.middleware.summarization import SummarizationMiddleware

__all__ = [
    "CompiledSubAgent",
    "FilesystemMiddleware",
    "MemoryMiddleware",
    "SkillsMiddleware",
    "SubAgent",
    "SubAgentMiddleware",
    "SummarizationMiddleware",
]
