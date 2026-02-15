"""TaskManager module for deep-worker agent task evaluation and correction.

This module provides TaskManager functionality similar to ai_lib TaskManager,
but adapted to run within the same process as the deep-worker agent using
a separate thread.
"""

from .task_manager import TaskManager
from .middleware import TaskManagerCallback

__all__ = ["TaskManager", "TaskManagerCallback"]
