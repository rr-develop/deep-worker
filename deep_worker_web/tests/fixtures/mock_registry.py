"""Mock ProcessRegistry for testing."""

from datetime import datetime
from pathlib import Path
from typing import Any


class MockProcessRegistry:
    """Mock ProcessRegistry for testing without real agents."""

    def __init__(self) -> None:
        """Initialize mock registry."""
        self._agents: dict[str, dict[str, Any]] = {}

    def register(self, agent_info: Any) -> None:
        """Register an agent."""
        self._agents[agent_info.name] = {
            "pid": agent_info.pid,
            "pid_file": str(agent_info.pid_file),
            "working_dir": agent_info.working_dir,
            "started_at": agent_info.started_at.isoformat(),
            "rpyc_port": agent_info.rpyc_port,
            "last_activity": datetime.now().isoformat(),
        }

    def unregister(self, agent_name: str) -> None:
        """Unregister an agent."""
        if agent_name in self._agents:
            del self._agents[agent_name]

    def lookup(self, agent_name: str) -> Any:
        """Look up an agent."""
        if agent_name not in self._agents:
            return None

        from deep_worker_cli.types import AgentProcessInfo

        entry = self._agents[agent_name]
        return AgentProcessInfo(
            name=agent_name,
            pid=entry["pid"],
            pid_file=Path(entry["pid_file"]),
            working_dir=entry["working_dir"],
            started_at=datetime.fromisoformat(entry["started_at"]),
            rpyc_port=entry["rpyc_port"],
            last_activity=datetime.fromisoformat(entry["last_activity"])
            if "last_activity" in entry
            else None,
        )

    def list_all(self) -> list[Any]:
        """List all agents."""
        from deep_worker_cli.types import AgentProcessInfo

        agents = []
        for name, entry in self._agents.items():
            agents.append(
                AgentProcessInfo(
                    name=name,
                    pid=entry["pid"],
                    pid_file=Path(entry["pid_file"]),
                    working_dir=entry["working_dir"],
                    started_at=datetime.fromisoformat(entry["started_at"]),
                    rpyc_port=entry["rpyc_port"],
                    last_activity=datetime.fromisoformat(entry["last_activity"])
                    if "last_activity" in entry
                    else None,
                )
            )
        return agents

    def add_mock_agent(
        self,
        name: str,
        pid: int,
        rpyc_port: int,
        working_dir: str = "/tmp/test",
    ) -> None:
        """Add a mock agent for testing."""
        from deep_worker_cli.types import AgentProcessInfo

        agent_info = AgentProcessInfo(
            name=name,
            pid=pid,
            pid_file=Path(f"/tmp/{name}.pid"),
            working_dir=working_dir,
            started_at=datetime.now(),
            rpyc_port=rpyc_port,
        )
        self.register(agent_info)
