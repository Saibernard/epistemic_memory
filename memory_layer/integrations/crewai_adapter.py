"""
CrewAI Memory Adapter.

Provides a memory interface compatible with CrewAI's agent memory system.
Agents can store and recall task-relevant memories.

Usage:
    from memory_layer.integrations import CrewAIMemory
    from memory_layer import MemoryManager

    brain = MemoryManager(db_path="my_brain.db")
    memory = CrewAIMemory(manager=brain)

    # Use memory.search() and memory.save() in CrewAI agents
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..core import MemoryManager


class CrewAIMemory:
    """
    CrewAI-compatible memory backed by MemoryManager.

    Provides search/save interface that CrewAI agents expect.
    """

    def __init__(
        self,
        manager: MemoryManager,
        namespace: str = "crewai",
        top_k: int = 5,
    ):
        self.manager = manager
        self.namespace = namespace
        self.top_k = top_k

    def search(self, query: str, top_k: int = None) -> List[Dict[str, Any]]:
        """Search memories relevant to the query."""
        k = top_k or self.top_k
        results = self.manager.recall(
            query=query,
            top_k=k,
            namespace=self.namespace,
        )
        return [
            {
                "content": r.memory.content,
                "score": r.relevance_score,
                "metadata": r.memory.metadata,
                "memory_type": r.memory.memory_type.value,
            }
            for r in results
        ]

    def save(
        self,
        content: str,
        metadata: Dict[str, Any] = None,
        agent_name: str = "",
    ) -> str:
        """Save a memory from agent execution."""
        meta = metadata or {}
        if agent_name:
            meta["agent"] = agent_name
        memory = self.manager.remember(
            content=content,
            metadata=meta,
            namespace=self.namespace,
            tags=["crewai", agent_name] if agent_name else ["crewai"],
        )
        return memory.id

    def save_task_result(
        self,
        task_description: str,
        result: str,
        agent_name: str = "",
    ) -> str:
        """Save a task result as a procedural memory."""
        from ..models import MemoryType
        memory = self.manager.remember(
            content=f"Task: {task_description}\nResult: {result}",
            memory_type=MemoryType.PROCEDURAL,
            metadata={"agent": agent_name, "task": task_description},
            namespace=self.namespace,
            tags=["crewai", "task_result"],
        )
        return memory.id

    def get_context(self, query: str, max_tokens: int = 2000) -> str:
        """Get formatted context string for agent prompt injection."""
        results = self.search(query)
        parts = []
        total = 0
        for r in results:
            text = r["content"]
            est = len(text) // 4
            if total + est > max_tokens:
                break
            parts.append(f"- {text}")
            total += est
        return "\n".join(parts) if parts else ""

    def clear(self) -> None:
        """Clear all CrewAI memories."""
        self.manager.storage.clear_working_memory()
