"""
AutoGen Memory Adapter.

Provides persistent memory for Microsoft AutoGen agent conversations.
Each agent gets its own namespace for memory isolation.

Usage:
    from memory_layer.integrations import AutoGenMemory
    from memory_layer import MemoryManager

    brain = MemoryManager(db_path="my_brain.db")
    memory = AutoGenMemory(manager=brain, agent_name="coder")

    memory.add("User prefers Python for data analysis")
    context = memory.search("What language should I use?")
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..core import MemoryManager


class AutoGenMemory:
    """
    AutoGen-compatible memory backed by MemoryManager.

    Each agent instance gets isolated memory via namespacing.
    """

    def __init__(
        self,
        manager: MemoryManager,
        agent_name: str = "default",
        top_k: int = 5,
    ):
        self.manager = manager
        self.agent_name = agent_name
        self.namespace = f"autogen_{agent_name}"
        self.top_k = top_k

    def add(self, content: str, metadata: Dict[str, Any] = None) -> str:
        """Add a memory for this agent."""
        meta = metadata or {}
        meta["agent"] = self.agent_name
        memory = self.manager.remember(
            content=content,
            metadata=meta,
            namespace=self.namespace,
            tags=["autogen", self.agent_name],
        )
        return memory.id

    def search(self, query: str, top_k: int = None) -> List[Dict[str, Any]]:
        """Search agent memories."""
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
            }
            for r in results
        ]

    def get_context_for_message(self, message: str) -> str:
        """Get relevant context string for a new message."""
        results = self.search(message)
        if not results:
            return ""
        return "\n".join(f"- {r['content']}" for r in results)

    def save_conversation(
        self, messages: List[Dict[str, str]]
    ) -> List[str]:
        """Save a batch of conversation messages."""
        ids = []
        for msg in messages:
            mid = self.add(
                content=msg.get("content", ""),
                metadata={"role": msg.get("role", "user")},
            )
            ids.append(mid)
        return ids

    def clear(self) -> None:
        """Clear this agent's working memory."""
        self.manager.storage.clear_working_memory()
