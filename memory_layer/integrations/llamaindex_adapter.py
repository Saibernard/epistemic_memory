"""
LlamaIndex Memory Adapter.

Wraps MemoryManager to work as a LlamaIndex memory module,
integrating with chat engines and agent pipelines.

Usage:
    from memory_layer.integrations import LlamaIndexMemory
    from memory_layer import MemoryManager

    brain = MemoryManager(db_path="my_brain.db")
    memory = LlamaIndexMemory(manager=brain)

    # Use with LlamaIndex chat engine
    context = memory.get(query="user preferences")
    memory.put({"role": "user", "content": "I prefer Python"})
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..core import MemoryManager


class LlamaIndexMemory:
    """
    LlamaIndex-compatible memory backed by MemoryManager.

    Implements get/put/get_all/reset interface.
    """

    def __init__(
        self,
        manager: MemoryManager,
        namespace: str = "llamaindex",
        top_k: int = 5,
    ):
        self.manager = manager
        self.namespace = namespace
        self.top_k = top_k

    def get(self, query: str, top_k: int = None) -> List[Dict[str, Any]]:
        """Retrieve relevant memories for a query."""
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
                "id": r.memory.id,
            }
            for r in results
        ]

    def get_all(self) -> List[Dict[str, Any]]:
        """Retrieve all memories in the namespace."""
        memories = self.manager.storage.get_all_memories(
            namespace=self.namespace, active_only=True,
        )
        return [
            {
                "content": m.content,
                "metadata": m.metadata,
                "id": m.id,
                "memory_type": m.memory_type.value,
            }
            for m in memories
        ]

    def put(self, message: Dict[str, Any]) -> str:
        """Store a chat message as memory."""
        content = message.get("content", "")
        role = message.get("role", "user")

        if role == "user":
            self.manager.add_to_working_memory(content, role="user")

        memory = self.manager.remember(
            content=content,
            metadata={"role": role},
            namespace=self.namespace,
            tags=["llamaindex", role],
        )
        return memory.id

    def put_interaction(
        self, user_message: str, assistant_response: str
    ) -> None:
        """Store a complete interaction."""
        self.manager.record_episode(
            user_message=user_message,
            assistant_response=assistant_response,
            namespace=self.namespace,
        )

    def reset(self) -> None:
        """Reset working memory."""
        self.manager.storage.clear_working_memory()

    def to_string(self, query: str = "") -> str:
        """Get memories as a formatted string for prompt injection."""
        if query:
            results = self.get(query)
            return "\n".join(f"- {r['content']}" for r in results)
        memories = self.get_all()
        return "\n".join(f"- {m['content']}" for m in memories[:20])
