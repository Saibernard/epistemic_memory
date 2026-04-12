"""
OpenAI Assistants / Thread Memory Adapter.

Maps OpenAI's thread-based conversation model to MemoryManager.
Each thread ID becomes a namespace, giving persistent cross-session
memory to OpenAI Assistants.

Usage:
    from memory_layer.integrations import OpenAIThreadMemory
    from memory_layer import MemoryManager

    brain = MemoryManager(db_path="my_brain.db")
    memory = OpenAIThreadMemory(manager=brain)

    # Store from thread messages
    memory.save_message(thread_id="thread_abc", role="user", content="I like Python")

    # Retrieve context for a new message
    context = memory.get_context(thread_id="thread_abc", query="preferences")
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..core import MemoryManager


class OpenAIThreadMemory:
    """
    OpenAI Assistants-compatible memory backed by MemoryManager.

    Thread IDs map to namespaces for isolation.
    """

    def __init__(
        self,
        manager: MemoryManager,
        top_k: int = 5,
    ):
        self.manager = manager
        self.top_k = top_k

    def save_message(
        self,
        thread_id: str,
        role: str,
        content: str,
        metadata: Dict[str, Any] = None,
    ) -> str:
        """Save a message from an OpenAI thread."""
        meta = metadata or {}
        meta["role"] = role
        meta["thread_id"] = thread_id
        memory = self.manager.remember(
            content=content,
            metadata=meta,
            namespace=f"openai_{thread_id}",
            tags=["openai", role],
        )
        return memory.id

    def save_run_result(
        self,
        thread_id: str,
        assistant_message: str,
        tool_outputs: List[Dict] = None,
    ) -> str:
        """Save an assistant run result."""
        meta = {"role": "assistant", "thread_id": thread_id}
        if tool_outputs:
            meta["tool_outputs"] = tool_outputs
        memory = self.manager.remember(
            content=assistant_message,
            metadata=meta,
            namespace=f"openai_{thread_id}",
            tags=["openai", "assistant"],
        )
        return memory.id

    def get_context(
        self,
        thread_id: str,
        query: str,
        top_k: int = None,
    ) -> List[Dict[str, Any]]:
        """Retrieve relevant context for an OpenAI thread."""
        k = top_k or self.top_k
        results = self.manager.recall(
            query=query,
            top_k=k,
            namespace=f"openai_{thread_id}",
        )
        return [
            {
                "content": r.memory.content,
                "role": r.memory.metadata.get("role", "assistant"),
                "score": r.relevance_score,
            }
            for r in results
        ]

    def get_system_prompt_context(
        self,
        thread_id: str,
        query: str,
        max_tokens: int = 1500,
    ) -> str:
        """Get formatted context for injection into system prompt."""
        results = self.get_context(thread_id, query)
        parts = []
        total = 0
        for r in results:
            text = r["content"]
            est = len(text) // 4
            if total + est > max_tokens:
                break
            parts.append(f"[{r['role']}] {text}")
            total += est
        return "\n".join(parts)
