"""
Vercel AI SDK Memory Adapter.

Provides a context provider for Vercel AI SDK applications,
returning relevant memories as system message context.

Usage:
    from memory_layer.integrations import VercelAIMemory
    from memory_layer import MemoryManager

    brain = MemoryManager(db_path="my_brain.db")
    memory = VercelAIMemory(manager=brain)

    # Get context for AI SDK
    context = memory.get_context("user question here")

    # Store interaction
    memory.save_interaction(user="question", assistant="answer")
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..core import MemoryManager


class VercelAIMemory:
    """
    Vercel AI SDK-compatible memory provider.

    Returns context as structured messages suitable for injection
    into Vercel AI SDK's message array.
    """

    def __init__(
        self,
        manager: MemoryManager,
        namespace: str = "vercel_ai",
        top_k: int = 5,
    ):
        self.manager = manager
        self.namespace = namespace
        self.top_k = top_k

    def get_context(self, query: str, top_k: int = None) -> str:
        """Get relevant memory context as a formatted string."""
        k = top_k or self.top_k
        results = self.manager.recall(
            query=query,
            top_k=k,
            namespace=self.namespace,
        )
        if not results:
            return ""
        return "\n".join(f"- {r.memory.content}" for r in results)

    def get_messages(self, query: str) -> List[Dict[str, str]]:
        """Get memories formatted as AI SDK messages."""
        results = self.manager.recall(
            query=query,
            top_k=self.top_k,
            namespace=self.namespace,
        )
        return [
            {
                "role": r.memory.metadata.get("role", "system"),
                "content": r.memory.content,
            }
            for r in results
        ]

    def save_interaction(
        self, user: str, assistant: str
    ) -> None:
        """Save a user-assistant interaction."""
        self.manager.record_episode(
            user_message=user,
            assistant_response=assistant,
            namespace=self.namespace,
        )

    def save(self, content: str, role: str = "system") -> str:
        """Save a single memory."""
        memory = self.manager.remember(
            content=content,
            metadata={"role": role},
            namespace=self.namespace,
            tags=["vercel_ai", role],
        )
        return memory.id
