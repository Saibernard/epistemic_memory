"""
LangChain Memory Adapter.

Wraps MemoryManager to implement LangChain's BaseMemory interface,
making it a drop-in replacement for ConversationBufferMemory or
ConversationSummaryMemory.

Usage:
    from memory_layer.integrations import LangChainMemory
    from memory_layer import MemoryManager

    brain = MemoryManager(db_path="my_brain.db")
    memory = LangChainMemory(manager=brain)

    # Use with any LangChain chain
    chain = ConversationChain(llm=llm, memory=memory)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..core import MemoryManager


class LangChainMemory:
    """
    LangChain-compatible memory backed by MemoryManager.

    Implements the essential BaseMemory interface:
    - memory_variables: list of keys injected into the prompt
    - load_memory_variables: retrieve relevant memories for context
    - save_context: store interaction after chain completion
    - clear: reset memory state
    """

    memory_key: str = "history"
    input_key: str = "input"
    output_key: str = "output"
    return_messages: bool = False

    def __init__(
        self,
        manager: MemoryManager,
        memory_key: str = "history",
        top_k: int = 5,
        namespace: str = "langchain",
    ):
        self.manager = manager
        self.memory_key = memory_key
        self.top_k = top_k
        self.namespace = namespace

    @property
    def memory_variables(self) -> List[str]:
        return [self.memory_key]

    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Recall relevant memories based on the input query."""
        query = inputs.get(self.input_key, "")
        if not query:
            return {self.memory_key: "" if not self.return_messages else []}

        results = self.manager.recall(
            query=query,
            top_k=self.top_k,
            namespace=self.namespace,
        )

        if self.return_messages:
            messages = []
            for r in results:
                meta = r.memory.metadata or {}
                role = meta.get("role", "assistant")
                messages.append({"role": role, "content": r.memory.content})
            return {self.memory_key: messages}

        context = "\n".join(
            f"- {r.memory.content}" for r in results
        )
        return {self.memory_key: context}

    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        """Store the interaction as episodic memory."""
        user_input = inputs.get(self.input_key, "")
        ai_output = outputs.get(self.output_key, "")

        if user_input:
            self.manager.record_episode(
                user_message=user_input,
                assistant_response=ai_output,
                namespace=self.namespace,
            )

    def clear(self) -> None:
        """Clear working memory."""
        self.manager.storage.clear_working_memory()
