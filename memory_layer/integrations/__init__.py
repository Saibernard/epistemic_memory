"""
Agent Framework Integrations for the Memory Layer.

Thin adapters that let any major AI agent framework use the Memory Layer
as its persistent memory backend. Each adapter wraps MemoryManager with
the interface expected by the target framework.

Supported frameworks:
  - LangChain (BaseMemory / BaseChatMemory)
  - CrewAI (memory interface)
  - LlamaIndex (BaseMemory)
  - OpenAI Assistants (thread-based memory)
  - AutoGen (agent memory)
  - Vercel AI SDK (context provider)
"""

from .langchain_adapter import LangChainMemory
from .crewai_adapter import CrewAIMemory
from .llamaindex_adapter import LlamaIndexMemory
from .openai_adapter import OpenAIThreadMemory
from .autogen_adapter import AutoGenMemory
from .vercel_ai_adapter import VercelAIMemory

__all__ = [
    "LangChainMemory",
    "CrewAIMemory",
    "LlamaIndexMemory",
    "OpenAIThreadMemory",
    "AutoGenMemory",
    "VercelAIMemory",
]
