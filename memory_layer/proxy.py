"""
Memory Proxy — transparent middleware that injects memories into any LLM call.

The proxy sits between your application and any LLM provider (OpenAI, Gemini,
Anthropic, Ollama, or any OpenAI-compatible API). It intercepts the user's
message, retrieves relevant memories, injects them as context, forwards the
enriched request to the LLM, records the episode, and returns the response.

Usage:

    # As a drop-in replacement for OpenAI client:
    from memory_layer.proxy import MemoryProxy

    proxy = MemoryProxy(db_path="~/.memory-layer/memory.db")
    response = proxy.chat("What language do I prefer?", provider="openai")

    # As a FastAPI server (any language can call it):
    python -m memory_layer.proxy --port 8585

    # OpenAI-compatible endpoint:
    curl http://localhost:8585/v1/chat/completions -d '{
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": "What do I prefer?"}]
    }'
"""

import os
import time
import json
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

from .core import MemoryManager
from .models import MemoryType


# ─────────────────────────────────────────────
# PROVIDER ADAPTERS
# ─────────────────────────────────────────────

def _call_openai(messages: list, model: str, api_key: str, base_url: str = None, **kw) -> dict:
    from openai import OpenAI
    client = OpenAI(api_key=api_key, base_url=base_url)
    resp = client.chat.completions.create(model=model, messages=messages, **kw)
    return {
        "content": resp.choices[0].message.content,
        "model": resp.model,
        "usage": {
            "prompt_tokens": resp.usage.prompt_tokens if resp.usage else 0,
            "completion_tokens": resp.usage.completion_tokens if resp.usage else 0,
            "total_tokens": resp.usage.total_tokens if resp.usage else 0,
        },
        "finish_reason": resp.choices[0].finish_reason,
    }


def _call_gemini(messages: list, model: str, api_key: str, **kw) -> dict:
    from google import genai
    client = genai.Client(api_key=api_key)
    system_parts = [m["content"] for m in messages if m["role"] == "system"]
    user_parts = [m["content"] for m in messages if m["role"] != "system"]
    full_prompt = "\n\n".join(system_parts + user_parts)
    resp = client.models.generate_content(model=model, contents=full_prompt)
    text = resp.text if hasattr(resp, "text") else str(resp)
    return {
        "content": text,
        "model": model,
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        "finish_reason": "stop",
    }


def _call_anthropic(messages: list, model: str, api_key: str, **kw) -> dict:
    import anthropic
    client = anthropic.Anthropic(api_key=api_key)
    system_msg = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
    non_system = [m for m in messages if m["role"] != "system"]
    resp = client.messages.create(
        model=model,
        system=system_msg or "You are a helpful assistant.",
        messages=non_system,
        max_tokens=kw.get("max_tokens", 1024),
    )
    text = resp.content[0].text if resp.content else ""
    return {
        "content": text,
        "model": resp.model,
        "usage": {
            "prompt_tokens": resp.usage.input_tokens if resp.usage else 0,
            "completion_tokens": resp.usage.output_tokens if resp.usage else 0,
            "total_tokens": (resp.usage.input_tokens + resp.usage.output_tokens) if resp.usage else 0,
        },
        "finish_reason": resp.stop_reason or "stop",
    }


PROVIDERS = {
    "openai": {"call": _call_openai, "default_model": "gpt-4o-mini", "key_env": "OPENAI_API_KEY"},
    "gemini": {"call": _call_gemini, "default_model": "gemini-2.5-flash", "key_env": "GOOGLE_API_KEY"},
    "anthropic": {"call": _call_anthropic, "default_model": "claude-sonnet-4-20250514", "key_env": "ANTHROPIC_API_KEY"},
    "ollama": {"call": _call_openai, "default_model": "llama3", "key_env": None},
}


# ─────────────────────────────────────────────
# MEMORY CONTEXT BUILDER
# ─────────────────────────────────────────────

_MEMORY_SYSTEM_PROMPT = """You have access to a persistent memory system. Relevant memories from past interactions are provided below. Use them naturally — reference prior context when relevant, and if the user's question relates to something you remember, use that information.

If the memories are not relevant to the current question, just answer normally and ignore them.

--- MEMORIES ---
{memories}
--- END MEMORIES ---"""

_MEMORY_SYSTEM_PROMPT_EMPTY = """You are a helpful assistant. You have a persistent memory system, but no relevant memories were found for this question. Answer to the best of your ability."""


# ─────────────────────────────────────────────
# PROXY RESPONSE
# ─────────────────────────────────────────────

@dataclass
class ProxyResponse:
    """Response from the memory proxy."""
    content: str
    provider: str
    model: str
    memories_used: int
    memories_context: str
    retrieval_ms: float
    llm_ms: float
    total_ms: float
    usage: dict = field(default_factory=dict)
    memory_ids: List[str] = field(default_factory=list)
    auto_stored: bool = False


# ─────────────────────────────────────────────
# MEMORY PROXY
# ─────────────────────────────────────────────

class MemoryProxy:
    """
    Transparent memory middleware for any LLM.

    Intercepts messages, retrieves relevant memories, injects context,
    forwards to the LLM provider, and optionally records the episode.

    Args:
        brain: Existing MemoryManager instance (or pass db_path to create one).
        db_path: Path to SQLite database (creates MemoryManager automatically).
        embedding_mode: Embedding backend ('local', 'openai', 'gemini').
        namespace: Default namespace for memory operations.
        top_k: Max memories to retrieve per query.
        token_budget: Max tokens for memory context injection.
        auto_record: Automatically record episodes after each exchange.
        auto_extract: Use LLM to extract facts from conversations (requires OpenAI key).
        min_relevance: Minimum relevance score for memory inclusion.
    """

    def __init__(
        self,
        brain: MemoryManager = None,
        db_path: str = None,
        embedding_mode: str = None,
        embedding_model: str = None,
        namespace: str = "default",
        top_k: int = 8,
        token_budget: int = 4000,
        auto_record: bool = True,
        auto_extract: bool = False,
        min_relevance: float = 0.05,
    ):
        if brain:
            self.brain = brain
        else:
            from .config import get_db_path, load_config, ensure_home_dir
            try:
                ensure_home_dir()
                cfg = load_config()
                _db = db_path or get_db_path()
                _mode = embedding_mode or cfg.get("embeddings", "mode")
                _model = embedding_model or cfg.get("embeddings", "model")
                _extract = auto_extract or cfg.getboolean("llm", "extract")
            except Exception:
                _db = db_path or os.path.expanduser("~/.memory-layer/memory.db")
                _mode = embedding_mode or "local"
                _model = embedding_model or "all-mpnet-base-v2"
                _extract = auto_extract

            from .storage_factory import create_storage
            storage = create_storage("sqlite", sqlite_path=_db)
            self.brain = MemoryManager(
                db_path=_db,
                embedding_mode=_mode,
                embedding_model=_model,
                llm_extract=_extract,
                storage=storage,
            )

        self.namespace = namespace
        self.top_k = top_k
        self.token_budget = token_budget
        self.auto_record = auto_record
        self.auto_extract = auto_extract
        self.min_relevance = min_relevance

    def chat(
        self,
        message: str,
        provider: str = "openai",
        model: str = None,
        api_key: str = None,
        namespace: str = None,
        top_k: int = None,
        system_prompt: str = None,
        extra_messages: List[dict] = None,
        **llm_kwargs,
    ) -> ProxyResponse:
        """
        Send a message through the memory proxy.

        1. Retrieves relevant memories for the message
        2. Builds enriched system prompt with memory context
        3. Forwards to the specified LLM provider
        4. Optionally records the episode
        5. Returns response with metadata

        Args:
            message: The user's message.
            provider: LLM provider ('openai', 'gemini', 'anthropic', 'ollama').
            model: Model name (uses provider default if not specified).
            api_key: API key (reads from env if not provided).
            namespace: Override default namespace.
            top_k: Override default top_k.
            system_prompt: Custom system prompt (memories are appended).
            extra_messages: Additional conversation history messages.
            **llm_kwargs: Extra kwargs passed to the LLM call.
        """
        t_start = time.time()
        ns = namespace or self.namespace
        k = top_k or self.top_k

        # Step 1: Retrieve memories
        t_retrieve = time.time()
        results = self.brain.recall(
            query=message,
            top_k=k,
            min_strength=0.0,
            min_confidence=self.min_relevance,
            namespace=ns,
        )
        retrieval_ms = (time.time() - t_retrieve) * 1000

        relevant = [r for r in results if r.relevance_score >= self.min_relevance]
        memory_ids = [r.memory.id for r in relevant]

        # Step 2: Build memory context
        if relevant:
            context = self.brain.format_for_llm(
                message, top_k=k, token_budget=self.token_budget, namespace=ns,
            )
            mem_system = _MEMORY_SYSTEM_PROMPT.format(memories=context)
        else:
            context = ""
            mem_system = _MEMORY_SYSTEM_PROMPT_EMPTY

        if system_prompt:
            final_system = system_prompt + "\n\n" + mem_system
        else:
            final_system = mem_system

        # Step 3: Build messages
        messages = [{"role": "system", "content": final_system}]
        if extra_messages:
            messages.extend(extra_messages)
        messages.append({"role": "user", "content": message})

        # Step 4: Call LLM
        prov_config = PROVIDERS.get(provider)
        if not prov_config:
            if provider.startswith("http"):
                prov_config = {"call": _call_openai, "default_model": "gpt-4o-mini", "key_env": "OPENAI_API_KEY"}
                llm_kwargs["base_url"] = provider
            else:
                raise ValueError(
                    f"Unknown provider: {provider}. "
                    f"Options: {', '.join(PROVIDERS.keys())} or a base URL."
                )

        _model = model or prov_config["default_model"]
        _key = api_key
        if not _key and prov_config.get("key_env"):
            _key = os.environ.get(prov_config["key_env"], "")
        if not _key and provider == "ollama":
            _key = "ollama"
            llm_kwargs.setdefault("base_url", "http://localhost:11434/v1")

        t_llm = time.time()
        llm_result = prov_config["call"](
            messages=messages, model=_model, api_key=_key, **llm_kwargs,
        )
        llm_ms = (time.time() - t_llm) * 1000

        # Step 5: Record episode
        auto_stored = False
        if self.auto_record:
            try:
                self.brain.record_episode(
                    user_message=message,
                    assistant_response=llm_result["content"],
                    namespace=ns,
                )
                auto_stored = True
            except Exception:
                pass

        total_ms = (time.time() - t_start) * 1000

        return ProxyResponse(
            content=llm_result["content"],
            provider=provider,
            model=llm_result.get("model", _model),
            memories_used=len(relevant),
            memories_context=context,
            retrieval_ms=round(retrieval_ms, 1),
            llm_ms=round(llm_ms, 1),
            total_ms=round(total_ms, 1),
            usage=llm_result.get("usage", {}),
            memory_ids=memory_ids,
            auto_stored=auto_stored,
        )

    def remember(self, content: str, importance: float = 0.7, tags: list = None, namespace: str = None):
        """Directly store a memory."""
        return self.brain.remember(
            content=content,
            memory_type=MemoryType.SEMANTIC,
            importance=importance,
            tags=tags or [],
            namespace=namespace or self.namespace,
        )

    def recall(self, query: str, top_k: int = None, namespace: str = None):
        """Directly recall memories."""
        return self.brain.recall(
            query=query,
            top_k=top_k or self.top_k,
            namespace=namespace or self.namespace,
        )

    def get_stats(self, namespace: str = None):
        """Get memory statistics."""
        return self.brain.get_stats(namespace=namespace)


# ─────────────────────────────────────────────
# OPENAI-COMPATIBLE PROXY SERVER
# ─────────────────────────────────────────────

def create_proxy_app(brain: MemoryManager = None, **proxy_kwargs):
    """
    Create a FastAPI app that serves an OpenAI-compatible API with memory injection.

    Any OpenAI SDK client can point to this server and get automatic memory.
    """
    from fastapi import FastAPI, Request
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel

    proxy = MemoryProxy(brain=brain, **proxy_kwargs)
    app = FastAPI(
        title="Memory Proxy",
        description="OpenAI-compatible API with automatic memory injection",
        version="1.0.0",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
    )

    class ChatCompletionRequest(BaseModel):
        model: str = "gpt-4o-mini"
        messages: list
        temperature: float = 0.7
        max_tokens: int = 1024
        provider: str = "openai"
        namespace: str = "default"
        memory_enabled: bool = True
        top_k: int = 8

    @app.post("/v1/chat/completions")
    async def chat_completions(req: ChatCompletionRequest):
        user_msgs = [m for m in req.messages if m.get("role") == "user"]
        if not user_msgs:
            return {"error": "No user message found"}

        last_user_msg = user_msgs[-1]["content"]
        system_msgs = [m for m in req.messages if m.get("role") == "system"]
        custom_system = system_msgs[0]["content"] if system_msgs else None

        history = [
            m for m in req.messages
            if m.get("role") in ("user", "assistant") and m != user_msgs[-1]
        ]

        if req.memory_enabled:
            result = proxy.chat(
                message=last_user_msg,
                provider=req.provider,
                model=req.model,
                namespace=req.namespace,
                top_k=req.top_k,
                system_prompt=custom_system,
                extra_messages=history if history else None,
                temperature=req.temperature,
                max_tokens=req.max_tokens,
            )
            return {
                "id": f"chatcmpl-mem-{int(time.time())}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": result.model,
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": result.content},
                    "finish_reason": "stop",
                }],
                "usage": result.usage,
                "memory_metadata": {
                    "memories_used": result.memories_used,
                    "memory_ids": result.memory_ids,
                    "retrieval_ms": result.retrieval_ms,
                    "auto_stored": result.auto_stored,
                },
            }
        else:
            prov = PROVIDERS.get(req.provider, PROVIDERS["openai"])
            _key = os.environ.get(prov["key_env"] or "", "")
            llm_result = prov["call"](
                messages=req.messages, model=req.model, api_key=_key,
                temperature=req.temperature, max_tokens=req.max_tokens,
            )
            return {
                "id": f"chatcmpl-{int(time.time())}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": llm_result.get("model", req.model),
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": llm_result["content"]},
                    "finish_reason": llm_result.get("finish_reason", "stop"),
                }],
                "usage": llm_result.get("usage", {}),
            }

    @app.post("/v1/memory/remember")
    async def api_remember(request: Request):
        body = await request.json()
        mem = proxy.remember(
            content=body["content"],
            importance=body.get("importance", 0.7),
            tags=body.get("tags", []),
            namespace=body.get("namespace", "default"),
        )
        return {"id": mem.id, "content": mem.content, "status": "stored"}

    @app.post("/v1/memory/recall")
    async def api_recall(request: Request):
        body = await request.json()
        results = proxy.recall(
            query=body["query"],
            top_k=body.get("top_k", 8),
            namespace=body.get("namespace", "default"),
        )
        return {
            "results": [
                {
                    "id": r.memory.id,
                    "content": r.memory.content,
                    "relevance": round(r.relevance_score, 4),
                    "type": r.memory.memory_type.value,
                    "tags": r.memory.tags,
                }
                for r in results
            ]
        }

    @app.get("/v1/memory/stats")
    async def api_stats(namespace: str = None):
        stats = proxy.get_stats(namespace=namespace)
        return {
            "total": stats.total_memories,
            "episodic": stats.episodic_count,
            "semantic": stats.semantic_count,
            "procedural": stats.procedural_count,
            "links": stats.total_links,
            "avg_strength": round(stats.avg_strength, 3),
        }

    @app.get("/health")
    async def health():
        return {"status": "healthy", "service": "memory-proxy", "version": "1.0.0"}

    return app


# ─────────────────────────────────────────────
# CLI ENTRY POINT
# ─────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Memory Proxy — LLM middleware with persistent memory")
    parser.add_argument("--port", type=int, default=8585)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--db", default=None)
    parser.add_argument("--embedding-mode", default=None)
    parser.add_argument("--namespace", default="default")
    args = parser.parse_args()

    print()
    print("  ╔══════════════════════════════════════════════╗")
    print("  ║           MEMORY PROXY FOR AI                ║")
    print("  ╠══════════════════════════════════════════════╣")
    print(f"  ║  Endpoint : http://{args.host}:{args.port}/v1   ║")
    print(f"  ║  Docs     : http://{args.host}:{args.port}/docs ║")
    print("  ║                                              ║")
    print("  ║  Drop-in replacement for OpenAI API.         ║")
    print("  ║  All calls get automatic memory injection.   ║")
    print("  ╚══════════════════════════════════════════════╝")
    print()

    app = create_proxy_app(
        db_path=args.db,
        embedding_mode=args.embedding_mode,
        namespace=args.namespace,
    )

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
