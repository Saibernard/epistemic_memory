"""
Chat Engine — answer questions from the memory graph.

Three modes:
    - local:   Pure retrieval, no LLM. Returns ranked memory excerpts.
    - llm:     Retrieves memories, then uses OpenAI to synthesize a
               natural language answer. Requires OPENAI_API_KEY.
    - ollama:  Same as llm but routes through the enrichment pipeline
               (Ollama / any local LLM). No API key needed.

Usage:
    engine = ChatEngine(brain, mode="local")
    response = engine.ask("What tech stack does the project use?")
    print(response.answer)
    print(response.sources)
"""

import re
from dataclasses import dataclass, field
from typing import List, Optional

from .models import RecallResult


# Relevance threshold — below this, we consider the memory graph
# doesn't have the answer and refuse to respond.
MIN_RELEVANCE_FOR_ANSWER = 0.08
MIN_SOURCES_FOR_LLM = 1

_SYSTEM_PROMPT = """You are a knowledge assistant that answers questions ONLY using the provided memory excerpts.

STRICT RULES:
1. ONLY use information from the provided memory excerpts to answer.
2. If the excerpts don't contain the answer, say exactly: "I don't have that information in my knowledge base."
3. Do NOT use your own knowledge. Do NOT guess. Do NOT infer beyond what's explicitly stated.
4. Keep answers concise and factual.
5. After your answer, list the source memory IDs you used in [Sources: ...] format.
6. If a question is partially answerable, answer what you can and state what you don't know.
"""


@dataclass
class ChatResponse:
    """Response from the chat engine."""
    answer: str
    sources: List[dict] = field(default_factory=list)
    mode: str = "local"
    has_answer: bool = True
    relevance_scores: List[float] = field(default_factory=list)


class ChatEngine:
    """
    Answers questions from a memory graph.

    Args:
        brain: A MemoryManager instance (loaded with memories).
        mode: "local" (no LLM), "llm" (OpenAI), or "ollama" (local LLM).
        llm_model: Model name for LLM mode (default: gpt-4o-mini).
        api_key: OpenAI API key (reads OPENAI_API_KEY env var if not set).
        min_relevance: Minimum relevance score to consider a memory useful.
        top_k: Max memories to retrieve per question.
        namespace: Namespace to search in.
    """

    def __init__(
        self,
        brain,
        mode: str = "local",
        llm_model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        min_relevance: float = MIN_RELEVANCE_FOR_ANSWER,
        top_k: int = 8,
        namespace: str = "default",
    ):
        self.brain = brain
        self.mode = mode
        self.llm_model = llm_model
        self.min_relevance = min_relevance
        self.top_k = top_k
        self.namespace = namespace

        self._client = None
        self._enrichment = None

        if mode == "llm":
            import os
            api_key = api_key or os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise ValueError(
                    "LLM mode requires an OpenAI API key.\n"
                    "Set OPENAI_API_KEY or pass api_key= to ChatEngine."
                )
            try:
                from openai import OpenAI
                self._client = OpenAI(api_key=api_key)
            except ImportError:
                raise ImportError(
                    "LLM mode requires openai package.\n"
                    "Install it: pip install 'memory-layer[openai]'"
                )
        elif mode == "ollama":
            if hasattr(brain, "enrichment") and brain.enrichment and brain.enrichment.has_llm:
                self._enrichment = brain.enrichment
            else:
                raise ValueError(
                    "Ollama chat mode requires an active enrichment LLM.\n"
                    "Set enrichment_backend='ollama' on MemoryManager."
                )

    def ask(self, question: str) -> ChatResponse:
        """
        Ask a question against the memory graph.

        Returns a ChatResponse with the answer and source memories.
        """
        results = self.brain.recall(
            query=question,
            top_k=self.top_k,
            min_strength=0.0,
            min_confidence=0.0,
            namespace=self.namespace,
        )

        relevant = [
            r for r in results
            if r.relevance_score >= self.min_relevance
        ]

        sources = [
            {
                "id": r.memory.id[:8],
                "content": r.memory.content,
                "relevance": round(r.relevance_score, 3),
                "tags": r.memory.tags,
                "type": r.memory.memory_type.value,
            }
            for r in relevant
        ]
        scores = [r.relevance_score for r in relevant]

        if not relevant:
            return ChatResponse(
                answer="I don't have that information in my knowledge base.",
                sources=[],
                mode=self.mode,
                has_answer=False,
                relevance_scores=[],
            )

        if self.mode == "llm" and self._client:
            return self._ask_llm(question, relevant, sources, scores)
        elif self.mode == "ollama" and self._enrichment:
            return self._ask_ollama(question, relevant, sources, scores)
        else:
            return self._ask_local(question, relevant, sources, scores)

    def _ask_local(
        self,
        question: str,
        results: List[RecallResult],
        sources: List[dict],
        scores: List[float],
    ) -> ChatResponse:
        """Pure retrieval mode — format memory excerpts as the answer."""
        lines = []
        for i, r in enumerate(results):
            mem = r.memory
            tags_str = f"  [{', '.join(mem.tags)}]" if mem.tags else ""
            lines.append(
                f"{i+1}. {mem.content}{tags_str}\n"
                f"   (relevance: {r.relevance_score:.3f}, "
                f"type: {mem.memory_type.value}, "
                f"id: {mem.id[:8]})"
            )

        answer = "\n\n".join(lines)

        return ChatResponse(
            answer=answer,
            sources=sources,
            mode="local",
            has_answer=True,
            relevance_scores=scores,
        )

    def _ask_llm(
        self,
        question: str,
        results: List[RecallResult],
        sources: List[dict],
        scores: List[float],
    ) -> ChatResponse:
        """LLM mode — synthesize a grounded answer from memory excerpts."""
        excerpts = []
        for r in results:
            mem = r.memory
            tags_str = f" [tags: {', '.join(mem.tags)}]" if mem.tags else ""
            excerpts.append(
                f"[Memory {mem.id[:8]}] (relevance: {r.relevance_score:.3f}){tags_str}\n"
                f"{mem.content}"
            )

        context = "\n\n".join(excerpts)

        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Memory excerpts:\n\n{context}\n\n"
                    f"---\n\n"
                    f"Question: {question}"
                ),
            },
        ]

        try:
            response = self._client.chat.completions.create(
                model=self.llm_model,
                messages=messages,
                temperature=0.1,
                max_tokens=500,
            )
            raw_answer = response.choices[0].message.content.strip()
        except Exception:
            return self._ask_local(question, results, sources, scores)

        no_info_phrases = [
            "i don't have that information",
            "i don't have information",
            "not in my knowledge base",
            "no information available",
            "i cannot find",
        ]
        has_answer = not any(p in raw_answer.lower() for p in no_info_phrases)

        if has_answer:
            has_answer = self._verify_grounding(raw_answer, results)
            if not has_answer:
                raw_answer = "I don't have that information in my knowledge base."

        return ChatResponse(
            answer=raw_answer,
            sources=sources,
            mode="llm",
            has_answer=has_answer,
            relevance_scores=scores,
        )

    def _ask_ollama(
        self,
        question: str,
        results: List[RecallResult],
        sources: List[dict],
        scores: List[float],
    ) -> ChatResponse:
        """Ollama mode — synthesize answer via local LLM."""
        excerpts = []
        for r in results:
            mem = r.memory
            tags_str = f" [tags: {', '.join(mem.tags)}]" if mem.tags else ""
            excerpts.append(
                f"[Memory {mem.id[:8]}]{tags_str}\n{mem.content}"
            )

        context = "\n\n".join(excerpts)
        prompt = (
            f"{_SYSTEM_PROMPT}\n\n"
            f"Memory excerpts:\n\n{context}\n\n---\n\n"
            f"Question: {question}\n\nAnswer:"
        )

        raw_answer = self._enrichment.generate(prompt, max_tokens=500)

        if not raw_answer or len(raw_answer.strip()) < 5:
            return self._ask_local(question, results, sources, scores)

        raw_answer = raw_answer.strip()

        no_info_phrases = [
            "i don't have that information",
            "i don't have information",
            "not in my knowledge base",
            "no information available",
            "i cannot find",
        ]
        has_answer = not any(p in raw_answer.lower() for p in no_info_phrases)

        if has_answer:
            has_answer = self._verify_grounding(raw_answer, results)
            if not has_answer:
                raw_answer = "I don't have that information in my knowledge base."

        return ChatResponse(
            answer=raw_answer,
            sources=sources,
            mode="ollama",
            has_answer=has_answer,
            relevance_scores=scores,
        )

    def _verify_grounding(
        self, answer: str, results: List[RecallResult]
    ) -> bool:
        """
        Verify the LLM answer is actually grounded in the retrieved memories.

        Checks that at least some key terms from the answer appear in the
        source memories. If the answer introduces completely novel content,
        it's likely hallucinated.
        """
        answer_tokens = set(re.findall(r"[a-z0-9]+", answer.lower()))
        stopwords = {
            "the", "a", "an", "is", "are", "was", "were", "be", "been",
            "has", "have", "had", "do", "does", "did", "will", "would",
            "can", "could", "may", "might", "shall", "should", "must",
            "to", "of", "in", "for", "on", "with", "at", "by", "from",
            "as", "into", "about", "it", "its", "this", "that", "these",
            "those", "not", "no", "and", "or", "but", "if", "then",
            "than", "so", "very", "just", "also", "only", "i", "my",
            "based", "sources", "memory", "information", "knowledge",
        }
        answer_keywords = answer_tokens - stopwords
        if len(answer_keywords) < 3:
            return True

        source_tokens = set()
        for r in results:
            source_tokens.update(re.findall(r"[a-z0-9]+", r.memory.content.lower()))

        overlap = answer_keywords & source_tokens
        coverage = len(overlap) / len(answer_keywords) if answer_keywords else 0

        return coverage >= 0.4
