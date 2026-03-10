"""
Reasoning Engine for the Memory Layer.

Honcho-inspired formal logic reasoning that runs in the background on
batched memories. Produces deductive, inductive, and abductive conclusions
stored as higher-abstraction memories in the existing hierarchy.

Tiered system — the tier controls which steps run, the model controls
which LLM runs them:

    ReasoningEngine(manager, mode="standard", model="gpt-4o-mini")

Tiers:
    none      — disabled (default)
    local     — Steps 1+5 only (premise extraction + peer card)
    standard  — Full 5-step pipeline
    advanced  — Full pipeline with richer chain-of-thought prompts

Models:
    "local"       — uses enrichment LLM (Ollama)
    "gpt-4o-mini" — default for standard tier
    "o3-mini"     — default for advanced tier
    Any OpenAI model string accepted.
"""

from __future__ import annotations

import json
import os
import time
import uuid
import threading
from typing import List, Dict, Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .core import MemoryManager

from .models import Memory, MemoryType, MemoryLink, LinkType

# ── Tier → steps mapping ──────────────────────────────────────────

TIER_STEPS = {
    "none":     [],
    "local":    [1, 5],
    "standard": [1, 2, 3, 4, 5],
    "advanced": [1, 2, 3, 4, 5],
}

TIER_DEFAULT_MODEL = {
    "none":     None,
    "local":    "local",
    "standard": "gpt-4o-mini",
    "advanced": "o3-mini",
}

# ── Prompts: Local (simple, tolerant) ─────────────────────────────

_LOCAL_EXTRACT_PROMPT = """List the key facts explicitly stated in these messages. One fact per line, be specific.

Messages:
{batch}

Facts:"""

_LOCAL_PEER_CARD_PROMPT = """Given existing profile and new facts, produce a concise updated profile.
Include: name, role, preferences, current projects, key traits.

Existing profile:
{existing_card}

New facts:
{new_facts}

Updated profile:"""

# ── Prompts: API (structured JSON) ────────────────────────────────

_API_EXTRACT_DEDUCTIVE_PROMPT = """You are a formal reasoning engine. Given these messages, extract:
1. "explicit" — things directly stated as fact
2. "deductive" — conclusions that NECESSARILY follow from combining 2+ premises

Rules:
- Each premise must be a single self-contained fact
- Deductive conclusions must follow from stated premises only — no speculation
- Include source_ids (the indices of messages each derives from)
- Return at most 10 explicit and 5 deductive conclusions

Messages:
{batch}

Return ONLY valid JSON:
{{"explicit": [{{"content": "...", "source_ids": [0, 2]}}], "deductive": [{{"premises": ["premise1", "premise2"], "conclusion": "...", "source_ids": [0, 2]}}]}}"""

_API_INDUCTIVE_PROMPT = """You are a pattern recognition engine. Given these conclusions drawn about a user over time, identify recurring patterns — themes, behavioral tendencies, or consistent preferences that appear across multiple conclusions.

Conclusions:
{conclusions}

Rules:
- Only identify patterns supported by 2+ conclusions
- Rate confidence 0.0-1.0
- Return at most 5 patterns

Return ONLY valid JSON:
{{"patterns": [{{"pattern": "...", "supporting": ["conclusion1", "conclusion2"], "confidence": 0.8}}]}}"""

_API_ABDUCTIVE_PROMPT = """You are an inference engine. Given these observed patterns about a user, infer the simplest explanations for their behavior. What motivates these patterns? What can be predicted?

Patterns:
{patterns}

Rules:
- Infer only what the evidence supports
- Each explanation must cite the pattern it explains
- Return at most 3 explanations

Return ONLY valid JSON:
{{"explanations": [{{"observation": "...", "explanation": "...", "prediction": "..."}}]}}"""

_API_PEER_CARD_PROMPT = """Given the existing profile and new conclusions, produce an updated concise profile of this person. Include name, role, key preferences, current projects, behavioral traits, and any notable patterns.

Existing profile:
{existing_card}

New conclusions:
{new_conclusions}

Return ONLY valid JSON:
{{"peer_card": "concise updated profile text"}}"""

# ── Advanced tier: chain-of-thought variants ──────────────────────

_ADV_EXTRACT_DEDUCTIVE_PROMPT = """You are a formal reasoning engine performing rigorous logical analysis.

Step 1 — Read each message carefully and extract every explicit premise (stated fact).
Step 2 — For each pair or group of premises, check if a deductive conclusion necessarily follows.
Step 3 — Only include conclusions where the inference is certain, not merely plausible.

Messages:
{batch}

Think step by step, then return ONLY valid JSON:
{{"explicit": [{{"content": "...", "source_ids": [0]}}], "deductive": [{{"premises": ["p1", "p2"], "conclusion": "...", "source_ids": [0, 2]}}]}}"""

_ADV_INDUCTIVE_PROMPT = """You are performing rigorous inductive analysis.

Step 1 — Review all conclusions and group them by theme.
Step 2 — For each group of 2+ related conclusions, identify the underlying pattern.
Step 3 — Rate confidence based on how many conclusions support the pattern and how strong the evidence is.

Conclusions:
{conclusions}

Think step by step, then return ONLY valid JSON:
{{"patterns": [{{"pattern": "...", "supporting": ["c1", "c2"], "confidence": 0.8}}]}}"""


class ReasoningEngine:
    """
    Background reasoning engine that produces formal logical conclusions
    from batched memories. Conclusions are stored as higher-abstraction
    memories in the existing hierarchy.
    """

    def __init__(
        self,
        manager: "MemoryManager",
        mode: str = "none",
        model: Optional[str] = None,
        batch_threshold: int = 1000,
        dreaming_interval: int = 200,
        local_full_reasoning: bool = False,
    ):
        self.manager = manager
        self.tier = mode if mode in TIER_STEPS else "none"

        # Resolve model
        if model:
            self.model = model
        else:
            self.model = TIER_DEFAULT_MODEL.get(self.tier)

        self.batch_threshold = batch_threshold
        self.dreaming_interval = dreaming_interval
        self.local_full_reasoning = local_full_reasoning

        # Resolve steps
        steps = list(TIER_STEPS.get(self.tier, []))
        if self.tier == "local" and local_full_reasoning:
            steps = [1, 2, 3, 4, 5]
        self.steps = steps

        self.enabled = self.tier != "none" and len(self.steps) > 0
        self._use_openai = self.model and self.model != "local"
        self._openai_client = None
        self._op_count = 0
        self._last_dream_time = 0.0
        self._lock = threading.Lock()

        self.stats = {
            "batches_processed": 0,
            "conclusions_created": 0,
            "dreams_completed": 0,
            "last_batch_time": 0.0,
            "last_dream_time": 0.0,
        }

        if self.enabled:
            if self._use_openai:
                self._init_openai()
            print(f"  + Reasoning engine: tier={self.tier}, model={self.model}, steps={self.steps}")

    def _init_openai(self):
        """Initialize OpenAI client for API-based models."""
        api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            print(f"  ! Reasoning engine: OpenAI key missing, falling back to local")
            self._use_openai = False
            self.model = "local"
            if self.tier in ("standard", "advanced"):
                self.steps = [1, 5]
            return
        try:
            from openai import OpenAI
            self._openai_client = OpenAI(api_key=api_key)
        except ImportError:
            print(f"  ! Reasoning engine: openai package not installed, falling back to local")
            self._use_openai = False
            self.model = "local"
            if self.tier in ("standard", "advanced"):
                self.steps = [1, 5]

    # ── LLM call routing ──────────────────────────────────────────

    _MAX_RETRIES = 2
    _RETRY_DELAY = 1.0

    def _generate(self, prompt: str, max_tokens: int = 1000) -> Optional[str]:
        """Route LLM call with retry logic."""
        for attempt in range(self._MAX_RETRIES + 1):
            try:
                if self._use_openai and self._openai_client:
                    result = self._call_openai(prompt, max_tokens)
                elif self.manager.enrichment and self.manager.enrichment.has_llm:
                    result = self.manager.enrichment.generate(prompt, max_tokens=max_tokens)
                else:
                    return None
                if result:
                    return result
            except Exception as e:
                if attempt < self._MAX_RETRIES:
                    time.sleep(self._RETRY_DELAY * (attempt + 1))
                else:
                    print(f"  ! Reasoning engine: LLM call failed after {self._MAX_RETRIES + 1} attempts: {e}")
        return None

    def _call_openai(self, prompt: str, max_tokens: int = 1000) -> Optional[str]:
        """Call OpenAI API directly."""
        is_reasoning_model = self.model and self.model.startswith(("o1", "o3"))
        kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
        }
        if is_reasoning_model:
            kwargs["max_completion_tokens"] = max_tokens
        else:
            kwargs["max_tokens"] = max_tokens
            kwargs["temperature"] = 0.1

        response = self._openai_client.chat.completions.create(**kwargs)
        return response.choices[0].message.content.strip()

    # ── Queue management ──────────────────────────────────────────

    def enqueue(self, memory: Memory):
        """Add a memory to the reasoning queue. Triggers batch if threshold met."""
        if not self.enabled:
            return
        token_est = max(1, len(memory.content) // 4)
        self.manager.storage.enqueue_for_reasoning(
            memory_id=memory.id,
            content=memory.content,
            token_estimate=token_est,
        )
        self._maybe_process_batch()

    def _maybe_process_batch(self):
        """Check if queue has enough tokens and trigger background processing."""
        stats = self.manager.storage.get_reasoning_queue_stats()
        if stats.get("pending_tokens", 0) >= self.batch_threshold:
            if hasattr(self.manager, '_bg_pool') and not self.manager._shutting_down:
                self.manager._bg_pool.submit(self._process_batch_safe)
            else:
                threading.Thread(
                    target=self._process_batch_safe, daemon=True,
                ).start()

    def _process_batch_safe(self):
        """Thread-safe batch processing."""
        try:
            self._process_batch()
        except Exception as e:
            import sys
            print(f"  ! Reasoning engine batch error: {e}", file=sys.stderr)

    def _process_batch(self):
        """Process a batch of queued memories through the reasoning pipeline."""
        batch = self.manager.storage.get_pending_reasoning_batch(
            self.batch_threshold,
        )
        if not batch:
            return

        memory_ids = [row["memory_id"] for row in batch]
        contents = [row["content"] for row in batch]
        batch_ids = [row["id"] for row in batch]

        batch_text = "\n".join(
            f"[{i}] {c[:300]}" for i, c in enumerate(contents)
        )

        conclusions_created = 0

        # Step 1: Explicit extraction (+ deduction for API tiers)
        if 1 in self.steps:
            premises, deductions = self._step1_extract(batch_text, memory_ids)

            # Step 2: Deductive reasoning (store conclusions)
            if 2 in self.steps and deductions:
                for ded in deductions:
                    self._store_conclusion(
                        content=ded["conclusion"],
                        reasoning_type="deductive",
                        abstraction_level=1,
                        metadata={
                            "premises": ded.get("premises", []),
                            "source_memory_ids": ded.get("source_memory_ids", memory_ids[:3]),
                        },
                        source_memory_ids=ded.get("source_memory_ids", memory_ids[:3]),
                    )
                    conclusions_created += 1

        # Step 5: Peer card (runs on every batch for all tiers)
        if 5 in self.steps:
            new_facts = "\n".join(f"- {c[:200]}" for c in contents[:10])
            self._step5_peer_card(new_facts)
            conclusions_created += 1

        self.manager.storage.mark_reasoning_processed(batch_ids)
        self.stats["batches_processed"] += 1
        self.stats["conclusions_created"] += conclusions_created
        self.stats["last_batch_time"] = time.time()

    # ── Step 1: Explicit extraction ───────────────────────────────

    def _step1_extract(
        self, batch_text: str, memory_ids: List[str],
    ) -> tuple:
        """Extract premises and (for API tiers) deductive conclusions."""
        if self._use_openai and 2 in self.steps:
            prompt_template = (
                _ADV_EXTRACT_DEDUCTIVE_PROMPT if self.tier == "advanced"
                else _API_EXTRACT_DEDUCTIVE_PROMPT
            )
            prompt = prompt_template.format(batch=batch_text)
            raw = self._generate(prompt, max_tokens=1500)
            parsed = _parse_json_safe(raw)
            if parsed:
                premises = parsed.get("explicit", [])
                deductions = []
                for d in parsed.get("deductive", []):
                    if isinstance(d, dict) and "conclusion" in d:
                        src_ids = d.get("source_ids", [])
                        mapped_ids = [
                            memory_ids[i] for i in src_ids
                            if isinstance(i, int) and i < len(memory_ids)
                        ] or memory_ids[:2]
                        deductions.append({
                            "premises": d.get("premises", []),
                            "conclusion": d["conclusion"],
                            "source_memory_ids": mapped_ids,
                        })
                return premises, deductions
            return [], []
        else:
            prompt = _LOCAL_EXTRACT_PROMPT.format(batch=batch_text)
            raw = self._generate(prompt, max_tokens=500)
            if raw:
                facts = [
                    line.strip().lstrip("- •*")
                    for line in raw.strip().split("\n")
                    if line.strip() and len(line.strip()) > 5
                ]
                premises = [{"content": f} for f in facts[:10]]
                return premises, []
            return [], []

    # ── Step 5: Peer card ─────────────────────────────────────────

    def _step5_peer_card(self, new_facts: str):
        """Update the peer card (L3 identity summary)."""
        existing_card = self._get_existing_peer_card()
        card_text = existing_card.content if existing_card else "(no existing profile)"

        if self._use_openai:
            prompt = _API_PEER_CARD_PROMPT.format(
                existing_card=card_text, new_conclusions=new_facts,
            )
            raw = self._generate(prompt, max_tokens=500)
            parsed = _parse_json_safe(raw)
            new_card = parsed.get("peer_card", "") if parsed else ""
            if not new_card and raw:
                new_card = raw.strip()
        else:
            prompt = _LOCAL_PEER_CARD_PROMPT.format(
                existing_card=card_text, new_facts=new_facts,
            )
            new_card = self._generate(prompt, max_tokens=500) or ""

        if new_card and len(new_card) > 10:
            if existing_card:
                existing_card.is_current = False
                existing_card.strength = 0.3
                self.manager.storage.update_memory(existing_card)
            self._store_conclusion(
                content=new_card.strip(),
                reasoning_type="peer_card",
                abstraction_level=3,
                metadata={"reasoning_type": "peer_card"},
                tags=["peer_card", "reasoning"],
            )

    def _get_existing_peer_card(self) -> Optional[Memory]:
        """Find the current peer card memory."""
        memories = self.manager.storage.get_all_memories(
            memory_type=MemoryType.SEMANTIC,
            tags=["peer_card"],
            active_only=True,
        )
        for m in memories:
            if m.is_current and (m.metadata or {}).get("reasoning_type") == "peer_card":
                return m
        return None

    # ── Dreaming (periodic background reasoning) ──────────────────

    def maybe_dream(self):
        """Called from _increment_operations. Triggers dream if interval met."""
        if not self.enabled:
            return
        with self._lock:
            self._op_count += 1
            if self._op_count % self.dreaming_interval != 0:
                return
        if 3 not in self.steps and 4 not in self.steps:
            return
        if hasattr(self.manager, '_bg_pool') and not self.manager._shutting_down:
            self.manager._bg_pool.submit(self._dream_safe)
        else:
            threading.Thread(target=self._dream_safe, daemon=True).start()

    def _dream_safe(self):
        """Thread-safe wrapper for dreaming."""
        try:
            self._dream()
        except Exception as e:
            import sys
            print(f"  ! Reasoning engine dream error: {e}", file=sys.stderr)

    def _dream(self):
        """Run inductive + abductive reasoning over recent deductive conclusions."""
        recent_conclusions = self.manager.storage.get_all_memories(
            memory_type=MemoryType.SEMANTIC,
            tags=["deductive"],
            active_only=True,
        )[:30]
        if len(recent_conclusions) < 2:
            return

        conclusion_texts = [m.content for m in recent_conclusions]
        conclusions_block = "\n".join(
            f"- {c[:200]}" for c in conclusion_texts
        )

        # Step 3: Inductive reasoning
        if 3 in self.steps:
            prompt_template = (
                _ADV_INDUCTIVE_PROMPT if self.tier == "advanced"
                else _API_INDUCTIVE_PROMPT
            )
            prompt = prompt_template.format(conclusions=conclusions_block)
            raw = self._generate(prompt, max_tokens=1000)
            parsed = _parse_json_safe(raw)
            if parsed and "patterns" in parsed:
                for p in parsed["patterns"]:
                    if isinstance(p, dict) and "pattern" in p:
                        self._store_conclusion(
                            content=p["pattern"],
                            reasoning_type="inductive",
                            abstraction_level=2,
                            metadata={
                                "supporting": p.get("supporting", []),
                                "confidence": p.get("confidence", 0.5),
                            },
                        )
                        self.stats["conclusions_created"] += 1

        # Step 4: Abductive reasoning
        if 4 in self.steps:
            patterns = self.manager.storage.get_all_memories(
                memory_type=MemoryType.SEMANTIC,
                tags=["inductive"],
                active_only=True,
            )[:10]
            if patterns:
                patterns_block = "\n".join(
                    f"- {m.content[:200]}" for m in patterns
                )
                prompt = _API_ABDUCTIVE_PROMPT.format(patterns=patterns_block)
                raw = self._generate(prompt, max_tokens=1000)
                parsed = _parse_json_safe(raw)
                if parsed and "explanations" in parsed:
                    for ex in parsed["explanations"]:
                        if isinstance(ex, dict) and "explanation" in ex:
                            content = ex["explanation"]
                            if ex.get("prediction"):
                                content += f" Prediction: {ex['prediction']}"
                            self._store_conclusion(
                                content=content,
                                reasoning_type="abductive",
                                abstraction_level=2,
                                metadata={
                                    "observation": ex.get("observation", ""),
                                    "prediction": ex.get("prediction", ""),
                                },
                            )
                            self.stats["conclusions_created"] += 1

        # Step 5: Update peer card during dream too
        if 5 in self.steps:
            new_facts = "\n".join(f"- {c[:200]}" for c in conclusion_texts[:10])
            self._step5_peer_card(new_facts)

        self.stats["dreams_completed"] += 1
        self.stats["last_dream_time"] = time.time()
        self._last_dream_time = time.time()

    def trigger_dream(self):
        """Manually trigger a dream cycle (for API/testing)."""
        if not self.enabled:
            return {"status": "disabled"}
        if 3 not in self.steps and 4 not in self.steps:
            return {"status": "tier_does_not_support_dreaming"}
        if hasattr(self.manager, '_bg_pool') and not self.manager._shutting_down:
            self.manager._bg_pool.submit(self._dream_safe)
        else:
            threading.Thread(target=self._dream_safe, daemon=True).start()
        return {"status": "dreaming_started"}

    # ── Conclusion storage ────────────────────────────────────────

    _MAX_CONCLUSIONS_PER_TYPE = 200

    def _store_conclusion(
        self,
        content: str,
        reasoning_type: str,
        abstraction_level: int,
        metadata: Optional[Dict[str, Any]] = None,
        source_memory_ids: Optional[List[str]] = None,
        tags: Optional[List[str]] = None,
    ):
        """Store a reasoning conclusion as a semantic memory with deduplication."""
        if not content or len(content.strip()) < 10:
            return

        # Dedup: check if a near-identical conclusion already exists
        try:
            embedding = self.manager.embeddings.embed(content)
            existing = self.manager.storage.get_all_memories(
                memory_type=MemoryType.SEMANTIC,
                tags=[reasoning_type],
                active_only=True,
            )
            for ex in existing:
                if ex.embedding:
                    import numpy as np
                    ev = np.array(ex.embedding, dtype=np.float32)
                    qv = np.array(embedding, dtype=np.float32)
                    sim = float(np.dot(ev, qv) / (np.linalg.norm(ev) * np.linalg.norm(qv) + 1e-9))
                    if sim >= 0.92:
                        return  # duplicate conclusion, skip
        except Exception:
            embedding = None

        meta = metadata or {}
        meta["reasoning_type"] = reasoning_type
        meta["reasoning_tier"] = self.tier
        meta["reasoning_model"] = self.model

        if tags is None:
            tags = ["reasoning", reasoning_type]

        memory = Memory(
            id=str(uuid.uuid4()),
            memory_type=MemoryType.SEMANTIC,
            content=content.strip(),
            importance=0.7 if reasoning_type != "peer_card" else 0.9,
            tags=tags,
            metadata=meta,
            abstraction_level=abstraction_level,
            is_current=True,
        )

        try:
            if embedding:
                memory.embedding = embedding
            else:
                memory.embedding = self.manager.embeddings.embed(content)

            self.manager.storage.store_memory(memory)
            self.manager.memory_index.add(memory.id, memory.embedding)

            if source_memory_ids:
                for src_id in source_memory_ids[:5]:
                    try:
                        link = MemoryLink(
                            source_id=src_id,
                            target_id=memory.id,
                            link_type=LinkType.DERIVED,
                            weight=0.8,
                        )
                        self.manager.storage.store_link(link)
                    except Exception:
                        pass
        except Exception as e:
            print(f"  ! Reasoning: failed to store conclusion: {e}")

    # ── Stats ─────────────────────────────────────────────────────

    def get_stats(self) -> Dict[str, Any]:
        """Return reasoning engine statistics."""
        queue_stats = {}
        if self.enabled:
            try:
                queue_stats = self.manager.storage.get_reasoning_queue_stats()
            except Exception:
                pass

        conclusion_counts = {}
        if self.enabled:
            for rtype in ("deductive", "inductive", "abductive", "peer_card"):
                try:
                    mems = self.manager.storage.get_all_memories(
                        memory_type=MemoryType.SEMANTIC,
                        tags=[rtype],
                        active_only=True,
                    )
                    mems = [m for m in mems if (m.metadata or {}).get("reasoning_type") == rtype]
                    conclusion_counts[rtype] = len(mems)
                except Exception:
                    conclusion_counts[rtype] = 0

        return {
            "enabled": self.enabled,
            "tier": self.tier,
            "model": self.model,
            "steps": self.steps,
            **self.stats,
            "queue": queue_stats,
            "conclusions": conclusion_counts,
        }


# ── JSON parsing helpers ──────────────────────────────────────────

def _parse_json_safe(text: Optional[str]) -> Optional[Dict]:
    """Parse JSON from LLM output with multiple fallback strategies."""
    if not text:
        return None
    text = text.strip()

    # Strategy 1: direct parse
    try:
        result = json.loads(text)
        if isinstance(result, dict):
            return result
    except json.JSONDecodeError:
        pass

    # Strategy 2: extract from markdown code fence
    import re
    fence_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if fence_match:
        try:
            result = json.loads(fence_match.group(1).strip())
            if isinstance(result, dict):
                return result
        except json.JSONDecodeError:
            pass

    # Strategy 3: find first { ... } block
    start = text.find("{")
    end = text.rfind("}") + 1
    if start >= 0 and end > start:
        try:
            result = json.loads(text[start:end])
            if isinstance(result, dict):
                return result
        except json.JSONDecodeError:
            pass

    return None
