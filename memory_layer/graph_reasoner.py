"""
Multi-Hop Graph Reasoning for the Memory Layer.

Goes beyond single-hop retrieval by following association and entity
links through the memory graph. LLM-guided at each hop to decide
which branches to explore, building a chain of evidence that answers
complex questions like "Why did I switch from React to Vue?"

Cost: ~200 tokens per hop × max 3 hops = ~600 tokens total.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Set, TYPE_CHECKING

if TYPE_CHECKING:
    from .core import MemoryManager

from .models import Memory, MemoryLink, LinkType


@dataclass
class ReasoningNode:
    """A node in the reasoning chain."""
    memory_id: str
    content: str
    hop: int
    reason: str = ""
    link_type: str = ""


@dataclass
class ReasoningChain:
    """Complete reasoning chain result."""
    nodes: List[ReasoningNode] = field(default_factory=list)
    synthesis: str = ""
    confidence: float = 0.0
    hops_taken: int = 0
    total_tokens_approx: int = 0


_HOP_PROMPT = """Given the question: "{query}"

What we've found so far:
{chain_so_far}

Linked memories available to explore:
{candidates}

Which of these linked memories (up to {max_branches}) are worth exploring to answer the question? If we already have enough information, say "SUFFICIENT".

Respond ONLY with JSON:
{{
  "selected": ["id1", "id2"],
  "reasoning": "why these are relevant",
  "sufficient": false
}}"""

_SYNTHESIS_PROMPT = """Given the question: "{query}"

Chain of evidence:
{chain}

Synthesize a clear answer based on this evidence chain. Be concise and cite the evidence.

Answer:"""


class GraphReasoner:
    """
    Multi-hop graph reasoning over the memory graph.

    Algorithm:
    1. Start with seed memories from standard recall
    2. At each hop, load linked memories
    3. Ask LLM which links to follow (compact prompt ~200 tokens)
    4. After max_hops or LLM says "sufficient", synthesize answer
    """

    def __init__(
        self,
        manager: "MemoryManager",
        max_hops: int = 3,
        max_branches: int = 3,
    ):
        self.manager = manager
        self.max_hops = max_hops
        self.max_branches = max_branches

    def reason(
        self,
        query: str,
        seed_memories: List[Memory],
        max_hops: int = None,
        max_branches: int = None,
    ) -> ReasoningChain:
        """
        Perform multi-hop reasoning starting from seed memories.

        Args:
            query: The user's question.
            seed_memories: Initial memories from standard recall.
            max_hops: Override default max hops.
            max_branches: Override default max branches per hop.

        Returns:
            ReasoningChain with nodes, synthesis, and confidence.
        """
        max_hops = max_hops or self.max_hops
        max_branches = max_branches or self.max_branches

        chain = ReasoningChain()
        visited: Set[str] = set()
        total_tokens = 0

        # Seed the chain with top seed memories
        current_ids = []
        for mem in seed_memories[:5]:
            node = ReasoningNode(
                memory_id=mem.id,
                content=mem.content[:200],
                hop=0,
                reason="seed from initial recall",
            )
            chain.nodes.append(node)
            visited.add(mem.id)
            current_ids.append(mem.id)

        if not current_ids:
            return chain

        has_llm = (
            self.manager.enrichment is not None
            and self.manager.enrichment.has_llm
        )

        for hop in range(1, max_hops + 1):
            # Get all linked memories for current frontier
            links = self.manager.storage.get_links_for_ids(current_ids)
            if not links:
                break

            # Collect candidate memory IDs
            candidate_ids = set()
            link_info: Dict[str, str] = {}
            for link in links:
                other_id = (
                    link.target_id if link.source_id in current_ids
                    else link.source_id
                )
                if other_id not in visited:
                    candidate_ids.add(other_id)
                    link_info[other_id] = link.link_type.value

            if not candidate_ids:
                break

            # Load candidate memories
            candidates = self.manager.storage.get_memories_by_ids(
                list(candidate_ids)
            )
            candidates = [m for m in candidates if m.is_active]

            if not candidates:
                break

            # LLM-guided selection or fallback to top by importance
            if has_llm:
                selected, sufficient, tokens = self._llm_select(
                    query, chain, candidates, link_info,
                    max_branches, hop,
                )
                total_tokens += tokens
                if sufficient:
                    break
            else:
                candidates.sort(key=lambda m: m.importance, reverse=True)
                selected = candidates[:max_branches]

            next_ids = []
            for mem in selected:
                lt = link_info.get(mem.id, "related")
                node = ReasoningNode(
                    memory_id=mem.id,
                    content=mem.content[:200],
                    hop=hop,
                    reason=f"hop {hop} via {lt}",
                    link_type=lt,
                )
                chain.nodes.append(node)
                visited.add(mem.id)
                next_ids.append(mem.id)

            current_ids = next_ids
            chain.hops_taken = hop

            if not next_ids:
                break

        # Synthesize answer
        if has_llm and len(chain.nodes) > 1:
            chain.synthesis, synth_tokens = self._synthesize(query, chain)
            total_tokens += synth_tokens
        else:
            chain.synthesis = self._heuristic_synthesis(chain)

        chain.total_tokens_approx = total_tokens
        chain.confidence = min(1.0, len(chain.nodes) / 5 * 0.8)

        return chain

    def _llm_select(
        self,
        query: str,
        chain: ReasoningChain,
        candidates: List[Memory],
        link_info: Dict[str, str],
        max_branches: int,
        hop: int,
    ) -> tuple:
        """Use LLM to select which linked memories to explore."""
        chain_lines = []
        for n in chain.nodes:
            chain_lines.append(f"  [{n.hop}] {n.content[:150]} ({n.reason})")

        cand_lines = []
        for m in candidates[:10]:
            lt = link_info.get(m.id, "related")
            cand_lines.append(
                f"  [{m.id[:8]}] ({lt}): {m.content[:150]}"
            )

        prompt = _HOP_PROMPT.format(
            query=query,
            chain_so_far="\n".join(chain_lines) or "  (starting)",
            candidates="\n".join(cand_lines),
            max_branches=max_branches,
        )

        approx_tokens = len(prompt) // 4 + 100

        try:
            raw = self.manager.enrichment.generate(prompt, max_tokens=200)
            result = _parse_json(raw)

            if result and result.get("sufficient"):
                return [], True, approx_tokens

            if result and result.get("selected"):
                id_map = {m.id[:8]: m for m in candidates}
                selected = []
                for sid in result["selected"][:max_branches]:
                    sid = str(sid).strip()
                    for key, mem in id_map.items():
                        if key.startswith(sid) or sid.startswith(key):
                            selected.append(mem)
                            break
                return selected, False, approx_tokens

        except Exception:
            pass

        # Fallback: select by importance
        candidates.sort(key=lambda m: m.importance, reverse=True)
        return candidates[:max_branches], False, approx_tokens

    def _synthesize(
        self, query: str, chain: ReasoningChain,
    ) -> tuple:
        """Synthesize a final answer from the reasoning chain."""
        chain_lines = []
        for n in chain.nodes:
            chain_lines.append(
                f"  [Hop {n.hop}, {n.link_type or 'seed'}] {n.content}"
            )

        prompt = _SYNTHESIS_PROMPT.format(
            query=query,
            chain="\n".join(chain_lines),
        )

        approx_tokens = len(prompt) // 4 + 150

        try:
            result = self.manager.enrichment.generate(prompt, max_tokens=300)
            if result and len(result.strip()) > 10:
                return result.strip(), approx_tokens
        except Exception:
            pass

        return self._heuristic_synthesis(chain), approx_tokens

    @staticmethod
    def _heuristic_synthesis(chain: ReasoningChain) -> str:
        """Fallback synthesis without LLM."""
        if not chain.nodes:
            return "No evidence found."
        contents = [n.content for n in chain.nodes]
        return "Based on memory chain: " + " → ".join(
            c[:100] for c in contents[:5]
        )


def _parse_json(text: str) -> Optional[Dict]:
    """Safely parse JSON from LLM output."""
    if not text:
        return None
    text = text.strip()
    start = text.find("{")
    end = text.rfind("}") + 1
    if start >= 0 and end > start:
        try:
            return json.loads(text[start:end])
        except json.JSONDecodeError:
            pass
    return None
