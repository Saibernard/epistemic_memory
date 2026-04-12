"""
Knowledge Page Manager for the Memory Layer.

Inspired by Karpathy's LLM Wiki pattern: auto-generates and maintains
wiki-style knowledge pages per entity/concept. Each page is a synthesized
summary of all memories related to that entity.

Pages are:
  - Auto-created when enough memories mention an entity
  - Auto-updated when new memories are added
  - Queryable by entity name, ID, or page type
  - Exportable as Obsidian-compatible markdown (see wiki_export.py)

The manager uses the enrichment LLM (if available) to synthesize
summaries. Falls back to keyword-based concatenation when no LLM.
"""

from __future__ import annotations

import time
import uuid
from typing import Optional, List, Dict, Any

from .models import KnowledgePage, Memory
from .entity_graph import extract_entities, _entity_id


_PAGE_SYNTHESIS_PROMPT = """You are a wiki page generator. Given these memory excerpts about '{entity}' ({entity_type}), write a concise, factual wiki-style summary.

Rules:
1. Synthesize the information into a coherent narrative — do NOT just list the excerpts
2. Keep it under 300 words
3. Focus on facts, preferences, relationships, and patterns
4. Use present tense for current facts, past tense for historical events
5. Do NOT add information not present in the excerpts
6. Start directly with the summary — no title or preamble

Memory excerpts:
{excerpts}

Wiki summary:"""


_MIN_MENTIONS_FOR_PAGE = 2  # Entity needs this many mentions to get a page


class KnowledgePageManager:
    """
    Manages auto-generated knowledge pages per entity.

    Uses:
      - storage: for page CRUD and memory retrieval
      - entity_graph: for entity-memory linkage lookups
      - enrichment: for LLM-powered summary synthesis
    """

    def __init__(self, storage, entity_graph, enrichment=None):
        self.storage = storage
        self.entity_graph = entity_graph
        self.enrichment = enrichment

    def propagate_from_memory(self, memory_id: str, content: str):
        """
        Called after storing a memory. Extracts entities and
        queues page updates for each affected entity.
        """
        entities = extract_entities(content)
        for ent in entities:
            eid = _entity_id(ent["name"].lower())
            try:
                self._maybe_update_page(eid, ent["name"], ent["entity_type"])
            except Exception:
                pass

    def _maybe_update_page(self, entity_id: str, entity_name: str, entity_type: str):
        """Create or update a page if the entity has enough mentions."""
        # Check mention count from entity_nodes
        conn = self.storage._conn()
        row = conn.execute(
            "SELECT mention_count FROM entity_nodes WHERE entity_id = ?",
            (entity_id,),
        ).fetchone()

        if not row or row[0] < _MIN_MENTIONS_FOR_PAGE:
            return

        # Get all memory IDs linked to this entity
        linked = conn.execute(
            "SELECT el.memory_id FROM entity_links el "
            "JOIN memories m ON m.id = el.memory_id "
            "WHERE el.entity_id = ? AND m.is_active = 1",
            (entity_id,),
        ).fetchall()
        memory_ids = [r[0] for r in linked]

        if not memory_ids:
            return

        # Load memories
        memories = self.storage.get_memories_by_ids(memory_ids)
        if not memories:
            return

        # Check if page already exists
        existing = self.storage.get_knowledge_page_by_entity(entity_id)

        # Synthesize summary
        summary = self._synthesize_summary(entity_name, entity_type, memories)

        now = time.time()
        if existing:
            existing.summary = summary
            existing.version += 1
            existing.last_updated = now
            existing.memory_ids = memory_ids
            self.storage.update_knowledge_page(existing)
            page_id = existing.page_id
        else:
            page = KnowledgePage(
                page_id=str(uuid.uuid4()),
                entity_id=entity_id,
                title=entity_name,
                page_type=self._infer_page_type(entity_type),
                summary=summary,
                memory_ids=memory_ids,
                version=1,
                last_updated=now,
                created_at=now,
            )
            page_id = self.storage.store_knowledge_page(page)

        # Update junction table
        for mid in memory_ids:
            self.storage.link_memory_to_page(page_id, mid)

    def _synthesize_summary(
        self, entity_name: str, entity_type: str, memories: List[Memory]
    ) -> str:
        """Synthesize a wiki summary from memories using LLM or fallback."""
        excerpts = "\n---\n".join(
            m.content[:500] for m in memories[:20]
        )

        if self.enrichment and self.enrichment.has_llm:
            prompt = _PAGE_SYNTHESIS_PROMPT.format(
                entity=entity_name,
                entity_type=entity_type,
                excerpts=excerpts,
            )
            result = self.enrichment.generate(prompt, max_tokens=600)
            if result and len(result.strip()) > 20:
                return result.strip()

        # Fallback: concatenate unique content snippets
        seen = set()
        parts = []
        for m in memories[:15]:
            snippet = m.content[:200].strip()
            if snippet not in seen:
                seen.add(snippet)
                parts.append(f"- {snippet}")
        return f"Knowledge about {entity_name}:\n" + "\n".join(parts)

    def _infer_page_type(self, entity_type: str) -> str:
        """Map entity_graph entity_type to knowledge page page_type."""
        mapping = {
            "person": "entity",
            "tech": "topic",
            "org": "entity",
            "concept": "concept",
            "identifier": "topic",
        }
        return mapping.get(entity_type, "entity")

    def create_or_update_page(
        self, entity_id: str, entity_name: str, entity_type: str = "concept"
    ) -> Optional[KnowledgePage]:
        """
        Explicitly create or update a knowledge page for an entity.
        Returns the page, or None if no memories exist for it.
        """
        conn = self.storage._conn()
        linked = conn.execute(
            "SELECT el.memory_id FROM entity_links el "
            "JOIN memories m ON m.id = el.memory_id "
            "WHERE el.entity_id = ? AND m.is_active = 1",
            (entity_id,),
        ).fetchall()
        memory_ids = [r[0] for r in linked]

        if not memory_ids:
            return None

        memories = self.storage.get_memories_by_ids(memory_ids)
        if not memories:
            return None

        summary = self._synthesize_summary(entity_name, entity_type, memories)
        existing = self.storage.get_knowledge_page_by_entity(entity_id)

        now = time.time()
        if existing:
            existing.summary = summary
            existing.version += 1
            existing.last_updated = now
            existing.memory_ids = memory_ids
            self.storage.update_knowledge_page(existing)
            for mid in memory_ids:
                self.storage.link_memory_to_page(existing.page_id, mid)
            return existing

        page = KnowledgePage(
            page_id=str(uuid.uuid4()),
            entity_id=entity_id,
            title=entity_name,
            page_type=self._infer_page_type(entity_type),
            summary=summary,
            memory_ids=memory_ids,
            version=1,
            last_updated=now,
            created_at=now,
        )
        self.storage.store_knowledge_page(page)
        for mid in memory_ids:
            self.storage.link_memory_to_page(page.page_id, mid)
        return page

    def get_page(self, page_id: str) -> Optional[KnowledgePage]:
        """Get a knowledge page by ID."""
        return self.storage.get_knowledge_page(page_id)

    def get_page_for_entity(self, entity_name: str) -> Optional[KnowledgePage]:
        """Get the knowledge page for an entity by name."""
        eid = _entity_id(entity_name.lower())
        return self.storage.get_knowledge_page_by_entity(eid)

    def get_all_pages(self, page_type: Optional[str] = None) -> List[KnowledgePage]:
        """Get all knowledge pages, optionally filtered by type."""
        return self.storage.get_all_knowledge_pages(page_type=page_type)

    def delete_page(self, page_id: str):
        """Delete a knowledge page."""
        self.storage.delete_knowledge_page(page_id)

    def rebuild_all_pages(self) -> Dict[str, Any]:
        """
        Rebuild all knowledge pages from scratch.
        Iterates all entities with sufficient mentions and regenerates pages.
        Returns summary stats.
        """
        conn = self.storage._conn()
        entities = conn.execute(
            "SELECT entity_id, name, entity_type FROM entity_nodes "
            "WHERE mention_count >= ?",
            (_MIN_MENTIONS_FOR_PAGE,),
        ).fetchall()

        created = 0
        updated = 0
        for eid, name, etype in entities:
            existing = self.storage.get_knowledge_page_by_entity(eid)
            result = self.create_or_update_page(eid, name, etype)
            if result:
                if existing:
                    updated += 1
                else:
                    created += 1

        return {
            "entities_checked": len(entities),
            "pages_created": created,
            "pages_updated": updated,
            "total_pages": len(self.get_all_pages()),
        }

    def get_outdated_pages(self, max_age_hours: float = 168) -> List[KnowledgePage]:
        """Get pages not updated since their newest linked memory was added."""
        all_pages = self.get_all_pages()
        outdated = []
        for page in all_pages:
            memory_ids = self.storage.get_memories_for_page(page.page_id)
            if not memory_ids:
                outdated.append(page)
                continue
            memories = self.storage.get_memories_by_ids(memory_ids)
            if memories:
                newest = max(m.created_at for m in memories)
                if newest > page.last_updated:
                    outdated.append(page)
        return outdated
