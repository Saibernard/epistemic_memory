"""
Chunk-Level Entity Graph for the Memory Layer.

Pre-links memory chunks to extracted entity nodes so that at retrieval
time we can expand entity neighborhoods — if a query mentions "Python",
we find all chunks linked to the "python" entity and surface them as
extra candidates.

This is the feature HydraDB calls "chunk-level graph expansion":
  1. At ingestion: extract entities → link chunk → entity nodes
  2. At retrieval: query → entities → entity → chunk fan-out

Entity extraction uses lightweight regex + heuristics (no LLM needed).
When the enrichment LLM is available, it can be used for better NER.

Storage is in SQLite alongside the main database:
  - entity_nodes:   (entity_id, name, entity_type, mention_count)
  - entity_links:   (memory_id, entity_id, weight)
"""

from __future__ import annotations

import re
import sqlite3
import hashlib
from typing import List, Dict, Set, Tuple, Optional


_STOPWORDS = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "need", "dare", "ought",
    "used", "to", "of", "in", "for", "on", "with", "at", "by", "from",
    "as", "into", "through", "during", "before", "after", "above", "below",
    "between", "out", "off", "over", "under", "again", "further", "then",
    "once", "here", "there", "when", "where", "why", "how", "all", "each",
    "every", "both", "few", "more", "most", "other", "some", "such", "no",
    "nor", "not", "only", "own", "same", "so", "than", "too", "very",
    "just", "because", "but", "and", "or", "if", "while", "that", "this",
    "what", "which", "who", "whom", "these", "those", "i", "me", "my",
    "we", "our", "you", "your", "he", "him", "his", "she", "her", "it",
    "its", "they", "them", "their", "about", "up", "also", "like",
    "user", "memory", "content", "really", "think", "know", "want",
    "get", "got", "make", "said", "going", "using", "use", "thing",
})


def _entity_id(name: str) -> str:
    return hashlib.md5(name.encode()).hexdigest()[:16]


def extract_entities(text: str) -> List[Dict[str, str]]:
    """
    Extract entities from text using heuristic rules.

    Returns list of dicts with keys: name, entity_type
    Entity types: person, tech, org, concept, identifier
    """
    entities: List[Dict[str, str]] = []
    seen: Set[str] = set()

    # Capitalized multi-word names (e.g. "Bernard Smith", "Wolf AI")
    for m in re.finditer(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b", text):
        name = m.group(1).strip()
        key = name.lower()
        if key not in seen and len(name) > 2:
            seen.add(key)
            entities.append({"name": name, "entity_type": "person"})

    # Tech identifiers: camelCase, snake_case, version strings
    for m in re.finditer(
        r"\b([a-z]+[A-Z][a-zA-Z]+|[a-z]+_[a-z_]+|[A-Za-z]+[\-\.]\d[\d\.]*[a-z]*)\b",
        text,
    ):
        name = m.group(1).strip()
        key = name.lower()
        if key not in seen and key not in _STOPWORDS and len(name) > 2:
            seen.add(key)
            entities.append({"name": name, "entity_type": "tech"})

    # Known tech terms (languages, frameworks, tools)
    tech_patterns = [
        r"\b(Python|JavaScript|TypeScript|Rust|Go|Java|C\+\+|Ruby|Swift|Kotlin)\b",
        r"\b(React|Vue|Angular|Next\.?js|Django|Flask|FastAPI|Express|Spring)\b",
        r"\b(Docker|Kubernetes|AWS|Azure|GCP|PostgreSQL|MySQL|Redis|MongoDB)\b",
        r"\b(Git|GitHub|GitLab|VS\s?Code|Cursor|Vim|Emacs|IntelliJ)\b",
        r"\b(GPT|Claude|Gemini|LLaMA|Mistral|BERT|FAISS|OpenAI)\b",
        r"\b(Linux|macOS|Windows|Ubuntu|iOS|Android)\b",
        r"\b(SQL|NoSQL|REST|GraphQL|gRPC|WebSocket|HTTP|TCP)\b",
    ]
    for pat in tech_patterns:
        for m in re.finditer(pat, text, re.IGNORECASE):
            name = m.group(1).strip()
            key = name.lower()
            if key not in seen:
                seen.add(key)
                entities.append({"name": name, "entity_type": "tech"})

    # Org names (all-caps 2+ letters, e.g. NASA, IBM)
    for m in re.finditer(r"\b([A-Z]{2,6})\b", text):
        name = m.group(1)
        key = name.lower()
        if key not in seen and key not in _STOPWORDS and len(name) >= 2:
            seen.add(key)
            entities.append({"name": name, "entity_type": "org"})

    # Single capitalized words that are likely proper nouns (names, brands)
    for m in re.finditer(r"(?<![.!?]\s)\b([A-Z][a-z]{2,15})\b", text):
        name = m.group(1)
        key = name.lower()
        if key not in seen and key not in _STOPWORDS:
            seen.add(key)
            entities.append({"name": name, "entity_type": "concept"})

    return entities


class EntityGraph:
    """
    Manages entity nodes and their links to memory chunks.

    Uses the same SQLite connection pool as MemoryStorage for
    zero-overhead integration.
    """

    def __init__(self, storage):
        """
        storage: a MemoryStorage instance (reuses its connection).
        """
        self.storage = storage
        self._ensure_tables()

    def _ensure_tables(self):
        conn = self.storage._conn()
        cursor = conn.cursor()
        cursor.executescript("""
            CREATE TABLE IF NOT EXISTS entity_nodes (
                entity_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                entity_type TEXT DEFAULT 'concept',
                mention_count INTEGER DEFAULT 1
            );

            CREATE TABLE IF NOT EXISTS entity_links (
                memory_id TEXT NOT NULL,
                entity_id TEXT NOT NULL,
                weight REAL DEFAULT 1.0,
                PRIMARY KEY (memory_id, entity_id),
                FOREIGN KEY (memory_id) REFERENCES memories(id),
                FOREIGN KEY (entity_id) REFERENCES entity_nodes(entity_id)
            );

            CREATE INDEX IF NOT EXISTS idx_entity_links_entity
                ON entity_links(entity_id);
            CREATE INDEX IF NOT EXISTS idx_entity_links_memory
                ON entity_links(memory_id);
            CREATE INDEX IF NOT EXISTS idx_entity_name
                ON entity_nodes(name);
        """)
        conn.commit()

    def index_memory(self, memory_id: str, content: str):
        """
        Extract entities from content and create entity→chunk links.
        """
        entities = extract_entities(content)
        if not entities:
            return

        with self.storage._write_lock:
            conn = self.storage._conn()
            for ent in entities:
                eid = _entity_id(ent["name"].lower())
                conn.execute(
                    "INSERT INTO entity_nodes (entity_id, name, entity_type, mention_count) "
                    "VALUES (?, ?, ?, 1) "
                    "ON CONFLICT(entity_id) DO UPDATE SET "
                    "mention_count = mention_count + 1",
                    (eid, ent["name"], ent["entity_type"]),
                )
                conn.execute(
                    "INSERT OR IGNORE INTO entity_links (memory_id, entity_id, weight) "
                    "VALUES (?, ?, 1.0)",
                    (memory_id, eid),
                )
            conn.commit()

    def _ensure_entity(self, entity_id: str, name: str, entity_type: str = "concept"):
        """Ensure an entity node exists, creating it if needed."""
        with self.storage._write_lock:
            conn = self.storage._conn()
            conn.execute(
                "INSERT INTO entity_nodes (entity_id, name, entity_type, mention_count) "
                "VALUES (?, ?, ?, 1) "
                "ON CONFLICT(entity_id) DO UPDATE SET "
                "mention_count = mention_count + 1",
                (entity_id, name, entity_type),
            )
            conn.commit()

    def remove_memory(self, memory_id: str):
        """Remove all entity links for a memory."""
        with self.storage._write_lock:
            conn = self.storage._conn()
            conn.execute(
                "DELETE FROM entity_links WHERE memory_id = ?", (memory_id,)
            )
            conn.commit()

    def expand_from_query(
        self, query: str, limit: int = 50
    ) -> List[Tuple[str, float]]:
        """
        Given a query, extract entities and find all memory IDs linked
        to those entities. Returns (memory_id, score) pairs.

        Score is based on how many query entities the memory shares
        multiplied by the entity's inverse mention frequency (rare
        entities are more informative).
        """
        entities = extract_entities(query)
        if not entities:
            return []

        entity_ids = [_entity_id(e["name"].lower()) for e in entities]
        if not entity_ids:
            return []

        placeholders = ",".join("?" for _ in entity_ids)
        cursor = self.storage._conn().cursor()

        cursor.execute(
            f"SELECT entity_id, mention_count FROM entity_nodes "
            f"WHERE entity_id IN ({placeholders})",
            entity_ids,
        )
        mention_counts = {row[0]: max(1, row[1]) for row in cursor.fetchall()}
        if not mention_counts:
            return []

        found_ids = list(mention_counts.keys())
        placeholders2 = ",".join("?" for _ in found_ids)
        cursor.execute(
            f"SELECT el.memory_id, el.entity_id, el.weight "
            f"FROM entity_links el "
            f"INNER JOIN memories m ON m.id = el.memory_id "
            f"WHERE el.entity_id IN ({placeholders2}) AND m.is_active = 1",
            found_ids,
        )

        memory_scores: Dict[str, float] = {}
        for mid, eid, weight in cursor.fetchall():
            idf = 1.0 / mention_counts.get(eid, 1)
            score = weight * min(1.0, idf * 5.0)
            memory_scores[mid] = memory_scores.get(mid, 0.0) + score

        results = sorted(memory_scores.items(), key=lambda x: x[1], reverse=True)
        return results[:limit]

    def get_entity_count(self) -> int:
        cursor = self.storage._conn().cursor()
        cursor.execute("SELECT COUNT(*) FROM entity_nodes")
        return cursor.fetchone()[0]

    def get_link_count(self) -> int:
        cursor = self.storage._conn().cursor()
        cursor.execute("SELECT COUNT(*) FROM entity_links")
        return cursor.fetchone()[0]

    def get_entities_for_memory(self, memory_id: str) -> List[Dict]:
        """Get all entities linked to a specific memory."""
        cursor = self.storage._conn().cursor()
        cursor.execute(
            "SELECT en.name, en.entity_type, en.mention_count, el.weight "
            "FROM entity_links el "
            "INNER JOIN entity_nodes en ON en.entity_id = el.entity_id "
            "WHERE el.memory_id = ?",
            (memory_id,),
        )
        return [
            {"name": r[0], "type": r[1], "mentions": r[2], "weight": r[3]}
            for r in cursor.fetchall()
        ]
