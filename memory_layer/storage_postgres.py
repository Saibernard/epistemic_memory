"""
PostgreSQL Storage Backend for the Memory Layer.

Same schema as SQLite, but using Postgres for multi-user concurrent access.
Tables are auto-created on first connection.

Requires: pip install psycopg2-binary  (or psycopg2 for production)
"""

import json
import time
import uuid
import threading
from typing import List, Optional, Dict, Set, Tuple

import numpy as np

from .models import (
    Memory, MemoryType, MemoryLink, LinkType, WorkingMemoryItem,
    KnowledgePage, ProvenanceEntry, MemoryVersion,
)


class PostgresStorage:
    """
    PostgreSQL storage backend.

    Thread-safe via connection-per-thread. Supports full concurrent
    read/write access from multiple processes.
    """

    _SCHEMA_VERSION = "4"

    def __init__(self, database_url: str):
        try:
            import psycopg2
        except ImportError:
            raise ImportError(
                "Postgres backend requires psycopg2.\n"
                "Install it: pip install 'memory-layer[postgres]'  (or: pip install psycopg2-binary)"
            )

        self._database_url = database_url
        self._local = threading.local()
        self._ensure_schema()

    def _conn(self):
        import psycopg2
        conn = getattr(self._local, "conn", None)
        if conn is None or conn.closed:
            conn = psycopg2.connect(self._database_url)
            conn.autocommit = False
            self._local.conn = conn
        return conn

    def _execute(self, sql: str, params: tuple = (), *, fetch: bool = False):
        conn = self._conn()
        cur = conn.cursor()
        cur.execute(sql, params)
        if fetch:
            rows = cur.fetchall()
            conn.commit()
            return rows
        conn.commit()

    def _executemany(self, sql: str, rows: list):
        conn = self._conn()
        cur = conn.cursor()
        cur.executemany(sql, rows)
        conn.commit()

    def _fetchone(self, sql: str, params: tuple = ()):
        conn = self._conn()
        cur = conn.cursor()
        cur.execute(sql, params)
        row = cur.fetchone()
        conn.commit()
        return row

    def _fetchall(self, sql: str, params: tuple = ()):
        conn = self._conn()
        cur = conn.cursor()
        cur.execute(sql, params)
        rows = cur.fetchall()
        conn.commit()
        return rows

    # ─────────────────────────────────────────────
    # SCHEMA
    # ─────────────────────────────────────────────

    def _ensure_schema(self):
        conn = self._conn()
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY,
                memory_type TEXT NOT NULL,
                content TEXT NOT NULL,
                embedding BYTEA,
                created_at DOUBLE PRECISION NOT NULL,
                last_accessed DOUBLE PRECISION NOT NULL,
                access_count INTEGER DEFAULT 0,
                strength DOUBLE PRECISION DEFAULT 1.0,
                importance DOUBLE PRECISION DEFAULT 0.5,
                metadata TEXT DEFAULT '{}',
                tags TEXT DEFAULT '[]',
                source_episode_ids TEXT DEFAULT '[]',
                is_active INTEGER DEFAULT 1,
                namespace TEXT DEFAULT 'default'
            );

            CREATE TABLE IF NOT EXISTS memory_links (
                id TEXT PRIMARY KEY,
                source_id TEXT NOT NULL REFERENCES memories(id),
                target_id TEXT NOT NULL REFERENCES memories(id),
                link_type TEXT NOT NULL,
                weight DOUBLE PRECISION DEFAULT 0.5,
                created_at DOUBLE PRECISION NOT NULL
            );

            CREATE TABLE IF NOT EXISTS working_memory (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                role TEXT DEFAULT 'user',
                created_at DOUBLE PRECISION NOT NULL,
                metadata TEXT DEFAULT '{}'
            );

            CREATE TABLE IF NOT EXISTS consolidation_log (
                id TEXT PRIMARY KEY,
                source_ids TEXT NOT NULL,
                result_id TEXT NOT NULL,
                created_at DOUBLE PRECISION NOT NULL,
                strategy TEXT DEFAULT 'pattern_extraction'
            );

            CREATE TABLE IF NOT EXISTS _metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS memory_passages (
                id TEXT PRIMARY KEY,
                memory_id TEXT NOT NULL REFERENCES memories(id),
                chunk_index INTEGER NOT NULL,
                content_preview TEXT NOT NULL,
                embedding BYTEA NOT NULL
            );
        """)

        # New columns for confidence/epistemic (idempotent via DO NOTHING pattern)
        for col, default in [
            ("confidence", "0.5"),
            ("epistemic_status", "'inferred'"),
        ]:
            try:
                cur.execute(
                    f"ALTER TABLE memories ADD COLUMN {col} "
                    f"{'DOUBLE PRECISION' if col == 'confidence' else 'TEXT'} DEFAULT {default}"
                )
            except Exception:
                conn.rollback()

        cur.execute("""
            CREATE TABLE IF NOT EXISTS knowledge_pages (
                page_id TEXT PRIMARY KEY,
                entity_id TEXT NOT NULL,
                title TEXT NOT NULL,
                page_type TEXT DEFAULT 'entity',
                summary TEXT DEFAULT '',
                version INTEGER DEFAULT 1,
                last_updated DOUBLE PRECISION NOT NULL,
                created_at DOUBLE PRECISION NOT NULL,
                metadata TEXT DEFAULT '{}'
            );
            CREATE TABLE IF NOT EXISTS knowledge_page_memories (
                page_id TEXT NOT NULL REFERENCES knowledge_pages(page_id),
                memory_id TEXT NOT NULL REFERENCES memories(id),
                PRIMARY KEY (page_id, memory_id)
            );
            CREATE TABLE IF NOT EXISTS provenance_log (
                id TEXT PRIMARY KEY,
                memory_id TEXT NOT NULL,
                parent_memory_ids TEXT DEFAULT '[]',
                operation TEXT NOT NULL,
                reason TEXT DEFAULT '',
                source_url TEXT DEFAULT '',
                created_at DOUBLE PRECISION NOT NULL
            );
            CREATE TABLE IF NOT EXISTS memory_versions (
                version_id TEXT PRIMARY KEY,
                memory_id TEXT NOT NULL,
                content TEXT NOT NULL,
                strength DOUBLE PRECISION DEFAULT 1.0,
                importance DOUBLE PRECISION DEFAULT 0.5,
                confidence DOUBLE PRECISION DEFAULT 0.5,
                changed_at DOUBLE PRECISION NOT NULL,
                change_reason TEXT DEFAULT ''
            );
        """)
        conn.commit()

        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_pg_mem_type ON memories(memory_type);
            CREATE INDEX IF NOT EXISTS idx_pg_mem_active ON memories(is_active);
            CREATE INDEX IF NOT EXISTS idx_pg_mem_strength ON memories(strength);
            CREATE INDEX IF NOT EXISTS idx_pg_mem_ns ON memories(namespace);
            CREATE INDEX IF NOT EXISTS idx_pg_mem_accessed ON memories(last_accessed);
            CREATE INDEX IF NOT EXISTS idx_pg_mem_confidence ON memories(confidence);
            CREATE INDEX IF NOT EXISTS idx_pg_mem_epistemic ON memories(epistemic_status);
            CREATE INDEX IF NOT EXISTS idx_pg_link_src ON memory_links(source_id);
            CREATE INDEX IF NOT EXISTS idx_pg_link_tgt ON memory_links(target_id);
            CREATE INDEX IF NOT EXISTS idx_pg_wm_created ON working_memory(created_at);
            CREATE INDEX IF NOT EXISTS idx_pg_pass_mid ON memory_passages(memory_id);
            CREATE INDEX IF NOT EXISTS idx_pg_kp_entity ON knowledge_pages(entity_id);
            CREATE INDEX IF NOT EXISTS idx_pg_kp_type ON knowledge_pages(page_type);
            CREATE INDEX IF NOT EXISTS idx_pg_kp_title ON knowledge_pages(title);
            CREATE INDEX IF NOT EXISTS idx_pg_kpm_mem ON knowledge_page_memories(memory_id);
            CREATE INDEX IF NOT EXISTS idx_pg_prov_mem ON provenance_log(memory_id);
            CREATE INDEX IF NOT EXISTS idx_pg_prov_op ON provenance_log(operation);
            CREATE INDEX IF NOT EXISTS idx_pg_mv_mem ON memory_versions(memory_id);
            CREATE INDEX IF NOT EXISTS idx_pg_mv_changed ON memory_versions(changed_at);
        """)
        conn.commit()

        stored = self.get_meta("schema_version")
        if stored != self._SCHEMA_VERSION:
            self.set_meta("schema_version", self._SCHEMA_VERSION)

    # ─────────────────────────────────────────────
    # METADATA
    # ─────────────────────────────────────────────

    def get_meta(self, key: str) -> Optional[str]:
        row = self._fetchone("SELECT value FROM _metadata WHERE key = %s", (key,))
        return row[0] if row else None

    def set_meta(self, key: str, value: str) -> None:
        self._execute(
            "INSERT INTO _metadata (key, value) VALUES (%s, %s) "
            "ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value",
            (key, value),
        )

    def has_memories(self) -> bool:
        row = self._fetchone("SELECT COUNT(*) FROM memories WHERE is_active = 1")
        return row[0] > 0

    def count_active_memories_with_embeddings(self) -> int:
        row = self._fetchone(
            "SELECT COUNT(*) FROM memories WHERE is_active = 1 AND embedding IS NOT NULL"
        )
        return row[0]

    def get_sample_embedding_dimension(self) -> Optional[int]:
        row = self._fetchone(
            "SELECT embedding FROM memories WHERE is_active = 1 AND embedding IS NOT NULL LIMIT 1"
        )
        if row and row[0]:
            data = bytes(row[0]) if not isinstance(row[0], bytes) else row[0]
            return len(np.frombuffer(data, dtype=np.float32))
        return None

    def clear_all_embeddings(self) -> None:
        self._execute("UPDATE memories SET embedding = NULL")
        self._execute("DELETE FROM memory_passages")

    # ─────────────────────────────────────────────
    # MEMORY CRUD
    # ─────────────────────────────────────────────

    def _row_to_memory(self, row) -> Memory:
        embedding = None
        if row[3]:
            data = bytes(row[3]) if not isinstance(row[3], bytes) else row[3]
            embedding = np.frombuffer(data, dtype=np.float32).copy().tolist()

        return Memory(
            id=row[0],
            memory_type=MemoryType(row[1]),
            content=row[2],
            embedding=embedding,
            created_at=row[4],
            last_accessed=row[5],
            access_count=row[6],
            strength=row[7],
            importance=row[8],
            metadata=json.loads(row[9]) if row[9] else {},
            tags=json.loads(row[10]) if row[10] else [],
            source_episode_ids=json.loads(row[11]) if row[11] else [],
            is_active=bool(row[12]),
            namespace=row[13] if len(row) > 13 else "default",
        )

    def store_memory(self, memory: Memory) -> str:
        emb_blob = None
        if memory.embedding:
            emb_blob = np.array(memory.embedding, dtype=np.float32).tobytes()

        self._execute("""
            INSERT INTO memories
            (id, memory_type, content, embedding, created_at, last_accessed,
             access_count, strength, importance, metadata, tags,
             source_episode_ids, is_active, namespace)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (id) DO UPDATE SET
                memory_type = EXCLUDED.memory_type,
                content = EXCLUDED.content,
                embedding = EXCLUDED.embedding,
                last_accessed = EXCLUDED.last_accessed,
                access_count = EXCLUDED.access_count,
                strength = EXCLUDED.strength,
                importance = EXCLUDED.importance,
                metadata = EXCLUDED.metadata,
                tags = EXCLUDED.tags,
                source_episode_ids = EXCLUDED.source_episode_ids,
                is_active = EXCLUDED.is_active,
                namespace = EXCLUDED.namespace
        """, (
            memory.id, memory.memory_type.value, memory.content, emb_blob,
            memory.created_at, memory.last_accessed, memory.access_count,
            memory.strength, memory.importance,
            json.dumps(memory.metadata), json.dumps(memory.tags),
            json.dumps(memory.source_episode_ids),
            1 if memory.is_active else 0, memory.namespace,
        ))
        return memory.id

    def get_memory(self, memory_id: str) -> Optional[Memory]:
        row = self._fetchone("SELECT * FROM memories WHERE id = %s", (memory_id,))
        return self._row_to_memory(row) if row else None

    def get_memories_by_ids(self, memory_ids: List[str]) -> List[Memory]:
        if not memory_ids:
            return []
        placeholders = ",".join(["%s"] * len(memory_ids))
        rows = self._fetchall(
            f"SELECT * FROM memories WHERE id IN ({placeholders})", tuple(memory_ids)
        )
        return [self._row_to_memory(r) for r in rows]

    def get_all_memories(
        self,
        memory_type: Optional[MemoryType] = None,
        active_only: bool = True,
        namespace: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> List[Memory]:
        query = "SELECT * FROM memories WHERE 1=1"
        params: list = []

        if active_only:
            query += " AND is_active = 1"
        if memory_type:
            query += " AND memory_type = %s"
            params.append(memory_type.value)
        if namespace is not None:
            query += " AND namespace = %s"
            params.append(namespace)

        query += " ORDER BY last_accessed DESC"
        rows = self._fetchall(query, tuple(params))
        memories = [self._row_to_memory(r) for r in rows]

        if tags:
            tag_set = set(tags)
            memories = [m for m in memories if tag_set & set(m.tags)]

        return memories

    def update_memory(self, memory: Memory) -> None:
        self.store_memory(memory)

    def deactivate_memory(self, memory_id: str) -> None:
        self._execute("UPDATE memories SET is_active = 0 WHERE id = %s", (memory_id,))
        self._execute("DELETE FROM memory_passages WHERE memory_id = %s", (memory_id,))

    def forget_memory(self, memory_id: str, hard: bool = False) -> None:
        if hard:
            self._execute("DELETE FROM memory_passages WHERE memory_id = %s", (memory_id,))
            self._execute(
                "DELETE FROM memory_links WHERE source_id = %s OR target_id = %s",
                (memory_id, memory_id),
            )
            self._execute("DELETE FROM memories WHERE id = %s", (memory_id,))
        else:
            self.deactivate_memory(memory_id)

    def bulk_forget(
        self,
        *,
        namespace: Optional[str] = None,
        tags: Optional[List[str]] = None,
        memory_type: Optional[MemoryType] = None,
        older_than_hours: Optional[float] = None,
        hard: bool = False,
    ) -> int:
        if not any([namespace, tags, memory_type, older_than_hours]):
            raise ValueError("At least one filter is required for bulk_forget")

        where = ["is_active = 1"]
        params: list = []

        if namespace is not None:
            where.append("namespace = %s")
            params.append(namespace)
        if memory_type is not None:
            where.append("memory_type = %s")
            params.append(memory_type.value)
        if older_than_hours is not None:
            cutoff = time.time() - (older_than_hours * 3600)
            where.append("created_at < %s")
            params.append(cutoff)

        clause = " AND ".join(where)
        rows = self._fetchall(f"SELECT id, tags FROM memories WHERE {clause}", tuple(params))

        tag_set = set(tags) if tags else None
        ids_to_forget = []
        for row_id, row_tags in rows:
            if tag_set:
                mem_tags = set(json.loads(row_tags) if row_tags else [])
                if not (tag_set & mem_tags):
                    continue
            ids_to_forget.append(row_id)

        for mid in ids_to_forget:
            self.forget_memory(mid, hard=hard)

        return len(ids_to_forget)

    # ─────────────────────────────────────────────
    # PASSAGES
    # ─────────────────────────────────────────────

    def store_passages(self, memory_id: str, passages: List[Dict]) -> None:
        rows = []
        for p in passages:
            emb_blob = np.array(p["embedding"], dtype=np.float32).tobytes()
            rows.append((
                str(uuid.uuid4()), memory_id, p["chunk_index"],
                p["content_preview"][:500], emb_blob,
            ))
        self._executemany(
            "INSERT INTO memory_passages (id, memory_id, chunk_index, content_preview, embedding) "
            "VALUES (%s, %s, %s, %s, %s) "
            "ON CONFLICT (id) DO NOTHING",
            rows,
        )

    def get_all_passage_embeddings(
        self, min_strength: float = 0.0,
    ) -> List[Tuple[str, np.ndarray]]:
        rows = self._fetchall("""
            SELECT p.memory_id, p.embedding
            FROM memory_passages p
            INNER JOIN memories m ON m.id = p.memory_id
            WHERE m.is_active = 1 AND m.strength >= %s
        """, (min_strength,))
        results = []
        for mid, blob in rows:
            data = bytes(blob) if not isinstance(blob, bytes) else blob
            emb = np.frombuffer(data, dtype=np.float32).copy()
            results.append((mid, emb))
        return results

    def delete_passages_for_memory(self, memory_id: str) -> None:
        self._execute("DELETE FROM memory_passages WHERE memory_id = %s", (memory_id,))

    def get_memories_with_embeddings(
        self,
        memory_type: Optional[MemoryType] = None,
        min_strength: float = 0.0,
        namespace: Optional[str] = None,
    ) -> List[Tuple[Memory, np.ndarray]]:
        query = (
            "SELECT * FROM memories "
            "WHERE is_active = 1 AND embedding IS NOT NULL AND strength >= %s"
        )
        params: list = [min_strength]

        if memory_type:
            query += " AND memory_type = %s"
            params.append(memory_type.value)
        if namespace is not None:
            query += " AND namespace = %s"
            params.append(namespace)

        rows = self._fetchall(query, tuple(params))
        results = []
        for row in rows:
            memory = self._row_to_memory(row)
            if row[3]:
                data = bytes(row[3]) if not isinstance(row[3], bytes) else row[3]
                emb = np.frombuffer(data, dtype=np.float32).copy()
                results.append((memory, emb))
        return results

    # ─────────────────────────────────────────────
    # LINKS
    # ─────────────────────────────────────────────

    def _row_to_link(self, r) -> MemoryLink:
        return MemoryLink(
            id=r[0], source_id=r[1], target_id=r[2],
            link_type=LinkType(r[3]), weight=r[4], created_at=r[5],
        )

    def store_link(self, link: MemoryLink) -> str:
        self._execute("""
            INSERT INTO memory_links (id, source_id, target_id, link_type, weight, created_at)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (id) DO UPDATE SET weight = EXCLUDED.weight
        """, (
            link.id, link.source_id, link.target_id,
            link.link_type.value, link.weight, link.created_at,
        ))
        return link.id

    def get_links_for(self, memory_id: str) -> List[MemoryLink]:
        rows = self._fetchall(
            "SELECT * FROM memory_links WHERE source_id = %s OR target_id = %s",
            (memory_id, memory_id),
        )
        return [self._row_to_link(r) for r in rows]

    def get_links_for_ids(self, memory_ids: List[str]) -> List[MemoryLink]:
        if not memory_ids:
            return []
        ph = ",".join(["%s"] * len(memory_ids))
        rows = self._fetchall(
            f"SELECT * FROM memory_links WHERE source_id IN ({ph}) OR target_id IN ({ph})",
            tuple(memory_ids) + tuple(memory_ids),
        )
        return [self._row_to_link(r) for r in rows]

    def get_all_links(self) -> List[MemoryLink]:
        rows = self._fetchall("SELECT * FROM memory_links")
        return [self._row_to_link(r) for r in rows]

    # ─────────────────────────────────────────────
    # WORKING MEMORY
    # ─────────────────────────────────────────────

    def store_working_item(self, item: WorkingMemoryItem) -> None:
        self._execute("""
            INSERT INTO working_memory (id, content, role, created_at, metadata)
            VALUES (%s, %s, %s, %s, %s)
        """, (
            item.id, item.content, item.role,
            item.created_at, json.dumps(item.metadata),
        ))

    def get_working_memory(self, limit: int = 20) -> List[WorkingMemoryItem]:
        rows = self._fetchall(
            "SELECT * FROM working_memory ORDER BY created_at DESC LIMIT %s", (limit,)
        )
        return [
            WorkingMemoryItem(
                id=r[0], content=r[1], role=r[2],
                created_at=r[3], metadata=json.loads(r[4]) if r[4] else {},
            )
            for r in reversed(rows)
        ]

    def clear_working_memory(self) -> None:
        self._execute("DELETE FROM working_memory")

    def trim_working_memory(self, keep_last: int = 20) -> None:
        self._execute("""
            DELETE FROM working_memory WHERE id NOT IN (
                SELECT id FROM working_memory ORDER BY created_at DESC LIMIT %s
            )
        """, (keep_last,))

    # ─────────────────────────────────────────────
    # CONSOLIDATION
    # ─────────────────────────────────────────────

    def log_consolidation(
        self,
        consolidation_id: str,
        source_ids: List[str],
        result_id: str,
        strategy: str = "pattern_extraction",
    ) -> None:
        self._execute("""
            INSERT INTO consolidation_log (id, source_ids, result_id, created_at, strategy)
            VALUES (%s, %s, %s, %s, %s)
        """, (
            consolidation_id, json.dumps(source_ids),
            result_id, time.time(), strategy,
        ))

    def get_consolidated_episode_ids(self) -> Set[str]:
        rows = self._fetchall("SELECT source_ids FROM consolidation_log")
        ids: Set[str] = set()
        for (blob,) in rows:
            try:
                ids.update(json.loads(blob))
            except (json.JSONDecodeError, TypeError):
                pass
        return ids

    def get_consolidation_count(self) -> int:
        row = self._fetchone("SELECT COUNT(*) FROM consolidation_log")
        return row[0] if row else 0

    # ─────────────────────────────────────────────
    # STATISTICS
    # ─────────────────────────────────────────────

    def get_counts(self, namespace: Optional[str] = None) -> Dict[str, int]:
        ns_filter = ""
        params: list = []
        if namespace is not None:
            ns_filter = " AND namespace = %s"
            params = [namespace]

        counts: Dict[str, int] = {}
        row = self._fetchone(
            f"SELECT COUNT(*) FROM memories WHERE is_active = 1{ns_filter}",
            tuple(params),
        )
        counts["total"] = row[0]

        for mt in MemoryType:
            row = self._fetchone(
                f"SELECT COUNT(*) FROM memories WHERE is_active = 1 AND memory_type = %s{ns_filter}",
                (mt.value,) + tuple(params),
            )
            counts[mt.value] = row[0]

        row = self._fetchone("SELECT COUNT(*) FROM memory_links")
        counts["links"] = row[0]

        row = self._fetchone("SELECT COUNT(*) FROM working_memory")
        counts["working"] = row[0]

        return counts

    def get_avg_strength(self) -> float:
        row = self._fetchone("SELECT AVG(strength) FROM memories WHERE is_active = 1")
        return float(row[0]) if row and row[0] else 0.0

    def get_avg_importance(self) -> float:
        row = self._fetchone("SELECT AVG(importance) FROM memories WHERE is_active = 1")
        return float(row[0]) if row and row[0] else 0.0

    def get_oldest_memory_age_hours(self) -> float:
        row = self._fetchone("SELECT MIN(created_at) FROM memories WHERE is_active = 1")
        if row and row[0]:
            return (time.time() - row[0]) / 3600.0
        return 0.0

    def get_most_accessed_memory_id(self) -> Optional[str]:
        row = self._fetchone(
            "SELECT id FROM memories WHERE is_active = 1 ORDER BY access_count DESC LIMIT 1"
        )
        return row[0] if row else None

    # ─────────────────────────────────────────────
    # KNOWLEDGE PAGES
    # ─────────────────────────────────────────────

    def store_knowledge_page(self, page: KnowledgePage) -> str:
        self._execute(
            "INSERT INTO knowledge_pages "
            "(page_id, entity_id, title, page_type, summary, version, "
            "last_updated, created_at, metadata) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s) "
            "ON CONFLICT (page_id) DO UPDATE SET "
            "title=EXCLUDED.title, summary=EXCLUDED.summary, version=EXCLUDED.version, "
            "last_updated=EXCLUDED.last_updated, metadata=EXCLUDED.metadata",
            (
                page.page_id, page.entity_id, page.title, page.page_type,
                page.summary, page.version, page.last_updated,
                page.created_at, json.dumps(page.metadata),
            ),
        )
        return page.page_id

    def _row_to_knowledge_page(self, row) -> KnowledgePage:
        return KnowledgePage(
            page_id=row[0], entity_id=row[1], title=row[2],
            page_type=row[3] or "entity", summary=row[4] or "",
            version=row[5] or 1, last_updated=row[6],
            created_at=row[7], metadata=json.loads(row[8]) if row[8] else {},
        )

    def get_knowledge_page(self, page_id: str) -> Optional[KnowledgePage]:
        row = self._fetchone("SELECT * FROM knowledge_pages WHERE page_id = %s", (page_id,))
        return self._row_to_knowledge_page(row) if row else None

    def get_knowledge_page_by_entity(self, entity_id: str) -> Optional[KnowledgePage]:
        row = self._fetchone(
            "SELECT * FROM knowledge_pages WHERE entity_id = %s", (entity_id,)
        )
        return self._row_to_knowledge_page(row) if row else None

    def get_knowledge_page_by_title(self, title: str) -> Optional[KnowledgePage]:
        row = self._fetchone(
            "SELECT * FROM knowledge_pages WHERE LOWER(title) = LOWER(%s)", (title,)
        )
        return self._row_to_knowledge_page(row) if row else None

    def get_all_knowledge_pages(self, page_type: Optional[str] = None) -> List[KnowledgePage]:
        if page_type:
            rows = self._fetchall(
                "SELECT * FROM knowledge_pages WHERE page_type = %s ORDER BY last_updated DESC",
                (page_type,),
            )
        else:
            rows = self._fetchall("SELECT * FROM knowledge_pages ORDER BY last_updated DESC")
        return [self._row_to_knowledge_page(r) for r in rows]

    def update_knowledge_page(self, page: KnowledgePage) -> None:
        self.store_knowledge_page(page)

    def delete_knowledge_page(self, page_id: str) -> None:
        self._execute("DELETE FROM knowledge_page_memories WHERE page_id = %s", (page_id,))
        self._execute("DELETE FROM knowledge_pages WHERE page_id = %s", (page_id,))

    def link_memory_to_page(self, page_id: str, memory_id: str) -> None:
        self._execute(
            "INSERT INTO knowledge_page_memories (page_id, memory_id) VALUES (%s, %s) "
            "ON CONFLICT DO NOTHING",
            (page_id, memory_id),
        )

    def get_memories_for_page(self, page_id: str) -> List[str]:
        rows = self._fetchall(
            "SELECT memory_id FROM knowledge_page_memories WHERE page_id = %s",
            (page_id,),
        )
        return [r[0] for r in rows]

    def get_pages_for_memory(self, memory_id: str) -> List[KnowledgePage]:
        rows = self._fetchall(
            "SELECT kp.* FROM knowledge_pages kp "
            "JOIN knowledge_page_memories kpm ON kp.page_id = kpm.page_id "
            "WHERE kpm.memory_id = %s",
            (memory_id,),
        )
        return [self._row_to_knowledge_page(r) for r in rows]

    # ─────────────────────────────────────────────
    # PROVENANCE
    # ─────────────────────────────────────────────

    def store_provenance(self, entry: ProvenanceEntry) -> str:
        self._execute(
            "INSERT INTO provenance_log "
            "(id, memory_id, parent_memory_ids, operation, reason, source_url, created_at) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s)",
            (
                entry.id, entry.memory_id,
                json.dumps(entry.parent_memory_ids),
                entry.operation, entry.reason,
                entry.source_url, entry.created_at,
            ),
        )
        return entry.id

    def _row_to_provenance(self, row) -> ProvenanceEntry:
        return ProvenanceEntry(
            id=row[0], memory_id=row[1],
            parent_memory_ids=json.loads(row[2]) if row[2] else [],
            operation=row[3], reason=row[4] or "",
            source_url=row[5] or "", created_at=row[6],
        )

    def get_provenance(self, memory_id: str) -> List[ProvenanceEntry]:
        rows = self._fetchall(
            "SELECT * FROM provenance_log WHERE memory_id = %s ORDER BY created_at ASC",
            (memory_id,),
        )
        return [self._row_to_provenance(r) for r in rows]

    def get_provenance_chain(self, memory_id: str) -> List[ProvenanceEntry]:
        visited: set = set()
        queue = [memory_id]
        all_entries: List[ProvenanceEntry] = []
        for _ in range(10):
            if not queue:
                break
            next_queue: list = []
            for mid in queue:
                if mid in visited:
                    continue
                visited.add(mid)
                entries = self.get_provenance(mid)
                all_entries.extend(entries)
                for e in entries:
                    for pid in e.parent_memory_ids:
                        if pid not in visited:
                            next_queue.append(pid)
            queue = next_queue
        all_entries.sort(key=lambda e: e.created_at)
        return all_entries

    # ─────────────────────────────────────────────
    # MEMORY VERSIONS
    # ─────────────────────────────────────────────

    def store_memory_version(self, version: MemoryVersion) -> str:
        self._execute(
            "INSERT INTO memory_versions "
            "(version_id, memory_id, content, strength, importance, confidence, "
            "changed_at, change_reason) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
            (
                version.version_id, version.memory_id, version.content,
                version.strength, version.importance, version.confidence,
                version.changed_at, version.change_reason,
            ),
        )
        return version.version_id

    def get_version_history(self, memory_id: str) -> List[MemoryVersion]:
        rows = self._fetchall(
            "SELECT * FROM memory_versions WHERE memory_id = %s ORDER BY changed_at ASC",
            (memory_id,),
        )
        return [
            MemoryVersion(
                version_id=r[0], memory_id=r[1], content=r[2],
                strength=r[3], importance=r[4], confidence=r[5],
                changed_at=r[6], change_reason=r[7] or "",
            )
            for r in rows
        ]

    # ─────────────────────────────────────────────
    # LINT HELPERS
    # ──────────────────────────────────────────���──

    def get_stale_memories(self, max_age_days: int = 14) -> List[Memory]:
        cutoff = time.time() - (max_age_days * 86400)
        rows = self._fetchall(
            "SELECT * FROM memories "
            "WHERE is_active = 1 AND access_count = 0 AND created_at < %s",
            (cutoff,),
        )
        return [self._row_to_memory(r) for r in rows]

    def get_orphan_memories(self) -> List[Memory]:
        rows = self._fetchall(
            "SELECT m.* FROM memories m "
            "WHERE m.is_active = 1 "
            "AND NOT EXISTS ("
            "  SELECT 1 FROM memory_links ml "
            "  WHERE ml.source_id = m.id OR ml.target_id = m.id"
            ")"
        )
        return [self._row_to_memory(r) for r in rows]
