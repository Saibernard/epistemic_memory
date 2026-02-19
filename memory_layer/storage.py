"""
SQLite Storage Layer for the Memory Layer.

All memories are persisted locally in a single SQLite database file.
Zero configuration, fully portable - just copy the .db file to move
your entire memory to another machine.

Production features:
- WAL mode for concurrent read/write safety
- Atomic multi-row transactions via transaction() context manager
- Auto-backup on configurable interval
- Startup integrity check and repair
- Growth management (stats, pruning helpers)
"""

import sqlite3
import json
import os
import shutil
import time
import threading
import numpy as np
from contextlib import contextmanager
from typing import List, Optional, Dict, Set, Tuple, Any
from pathlib import Path

from .models import (
    Memory, MemoryType, MemoryLink, LinkType, WorkingMemoryItem,
    KnowledgePage, ProvenanceEntry, MemoryVersion,
)


class MemoryStorage:
    """
    SQLite-based persistent storage for all memory types.

    Uses thread-local connections for safe concurrent access and WAL mode
    for reader/writer concurrency.

    Schema:
    - memories: All long-term memories (episodic, semantic, procedural)
    - memory_links: Associations between memories
    - working_memory: Short-term context buffer
    - consolidation_log: Record of memory consolidation events
    - _metadata: System metadata (embedding model info, schema version)
    - memory_passages: Passage-level embeddings for long memories
    """

    _SCHEMA_VERSION = "4"

    _BACKUP_INTERVAL = 500  # auto-backup every N write operations

    def __init__(self, db_path: str = "memory.db"):
        self.db_path = db_path
        self._local = threading.local()
        self._write_lock = threading.Lock()
        self._write_count = 0
        self._last_backup_time = 0.0
        self._ensure_db()

    # ─────────────────────────────────────────────
    # CONNECTION POOL (thread-local)
    # ─────────────────────────────────────────────

    def _conn(self) -> sqlite3.Connection:
        """Return a thread-local connection, creating one if needed."""
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self.db_path, timeout=30)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA foreign_keys=ON")
            conn.execute("PRAGMA busy_timeout=5000")
            self._local.conn = conn
        return conn

    def _write(self, sql: str, params: tuple = (), *, script: bool = False):
        """Execute a write operation under the write lock."""
        with self._write_lock:
            conn = self._conn()
            cursor = conn.cursor()
            if script:
                cursor.executescript(sql)
            else:
                cursor.execute(sql, params)
            conn.commit()

    def _writemany(self, sql: str, rows: List[tuple]):
        """Execute multiple write operations under the write lock."""
        with self._write_lock:
            conn = self._conn()
            cursor = conn.cursor()
            cursor.executemany(sql, rows)
            conn.commit()

    @contextmanager
    def transaction(self):
        """
        Atomic multi-statement transaction. All writes inside the block
        are committed together or rolled back on error.

        Usage:
            with storage.transaction() as conn:
                conn.execute("INSERT INTO ...", (...))
                conn.execute("INSERT INTO ...", (...))
        """
        with self._write_lock:
            conn = self._conn()
            conn.execute("BEGIN IMMEDIATE")
            try:
                yield conn
                conn.commit()
                self._write_count += 1
                if self._write_count % self._BACKUP_INTERVAL == 0:
                    self._auto_backup()
            except Exception:
                conn.rollback()
                raise

    # ─────────────────────────────────────────────
    # BACKUP & RECOVERY
    # ─────────────────────────────────────────────

    def backup(self, dest_path: str = None) -> str:
        """
        Create a backup of the database. Returns the backup file path.
        Uses SQLite's built-in backup API for a consistent snapshot.
        """
        if dest_path is None:
            ts = time.strftime("%Y%m%d_%H%M%S")
            base = self.db_path.rsplit(".", 1)[0] if "." in self.db_path else self.db_path
            dest_path = f"{base}_backup_{ts}.db"

        src_conn = self._conn()
        dst_conn = sqlite3.connect(dest_path)
        try:
            src_conn.backup(dst_conn)
        finally:
            dst_conn.close()

        return dest_path

    def _auto_backup(self):
        """Background auto-backup, at most once per 10 minutes."""
        now = time.time()
        if now - self._last_backup_time < 600:
            return
        self._last_backup_time = now
        try:
            base = self.db_path.rsplit(".", 1)[0] if "." in self.db_path else self.db_path
            dest = f"{base}_auto_backup.db"
            src_conn = self._conn()
            dst_conn = sqlite3.connect(dest)
            try:
                src_conn.backup(dst_conn)
            finally:
                dst_conn.close()
        except Exception:
            pass

    # ─────────────────────────────────────────────
    # INTEGRITY CHECK & REPAIR
    # ─────────────────────────────────────────────

    def integrity_check(self) -> Dict[str, Any]:
        """
        Run integrity checks and return a report. Call on startup to
        detect and optionally repair inconsistencies.
        """
        report: Dict[str, Any] = {
            "sqlite_ok": False,
            "orphaned_links": 0,
            "orphaned_passages": 0,
            "dead_peer_cards": 0,
            "null_embeddings": 0,
            "total_memories": 0,
            "active_memories": 0,
        }

        conn = self._conn()

        # SQLite integrity
        try:
            result = conn.execute("PRAGMA integrity_check").fetchone()
            report["sqlite_ok"] = result[0] == "ok"
        except Exception:
            report["sqlite_ok"] = False

        # Memory counts
        row = conn.execute(
            "SELECT COUNT(*), SUM(CASE WHEN is_active = 1 THEN 1 ELSE 0 END) FROM memories"
        ).fetchone()
        report["total_memories"] = row[0] or 0
        report["active_memories"] = row[1] or 0

        # Null embeddings on active memories
        row = conn.execute(
            "SELECT COUNT(*) FROM memories WHERE is_active = 1 AND embedding IS NULL"
        ).fetchone()
        report["null_embeddings"] = row[0] or 0

        # Orphaned links (links pointing to non-existent memories)
        row = conn.execute(
            "SELECT COUNT(*) FROM memory_links ml "
            "WHERE NOT EXISTS (SELECT 1 FROM memories m WHERE m.id = ml.source_id) "
            "OR NOT EXISTS (SELECT 1 FROM memories m WHERE m.id = ml.target_id)"
        ).fetchone()
        report["orphaned_links"] = row[0] or 0

        # Orphaned passages
        row = conn.execute(
            "SELECT COUNT(*) FROM memory_passages mp "
            "WHERE NOT EXISTS (SELECT 1 FROM memories m WHERE m.id = mp.memory_id)"
        ).fetchone()
        report["orphaned_passages"] = row[0] or 0

        # Dead peer cards (non-current reasoning memories)
        try:
            row = conn.execute(
                "SELECT COUNT(*) FROM memories "
                "WHERE is_active = 1 AND is_current = 0 AND tags LIKE '%peer_card%'"
            ).fetchone()
            report["dead_peer_cards"] = row[0] or 0
        except Exception:
            pass

        return report

    def repair(self) -> Dict[str, int]:
        """
        Repair common inconsistencies found by integrity_check().
        Returns counts of items repaired.
        """
        repaired = {"orphaned_links": 0, "orphaned_passages": 0, "dead_peer_cards": 0}

        with self._write_lock:
            conn = self._conn()

            # Remove orphaned links
            cursor = conn.execute(
                "DELETE FROM memory_links "
                "WHERE NOT EXISTS (SELECT 1 FROM memories m WHERE m.id = memory_links.source_id) "
                "OR NOT EXISTS (SELECT 1 FROM memories m WHERE m.id = memory_links.target_id)"
            )
            repaired["orphaned_links"] = cursor.rowcount

            # Remove orphaned passages
            cursor = conn.execute(
                "DELETE FROM memory_passages "
                "WHERE NOT EXISTS (SELECT 1 FROM memories m WHERE m.id = memory_passages.memory_id)"
            )
            repaired["orphaned_passages"] = cursor.rowcount

            # Deactivate old dead peer cards (keep only the 3 most recent)
            try:
                ids_to_deactivate = conn.execute(
                    "SELECT id FROM memories "
                    "WHERE is_active = 1 AND is_current = 0 AND tags LIKE '%peer_card%' "
                    "ORDER BY created_at DESC "
                    "LIMIT -1 OFFSET 3"
                ).fetchall()
                if ids_to_deactivate:
                    placeholders = ",".join("?" for _ in ids_to_deactivate)
                    conn.execute(
                        f"UPDATE memories SET is_active = 0 WHERE id IN ({placeholders})",
                        [row[0] for row in ids_to_deactivate],
                    )
                    repaired["dead_peer_cards"] = len(ids_to_deactivate)
            except Exception:
                pass

            conn.commit()

        return repaired

    # ─────────────────────────────────────────────
    # GROWTH MANAGEMENT
    # ─────────────────────────────────────────────

    def get_storage_stats(self) -> Dict[str, Any]:
        """Comprehensive storage statistics for monitoring."""
        conn = self._conn()

        stats: Dict[str, Any] = {}

        row = conn.execute("SELECT COUNT(*) FROM memories WHERE is_active = 1").fetchone()
        stats["active_memories"] = row[0]

        row = conn.execute("SELECT COUNT(*) FROM memories WHERE is_active = 0").fetchone()
        stats["inactive_memories"] = row[0]

        row = conn.execute("SELECT COUNT(*) FROM memory_links").fetchone()
        stats["total_links"] = row[0]

        row = conn.execute("SELECT COUNT(*) FROM working_memory").fetchone()
        stats["working_memory_items"] = row[0]

        row = conn.execute("SELECT COUNT(*) FROM consolidation_log").fetchone()
        stats["consolidation_events"] = row[0]

        # Reasoning conclusions by type
        try:
            for rtype in ("deductive", "inductive", "abductive", "peer_card"):
                row = conn.execute(
                    "SELECT COUNT(*) FROM memories "
                    "WHERE is_active = 1 AND tags LIKE ?",
                    (f'%"{rtype}"%',),
                ).fetchone()
                stats[f"reasoning_{rtype}"] = row[0]
        except Exception:
            pass

        # Reasoning queue
        try:
            row = conn.execute(
                "SELECT COUNT(*), COALESCE(SUM(token_estimate), 0) "
                "FROM reasoning_queue WHERE processed = 0"
            ).fetchone()
            stats["reasoning_queue_pending"] = row[0]
            stats["reasoning_queue_tokens"] = row[1]
        except Exception:
            pass

        # DB file size
        try:
            stats["db_size_mb"] = round(os.path.getsize(self.db_path) / (1024 * 1024), 2)
        except Exception:
            stats["db_size_mb"] = 0

        return stats

    def prune_reasoning_conclusions(self, max_per_type: int = 200) -> int:
        """
        Cap reasoning conclusions per type. Keeps the most recent ones,
        deactivates the rest. Returns count pruned.
        """
        total_pruned = 0
        with self._write_lock:
            conn = self._conn()
            for rtype in ("deductive", "inductive", "abductive"):
                ids_to_prune = conn.execute(
                    "SELECT id FROM memories "
                    "WHERE is_active = 1 AND tags LIKE ? "
                    "ORDER BY created_at DESC "
                    "LIMIT -1 OFFSET ?",
                    (f'%"{rtype}"%', max_per_type),
                ).fetchall()
                if ids_to_prune:
                    placeholders = ",".join("?" for _ in ids_to_prune)
                    conn.execute(
                        f"UPDATE memories SET is_active = 0 WHERE id IN ({placeholders})",
                        [row[0] for row in ids_to_prune],
                    )
                    total_pruned += len(ids_to_prune)
            conn.commit()
        return total_pruned

    def prune_processed_reasoning_queue(self, keep_hours: float = 24) -> int:
        """Remove old processed reasoning queue entries."""
        cutoff = time.time() - (keep_hours * 3600)
        with self._write_lock:
            conn = self._conn()
            cursor = conn.execute(
                "DELETE FROM reasoning_queue WHERE processed = 1 AND created_at < ?",
                (cutoff,),
            )
            conn.commit()
            return cursor.rowcount

    def _ensure_db(self):
        """Create database tables and indices if they don't exist."""
        conn = self._conn()
        cursor = conn.cursor()

        cursor.executescript("""
            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY,
                memory_type TEXT NOT NULL,
                content TEXT NOT NULL,
                embedding BLOB,
                created_at REAL NOT NULL,
                last_accessed REAL NOT NULL,
                access_count INTEGER DEFAULT 0,
                strength REAL DEFAULT 1.0,
                importance REAL DEFAULT 0.5,
                metadata TEXT DEFAULT '{}',
                tags TEXT DEFAULT '[]',
                source_episode_ids TEXT DEFAULT '[]',
                is_active INTEGER DEFAULT 1
            );

            CREATE TABLE IF NOT EXISTS memory_links (
                id TEXT PRIMARY KEY,
                source_id TEXT NOT NULL,
                target_id TEXT NOT NULL,
                link_type TEXT NOT NULL,
                weight REAL DEFAULT 0.5,
                created_at REAL NOT NULL,
                FOREIGN KEY (source_id) REFERENCES memories(id),
                FOREIGN KEY (target_id) REFERENCES memories(id)
            );

            CREATE TABLE IF NOT EXISTS working_memory (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                role TEXT DEFAULT 'user',
                created_at REAL NOT NULL,
                metadata TEXT DEFAULT '{}'
            );

            CREATE TABLE IF NOT EXISTS consolidation_log (
                id TEXT PRIMARY KEY,
                source_ids TEXT NOT NULL,
                result_id TEXT NOT NULL,
                created_at REAL NOT NULL,
                strategy TEXT DEFAULT 'pattern_extraction'
            );

            CREATE TABLE IF NOT EXISTS _metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS memory_passages (
                id TEXT PRIMARY KEY,
                memory_id TEXT NOT NULL,
                chunk_index INTEGER NOT NULL,
                content_preview TEXT NOT NULL,
                embedding BLOB NOT NULL,
                FOREIGN KEY (memory_id) REFERENCES memories(id)
            );

            -- Indices for fast lookups
            CREATE INDEX IF NOT EXISTS idx_memories_type
                ON memories(memory_type);
            CREATE INDEX IF NOT EXISTS idx_memories_active
                ON memories(is_active);
            CREATE INDEX IF NOT EXISTS idx_memories_strength
                ON memories(strength);
            CREATE INDEX IF NOT EXISTS idx_memories_importance
                ON memories(importance);
            CREATE INDEX IF NOT EXISTS idx_memories_last_accessed
                ON memories(last_accessed);
            CREATE INDEX IF NOT EXISTS idx_links_source
                ON memory_links(source_id);
            CREATE INDEX IF NOT EXISTS idx_links_target
                ON memory_links(target_id);
            CREATE INDEX IF NOT EXISTS idx_working_created
                ON working_memory(created_at);
            CREATE INDEX IF NOT EXISTS idx_passages_memory_id
                ON memory_passages(memory_id);
        """)
        conn.commit()

        self._ensure_fts()
        self._run_migrations()

    def _ensure_fts(self):
        """Create FTS5 virtual table for full-text search if not exists."""
        conn = self._conn()
        cursor = conn.cursor()
        try:
            cursor.execute(
                "CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts "
                "USING fts5(content, memory_id UNINDEXED, "
                "tokenize='porter unicode61')"
            )
            conn.commit()
        except Exception:
            pass

    def fts_index_memory(self, memory_id: str, content: str):
        """Add or update a memory in the FTS5 index."""
        with self._write_lock:
            conn = self._conn()
            try:
                conn.execute(
                    "DELETE FROM memories_fts WHERE memory_id = ?", (memory_id,)
                )
                conn.execute(
                    "INSERT INTO memories_fts (content, memory_id) VALUES (?, ?)",
                    (content, memory_id),
                )
                conn.commit()
            except Exception:
                pass

    def fts_search(self, query: str, limit: int = 100) -> List[Tuple[str, float]]:
        """
        Full-text search via FTS5. Returns list of (memory_id, bm25_score).
        Score is negative (closer to 0 = better match) per SQLite BM25.
        """
        try:
            cursor = self._conn().cursor()
            safe_query = " OR ".join(
                t for t in query.split() if t and not t.startswith("-")
            )
            if not safe_query.strip():
                return []
            cursor.execute(
                "SELECT memory_id, bm25(memories_fts) AS score "
                "FROM memories_fts WHERE memories_fts MATCH ? "
                "ORDER BY score LIMIT ?",
                (safe_query, limit),
            )
            return [(row[0], row[1]) for row in cursor.fetchall()]
        except Exception:
            return []

    def fts_remove(self, memory_id: str):
        """Remove a memory from the FTS5 index."""
        with self._write_lock:
            conn = self._conn()
            try:
                conn.execute(
                    "DELETE FROM memories_fts WHERE memory_id = ?", (memory_id,)
                )
                conn.commit()
            except Exception:
                pass

    def fts_rebuild(self):
        """Rebuild FTS5 index from all active memories."""
        with self._write_lock:
            conn = self._conn()
            try:
                conn.execute("DELETE FROM memories_fts")
                conn.execute(
                    "INSERT INTO memories_fts (content, memory_id) "
                    "SELECT content, id FROM memories WHERE is_active = 1"
                )
                conn.commit()
            except Exception:
                pass

    def _run_migrations(self):
        """Apply schema migrations for new columns / tables."""
        conn = self._conn()
        cursor = conn.cursor()

        cursor.execute("PRAGMA table_info(memories)")
        existing_cols = {row[1] for row in cursor.fetchall()}

        if "namespace" not in existing_cols:
            cursor.execute(
                "ALTER TABLE memories ADD COLUMN namespace TEXT DEFAULT 'default'"
            )
            conn.commit()

        if "enriched_content" not in existing_cols:
            cursor.execute(
                "ALTER TABLE memories ADD COLUMN enriched_content TEXT"
            )
            conn.commit()

        if "superseded_by" not in existing_cols:
            cursor.execute(
                "ALTER TABLE memories ADD COLUMN superseded_by TEXT"
            )
            conn.commit()

        if "valid_from" not in existing_cols:
            cursor.execute(
                "ALTER TABLE memories ADD COLUMN valid_from REAL"
            )
            conn.commit()

        if "valid_until" not in existing_cols:
            cursor.execute(
                "ALTER TABLE memories ADD COLUMN valid_until REAL"
            )
            conn.commit()

        # --- Phase 1A: Temporal grounding columns ---
        if "document_date" not in existing_cols:
            cursor.execute("ALTER TABLE memories ADD COLUMN document_date REAL")
            conn.commit()
        if "event_dates" not in existing_cols:
            cursor.execute("ALTER TABLE memories ADD COLUMN event_dates TEXT")
            conn.commit()

        # --- Phase 1E: Append-only versioning ---
        if "is_current" not in existing_cols:
            cursor.execute(
                "ALTER TABLE memories ADD COLUMN is_current INTEGER DEFAULT 1"
            )
            conn.commit()

        # --- Phase 2A: Abstraction hierarchy ---
        if "abstraction_level" not in existing_cols:
            cursor.execute(
                "ALTER TABLE memories ADD COLUMN abstraction_level INTEGER DEFAULT 0"
            )
            conn.commit()

        # --- Phase 4: Confidence & epistemic status ---
        if "confidence" not in existing_cols:
            cursor.execute(
                "ALTER TABLE memories ADD COLUMN confidence REAL DEFAULT 0.5"
            )
            conn.commit()
        if "epistemic_status" not in existing_cols:
            cursor.execute(
                "ALTER TABLE memories ADD COLUMN epistemic_status TEXT DEFAULT 'inferred'"
            )
            conn.commit()

        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_memories_namespace "
            "ON memories(namespace)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_memories_is_current "
            "ON memories(is_current)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_memories_abstraction_level "
            "ON memories(abstraction_level)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_memories_confidence "
            "ON memories(confidence)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_memories_epistemic "
            "ON memories(epistemic_status)"
        )
        conn.commit()

        # --- Phase 1B: Source chunk injection tables ---
        cursor.executescript("""
            CREATE TABLE IF NOT EXISTS source_chunks (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                source_file TEXT,
                chunk_index INTEGER,
                created_at REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS memory_source_map (
                memory_id TEXT NOT NULL,
                chunk_id TEXT NOT NULL,
                PRIMARY KEY (memory_id, chunk_id)
            );
            CREATE INDEX IF NOT EXISTS idx_source_map_chunk
                ON memory_source_map(chunk_id);
        """)
        conn.commit()

        # --- Phase 1D: Typed entity-relationship knowledge graph ---
        cursor.executescript("""
            CREATE TABLE IF NOT EXISTS entity_relationships (
                id TEXT PRIMARY KEY,
                source_entity_id TEXT NOT NULL,
                target_entity_id TEXT NOT NULL,
                relation_type TEXT NOT NULL,
                context TEXT,
                reasoning TEXT,
                document_date REAL,
                event_date REAL,
                valid_from REAL,
                valid_until REAL,
                is_current INTEGER DEFAULT 1,
                memory_id TEXT,
                confidence REAL DEFAULT 1.0,
                created_at REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_entrel_source
                ON entity_relationships(source_entity_id);
            CREATE INDEX IF NOT EXISTS idx_entrel_target
                ON entity_relationships(target_entity_id);
            CREATE INDEX IF NOT EXISTS idx_entrel_type
                ON entity_relationships(relation_type);
            CREATE INDEX IF NOT EXISTS idx_entrel_memory
                ON entity_relationships(memory_id);
            CREATE INDEX IF NOT EXISTS idx_entrel_current
                ON entity_relationships(is_current);
        """)
        conn.commit()

        # --- Reasoning engine queue ---
        cursor.executescript("""
            CREATE TABLE IF NOT EXISTS reasoning_queue (
                id TEXT PRIMARY KEY,
                memory_id TEXT NOT NULL,
                content TEXT NOT NULL,
                token_estimate INTEGER DEFAULT 0,
                created_at REAL,
                processed INTEGER DEFAULT 0
            );
            CREATE INDEX IF NOT EXISTS idx_rq_processed
                ON reasoning_queue(processed);
        """)
        conn.commit()

        # --- Knowledge pages (Karpathy wiki-style) ---
        cursor.executescript("""
            CREATE TABLE IF NOT EXISTS knowledge_pages (
                page_id TEXT PRIMARY KEY,
                entity_id TEXT NOT NULL,
                title TEXT NOT NULL,
                page_type TEXT DEFAULT 'entity',
                summary TEXT DEFAULT '',
                version INTEGER DEFAULT 1,
                last_updated REAL NOT NULL,
                created_at REAL NOT NULL,
                metadata TEXT DEFAULT '{}'
            );
            CREATE INDEX IF NOT EXISTS idx_kp_entity
                ON knowledge_pages(entity_id);
            CREATE INDEX IF NOT EXISTS idx_kp_type
                ON knowledge_pages(page_type);
            CREATE INDEX IF NOT EXISTS idx_kp_title
                ON knowledge_pages(title);

            CREATE TABLE IF NOT EXISTS knowledge_page_memories (
                page_id TEXT NOT NULL,
                memory_id TEXT NOT NULL,
                PRIMARY KEY (page_id, memory_id),
                FOREIGN KEY (page_id) REFERENCES knowledge_pages(page_id),
                FOREIGN KEY (memory_id) REFERENCES memories(id)
            );
            CREATE INDEX IF NOT EXISTS idx_kpm_memory
                ON knowledge_page_memories(memory_id);
        """)
        conn.commit()

        # --- Provenance log ---
        cursor.executescript("""
            CREATE TABLE IF NOT EXISTS provenance_log (
                id TEXT PRIMARY KEY,
                memory_id TEXT NOT NULL,
                parent_memory_ids TEXT DEFAULT '[]',
                operation TEXT NOT NULL,
                reason TEXT DEFAULT '',
                source_url TEXT DEFAULT '',
                created_at REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_prov_memory
                ON provenance_log(memory_id);
            CREATE INDEX IF NOT EXISTS idx_prov_operation
                ON provenance_log(operation);
        """)
        conn.commit()

        # --- Memory versions (snapshots before mutations) ---
        cursor.executescript("""
            CREATE TABLE IF NOT EXISTS memory_versions (
                version_id TEXT PRIMARY KEY,
                memory_id TEXT NOT NULL,
                content TEXT NOT NULL,
                strength REAL DEFAULT 1.0,
                importance REAL DEFAULT 0.5,
                confidence REAL DEFAULT 0.5,
                changed_at REAL NOT NULL,
                change_reason TEXT DEFAULT ''
            );
            CREATE INDEX IF NOT EXISTS idx_mv_memory
                ON memory_versions(memory_id);
            CREATE INDEX IF NOT EXISTS idx_mv_changed
                ON memory_versions(changed_at);
        """)
        conn.commit()

        stored_version = self.get_meta("schema_version")
        if stored_version != self._SCHEMA_VERSION:
            self.set_meta("schema_version", self._SCHEMA_VERSION)

    # ─────────────────────────────────────────────
    # METADATA (model info, schema version, etc.)
    # ─────────────────────────────────────────────

    def get_meta(self, key: str) -> Optional[str]:
        """Get a metadata value by key."""
        cursor = self._conn().cursor()
        cursor.execute("SELECT value FROM _metadata WHERE key = ?", (key,))
        row = cursor.fetchone()
        return row[0] if row else None

    def set_meta(self, key: str, value: str):
        """Set a metadata value."""
        with self._write_lock:
            conn = self._conn()
            conn.execute(
                "INSERT OR REPLACE INTO _metadata (key, value) VALUES (?, ?)",
                (key, value),
            )
            conn.commit()

    def has_memories(self) -> bool:
        """Check if any memories exist in the database."""
        cursor = self._conn().cursor()
        cursor.execute("SELECT COUNT(*) FROM memories WHERE is_active = 1")
        return cursor.fetchone()[0] > 0

    def count_active_memories_with_embeddings(self) -> int:
        """Count active memories that have non-NULL embeddings."""
        cursor = self._conn().cursor()
        cursor.execute(
            "SELECT COUNT(*) FROM memories "
            "WHERE is_active = 1 AND embedding IS NOT NULL"
        )
        return cursor.fetchone()[0]

    def get_sample_embedding_dimension(self) -> Optional[int]:
        """Return the dimension of the first non-NULL embedding, or None."""
        cursor = self._conn().cursor()
        cursor.execute(
            "SELECT embedding FROM memories "
            "WHERE is_active = 1 AND embedding IS NOT NULL LIMIT 1"
        )
        row = cursor.fetchone()
        if row and row[0]:
            return len(np.frombuffer(row[0], dtype=np.float32))
        return None

    def clear_all_embeddings(self):
        """Null-out all embedding blobs (for re-embedding after model change)."""
        with self._write_lock:
            conn = self._conn()
            conn.execute("UPDATE memories SET embedding = NULL")
            conn.execute("DELETE FROM memory_passages")
            conn.commit()

    # ─────────────────────────────────────────────
    # MEMORY CRUD
    # ─────────────────────────────────────────────

    def _memory_to_row(self, memory: Memory) -> Tuple:
        """Convert a Memory to SQL row params. Shared by store_memory and transactional writes."""
        embedding_blob = None
        if memory.embedding:
            embedding_blob = np.array(memory.embedding, dtype=np.float32).tobytes()

        meta = dict(memory.metadata)
        enriched = meta.pop("enriched_content", None)
        superseded_by = meta.pop("superseded_by", None)
        valid_from = meta.pop("valid_from", None)
        valid_until = meta.pop("valid_until", None)

        event_dates_json = (
            json.dumps(memory.event_dates) if memory.event_dates else None
        )

        params = (
            memory.id,
            memory.memory_type.value,
            memory.content,
            embedding_blob,
            memory.created_at,
            memory.last_accessed,
            memory.access_count,
            memory.strength,
            memory.importance,
            json.dumps(meta),
            json.dumps(memory.tags),
            json.dumps(memory.source_episode_ids),
            1 if memory.is_active else 0,
            memory.namespace,
            enriched,
            superseded_by,
            valid_from,
            valid_until,
            memory.document_date,
            event_dates_json,
            1 if memory.is_current else 0,
            memory.abstraction_level,
            memory.confidence,
            memory.epistemic_status,
        )
        return params, enriched

    _STORE_SQL = """
        INSERT OR REPLACE INTO memories
        (id, memory_type, content, embedding, created_at, last_accessed,
         access_count, strength, importance, metadata, tags,
         source_episode_ids, is_active, namespace,
         enriched_content, superseded_by, valid_from, valid_until,
         document_date, event_dates, is_current, abstraction_level,
         confidence, epistemic_status)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                ?, ?, ?, ?, ?, ?)
    """

    _STORE_LINK_SQL = """
        INSERT OR REPLACE INTO memory_links
        (id, source_id, target_id, link_type, weight, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
    """

    def store_memory_on_conn(self, conn: sqlite3.Connection, memory: Memory):
        """Store a memory using an existing connection (for use inside a transaction)."""
        params, _ = self._memory_to_row(memory)
        conn.execute(self._STORE_SQL, params)

    def store_link_on_conn(self, conn: sqlite3.Connection, link: MemoryLink):
        """Store a link using an existing connection (for use inside a transaction)."""
        conn.execute(self._STORE_LINK_SQL, (
            link.id, link.source_id, link.target_id,
            link.link_type.value, link.weight, link.created_at,
        ))

    def store_memory(self, memory: Memory) -> str:
        """Store or update a memory."""
        params, enriched = self._memory_to_row(memory)

        with self._write_lock:
            conn = self._conn()
            conn.execute(self._STORE_SQL, params)
            conn.commit()

        fts_text = enriched or memory.content
        self.fts_index_memory(memory.id, fts_text)

        return memory.id

    def get_memory(self, memory_id: str) -> Optional[Memory]:
        """Retrieve a single memory by ID."""
        cursor = self._conn().cursor()
        cursor.execute("SELECT * FROM memories WHERE id = ?", (memory_id,))
        row = cursor.fetchone()
        if row is None:
            return None
        return self._row_to_memory(row)

    def get_memories_by_ids(self, memory_ids: List[str]) -> List[Memory]:
        """Batch-load memories by a list of IDs (single query, fast)."""
        if not memory_ids:
            return []
        cursor = self._conn().cursor()
        placeholders = ",".join("?" for _ in memory_ids)
        cursor.execute(
            f"SELECT * FROM memories WHERE id IN ({placeholders})", memory_ids,
        )
        return [self._row_to_memory(r) for r in cursor.fetchall()]

    def get_all_memories(
        self,
        memory_type: Optional[MemoryType] = None,
        active_only: bool = True,
        namespace: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> List[Memory]:
        """Retrieve all memories, optionally filtered by type/namespace/tags."""
        query = "SELECT * FROM memories WHERE 1=1"
        params: list = []

        if active_only:
            query += " AND is_active = 1"
        if memory_type:
            query += " AND memory_type = ?"
            params.append(memory_type.value)
        if namespace is not None:
            query += " AND namespace = ?"
            params.append(namespace)

        query += " ORDER BY last_accessed DESC"
        cursor = self._conn().cursor()
        cursor.execute(query, params)
        memories = [self._row_to_memory(r) for r in cursor.fetchall()]

        if tags:
            tag_set = set(tags)
            memories = [m for m in memories if tag_set & set(m.tags)]

        return memories

    def update_memory(self, memory: Memory):
        """Update an existing memory (alias for store_memory)."""
        self.store_memory(memory)

    def deactivate_memory(self, memory_id: str):
        """Soft-delete a memory by marking it inactive and removing its passages."""
        with self._write_lock:
            conn = self._conn()
            conn.execute(
                "UPDATE memories SET is_active = 0 WHERE id = ?", (memory_id,)
            )
            conn.execute(
                "DELETE FROM memory_passages WHERE memory_id = ?", (memory_id,)
            )
            conn.commit()
        self.fts_remove(memory_id)

    def forget_memory(self, memory_id: str, hard: bool = False):
        """
        Forget a memory.

        hard=False: soft-delete (deactivate) — reversible
        hard=True:  permanent delete from DB — irreversible
        """
        if hard:
            with self._write_lock:
                conn = self._conn()
                conn.execute("DELETE FROM memory_passages WHERE memory_id = ?", (memory_id,))
                conn.execute(
                    "DELETE FROM memory_links WHERE source_id = ? OR target_id = ?",
                    (memory_id, memory_id),
                )
                conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
                conn.commit()
            self.fts_remove(memory_id)
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
        """
        Bulk forget memories matching the given filters. Returns count affected.
        At least one filter is required to prevent accidental wipe.
        """
        if not any([namespace, tags, memory_type, older_than_hours]):
            raise ValueError("At least one filter is required for bulk_forget")

        where = ["is_active = 1"]
        params: list = []

        if namespace is not None:
            where.append("namespace = ?")
            params.append(namespace)
        if memory_type is not None:
            where.append("memory_type = ?")
            params.append(memory_type.value)
        if older_than_hours is not None:
            cutoff = time.time() - (older_than_hours * 3600)
            where.append("created_at < ?")
            params.append(cutoff)

        where_clause = " AND ".join(where)

        cursor = self._conn().cursor()
        cursor.execute(f"SELECT id, tags FROM memories WHERE {where_clause}", params)
        rows = cursor.fetchall()

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
    # PASSAGE-LEVEL EMBEDDINGS (for long memories)
    # ─────────────────────────────────────────────

    def store_passages(self, memory_id: str, passages: List[Dict]):
        """
        Store chunk-level embeddings for a long memory.

        Each dict in *passages* must have keys:
            chunk_index (int), content_preview (str), embedding (list[float])
        """
        import uuid as _uuid

        rows = []
        for p in passages:
            emb_blob = np.array(p["embedding"], dtype=np.float32).tobytes()
            rows.append((
                str(_uuid.uuid4()),
                memory_id,
                p["chunk_index"],
                p["content_preview"][:500],
                emb_blob,
            ))
        self._writemany(
            "INSERT OR REPLACE INTO memory_passages "
            "(id, memory_id, chunk_index, content_preview, embedding) "
            "VALUES (?, ?, ?, ?, ?)",
            rows,
        )

    def get_all_passage_embeddings(
        self, min_strength: float = 0.0
    ) -> List[Tuple[str, np.ndarray]]:
        """
        Retrieve all passage embeddings whose *parent* memory is active and
        meets the strength threshold.
        """
        cursor = self._conn().cursor()
        cursor.execute(
            """
            SELECT p.memory_id, p.embedding
            FROM memory_passages p
            INNER JOIN memories m ON m.id = p.memory_id
            WHERE m.is_active = 1 AND m.strength >= ?
            """,
            (min_strength,),
        )
        results: List[Tuple[str, np.ndarray]] = []
        for mid, blob in cursor.fetchall():
            emb = np.frombuffer(blob, dtype=np.float32).copy()
            results.append((mid, emb))
        return results

    def delete_passages_for_memory(self, memory_id: str):
        """Remove all passage embeddings for a given memory."""
        self._write(
            "DELETE FROM memory_passages WHERE memory_id = ?", (memory_id,)
        )

    def get_memories_with_embeddings(
        self,
        memory_type: Optional[MemoryType] = None,
        min_strength: float = 0.0,
        namespace: Optional[str] = None,
    ) -> List[Tuple[Memory, np.ndarray]]:
        """
        Get active memories along with their numpy embedding arrays.
        Used for similarity search operations.
        """
        query = (
            "SELECT * FROM memories "
            "WHERE is_active = 1 AND embedding IS NOT NULL AND strength >= ?"
        )
        params: list = [min_strength]

        if memory_type:
            query += " AND memory_type = ?"
            params.append(memory_type.value)
        if namespace is not None:
            query += " AND namespace = ?"
            params.append(namespace)

        cursor = self._conn().cursor()
        cursor.execute(query, params)
        rows = cursor.fetchall()

        results = []
        for row in rows:
            memory = self._row_to_memory(row)
            if row[3]:  # embedding blob column
                emb = np.frombuffer(row[3], dtype=np.float32).copy()
                results.append((memory, emb))

        return results

    def _row_to_memory(self, row) -> Memory:
        """Convert a database row to a Memory object."""
        embedding = None
        if row[3]:
            embedding = np.frombuffer(row[3], dtype=np.float32).copy().tolist()

        n = len(row)
        namespace = row[13] if n > 13 else "default"
        enriched_content = row[14] if n > 14 else None
        superseded_by = row[15] if n > 15 else None
        valid_from = row[16] if n > 16 else None
        valid_until = row[17] if n > 17 else None
        document_date = row[18] if n > 18 else None
        event_dates_raw = row[19] if n > 19 else None
        is_current_raw = row[20] if n > 20 else 1
        abstraction_level = row[21] if n > 21 else 0
        confidence = row[22] if n > 22 else 0.5
        epistemic_status = row[23] if n > 23 else "inferred"

        event_dates = None
        if event_dates_raw:
            try:
                event_dates = json.loads(event_dates_raw)
            except (json.JSONDecodeError, TypeError):
                pass

        meta = json.loads(row[9]) if row[9] else {}
        if enriched_content:
            meta["enriched_content"] = enriched_content
        if superseded_by:
            meta["superseded_by"] = superseded_by
        if valid_from:
            meta["valid_from"] = valid_from
        if valid_until:
            meta["valid_until"] = valid_until

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
            metadata=meta,
            tags=json.loads(row[10]) if row[10] else [],
            source_episode_ids=json.loads(row[11]) if row[11] else [],
            is_active=bool(row[12]),
            namespace=namespace,
            document_date=document_date,
            event_dates=event_dates,
            is_current=bool(is_current_raw) if is_current_raw is not None else True,
            abstraction_level=abstraction_level or 0,
            confidence=confidence if confidence is not None else 0.5,
            epistemic_status=epistemic_status or "inferred",
        )

    # ─────────────────────────────────────────────
    # MEMORY LINKS
    # ─────────────────────────────────────────────

    def store_link(self, link: MemoryLink) -> str:
        """Store an association link between two memories."""
        with self._write_lock:
            conn = self._conn()
            conn.execute("""
                INSERT OR REPLACE INTO memory_links
                (id, source_id, target_id, link_type, weight, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                link.id, link.source_id, link.target_id,
                link.link_type.value, link.weight, link.created_at,
            ))
            conn.commit()
        return link.id

    def get_links_for(self, memory_id: str) -> List[MemoryLink]:
        """Get all links connected to a specific memory."""
        cursor = self._conn().cursor()
        cursor.execute("""
            SELECT * FROM memory_links
            WHERE source_id = ? OR target_id = ?
        """, (memory_id, memory_id))
        return [
            MemoryLink(
                id=r[0], source_id=r[1], target_id=r[2],
                link_type=LinkType(r[3]), weight=r[4], created_at=r[5],
            )
            for r in cursor.fetchall()
        ]

    def get_links_for_ids(self, memory_ids: List[str]) -> List[MemoryLink]:
        """Batch-get links involving any of the given memory IDs."""
        if not memory_ids:
            return []
        placeholders = ",".join("?" for _ in memory_ids)
        cursor = self._conn().cursor()
        cursor.execute(
            f"SELECT * FROM memory_links "
            f"WHERE source_id IN ({placeholders}) OR target_id IN ({placeholders})",
            memory_ids + memory_ids,
        )
        return [
            MemoryLink(
                id=r[0], source_id=r[1], target_id=r[2],
                link_type=LinkType(r[3]), weight=r[4], created_at=r[5],
            )
            for r in cursor.fetchall()
        ]

    def get_all_links(self) -> List[MemoryLink]:
        """Get all association links in the system."""
        cursor = self._conn().cursor()
        cursor.execute("SELECT * FROM memory_links")
        return [
            MemoryLink(
                id=r[0], source_id=r[1], target_id=r[2],
                link_type=LinkType(r[3]), weight=r[4], created_at=r[5],
            )
            for r in cursor.fetchall()
        ]

    def get_links_by_type(
        self, link_type: LinkType, limit: int = 10000
    ) -> List[MemoryLink]:
        """Get links filtered by type — avoids loading the entire graph."""
        cursor = self._conn().cursor()
        cursor.execute(
            "SELECT * FROM memory_links WHERE link_type = ? LIMIT ?",
            (link_type.value, limit),
        )
        return [
            MemoryLink(
                id=r[0], source_id=r[1], target_id=r[2],
                link_type=LinkType(r[3]), weight=r[4], created_at=r[5],
            )
            for r in cursor.fetchall()
        ]

    # ─────────────────────────────────────────────
    # WORKING MEMORY
    # ─────────────────────────────────────────────

    def store_working_item(self, item: WorkingMemoryItem):
        """Add an item to working memory."""
        with self._write_lock:
            conn = self._conn()
            conn.execute("""
                INSERT INTO working_memory (id, content, role, created_at, metadata)
                VALUES (?, ?, ?, ?, ?)
            """, (
                item.id, item.content, item.role,
                item.created_at, json.dumps(item.metadata),
            ))
            conn.commit()

    def get_working_memory(self, limit: int = 20) -> List[WorkingMemoryItem]:
        """Get recent working memory items (ordered chronologically)."""
        cursor = self._conn().cursor()
        cursor.execute("""
            SELECT * FROM working_memory
            ORDER BY created_at DESC LIMIT ?
        """, (limit,))
        rows = cursor.fetchall()
        return [
            WorkingMemoryItem(
                id=r[0], content=r[1], role=r[2],
                created_at=r[3],
                metadata=json.loads(r[4]) if r[4] else {},
            )
            for r in reversed(rows)
        ]

    def clear_working_memory(self):
        """Clear all items from working memory."""
        self._write("DELETE FROM working_memory")

    def trim_working_memory(self, keep_last: int = 20):
        """Trim working memory to keep only the most recent N items."""
        self._write("""
            DELETE FROM working_memory WHERE id NOT IN (
                SELECT id FROM working_memory
                ORDER BY created_at DESC LIMIT ?
            )
        """, (keep_last,))

    # ─────────────────────────────────────────────
    # CONSOLIDATION LOG
    # ─────────────────────────────────────────────

    def log_consolidation(
        self,
        consolidation_id: str,
        source_ids: List[str],
        result_id: str,
        strategy: str = "pattern_extraction",
    ):
        """Record a consolidation event."""
        with self._write_lock:
            conn = self._conn()
            conn.execute("""
                INSERT INTO consolidation_log
                (id, source_ids, result_id, created_at, strategy)
                VALUES (?, ?, ?, ?, ?)
            """, (
                consolidation_id, json.dumps(source_ids),
                result_id, time.time(), strategy,
            ))
            conn.commit()

    def get_consolidated_episode_ids(self) -> Set[str]:
        """Return the set of episode IDs that have already been consolidated."""
        cursor = self._conn().cursor()
        cursor.execute("SELECT source_ids FROM consolidation_log")
        ids: Set[str] = set()
        for (blob,) in cursor.fetchall():
            try:
                ids.update(json.loads(blob))
            except (json.JSONDecodeError, TypeError):
                pass
        return ids

    def get_consolidation_count(self) -> int:
        """Get the total number of consolidation events."""
        cursor = self._conn().cursor()
        cursor.execute("SELECT COUNT(*) FROM consolidation_log")
        return cursor.fetchone()[0]

    # ─────────────────────────────────────────────
    # STATISTICS
    # ─────────────────────────────────────────────

    def get_counts(self, namespace: Optional[str] = None) -> Dict[str, int]:
        """Get counts of all memory types and links."""
        ns_filter = ""
        params: list = []
        if namespace is not None:
            ns_filter = " AND namespace = ?"
            params = [namespace]

        cursor = self._conn().cursor()
        counts: Dict[str, int] = {}

        cursor.execute(
            f"SELECT COUNT(*) FROM memories WHERE is_active = 1{ns_filter}",
            params,
        )
        counts["total"] = cursor.fetchone()[0]

        for mt in MemoryType:
            cursor.execute(
                f"SELECT COUNT(*) FROM memories "
                f"WHERE is_active = 1 AND memory_type = ?{ns_filter}",
                [mt.value] + params,
            )
            counts[mt.value] = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM memory_links")
        counts["links"] = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM working_memory")
        counts["working"] = cursor.fetchone()[0]

        return counts

    def get_avg_strength(self) -> float:
        cursor = self._conn().cursor()
        cursor.execute("SELECT AVG(strength) FROM memories WHERE is_active = 1")
        result = cursor.fetchone()[0]
        return result or 0.0

    def get_avg_importance(self) -> float:
        cursor = self._conn().cursor()
        cursor.execute("SELECT AVG(importance) FROM memories WHERE is_active = 1")
        result = cursor.fetchone()[0]
        return result or 0.0

    def get_oldest_memory_age_hours(self) -> float:
        cursor = self._conn().cursor()
        cursor.execute("SELECT MIN(created_at) FROM memories WHERE is_active = 1")
        result = cursor.fetchone()[0]
        if result is None:
            return 0.0
        return (time.time() - result) / 3600.0

    def get_most_accessed_memory_id(self) -> Optional[str]:
        cursor = self._conn().cursor()
        cursor.execute(
            "SELECT id FROM memories WHERE is_active = 1 "
            "ORDER BY access_count DESC LIMIT 1"
        )
        result = cursor.fetchone()
        return result[0] if result else None

    # ─────────────────────────────────────────────
    # SOURCE CHUNKS (Phase 1B)
    # ─────────────────────────────────────────────

    def store_source_chunk(
        self, chunk_id: str, content: str,
        source_file: Optional[str] = None, chunk_index: int = 0,
    ) -> str:
        """Store an original source chunk before it was split into atomic facts."""
        with self._write_lock:
            conn = self._conn()
            conn.execute(
                "INSERT OR IGNORE INTO source_chunks "
                "(id, content, source_file, chunk_index, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (chunk_id, content, source_file, chunk_index, time.time()),
            )
            conn.commit()
        return chunk_id

    def link_memory_to_chunk(self, memory_id: str, chunk_id: str):
        """Link a memory to its source chunk."""
        with self._write_lock:
            conn = self._conn()
            conn.execute(
                "INSERT OR IGNORE INTO memory_source_map "
                "(memory_id, chunk_id) VALUES (?, ?)",
                (memory_id, chunk_id),
            )
            conn.commit()

    def get_source_chunks(self, memory_ids: List[str]) -> Dict[str, str]:
        """
        Get source chunk content for the given memory IDs.

        Returns:
            Dict mapping memory_id -> source chunk content.
        """
        if not memory_ids:
            return {}
        placeholders = ",".join("?" for _ in memory_ids)
        cursor = self._conn().cursor()
        cursor.execute(
            f"SELECT m.memory_id, s.content "
            f"FROM memory_source_map m "
            f"JOIN source_chunks s ON s.id = m.chunk_id "
            f"WHERE m.memory_id IN ({placeholders})",
            memory_ids,
        )
        result: Dict[str, str] = {}
        for mid, content in cursor.fetchall():
            if mid not in result:
                result[mid] = content
        return result

    # ─────────────────────────────────────────────
    # ENTITY RELATIONSHIPS (Phase 1D)
    # ─────────────────────────────────────────────

    def store_entity_relationship(self, rel: Dict[str, Any]) -> str:
        """Store a typed entity relationship."""
        with self._write_lock:
            conn = self._conn()
            conn.execute(
                "INSERT OR REPLACE INTO entity_relationships "
                "(id, source_entity_id, target_entity_id, relation_type, "
                "context, reasoning, document_date, event_date, "
                "valid_from, valid_until, is_current, memory_id, "
                "confidence, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    rel["id"], rel["source_entity_id"], rel["target_entity_id"],
                    rel["relation_type"], rel.get("context"),
                    rel.get("reasoning"), rel.get("document_date"),
                    rel.get("event_date"), rel.get("valid_from"),
                    rel.get("valid_until"), rel.get("is_current", 1),
                    rel.get("memory_id"), rel.get("confidence", 1.0),
                    rel.get("created_at", time.time()),
                ),
            )
            conn.commit()
        return rel["id"]

    def get_entity_relationships(
        self,
        entity_id: Optional[str] = None,
        relation_type: Optional[str] = None,
        current_only: bool = True,
        memory_id: Optional[str] = None,
        limit: int = 500,
    ) -> List[Dict[str, Any]]:
        """Query entity relationships with optional filters."""
        query = "SELECT * FROM entity_relationships WHERE 1=1"
        params: list = []
        if entity_id:
            query += " AND (source_entity_id = ? OR target_entity_id = ?)"
            params.extend([entity_id, entity_id])
        if relation_type:
            query += " AND relation_type = ?"
            params.append(relation_type)
        if current_only:
            query += " AND is_current = 1"
        if memory_id:
            query += " AND memory_id = ?"
            params.append(memory_id)
        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)

        cursor = self._conn().cursor()
        cursor.execute(query, params)
        cols = [
            "id", "source_entity_id", "target_entity_id", "relation_type",
            "context", "reasoning", "document_date", "event_date",
            "valid_from", "valid_until", "is_current", "memory_id",
            "confidence", "created_at",
        ]
        return [dict(zip(cols, row)) for row in cursor.fetchall()]

    def supersede_entity_relationships(
        self, source_entity_id: str, target_entity_id: str,
        relation_type: str,
    ):
        """Mark existing relationships of this type between entities as non-current."""
        with self._write_lock:
            conn = self._conn()
            conn.execute(
                "UPDATE entity_relationships "
                "SET is_current = 0, valid_until = ? "
                "WHERE source_entity_id = ? AND target_entity_id = ? "
                "AND relation_type = ? AND is_current = 1",
                (time.time(), source_entity_id, target_entity_id, relation_type),
            )
            conn.commit()

    def delete_entity_relationships_for_memory(self, memory_id: str):
        """Remove all entity relationships for a memory."""
        self._write(
            "DELETE FROM entity_relationships WHERE memory_id = ?",
            (memory_id,),
        )

    # ─────────────────────────────────────────────
    # REASONING QUEUE
    # ─────────────────────────────────────────────

    def enqueue_for_reasoning(
        self, memory_id: str, content: str, token_estimate: int,
    ):
        """Add a memory to the reasoning queue."""
        import uuid as _uuid
        self._write(
            "INSERT OR IGNORE INTO reasoning_queue "
            "(id, memory_id, content, token_estimate, created_at, processed) "
            "VALUES (?, ?, ?, ?, ?, 0)",
            (str(_uuid.uuid4()), memory_id, content, token_estimate, time.time()),
        )

    def get_pending_reasoning_batch(
        self, token_threshold: int,
    ) -> List[Dict[str, Any]]:
        """Return unprocessed queue rows up to token_threshold tokens."""
        conn = self._conn()
        cursor = conn.execute(
            "SELECT id, memory_id, content, token_estimate "
            "FROM reasoning_queue "
            "WHERE processed = 0 "
            "ORDER BY created_at ASC",
        )
        rows = []
        total_tokens = 0
        for row in cursor:
            entry = {
                "id": row[0],
                "memory_id": row[1],
                "content": row[2],
                "token_estimate": row[3],
            }
            rows.append(entry)
            total_tokens += row[3]
            if total_tokens >= token_threshold:
                break
        return rows

    def mark_reasoning_processed(self, ids: List[str]):
        """Mark queue entries as processed."""
        if not ids:
            return
        with self._write_lock:
            conn = self._conn()
            placeholders = ",".join("?" for _ in ids)
            conn.execute(
                f"UPDATE reasoning_queue SET processed = 1 WHERE id IN ({placeholders})",
                ids,
            )
            conn.commit()

    def get_reasoning_queue_stats(self) -> Dict[str, Any]:
        """Return pending count and total tokens in the queue."""
        conn = self._conn()
        row = conn.execute(
            "SELECT COUNT(*), COALESCE(SUM(token_estimate), 0) "
            "FROM reasoning_queue WHERE processed = 0",
        ).fetchone()
        return {
            "pending_count": row[0],
            "pending_tokens": row[1],
        }

    # ─────────────────────────────────────────────
    # KNOWLEDGE PAGES
    # ─────────────────────────────────────────────

    def store_knowledge_page(self, page: KnowledgePage) -> str:
        """Store or update a knowledge page."""
        with self._write_lock:
            conn = self._conn()
            conn.execute(
                "INSERT OR REPLACE INTO knowledge_pages "
                "(page_id, entity_id, title, page_type, summary, version, "
                "last_updated, created_at, metadata) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    page.page_id, page.entity_id, page.title, page.page_type,
                    page.summary, page.version, page.last_updated,
                    page.created_at, json.dumps(page.metadata),
                ),
            )
            conn.commit()
        return page.page_id

    def get_knowledge_page(self, page_id: str) -> Optional[KnowledgePage]:
        """Retrieve a knowledge page by ID."""
        cursor = self._conn().cursor()
        cursor.execute("SELECT * FROM knowledge_pages WHERE page_id = ?", (page_id,))
        row = cursor.fetchone()
        if not row:
            return None
        return self._row_to_knowledge_page(row)

    def get_knowledge_page_by_entity(self, entity_id: str) -> Optional[KnowledgePage]:
        """Retrieve a knowledge page by entity ID."""
        cursor = self._conn().cursor()
        cursor.execute(
            "SELECT * FROM knowledge_pages WHERE entity_id = ?", (entity_id,)
        )
        row = cursor.fetchone()
        if not row:
            return None
        return self._row_to_knowledge_page(row)

    def get_knowledge_page_by_title(self, title: str) -> Optional[KnowledgePage]:
        """Retrieve a knowledge page by title (case-insensitive)."""
        cursor = self._conn().cursor()
        cursor.execute(
            "SELECT * FROM knowledge_pages WHERE LOWER(title) = LOWER(?)", (title,)
        )
        row = cursor.fetchone()
        if not row:
            return None
        return self._row_to_knowledge_page(row)

    def get_all_knowledge_pages(
        self, page_type: Optional[str] = None
    ) -> List[KnowledgePage]:
        """Retrieve all knowledge pages, optionally filtered by type."""
        query = "SELECT * FROM knowledge_pages"
        params: list = []
        if page_type:
            query += " WHERE page_type = ?"
            params.append(page_type)
        query += " ORDER BY last_updated DESC"
        cursor = self._conn().cursor()
        cursor.execute(query, params)
        return [self._row_to_knowledge_page(r) for r in cursor.fetchall()]

    def update_knowledge_page(self, page: KnowledgePage):
        """Update an existing knowledge page (alias for store)."""
        self.store_knowledge_page(page)

    def delete_knowledge_page(self, page_id: str):
        """Delete a knowledge page and its memory links."""
        with self._write_lock:
            conn = self._conn()
            conn.execute(
                "DELETE FROM knowledge_page_memories WHERE page_id = ?", (page_id,)
            )
            conn.execute(
                "DELETE FROM knowledge_pages WHERE page_id = ?", (page_id,)
            )
            conn.commit()

    def link_memory_to_page(self, page_id: str, memory_id: str):
        """Link a memory to a knowledge page."""
        with self._write_lock:
            conn = self._conn()
            conn.execute(
                "INSERT OR IGNORE INTO knowledge_page_memories "
                "(page_id, memory_id) VALUES (?, ?)",
                (page_id, memory_id),
            )
            conn.commit()

    def get_memories_for_page(self, page_id: str) -> List[str]:
        """Get all memory IDs linked to a knowledge page."""
        cursor = self._conn().cursor()
        cursor.execute(
            "SELECT memory_id FROM knowledge_page_memories WHERE page_id = ?",
            (page_id,),
        )
        return [r[0] for r in cursor.fetchall()]

    def get_pages_for_memory(self, memory_id: str) -> List[KnowledgePage]:
        """Get all knowledge pages linked to a memory."""
        cursor = self._conn().cursor()
        cursor.execute(
            "SELECT kp.* FROM knowledge_pages kp "
            "JOIN knowledge_page_memories kpm ON kp.page_id = kpm.page_id "
            "WHERE kpm.memory_id = ?",
            (memory_id,),
        )
        return [self._row_to_knowledge_page(r) for r in cursor.fetchall()]

    def _row_to_knowledge_page(self, row) -> KnowledgePage:
        """Convert a database row to a KnowledgePage."""
        return KnowledgePage(
            page_id=row[0],
            entity_id=row[1],
            title=row[2],
            page_type=row[3] or "entity",
            summary=row[4] or "",
            version=row[5] or 1,
            last_updated=row[6],
            created_at=row[7],
            metadata=json.loads(row[8]) if row[8] else {},
        )

    # ─────────────────────────────────────────────
    # PROVENANCE LOG
    # ─────────────────────────────────────────────

    def store_provenance(self, entry: ProvenanceEntry) -> str:
        """Store a provenance log entry."""
        with self._write_lock:
            conn = self._conn()
            conn.execute(
                "INSERT INTO provenance_log "
                "(id, memory_id, parent_memory_ids, operation, reason, "
                "source_url, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    entry.id, entry.memory_id,
                    json.dumps(entry.parent_memory_ids),
                    entry.operation, entry.reason,
                    entry.source_url, entry.created_at,
                ),
            )
            conn.commit()
        return entry.id

    def get_provenance(self, memory_id: str) -> List[ProvenanceEntry]:
        """Get all provenance entries for a memory, ordered chronologically."""
        cursor = self._conn().cursor()
        cursor.execute(
            "SELECT * FROM provenance_log WHERE memory_id = ? ORDER BY created_at ASC",
            (memory_id,),
        )
        return [self._row_to_provenance(r) for r in cursor.fetchall()]

    def get_provenance_chain(self, memory_id: str) -> List[ProvenanceEntry]:
        """
        Get the full provenance chain for a memory, including parent entries.
        Follows parent_memory_ids recursively (up to 10 levels).
        """
        visited: Set[str] = set()
        queue = [memory_id]
        all_entries: List[ProvenanceEntry] = []

        for _ in range(10):
            if not queue:
                break
            next_queue: List[str] = []
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

    def _row_to_provenance(self, row) -> ProvenanceEntry:
        """Convert a database row to a ProvenanceEntry."""
        return ProvenanceEntry(
            id=row[0],
            memory_id=row[1],
            parent_memory_ids=json.loads(row[2]) if row[2] else [],
            operation=row[3],
            reason=row[4] or "",
            source_url=row[5] or "",
            created_at=row[6],
        )

    # ─────────────────────────────────────────────
    # MEMORY VERSIONS
    # ─────────────────────────────────────────────

    def store_memory_version(self, version: MemoryVersion) -> str:
        """Store a version snapshot of a memory."""
        with self._write_lock:
            conn = self._conn()
            conn.execute(
                "INSERT INTO memory_versions "
                "(version_id, memory_id, content, strength, importance, "
                "confidence, changed_at, change_reason) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    version.version_id, version.memory_id, version.content,
                    version.strength, version.importance, version.confidence,
                    version.changed_at, version.change_reason,
                ),
            )
            conn.commit()
        return version.version_id

    def get_version_history(self, memory_id: str) -> List[MemoryVersion]:
        """Get all version snapshots for a memory, ordered chronologically."""
        cursor = self._conn().cursor()
        cursor.execute(
            "SELECT * FROM memory_versions WHERE memory_id = ? ORDER BY changed_at ASC",
            (memory_id,),
        )
        return [
            MemoryVersion(
                version_id=r[0], memory_id=r[1], content=r[2],
                strength=r[3], importance=r[4], confidence=r[5],
                changed_at=r[6], change_reason=r[7] or "",
            )
            for r in cursor.fetchall()
        ]

    # ─────────────────────────────────────────────
    # LINT HELPERS
    # ─────────────────────────────────────────────

    def get_stale_memories(self, max_age_days: int = 14) -> List[Memory]:
        """Get active memories with zero access count older than max_age_days."""
        cutoff = time.time() - (max_age_days * 86400)
        cursor = self._conn().cursor()
        cursor.execute(
            "SELECT * FROM memories "
            "WHERE is_active = 1 AND access_count = 0 AND created_at < ?",
            (cutoff,),
        )
        return [self._row_to_memory(r) for r in cursor.fetchall()]

    def get_orphan_memories(self) -> List[Memory]:
        """Get active memories with zero links and zero entity associations."""
        cursor = self._conn().cursor()
        cursor.execute(
            "SELECT m.* FROM memories m "
            "WHERE m.is_active = 1 "
            "AND NOT EXISTS ("
            "  SELECT 1 FROM memory_links ml "
            "  WHERE ml.source_id = m.id OR ml.target_id = m.id"
            ") "
            "AND NOT EXISTS ("
            "  SELECT 1 FROM entity_links el WHERE el.memory_id = m.id"
            ")"
        )
        return [self._row_to_memory(r) for r in cursor.fetchall()]
