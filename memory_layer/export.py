"""
Export / Import — serialize the memory graph to portable JSON.

    memory-layer export brain.json          Export all memories + links
    memory-layer import brain.json          Import into current database

The JSON format is backend-agnostic: export from SQLite, import into
DynamoDB (or vice versa). Embeddings are excluded by default to keep
file sizes small — they are re-computed on import.
"""

import json
import time
from pathlib import Path
from typing import Optional, List

from .models import Memory, MemoryType, MemoryLink, LinkType
from .storage_protocol import StorageBackend


_EXPORT_VERSION = 1


def export_brain(
    storage: StorageBackend,
    output_path: str,
    *,
    namespace: Optional[str] = None,
    include_embeddings: bool = False,
    include_inactive: bool = False,
) -> dict:
    """
    Export memories and links to a JSON file.

    Returns summary dict with counts.
    """
    memories = storage.get_all_memories(
        active_only=not include_inactive,
        namespace=namespace,
    )
    links = storage.get_all_links()

    if namespace:
        memory_ids = {m.id for m in memories}
        links = [
            l for l in links
            if l.source_id in memory_ids or l.target_id in memory_ids
        ]

    mem_dicts = []
    for m in memories:
        d = {
            "id": m.id,
            "memory_type": m.memory_type.value,
            "content": m.content,
            "created_at": m.created_at,
            "last_accessed": m.last_accessed,
            "access_count": m.access_count,
            "strength": m.strength,
            "importance": m.importance,
            "metadata": m.metadata,
            "tags": m.tags,
            "source_episode_ids": m.source_episode_ids,
            "is_active": m.is_active,
            "namespace": m.namespace,
        }
        if include_embeddings and m.embedding:
            d["embedding"] = m.embedding
        mem_dicts.append(d)

    link_dicts = [
        {
            "id": l.id,
            "source_id": l.source_id,
            "target_id": l.target_id,
            "link_type": l.link_type.value,
            "weight": l.weight,
            "created_at": l.created_at,
        }
        for l in links
    ]

    export_data = {
        "version": _EXPORT_VERSION,
        "exported_at": time.time(),
        "namespace_filter": namespace,
        "includes_embeddings": include_embeddings,
        "memory_count": len(mem_dicts),
        "link_count": len(link_dicts),
        "memories": mem_dicts,
        "links": link_dicts,
    }

    with open(output_path, "w") as f:
        json.dump(export_data, f, indent=2)

    summary = {
        "output_path": output_path,
        "memories_exported": len(mem_dicts),
        "links_exported": len(link_dicts),
        "namespaces": list({m.namespace for m in memories}),
        "file_size_mb": round(Path(output_path).stat().st_size / (1024 * 1024), 2),
    }
    return summary


def import_brain(
    storage: StorageBackend,
    input_path: str,
    *,
    target_namespace: Optional[str] = None,
    skip_duplicates: bool = True,
    reembed: bool = True,
    embeddings_engine=None,
) -> dict:
    """
    Import memories and links from a JSON file.

    Args:
        storage: Target storage backend.
        input_path: Path to the JSON export file.
        target_namespace: Override namespace for all imported memories.
        skip_duplicates: Skip memories whose ID already exists.
        reembed: Re-compute embeddings (requires embeddings_engine).
        embeddings_engine: The embedding engine to use for re-embedding.

    Returns summary dict with counts.
    """
    with open(input_path, "r") as f:
        data = json.load(f)

    version = data.get("version", 1)
    if version > _EXPORT_VERSION:
        raise ValueError(
            f"Export file version {version} is newer than supported ({_EXPORT_VERSION}). "
            f"Please update memory-layer."
        )

    memories_data = data.get("memories", [])
    links_data = data.get("links", [])

    imported_memories = 0
    skipped = 0
    imported_links = 0

    for md in memories_data:
        if skip_duplicates:
            existing = storage.get_memory(md["id"])
            if existing:
                skipped += 1
                continue

        ns = target_namespace or md.get("namespace", "default")
        embedding = md.get("embedding")

        memory = Memory(
            id=md["id"],
            memory_type=MemoryType(md["memory_type"]),
            content=md["content"],
            embedding=embedding,
            created_at=md.get("created_at", time.time()),
            last_accessed=md.get("last_accessed", time.time()),
            access_count=md.get("access_count", 0),
            strength=md.get("strength", 1.0),
            importance=md.get("importance", 0.5),
            metadata=md.get("metadata", {}),
            tags=md.get("tags", []),
            source_episode_ids=md.get("source_episode_ids", []),
            is_active=md.get("is_active", True),
            namespace=ns,
        )

        if reembed and embeddings_engine and not embedding:
            memory.embedding = embeddings_engine.embed(memory.content)

        storage.store_memory(memory)
        imported_memories += 1

    memory_ids = {md["id"] for md in memories_data}
    for ld in links_data:
        if ld["source_id"] not in memory_ids or ld["target_id"] not in memory_ids:
            continue

        link = MemoryLink(
            id=ld["id"],
            source_id=ld["source_id"],
            target_id=ld["target_id"],
            link_type=LinkType(ld["link_type"]),
            weight=ld.get("weight", 0.5),
            created_at=ld.get("created_at", time.time()),
        )
        storage.store_link(link)
        imported_links += 1

    summary = {
        "input_path": input_path,
        "memories_imported": imported_memories,
        "memories_skipped": skipped,
        "links_imported": imported_links,
        "source_version": version,
    }
    return summary
