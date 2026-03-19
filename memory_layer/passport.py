"""
Universal Memory Passport — portable, encrypted, provider-agnostic memory format.

The Memory Passport is a JSON-based format that lets users:
  - Export their memories from any provider
  - Import them into any other provider or AI system
  - Carry their AI memory with them across tools and platforms
  - Encrypt sensitive memories with a user-controlled passphrase
  - Convert between formats (Mem0, Zep, ChatGPT Memory, Claude Projects)

Format Specification (v2):
{
    "passport_version": 2,
    "format": "memory-layer-universal",
    "created_at": "2026-03-09T12:00:00Z",
    "encrypted": false,
    "metadata": {
        "source_system": "memory-layer",
        "source_version": "0.3.0",
        "total_memories": 42,
        "total_links": 15,
        "namespaces": ["default", "work"],
        "export_options": {...}
    },
    "memories": [...],
    "links": [...],
    "user_profile": {...}   # optional user-level metadata
}
"""

import json
import time
import hashlib
import base64
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

from .models import Memory, MemoryType, MemoryLink, LinkType
from .storage_protocol import StorageBackend

PASSPORT_VERSION = 2
FORMAT_ID = "memory-layer-universal"


# ─────────────────────────────────────────────
# ENCRYPTION (AES-256-GCM via cryptography lib,
#              falls back to Fernet if unavailable)
# ─────────────────────────────────────────────

def _derive_key(passphrase: str, salt: bytes = None) -> Tuple[bytes, bytes]:
    """Derive a 256-bit key from a passphrase using PBKDF2."""
    if salt is None:
        salt = os.urandom(16)
    key = hashlib.pbkdf2_hmac("sha256", passphrase.encode("utf-8"), salt, iterations=600_000)
    return key, salt


def encrypt_data(data: str, passphrase: str) -> dict:
    """Encrypt a JSON string with AES-256-GCM."""
    key, salt = _derive_key(passphrase)
    try:
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
        nonce = os.urandom(12)
        aesgcm = AESGCM(key)
        ciphertext = aesgcm.encrypt(nonce, data.encode("utf-8"), None)
        return {
            "algorithm": "AES-256-GCM",
            "salt": base64.b64encode(salt).decode(),
            "nonce": base64.b64encode(nonce).decode(),
            "ciphertext": base64.b64encode(ciphertext).decode(),
            "kdf": "PBKDF2-SHA256",
            "kdf_iterations": 600_000,
        }
    except ImportError:
        from cryptography.fernet import Fernet
        fernet_key = base64.urlsafe_b64encode(key)
        f = Fernet(fernet_key)
        token = f.encrypt(data.encode("utf-8"))
        return {
            "algorithm": "Fernet-AES-128-CBC",
            "salt": base64.b64encode(salt).decode(),
            "ciphertext": base64.b64encode(token).decode(),
            "kdf": "PBKDF2-SHA256",
            "kdf_iterations": 600_000,
        }


def decrypt_data(enc: dict, passphrase: str) -> str:
    """Decrypt encrypted passport data."""
    salt = base64.b64decode(enc["salt"])
    key, _ = _derive_key(passphrase, salt)

    if enc["algorithm"] == "AES-256-GCM":
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
        nonce = base64.b64decode(enc["nonce"])
        ciphertext = base64.b64decode(enc["ciphertext"])
        aesgcm = AESGCM(key)
        plaintext = aesgcm.decrypt(nonce, ciphertext, None)
        return plaintext.decode("utf-8")
    elif enc["algorithm"] == "Fernet-AES-128-CBC":
        from cryptography.fernet import Fernet
        fernet_key = base64.urlsafe_b64encode(key)
        f = Fernet(fernet_key)
        token = base64.b64decode(enc["ciphertext"])
        return f.decrypt(token).decode("utf-8")
    else:
        raise ValueError(f"Unknown encryption algorithm: {enc['algorithm']}")


# ─────────────────────────────────────────────
# PASSPORT EXPORT
# ─────────────────────────────────────────────

def export_passport(
    storage: StorageBackend,
    output_path: str,
    *,
    namespace: Optional[str] = None,
    include_embeddings: bool = False,
    include_inactive: bool = False,
    passphrase: Optional[str] = None,
    user_profile: Optional[dict] = None,
) -> dict:
    """
    Export memories to Universal Memory Passport format.

    Args:
        storage: Storage backend to export from.
        output_path: Where to save the passport JSON.
        namespace: Filter to specific namespace (None = all).
        include_embeddings: Include embedding vectors (larger file).
        include_inactive: Include deactivated memories.
        passphrase: Encrypt the passport with this passphrase.
        user_profile: Optional user metadata to include.

    Returns:
        Summary dict with export statistics.
    """
    memories = storage.get_all_memories(
        active_only=not include_inactive,
        namespace=namespace,
    )
    links = storage.get_all_links()

    if namespace:
        memory_ids = {m.id for m in memories}
        links = [l for l in links if l.source_id in memory_ids or l.target_id in memory_ids]

    mem_dicts = []
    for m in memories:
        d = {
            "id": m.id,
            "type": m.memory_type.value,
            "content": m.content,
            "created_at": m.created_at,
            "last_accessed": m.last_accessed,
            "access_count": m.access_count,
            "strength": m.strength,
            "importance": m.importance,
            "metadata": m.metadata,
            "tags": m.tags,
            "namespace": m.namespace,
            "is_active": m.is_active,
            "source_episode_ids": m.source_episode_ids,
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

    namespaces = sorted({m.namespace for m in memories})

    passport = {
        "passport_version": PASSPORT_VERSION,
        "format": FORMAT_ID,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "encrypted": passphrase is not None,
        "metadata": {
            "source_system": "memory-layer",
            "source_version": "0.3.0",
            "total_memories": len(mem_dicts),
            "total_links": len(link_dicts),
            "namespaces": namespaces,
            "includes_embeddings": include_embeddings,
            "export_options": {
                "namespace_filter": namespace,
                "include_inactive": include_inactive,
            },
        },
    }

    if user_profile:
        passport["user_profile"] = user_profile

    if passphrase:
        payload = json.dumps({"memories": mem_dicts, "links": link_dicts})
        passport["encrypted_payload"] = encrypt_data(payload, passphrase)
    else:
        passport["memories"] = mem_dicts
        passport["links"] = link_dicts

    with open(output_path, "w") as f:
        json.dump(passport, f, indent=2)

    file_size = Path(output_path).stat().st_size

    return {
        "output_path": output_path,
        "memories_exported": len(mem_dicts),
        "links_exported": len(link_dicts),
        "namespaces": namespaces,
        "encrypted": passphrase is not None,
        "file_size_bytes": file_size,
        "file_size_mb": round(file_size / (1024 * 1024), 2),
    }


# ─────────────────────────────────────────────
# PASSPORT IMPORT
# ─────────────────────────────────────────────

def import_passport(
    storage: StorageBackend,
    input_path: str,
    *,
    passphrase: Optional[str] = None,
    target_namespace: Optional[str] = None,
    skip_duplicates: bool = True,
    reembed: bool = True,
    embeddings_engine=None,
) -> dict:
    """
    Import memories from a Universal Memory Passport.

    Supports:
    - memory-layer-universal (v1, v2)
    - memory-layer legacy export format
    - Mem0 export format
    - Zep/Graphiti export format
    - ChatGPT memory export format

    Args:
        storage: Target storage backend.
        input_path: Path to passport JSON.
        passphrase: Decryption passphrase (if encrypted).
        target_namespace: Override namespace for imported memories.
        skip_duplicates: Skip if memory ID already exists.
        reembed: Re-compute embeddings on import.
        embeddings_engine: Embedding engine for re-embedding.

    Returns:
        Summary dict with import statistics.
    """
    with open(input_path, "r") as f:
        raw = json.load(f)

    detected_format = detect_format(raw)

    if detected_format == "memory-layer-universal":
        return _import_universal(
            storage, raw, passphrase=passphrase,
            target_namespace=target_namespace, skip_duplicates=skip_duplicates,
            reembed=reembed, embeddings_engine=embeddings_engine,
        )
    elif detected_format == "memory-layer-legacy":
        return _import_legacy(
            storage, raw, target_namespace=target_namespace,
            skip_duplicates=skip_duplicates, reembed=reembed,
            embeddings_engine=embeddings_engine,
        )
    elif detected_format == "mem0":
        return _import_mem0(
            storage, raw, target_namespace=target_namespace,
            reembed=reembed, embeddings_engine=embeddings_engine,
        )
    elif detected_format == "zep":
        return _import_zep(
            storage, raw, target_namespace=target_namespace,
            reembed=reembed, embeddings_engine=embeddings_engine,
        )
    elif detected_format == "chatgpt":
        return _import_chatgpt(
            storage, raw, target_namespace=target_namespace,
            reembed=reembed, embeddings_engine=embeddings_engine,
        )
    elif detected_format == "claude":
        return _import_claude(
            storage, raw, target_namespace=target_namespace,
            reembed=reembed, embeddings_engine=embeddings_engine,
        )
    else:
        raise ValueError(
            f"Unrecognized passport format. Detected: {detected_format}. "
            f"Supported: memory-layer-universal, memory-layer-legacy, mem0, zep, chatgpt, claude"
        )


def detect_format(data) -> str:
    """Auto-detect the format of a passport/export file."""
    if isinstance(data, list):
        if data and isinstance(data[0], dict):
            first = data[0]
            if any(k in first for k in ("model_slug", "conversation_id")):
                return "chatgpt"
            if "content" in first or "text" in first or "memory" in first:
                return "chatgpt"
        return "unknown"

    if not isinstance(data, dict):
        return "unknown"

    if data.get("format") == FORMAT_ID:
        return "memory-layer-universal"
    if data.get("passport_version"):
        return "memory-layer-universal"
    if "version" in data and "memories" in data and "links" in data:
        return "memory-layer-legacy"

    if "results" in data and isinstance(data["results"], list):
        first = data["results"][0] if data["results"] else {}
        if isinstance(first, dict) and ("memory" in first or "hash" in first):
            return "mem0"

    if "facts" in data or "episodes" in data or "entities" in data:
        return "zep"

    if "chat_messages" in data or "project_knowledge" in data:
        return "claude"

    if "memories" in data or "model_memories" in data or "bio" in data:
        return "chatgpt"

    return "unknown"


# ─────────────────────────────────────────────
# FORMAT-SPECIFIC IMPORTERS
# ─────────────────────────────────────────────

def _import_universal(storage, data, *, passphrase=None, target_namespace=None,
                      skip_duplicates=True, reembed=True, embeddings_engine=None):
    if data.get("encrypted"):
        if not passphrase:
            raise ValueError("This passport is encrypted. Provide a passphrase to decrypt.")
        payload_str = decrypt_data(data["encrypted_payload"], passphrase)
        payload = json.loads(payload_str)
        mem_data = payload["memories"]
        link_data = payload["links"]
    else:
        mem_data = data.get("memories", [])
        link_data = data.get("links", [])

    imported, skipped = 0, 0
    for md in mem_data:
        if skip_duplicates:
            if storage.get_memory(md["id"]):
                skipped += 1
                continue

        ns = target_namespace or md.get("namespace", "default")
        mem = Memory(
            id=md["id"],
            memory_type=MemoryType(md.get("type", md.get("memory_type", "semantic"))),
            content=md["content"],
            embedding=md.get("embedding"),
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
        if reembed and embeddings_engine and not mem.embedding:
            mem.embedding = embeddings_engine.embed(mem.content)
        storage.store_memory(mem)
        imported += 1

    imported_links = 0
    mem_ids = {md["id"] for md in mem_data}
    for ld in link_data:
        if ld["source_id"] not in mem_ids or ld["target_id"] not in mem_ids:
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

    return {
        "format": "memory-layer-universal",
        "memories_imported": imported,
        "memories_skipped": skipped,
        "links_imported": imported_links,
    }


def _import_legacy(storage, data, *, target_namespace=None, skip_duplicates=True,
                   reembed=True, embeddings_engine=None):
    """Import from the old memory-layer export format (v1)."""
    mem_data = data.get("memories", [])
    link_data = data.get("links", [])
    imported, skipped = 0, 0

    for md in mem_data:
        if skip_duplicates and storage.get_memory(md["id"]):
            skipped += 1
            continue
        ns = target_namespace or md.get("namespace", "default")
        mem = Memory(
            id=md["id"],
            memory_type=MemoryType(md["memory_type"]),
            content=md["content"],
            embedding=md.get("embedding"),
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
        if reembed and embeddings_engine and not mem.embedding:
            mem.embedding = embeddings_engine.embed(mem.content)
        storage.store_memory(mem)
        imported += 1

    imported_links = 0
    for ld in link_data:
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

    return {
        "format": "memory-layer-legacy",
        "memories_imported": imported,
        "memories_skipped": skipped,
        "links_imported": imported_links,
    }


def _import_mem0(storage, data, *, target_namespace=None, reembed=True, embeddings_engine=None):
    """
    Import from Mem0 export format.

    Mem0 exports as: {"results": [{"memory": "...", "hash": "...", "metadata": {...}, ...}]}
    """
    results = data.get("results", data.get("memories", []))
    imported = 0

    for item in results:
        content = item.get("memory", item.get("content", ""))
        if not content:
            continue

        mem_id = item.get("hash", item.get("id", str(hash(content))))
        metadata = item.get("metadata", {})
        metadata["source_system"] = "mem0"
        metadata["original_id"] = item.get("id", "")

        tags = []
        if item.get("categories"):
            tags.extend(item["categories"])
        tags.append("imported-from-mem0")

        mem = Memory(
            id=str(mem_id),
            memory_type=MemoryType.SEMANTIC,
            content=content,
            created_at=_parse_timestamp(item.get("created_at", item.get("updated_at"))),
            last_accessed=time.time(),
            strength=1.0,
            importance=0.7,
            metadata=metadata,
            tags=tags,
            namespace=target_namespace or metadata.get("user_id", "default"),
        )
        if reembed and embeddings_engine:
            mem.embedding = embeddings_engine.embed(mem.content)
        storage.store_memory(mem)
        imported += 1

    return {"format": "mem0", "memories_imported": imported, "memories_skipped": 0, "links_imported": 0}


def _import_zep(storage, data, *, target_namespace=None, reembed=True, embeddings_engine=None):
    """
    Import from Zep/Graphiti export format.

    Zep uses: {"facts": [...], "episodes": [...], "entities": [...]}
    """
    imported = 0

    for fact in data.get("facts", []):
        content = fact.get("fact", fact.get("body", fact.get("content", "")))
        if not content:
            continue
        mem = Memory(
            id=fact.get("uuid", str(hash(content))),
            memory_type=MemoryType.SEMANTIC,
            content=content,
            created_at=_parse_timestamp(fact.get("created_at")),
            last_accessed=time.time(),
            strength=1.0,
            importance=0.7,
            metadata={"source_system": "zep", "original_id": fact.get("uuid", "")},
            tags=["imported-from-zep", "fact"],
            namespace=target_namespace or "default",
        )
        if reembed and embeddings_engine:
            mem.embedding = embeddings_engine.embed(mem.content)
        storage.store_memory(mem)
        imported += 1

    for episode in data.get("episodes", []):
        content = episode.get("content", episode.get("body", ""))
        if not content:
            continue
        mem = Memory(
            id=episode.get("uuid", str(hash(content))),
            memory_type=MemoryType.EPISODIC,
            content=content,
            created_at=_parse_timestamp(episode.get("created_at")),
            last_accessed=time.time(),
            strength=1.0,
            importance=0.5,
            metadata={"source_system": "zep", "original_id": episode.get("uuid", "")},
            tags=["imported-from-zep", "episode"],
            namespace=target_namespace or "default",
        )
        if reembed and embeddings_engine:
            mem.embedding = embeddings_engine.embed(mem.content)
        storage.store_memory(mem)
        imported += 1

    for entity in data.get("entities", []):
        name = entity.get("name", "")
        summary = entity.get("summary", entity.get("description", ""))
        if not name:
            continue
        content = f"{name}: {summary}" if summary else name
        mem = Memory(
            id=entity.get("uuid", str(hash(content))),
            memory_type=MemoryType.SEMANTIC,
            content=content,
            created_at=_parse_timestamp(entity.get("created_at")),
            last_accessed=time.time(),
            strength=1.0,
            importance=0.8,
            metadata={"source_system": "zep", "entity_type": entity.get("entity_type", "")},
            tags=["imported-from-zep", "entity"],
            namespace=target_namespace or "default",
        )
        if reembed and embeddings_engine:
            mem.embedding = embeddings_engine.embed(mem.content)
        storage.store_memory(mem)
        imported += 1

    return {"format": "zep", "memories_imported": imported, "memories_skipped": 0, "links_imported": 0}


def _import_chatgpt(storage, data, *, target_namespace=None, reembed=True, embeddings_engine=None):
    """
    Import from ChatGPT memory export.

    ChatGPT exports memories as a simple list or dict of facts the model has saved.
    Format varies but generally: [{"content": "..."}, ...] or list of strings.
    Also handles the Settings > Data Controls > Export format.
    """
    imported = 0
    items = []

    if isinstance(data, list):
        items = data
    elif "memories" in data:
        items = data["memories"]
    elif "model_memories" in data:
        items = data["model_memories"]
    elif "bio" in data:
        items = [{"content": data["bio"]}]

    for item in items:
        if isinstance(item, str):
            content = item
        elif isinstance(item, dict):
            content = item.get("content", item.get("text", item.get("memory", "")))
        else:
            continue

        if not content or not content.strip():
            continue

        mem = Memory(
            memory_type=MemoryType.SEMANTIC,
            content=content.strip(),
            created_at=_parse_timestamp(item.get("timestamp", item.get("created_at")) if isinstance(item, dict) else None),
            last_accessed=time.time(),
            strength=1.0,
            importance=0.7,
            metadata={"source_system": "chatgpt"},
            tags=["imported-from-chatgpt"],
            namespace=target_namespace or "default",
        )
        if reembed and embeddings_engine:
            mem.embedding = embeddings_engine.embed(mem.content)
        storage.store_memory(mem)
        imported += 1

    return {"format": "chatgpt", "memories_imported": imported, "memories_skipped": 0, "links_imported": 0}


def _import_claude(storage, data, *, target_namespace=None, reembed=True, embeddings_engine=None):
    """
    Import from Claude Projects / Claude memory export.

    Claude Projects can export project knowledge as JSON.
    """
    imported = 0
    items = []

    if "project_knowledge" in data:
        items = data["project_knowledge"]
    elif "chat_messages" in data:
        items = data["chat_messages"]
    elif isinstance(data, list):
        items = data

    for item in items:
        if isinstance(item, str):
            content = item
        elif isinstance(item, dict):
            content = item.get("content", item.get("text", item.get("knowledge", "")))
        else:
            continue

        if not content or not content.strip():
            continue

        mem_type = MemoryType.SEMANTIC
        if isinstance(item, dict) and item.get("role") in ("user", "assistant"):
            mem_type = MemoryType.EPISODIC

        mem = Memory(
            memory_type=mem_type,
            content=content.strip(),
            created_at=_parse_timestamp(item.get("created_at", item.get("timestamp")) if isinstance(item, dict) else None),
            last_accessed=time.time(),
            strength=1.0,
            importance=0.7,
            metadata={"source_system": "claude"},
            tags=["imported-from-claude"],
            namespace=target_namespace or "default",
        )
        if reembed and embeddings_engine:
            mem.embedding = embeddings_engine.embed(mem.content)
        storage.store_memory(mem)
        imported += 1

    return {"format": "claude", "memories_imported": imported, "memories_skipped": 0, "links_imported": 0}


# ─────────────────────────────────────────────
# PASSPORT CONVERTER
# ─────────────────────────────────────────────

def convert_passport(
    input_path: str,
    output_path: str,
    *,
    input_passphrase: Optional[str] = None,
    output_passphrase: Optional[str] = None,
    target_format: str = "memory-layer-universal",
) -> dict:
    """
    Convert between passport formats and optionally encrypt/decrypt.

    Useful for:
    - Encrypting an unencrypted passport
    - Decrypting a passport
    - Normalizing any supported format to universal format
    """
    with open(input_path, "r") as f:
        raw = json.load(f)

    detected = detect_format(raw)
    mem_data = []
    link_data = []

    if detected == "memory-layer-universal":
        if raw.get("encrypted"):
            if not input_passphrase:
                raise ValueError("Input passport is encrypted. Provide input_passphrase.")
            payload = json.loads(decrypt_data(raw["encrypted_payload"], input_passphrase))
            mem_data = payload["memories"]
            link_data = payload["links"]
        else:
            mem_data = raw.get("memories", [])
            link_data = raw.get("links", [])
    elif detected == "memory-layer-legacy":
        mem_data = [
            {**m, "type": m.pop("memory_type", "semantic")}
            for m in raw.get("memories", [])
        ]
        link_data = raw.get("links", [])
    else:
        raise ValueError(f"Cannot convert from {detected} without import→export cycle. Use import_passport() first.")

    passport = {
        "passport_version": PASSPORT_VERSION,
        "format": FORMAT_ID,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "encrypted": output_passphrase is not None,
        "metadata": {
            "source_system": "memory-layer",
            "source_version": "0.3.0",
            "total_memories": len(mem_data),
            "total_links": len(link_data),
            "namespaces": sorted({m.get("namespace", "default") for m in mem_data}),
            "converted_from": detected,
        },
    }

    if raw.get("user_profile"):
        passport["user_profile"] = raw["user_profile"]

    if output_passphrase:
        payload = json.dumps({"memories": mem_data, "links": link_data})
        passport["encrypted_payload"] = encrypt_data(payload, output_passphrase)
    else:
        passport["memories"] = mem_data
        passport["links"] = link_data

    with open(output_path, "w") as f:
        json.dump(passport, f, indent=2)

    return {
        "input_format": detected,
        "output_format": target_format,
        "memories": len(mem_data),
        "links": len(link_data),
        "encrypted": output_passphrase is not None,
        "output_path": output_path,
    }


def inspect_passport(path: str, passphrase: Optional[str] = None) -> dict:
    """
    Inspect a passport file without importing it.

    Returns metadata, format info, memory count, namespaces, etc.
    """
    with open(path, "r") as f:
        raw = json.load(f)

    detected = detect_format(raw)
    file_size = Path(path).stat().st_size

    info = {
        "path": str(path),
        "format": detected,
        "file_size_bytes": file_size,
        "file_size_mb": round(file_size / (1024 * 1024), 2),
    }

    if detected == "memory-layer-universal":
        info["passport_version"] = raw.get("passport_version")
        info["created_at"] = raw.get("created_at")
        info["encrypted"] = raw.get("encrypted", False)
        info["metadata"] = raw.get("metadata", {})
        info["user_profile"] = raw.get("user_profile")

        if not raw.get("encrypted"):
            mems = raw.get("memories", [])
            info["memory_count"] = len(mems)
            info["link_count"] = len(raw.get("links", []))
            info["namespaces"] = sorted({m.get("namespace", "default") for m in mems})
            info["memory_types"] = _count_types(mems, key="type")
            info["tag_summary"] = _count_tags(mems)
        elif passphrase:
            payload = json.loads(decrypt_data(raw["encrypted_payload"], passphrase))
            mems = payload["memories"]
            info["memory_count"] = len(mems)
            info["link_count"] = len(payload.get("links", []))
            info["namespaces"] = sorted({m.get("namespace", "default") for m in mems})
            info["memory_types"] = _count_types(mems, key="type")
        else:
            info["memory_count"] = raw.get("metadata", {}).get("total_memories", "?? (encrypted)")
            info["link_count"] = raw.get("metadata", {}).get("total_links", "?? (encrypted)")
    elif detected == "memory-layer-legacy":
        mems = raw.get("memories", [])
        info["memory_count"] = len(mems)
        info["link_count"] = len(raw.get("links", []))
        info["memory_types"] = _count_types(mems, key="memory_type")
    elif detected == "mem0":
        items = raw.get("results", raw.get("memories", []))
        info["memory_count"] = len(items)
    elif detected == "zep":
        info["facts"] = len(raw.get("facts", []))
        info["episodes"] = len(raw.get("episodes", []))
        info["entities"] = len(raw.get("entities", []))
        info["memory_count"] = info["facts"] + info["episodes"] + info["entities"]
    elif detected == "chatgpt":
        if isinstance(raw, list):
            info["memory_count"] = len(raw)
        else:
            info["memory_count"] = len(raw.get("memories", raw.get("model_memories", [])))
    elif detected == "claude":
        items = raw.get("project_knowledge", raw.get("chat_messages", []))
        info["memory_count"] = len(items) if isinstance(items, list) else 0

    return info


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def _parse_timestamp(value) -> float:
    """Parse various timestamp formats to epoch float."""
    if value is None:
        return time.time()
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        for fmt in ("%Y-%m-%dT%H:%M:%S.%fZ", "%Y-%m-%dT%H:%M:%SZ",
                     "%Y-%m-%dT%H:%M:%S.%f+00:00", "%Y-%m-%dT%H:%M:%S+00:00",
                     "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
            try:
                return datetime.strptime(value, fmt).replace(tzinfo=timezone.utc).timestamp()
            except ValueError:
                continue
    return time.time()


def _count_types(mems: list, key: str = "type") -> dict:
    counts = {}
    for m in mems:
        t = m.get(key, "unknown")
        counts[t] = counts.get(t, 0) + 1
    return counts


def _count_tags(mems: list) -> dict:
    counts = {}
    for m in mems:
        for tag in m.get("tags", []):
            counts[tag] = counts.get(tag, 0) + 1
    top = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:20]
    return dict(top)
