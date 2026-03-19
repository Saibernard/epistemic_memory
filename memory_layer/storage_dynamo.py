"""
DynamoDB Storage Backend for the Memory Layer.

Single-table design using composite keys:

    PK (partition key)          SK (sort key)           Entity
    ─────────────────────────   ─────────────────────   ──────────────
    MEM#{namespace}             {memory_id}             Memory
    LINK#{source_id}            {link_id}               MemoryLink
    PASS#{memory_id}            {passage_id}            Passage
    WM#global                   {item_id}               WorkingMemoryItem
    CLOG#global                 {consolidation_id}      ConsolidationLog
    META#global                 {key}                   Metadata

GSI-1 (gsi_type_active):
    GSI1PK = memory_type        GSI1SK = created_at     (for type-based queries)
GSI-2 (gsi_target):
    GSI2PK = LINK_TGT#{target}  GSI2SK = {link_id}      (for reverse link lookups)

Requires: pip install boto3
"""

import json
import time
import uuid
import base64
from typing import List, Optional, Dict, Set, Tuple
from decimal import Decimal

import numpy as np

from .models import (
    Memory, MemoryType, MemoryLink, LinkType, WorkingMemoryItem,
    KnowledgePage, ProvenanceEntry, MemoryVersion,
)


def _float_to_decimal(val):
    """DynamoDB doesn't support float; convert to Decimal."""
    return Decimal(str(val))


def _decimal_to_float(val):
    """Convert Decimal back to float."""
    if isinstance(val, Decimal):
        return float(val)
    return val


def _emb_to_b64(embedding) -> str:
    """Serialize a numpy/list embedding to base64 for DynamoDB storage."""
    arr = np.array(embedding, dtype=np.float32)
    return base64.b64encode(arr.tobytes()).decode("ascii")


def _b64_to_emb(b64: str) -> np.ndarray:
    """Deserialize a base64 string back to numpy array."""
    raw = base64.b64decode(b64)
    return np.frombuffer(raw, dtype=np.float32).copy()


class DynamoStorage:
    """
    AWS DynamoDB storage backend.

    Uses a single-table design. The table and GSIs are auto-created
    on first use if they don't exist (pay-per-request billing mode).
    """

    def __init__(self, region: str, table_name: str = "memory-layer"):
        try:
            import boto3
        except ImportError:
            raise ImportError(
                "DynamoDB backend requires boto3.\n"
                "Install it: pip install 'memory-layer[dynamodb]'  (or: pip install boto3)"
            )

        self._ddb = boto3.resource("dynamodb", region_name=region)
        self._client = boto3.client("dynamodb", region_name=region)
        self._table_name = table_name
        self._table = None
        self._ensure_table()

    # ─────────────────────────────────────────────
    # TABLE SETUP
    # ─────────────────────────────────────────────

    def _ensure_table(self):
        """Create the table + GSIs if they don't exist."""
        existing = [t.name for t in self._ddb.tables.all()]
        if self._table_name in existing:
            self._table = self._ddb.Table(self._table_name)
            return

        print(f"  Creating DynamoDB table: {self._table_name} ...")
        self._table = self._ddb.create_table(
            TableName=self._table_name,
            KeySchema=[
                {"AttributeName": "PK", "KeyType": "HASH"},
                {"AttributeName": "SK", "KeyType": "RANGE"},
            ],
            AttributeDefinitions=[
                {"AttributeName": "PK", "AttributeType": "S"},
                {"AttributeName": "SK", "AttributeType": "S"},
                {"AttributeName": "GSI1PK", "AttributeType": "S"},
                {"AttributeName": "GSI1SK", "AttributeType": "N"},
                {"AttributeName": "GSI2PK", "AttributeType": "S"},
            ],
            GlobalSecondaryIndexes=[
                {
                    "IndexName": "gsi_type_active",
                    "KeySchema": [
                        {"AttributeName": "GSI1PK", "KeyType": "HASH"},
                        {"AttributeName": "GSI1SK", "KeyType": "RANGE"},
                    ],
                    "Projection": {"ProjectionType": "ALL"},
                },
                {
                    "IndexName": "gsi_target",
                    "KeySchema": [
                        {"AttributeName": "GSI2PK", "KeyType": "HASH"},
                        {"AttributeName": "SK", "KeyType": "RANGE"},
                    ],
                    "Projection": {"ProjectionType": "ALL"},
                },
            ],
            BillingMode="PAY_PER_REQUEST",
        )
        self._table.wait_until_exists()
        print(f"  ✓ DynamoDB table created: {self._table_name}")

    # ─────────────────────────────────────────────
    # INTERNAL HELPERS
    # ─────────────────────────────────────────────

    def _mem_pk(self, namespace: str) -> str:
        return f"MEM#{namespace}"

    def _memory_to_item(self, memory: Memory) -> dict:
        item = {
            "PK": self._mem_pk(memory.namespace),
            "SK": memory.id,
            "entity": "memory",
            "memory_type": memory.memory_type.value,
            "content": memory.content,
            "created_at": _float_to_decimal(memory.created_at),
            "last_accessed": _float_to_decimal(memory.last_accessed),
            "access_count": memory.access_count,
            "strength": _float_to_decimal(memory.strength),
            "importance": _float_to_decimal(memory.importance),
            "metadata": json.dumps(memory.metadata),
            "tags": json.dumps(memory.tags),
            "source_episode_ids": json.dumps(memory.source_episode_ids),
            "is_active": 1 if memory.is_active else 0,
            "namespace": memory.namespace,
            "confidence": _float_to_decimal(memory.confidence),
            "epistemic_status": memory.epistemic_status,
            # GSI keys
            "GSI1PK": memory.memory_type.value,
            "GSI1SK": _float_to_decimal(memory.created_at),
        }
        if memory.embedding:
            item["embedding"] = _emb_to_b64(memory.embedding)
        return item

    def _item_to_memory(self, item: dict) -> Memory:
        embedding = None
        if "embedding" in item and item["embedding"]:
            embedding = _b64_to_emb(item["embedding"]).tolist()

        return Memory(
            id=item["SK"],
            memory_type=MemoryType(item["memory_type"]),
            content=item["content"],
            embedding=embedding,
            created_at=_decimal_to_float(item["created_at"]),
            last_accessed=_decimal_to_float(item["last_accessed"]),
            access_count=int(item.get("access_count", 0)),
            strength=_decimal_to_float(item.get("strength", 1)),
            importance=_decimal_to_float(item.get("importance", 0.5)),
            metadata=json.loads(item.get("metadata", "{}")),
            tags=json.loads(item.get("tags", "[]")),
            source_episode_ids=json.loads(item.get("source_episode_ids", "[]")),
            is_active=bool(int(item.get("is_active", 1))),
            namespace=item.get("namespace", "default"),
            confidence=_decimal_to_float(item.get("confidence", 0.5)),
            epistemic_status=item.get("epistemic_status", "inferred"),
        )

    def _item_to_link(self, item: dict) -> MemoryLink:
        return MemoryLink(
            id=item["SK"],
            source_id=item["source_id"],
            target_id=item["target_id"],
            link_type=LinkType(item["link_type"]),
            weight=_decimal_to_float(item.get("weight", 0.5)),
            created_at=_decimal_to_float(item["created_at"]),
        )

    def _query_all(self, **kwargs) -> list:
        """Query with automatic pagination."""
        items = []
        resp = self._table.query(**kwargs)
        items.extend(resp.get("Items", []))
        while "LastEvaluatedKey" in resp:
            kwargs["ExclusiveStartKey"] = resp["LastEvaluatedKey"]
            resp = self._table.query(**kwargs)
            items.extend(resp.get("Items", []))
        return items

    def _scan_all(self, **kwargs) -> list:
        items = []
        resp = self._table.scan(**kwargs)
        items.extend(resp.get("Items", []))
        while "LastEvaluatedKey" in resp:
            kwargs["ExclusiveStartKey"] = resp["LastEvaluatedKey"]
            resp = self._table.scan(**kwargs)
            items.extend(resp.get("Items", []))
        return items

    # ─────────────────────────────────────────────
    # METADATA
    # ─────────────────────────────────────────────

    def get_meta(self, key: str) -> Optional[str]:
        resp = self._table.get_item(Key={"PK": "META#global", "SK": key})
        item = resp.get("Item")
        return item["value"] if item else None

    def set_meta(self, key: str, value: str) -> None:
        self._table.put_item(Item={
            "PK": "META#global", "SK": key,
            "entity": "meta", "value": value,
        })

    def has_memories(self) -> bool:
        resp = self._table.scan(
            FilterExpression="entity = :e AND is_active = :a",
            ExpressionAttributeValues={":e": "memory", ":a": 1},
            Select="COUNT",
            Limit=1,
        )
        return resp["Count"] > 0

    def count_active_memories_with_embeddings(self) -> int:
        items = self._scan_all(
            FilterExpression="entity = :e AND is_active = :a AND attribute_exists(embedding)",
            ExpressionAttributeValues={":e": "memory", ":a": 1},
            Select="COUNT",
        )
        return len(items) if isinstance(items, list) and items and isinstance(items[0], dict) else 0

    def get_sample_embedding_dimension(self) -> Optional[int]:
        resp = self._table.scan(
            FilterExpression="entity = :e AND is_active = :a AND attribute_exists(embedding)",
            ExpressionAttributeValues={":e": "memory", ":a": 1},
            Limit=1,
        )
        items = resp.get("Items", [])
        if items and "embedding" in items[0]:
            return len(_b64_to_emb(items[0]["embedding"]))
        return None

    def clear_all_embeddings(self) -> None:
        items = self._scan_all(
            FilterExpression="entity = :e",
            ExpressionAttributeValues={":e": "memory"},
            ProjectionExpression="PK, SK",
        )
        for item in items:
            self._table.update_item(
                Key={"PK": item["PK"], "SK": item["SK"]},
                UpdateExpression="REMOVE embedding",
            )
        # Also delete all passages
        passages = self._scan_all(
            FilterExpression="entity = :e",
            ExpressionAttributeValues={":e": "passage"},
            ProjectionExpression="PK, SK",
        )
        with self._table.batch_writer() as batch:
            for p in passages:
                batch.delete_item(Key={"PK": p["PK"], "SK": p["SK"]})

    # ─────────────────────────────────────────────
    # MEMORY CRUD
    # ─────────────────────────────────────────────

    def store_memory(self, memory: Memory) -> str:
        self._table.put_item(Item=self._memory_to_item(memory))
        return memory.id

    def get_memory(self, memory_id: str) -> Optional[Memory]:
        # We don't know the namespace, so scan for it
        resp = self._table.scan(
            FilterExpression="entity = :e AND SK = :sk",
            ExpressionAttributeValues={":e": "memory", ":sk": memory_id},
            Limit=1,
        )
        items = resp.get("Items", [])
        return self._item_to_memory(items[0]) if items else None

    def get_memories_by_ids(self, memory_ids: List[str]) -> List[Memory]:
        if not memory_ids:
            return []
        # BatchGetItem requires exact keys; scan with filter for flexibility
        from boto3.dynamodb.conditions import Attr
        results = []
        # Process in chunks of 25 to avoid filter expression limits
        for i in range(0, len(memory_ids), 25):
            chunk = memory_ids[i:i + 25]
            items = self._scan_all(
                FilterExpression=Attr("entity").eq("memory") & Attr("SK").is_in(chunk),
            )
            results.extend(items)
        return [self._item_to_memory(item) for item in results]

    def get_all_memories(
        self,
        memory_type: Optional[MemoryType] = None,
        active_only: bool = True,
        namespace: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> List[Memory]:
        if namespace is not None:
            items = self._query_all(
                KeyConditionExpression="PK = :pk",
                ExpressionAttributeValues={":pk": self._mem_pk(namespace)},
            )
        else:
            items = self._scan_all(
                FilterExpression="entity = :e",
                ExpressionAttributeValues={":e": "memory"},
            )

        memories = [self._item_to_memory(item) for item in items]

        if active_only:
            memories = [m for m in memories if m.is_active]
        if memory_type:
            memories = [m for m in memories if m.memory_type == memory_type]
        if tags:
            tag_set = set(tags)
            memories = [m for m in memories if tag_set & set(m.tags)]

        memories.sort(key=lambda m: m.last_accessed, reverse=True)
        return memories

    def update_memory(self, memory: Memory) -> None:
        self.store_memory(memory)

    def deactivate_memory(self, memory_id: str) -> None:
        mem = self.get_memory(memory_id)
        if mem:
            self._table.update_item(
                Key={"PK": self._mem_pk(mem.namespace), "SK": memory_id},
                UpdateExpression="SET is_active = :a",
                ExpressionAttributeValues={":a": 0},
            )
            self.delete_passages_for_memory(memory_id)

    def forget_memory(self, memory_id: str, hard: bool = False) -> None:
        if hard:
            mem = self.get_memory(memory_id)
            if not mem:
                return
            self._table.delete_item(
                Key={"PK": self._mem_pk(mem.namespace), "SK": memory_id}
            )
            self.delete_passages_for_memory(memory_id)
            # Delete associated links
            links = self.get_links_for(memory_id)
            with self._table.batch_writer() as batch:
                for link in links:
                    batch.delete_item(
                        Key={"PK": f"LINK#{link.source_id}", "SK": link.id}
                    )
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

        memories = self.get_all_memories(
            memory_type=memory_type,
            active_only=True,
            namespace=namespace,
            tags=tags,
        )

        if older_than_hours is not None:
            cutoff = time.time() - (older_than_hours * 3600)
            memories = [m for m in memories if m.created_at < cutoff]

        for mem in memories:
            self.forget_memory(mem.id, hard=hard)

        return len(memories)

    # ─────────────────────────────────────────────
    # PASSAGES
    # ─────────────────────────────────────────────

    def store_passages(self, memory_id: str, passages: List[Dict]) -> None:
        with self._table.batch_writer() as batch:
            for p in passages:
                passage_id = str(uuid.uuid4())
                batch.put_item(Item={
                    "PK": f"PASS#{memory_id}",
                    "SK": passage_id,
                    "entity": "passage",
                    "memory_id": memory_id,
                    "chunk_index": p["chunk_index"],
                    "content_preview": p["content_preview"][:500],
                    "embedding": _emb_to_b64(p["embedding"]),
                })

    def get_all_passage_embeddings(
        self, min_strength: float = 0.0,
    ) -> List[Tuple[str, np.ndarray]]:
        passages = self._scan_all(
            FilterExpression="entity = :e",
            ExpressionAttributeValues={":e": "passage"},
        )
        results = []
        for p in passages:
            mid = p["memory_id"]
            mem = self.get_memory(mid)
            if mem and mem.is_active and mem.strength >= min_strength:
                emb = _b64_to_emb(p["embedding"])
                results.append((mid, emb))
        return results

    def delete_passages_for_memory(self, memory_id: str) -> None:
        items = self._query_all(
            KeyConditionExpression="PK = :pk",
            ExpressionAttributeValues={":pk": f"PASS#{memory_id}"},
            ProjectionExpression="PK, SK",
        )
        if items:
            with self._table.batch_writer() as batch:
                for item in items:
                    batch.delete_item(Key={"PK": item["PK"], "SK": item["SK"]})

    def get_memories_with_embeddings(
        self,
        memory_type: Optional[MemoryType] = None,
        min_strength: float = 0.0,
        namespace: Optional[str] = None,
    ) -> List[Tuple[Memory, np.ndarray]]:
        memories = self.get_all_memories(
            memory_type=memory_type,
            active_only=True,
            namespace=namespace,
        )
        results = []
        for mem in memories:
            if mem.embedding and mem.strength >= min_strength:
                emb = np.array(mem.embedding, dtype=np.float32)
                results.append((mem, emb))
        return results

    # ─────────────────────────────────────────────
    # LINKS
    # ─────────────────────────────────────────────

    def store_link(self, link: MemoryLink) -> str:
        self._table.put_item(Item={
            "PK": f"LINK#{link.source_id}",
            "SK": link.id,
            "entity": "link",
            "source_id": link.source_id,
            "target_id": link.target_id,
            "link_type": link.link_type.value,
            "weight": _float_to_decimal(link.weight),
            "created_at": _float_to_decimal(link.created_at),
            "GSI2PK": f"LINK_TGT#{link.target_id}",
        })
        return link.id

    def get_links_for(self, memory_id: str) -> List[MemoryLink]:
        # Source links
        source_items = self._query_all(
            KeyConditionExpression="PK = :pk",
            ExpressionAttributeValues={":pk": f"LINK#{memory_id}"},
        )
        # Target links (via GSI)
        target_items = self._query_all(
            IndexName="gsi_target",
            KeyConditionExpression="GSI2PK = :pk",
            ExpressionAttributeValues={":pk": f"LINK_TGT#{memory_id}"},
        )
        seen = set()
        links = []
        for item in source_items + target_items:
            lid = item["SK"]
            if lid not in seen:
                seen.add(lid)
                links.append(self._item_to_link(item))
        return links

    def get_links_for_ids(self, memory_ids: List[str]) -> List[MemoryLink]:
        if not memory_ids:
            return []
        seen = set()
        links = []
        for mid in memory_ids:
            for link in self.get_links_for(mid):
                if link.id not in seen:
                    seen.add(link.id)
                    links.append(link)
        return links

    def get_all_links(self) -> List[MemoryLink]:
        items = self._scan_all(
            FilterExpression="entity = :e",
            ExpressionAttributeValues={":e": "link"},
        )
        return [self._item_to_link(item) for item in items]

    # ─────────────────────────────────────────────
    # WORKING MEMORY
    # ─────────────────────────────────────────────

    def store_working_item(self, item: WorkingMemoryItem) -> None:
        self._table.put_item(Item={
            "PK": "WM#global",
            "SK": item.id,
            "entity": "working",
            "content": item.content,
            "role": item.role,
            "created_at": _float_to_decimal(item.created_at),
            "metadata": json.dumps(item.metadata),
        })

    def get_working_memory(self, limit: int = 20) -> List[WorkingMemoryItem]:
        items = self._query_all(
            KeyConditionExpression="PK = :pk",
            ExpressionAttributeValues={":pk": "WM#global"},
        )
        wm_items = []
        for item in items:
            wm_items.append(WorkingMemoryItem(
                id=item["SK"],
                content=item["content"],
                role=item.get("role", "user"),
                created_at=_decimal_to_float(item["created_at"]),
                metadata=json.loads(item.get("metadata", "{}")),
            ))
        wm_items.sort(key=lambda x: x.created_at)
        return wm_items[-limit:]

    def clear_working_memory(self) -> None:
        items = self._query_all(
            KeyConditionExpression="PK = :pk",
            ExpressionAttributeValues={":pk": "WM#global"},
            ProjectionExpression="PK, SK",
        )
        if items:
            with self._table.batch_writer() as batch:
                for item in items:
                    batch.delete_item(Key={"PK": item["PK"], "SK": item["SK"]})

    def trim_working_memory(self, keep_last: int = 20) -> None:
        wm = self.get_working_memory(limit=999999)
        if len(wm) <= keep_last:
            return
        to_remove = wm[:-keep_last]
        with self._table.batch_writer() as batch:
            for item in to_remove:
                batch.delete_item(Key={"PK": "WM#global", "SK": item.id})

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
        self._table.put_item(Item={
            "PK": "CLOG#global",
            "SK": consolidation_id,
            "entity": "consolidation",
            "source_ids": json.dumps(source_ids),
            "result_id": result_id,
            "created_at": _float_to_decimal(time.time()),
            "strategy": strategy,
        })

    def get_consolidated_episode_ids(self) -> Set[str]:
        items = self._query_all(
            KeyConditionExpression="PK = :pk",
            ExpressionAttributeValues={":pk": "CLOG#global"},
        )
        ids: Set[str] = set()
        for item in items:
            try:
                ids.update(json.loads(item["source_ids"]))
            except (json.JSONDecodeError, TypeError, KeyError):
                pass
        return ids

    def get_consolidation_count(self) -> int:
        items = self._query_all(
            KeyConditionExpression="PK = :pk",
            ExpressionAttributeValues={":pk": "CLOG#global"},
            Select="COUNT",
        )
        return len(items) if isinstance(items, list) else 0

    # ─────────────────────────────────────────────
    # STATISTICS
    # ─────────────────────────────────────────────

    def get_counts(self, namespace: Optional[str] = None) -> Dict[str, int]:
        memories = self.get_all_memories(active_only=True, namespace=namespace)
        counts: Dict[str, int] = {"total": len(memories)}
        for mt in MemoryType:
            counts[mt.value] = sum(1 for m in memories if m.memory_type == mt)

        links = self.get_all_links()
        counts["links"] = len(links)

        wm = self.get_working_memory(limit=999999)
        counts["working"] = len(wm)

        return counts

    def get_avg_strength(self) -> float:
        memories = self.get_all_memories(active_only=True)
        if not memories:
            return 0.0
        return sum(m.strength for m in memories) / len(memories)

    def get_avg_importance(self) -> float:
        memories = self.get_all_memories(active_only=True)
        if not memories:
            return 0.0
        return sum(m.importance for m in memories) / len(memories)

    def get_oldest_memory_age_hours(self) -> float:
        memories = self.get_all_memories(active_only=True)
        if not memories:
            return 0.0
        oldest = min(m.created_at for m in memories)
        return (time.time() - oldest) / 3600.0

    def get_most_accessed_memory_id(self) -> Optional[str]:
        memories = self.get_all_memories(active_only=True)
        if not memories:
            return None
        return max(memories, key=lambda m: m.access_count).id

    # ─────────────────────────────────────────────
    # KNOWLEDGE PAGES  PK=KP#{entity_id}  SK={page_id}
    # ─────────────────────────────────────────────

    def _kp_to_item(self, page: KnowledgePage) -> dict:
        return {
            "PK": f"KP#{page.entity_id}",
            "SK": page.page_id,
            "entity": "knowledge_page",
            "entity_id": page.entity_id,
            "title": page.title,
            "page_type": page.page_type,
            "summary": page.summary,
            "version": page.version,
            "last_updated": _float_to_decimal(page.last_updated),
            "created_at": _float_to_decimal(page.created_at),
            "metadata": json.dumps(page.metadata),
            "memory_ids": json.dumps(page.memory_ids),
        }

    def _item_to_kp(self, item: dict) -> KnowledgePage:
        return KnowledgePage(
            page_id=item["SK"],
            entity_id=item["entity_id"],
            title=item["title"],
            page_type=item.get("page_type", "entity"),
            summary=item.get("summary", ""),
            memory_ids=json.loads(item.get("memory_ids", "[]")),
            version=int(item.get("version", 1)),
            last_updated=_decimal_to_float(item["last_updated"]),
            created_at=_decimal_to_float(item["created_at"]),
            metadata=json.loads(item.get("metadata", "{}")),
        )

    def store_knowledge_page(self, page: KnowledgePage) -> str:
        self._table.put_item(Item=self._kp_to_item(page))
        return page.page_id

    def get_knowledge_page(self, page_id: str) -> Optional[KnowledgePage]:
        items = self._scan_all(
            FilterExpression="entity = :e AND SK = :sk",
            ExpressionAttributeValues={":e": "knowledge_page", ":sk": page_id},
        )
        return self._item_to_kp(items[0]) if items else None

    def get_knowledge_page_by_entity(self, entity_id: str) -> Optional[KnowledgePage]:
        items = self._query_all(
            KeyConditionExpression="PK = :pk",
            ExpressionAttributeValues={":pk": f"KP#{entity_id}"},
        )
        kp_items = [i for i in items if i.get("entity") == "knowledge_page"]
        return self._item_to_kp(kp_items[0]) if kp_items else None

    def get_knowledge_page_by_title(self, title: str) -> Optional[KnowledgePage]:
        items = self._scan_all(
            FilterExpression="entity = :e",
            ExpressionAttributeValues={":e": "knowledge_page"},
        )
        for item in items:
            if item.get("title", "").lower() == title.lower():
                return self._item_to_kp(item)
        return None

    def get_all_knowledge_pages(self, page_type: Optional[str] = None) -> List[KnowledgePage]:
        items = self._scan_all(
            FilterExpression="entity = :e",
            ExpressionAttributeValues={":e": "knowledge_page"},
        )
        pages = [self._item_to_kp(i) for i in items]
        if page_type:
            pages = [p for p in pages if p.page_type == page_type]
        pages.sort(key=lambda p: p.last_updated, reverse=True)
        return pages

    def update_knowledge_page(self, page: KnowledgePage) -> None:
        self.store_knowledge_page(page)

    def delete_knowledge_page(self, page_id: str) -> None:
        page = self.get_knowledge_page(page_id)
        if page:
            self._table.delete_item(Key={"PK": f"KP#{page.entity_id}", "SK": page_id})

    def link_memory_to_page(self, page_id: str, memory_id: str) -> None:
        # For Dynamo we track memory_ids as a list on the page item itself
        page = self.get_knowledge_page(page_id)
        if page and memory_id not in page.memory_ids:
            page.memory_ids.append(memory_id)
            page.last_updated = time.time()
            self.store_knowledge_page(page)

    def get_memories_for_page(self, page_id: str) -> List[str]:
        page = self.get_knowledge_page(page_id)
        return page.memory_ids if page else []

    def get_pages_for_memory(self, memory_id: str) -> List[KnowledgePage]:
        all_pages = self.get_all_knowledge_pages()
        return [p for p in all_pages if memory_id in p.memory_ids]

    # ─────────────────────────────────────────────
    # PROVENANCE  PK=PROV#{memory_id}  SK={entry_id}
    # ─────────────────────────────────────────────

    def store_provenance(self, entry: ProvenanceEntry) -> str:
        self._table.put_item(Item={
            "PK": f"PROV#{entry.memory_id}",
            "SK": entry.id,
            "entity": "provenance",
            "memory_id": entry.memory_id,
            "parent_memory_ids": json.dumps(entry.parent_memory_ids),
            "operation": entry.operation,
            "reason": entry.reason,
            "source_url": entry.source_url,
            "created_at": _float_to_decimal(entry.created_at),
        })
        return entry.id

    def _item_to_provenance(self, item: dict) -> ProvenanceEntry:
        return ProvenanceEntry(
            id=item["SK"],
            memory_id=item["memory_id"],
            parent_memory_ids=json.loads(item.get("parent_memory_ids", "[]")),
            operation=item["operation"],
            reason=item.get("reason", ""),
            source_url=item.get("source_url", ""),
            created_at=_decimal_to_float(item["created_at"]),
        )

    def get_provenance(self, memory_id: str) -> List[ProvenanceEntry]:
        items = self._query_all(
            KeyConditionExpression="PK = :pk",
            ExpressionAttributeValues={":pk": f"PROV#{memory_id}"},
        )
        entries = [self._item_to_provenance(i) for i in items if i.get("entity") == "provenance"]
        entries.sort(key=lambda e: e.created_at)
        return entries

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
    # MEMORY VERSIONS  PK=VER#{memory_id}  SK={version_id}
    # ─────────────────────────────────────────────

    def store_memory_version(self, version: MemoryVersion) -> str:
        self._table.put_item(Item={
            "PK": f"VER#{version.memory_id}",
            "SK": version.version_id,
            "entity": "version",
            "memory_id": version.memory_id,
            "content": version.content,
            "strength": _float_to_decimal(version.strength),
            "importance": _float_to_decimal(version.importance),
            "confidence": _float_to_decimal(version.confidence),
            "changed_at": _float_to_decimal(version.changed_at),
            "change_reason": version.change_reason,
        })
        return version.version_id

    def get_version_history(self, memory_id: str) -> List[MemoryVersion]:
        items = self._query_all(
            KeyConditionExpression="PK = :pk",
            ExpressionAttributeValues={":pk": f"VER#{memory_id}"},
        )
        versions = [
            MemoryVersion(
                version_id=i["SK"],
                memory_id=i["memory_id"],
                content=i["content"],
                strength=_decimal_to_float(i.get("strength", 1)),
                importance=_decimal_to_float(i.get("importance", 0.5)),
                confidence=_decimal_to_float(i.get("confidence", 0.5)),
                changed_at=_decimal_to_float(i["changed_at"]),
                change_reason=i.get("change_reason", ""),
            )
            for i in items if i.get("entity") == "version"
        ]
        versions.sort(key=lambda v: v.changed_at)
        return versions

    # ─────────────────────────────────────────────
    # LINT HELPERS
    # ─────────────────────────────────────────────

    def get_stale_memories(self, max_age_days: int = 14) -> List[Memory]:
        cutoff = time.time() - (max_age_days * 86400)
        memories = self.get_all_memories(active_only=True)
        return [m for m in memories if m.access_count == 0 and m.created_at < cutoff]

    def get_orphan_memories(self) -> List[Memory]:
        memories = self.get_all_memories(active_only=True)
        all_links = self.get_all_links()
        linked_ids: set = set()
        for link in all_links:
            linked_ids.add(link.source_id)
            linked_ids.add(link.target_id)
        return [m for m in memories if m.id not in linked_ids]
