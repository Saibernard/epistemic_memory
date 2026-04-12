"""
Wiki Export for the Memory Layer.

Exports knowledge pages as Obsidian-compatible markdown files with
YAML frontmatter and [[wikilinks]] for cross-referencing.

Output structure:
  output_dir/
    index.md              — Table of contents
    entities/             — Person, org pages
    concepts/             — Abstract concept pages
    topics/               — Tech, tool, topic pages
"""

from __future__ import annotations

import os
import re
import time
from typing import Optional, Dict, Any, List

from .models import KnowledgePage


def _sanitize_filename(name: str) -> str:
    """Convert a title to a safe filename."""
    safe = re.sub(r'[^\w\s\-]', '', name)
    safe = re.sub(r'\s+', '_', safe.strip())
    return safe[:80] or "untitled"


def _page_dir(page_type: str) -> str:
    """Map page_type to subdirectory."""
    return {
        "entity": "entities",
        "concept": "concepts",
        "topic": "topics",
    }.get(page_type, "entities")


def export_wiki(
    storage,
    entity_graph,
    knowledge_page_manager,
    output_dir: str,
    format: str = "obsidian",
) -> Dict[str, Any]:
    """
    Export all knowledge pages as markdown files.

    Args:
        storage: Storage backend
        entity_graph: EntityGraph instance
        knowledge_page_manager: KnowledgePageManager instance
        output_dir: Directory to write files to
        format: Export format (currently only "obsidian")

    Returns:
        Summary dict with export stats
    """
    pages = knowledge_page_manager.get_all_pages()

    # Build title lookup for wikilinks
    title_map: Dict[str, str] = {}
    for page in pages:
        title_map[page.entity_id] = page.title

    # Create directory structure
    for subdir in ("entities", "concepts", "topics"):
        os.makedirs(os.path.join(output_dir, subdir), exist_ok=True)

    stats = {
        "pages_exported": 0,
        "entities": 0,
        "concepts": 0,
        "topics": 0,
        "files": [],
    }

    for page in pages:
        subdir = _page_dir(page.page_type)
        filename = _sanitize_filename(page.title) + ".md"
        filepath = os.path.join(output_dir, subdir, filename)

        # Get linked memories for related info
        memory_ids = storage.get_memories_for_page(page.page_id)
        memories = storage.get_memories_by_ids(memory_ids) if memory_ids else []

        # Find related entities for wikilinks
        related_entities = _find_related_entities(
            page, memories, entity_graph, title_map
        )

        content = _render_page(page, memories, related_entities, format)

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)

        stats["pages_exported"] += 1
        stats[subdir.rstrip("s") if subdir.endswith("ies") else subdir.rstrip("s")] = (
            stats.get(subdir.rstrip("s") if subdir.endswith("ies") else subdir.rstrip("s"), 0) + 1
        )
        stats["files"].append(filepath)

    # Count by type properly
    stats["entities"] = sum(1 for p in pages if p.page_type == "entity")
    stats["concepts"] = sum(1 for p in pages if p.page_type == "concept")
    stats["topics"] = sum(1 for p in pages if p.page_type == "topic")

    # Write index
    index_path = os.path.join(output_dir, "index.md")
    _write_index(index_path, pages, stats)
    stats["files"].append(index_path)

    return stats


def _find_related_entities(
    page: KnowledgePage,
    memories: list,
    entity_graph,
    title_map: Dict[str, str],
) -> List[str]:
    """Find entity names that appear in the same memories as this page's entity."""
    related: set = set()
    conn = entity_graph.storage._conn()

    for mem in memories[:20]:
        rows = conn.execute(
            "SELECT el.entity_id FROM entity_links el "
            "WHERE el.memory_id = ? AND el.entity_id != ?",
            (mem.id, page.entity_id),
        ).fetchall()
        for (eid,) in rows:
            if eid in title_map:
                related.add(title_map[eid])

    return sorted(related)[:15]


def _render_page(
    page: KnowledgePage,
    memories: list,
    related_entities: List[str],
    format: str,
) -> str:
    """Render a knowledge page as markdown."""
    lines = []

    # YAML frontmatter
    lines.append("---")
    lines.append(f"title: \"{page.title}\"")
    lines.append(f"type: {page.page_type}")
    lines.append(f"entity_id: {page.entity_id}")
    lines.append(f"version: {page.version}")
    lines.append(f"last_updated: {time.strftime('%Y-%m-%d %H:%M', time.localtime(page.last_updated))}")
    lines.append(f"created: {time.strftime('%Y-%m-%d %H:%M', time.localtime(page.created_at))}")
    lines.append(f"memory_count: {len(memories)}")
    lines.append("---")
    lines.append("")

    # Title
    lines.append(f"# {page.title}")
    lines.append("")

    # Summary
    if page.summary:
        lines.append(page.summary)
        lines.append("")

    # Related memories
    if memories:
        lines.append("## Source Memories")
        lines.append("")
        for mem in memories[:20]:
            snippet = mem.content[:150].replace("\n", " ")
            tags = ", ".join(mem.tags[:3]) if mem.tags else ""
            tag_str = f" `{tags}`" if tags else ""
            lines.append(f"- {snippet}{tag_str}")
        lines.append("")

    # Related entities (wikilinks)
    if related_entities:
        lines.append("## Related")
        lines.append("")
        for name in related_entities:
            lines.append(f"- [[{name}]]")
        lines.append("")

    return "\n".join(lines)


def _write_index(
    index_path: str,
    pages: List[KnowledgePage],
    stats: Dict[str, Any],
):
    """Write the index.md table of contents."""
    lines = []
    lines.append("---")
    lines.append("title: Memory Wiki Index")
    lines.append(f"exported: {time.strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"total_pages: {stats['pages_exported']}")
    lines.append("---")
    lines.append("")
    lines.append("# Memory Wiki")
    lines.append("")
    lines.append(f"**{stats['pages_exported']}** knowledge pages | "
                 f"**{stats['entities']}** entities | "
                 f"**{stats['concepts']}** concepts | "
                 f"**{stats['topics']}** topics")
    lines.append("")

    # Group by type
    for ptype, label in [("entity", "Entities"), ("concept", "Concepts"), ("topic", "Topics")]:
        typed = [p for p in pages if p.page_type == ptype]
        if typed:
            lines.append(f"## {label}")
            lines.append("")
            for page in sorted(typed, key=lambda p: p.title.lower()):
                subdir = _page_dir(page.page_type)
                fname = _sanitize_filename(page.title) + ".md"
                lines.append(f"- [[{page.title}]] — v{page.version}")
            lines.append("")

    with open(index_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
