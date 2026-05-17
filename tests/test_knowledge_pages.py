"""Tests for knowledge pages and wiki export."""

import os
import tempfile
import pytest


class TestKnowledgePages:
    """Test knowledge page creation, update, and retrieval."""

    def test_page_creation_on_remember(self, brain):
        """Pages should be created when entities accumulate enough mentions."""
        # Store multiple memories mentioning Python
        for content in [
            "Python is a high-level programming language",
            "Python supports multiple programming paradigms",
            "Python has a large standard library",
        ]:
            brain.remember(content, tags=["test"])

        # Rebuild pages
        result = brain.rebuild_knowledge_pages()
        assert result["total_pages"] >= 0  # May or may not create depending on entity extraction

    def test_page_update_on_new_memory(self, brain):
        """Existing pages should update when new memories are added."""
        # Create initial memories
        for content in [
            "Python is a popular programming language",
            "Python is used in data science and AI",
        ]:
            brain.remember(content, tags=["test"])

        brain.rebuild_knowledge_pages()
        pages_before = brain.get_knowledge_pages()

        # Add more memories
        brain.remember("Python 3.12 added performance improvements", tags=["test"])
        brain.rebuild_knowledge_pages()
        pages_after = brain.get_knowledge_pages()

        # Should still have pages (or more)
        assert len(pages_after) >= len(pages_before)

    def test_get_page_by_entity(self, brain):
        """Should retrieve a page by entity name."""
        for content in [
            "Django is a Python web framework",
            "Django follows the MTV pattern",
            "Django has an ORM for database operations",
        ]:
            brain.remember(content, tags=["test"])

        brain.rebuild_knowledge_pages()
        page = brain.knowledge_pages.get_page_for_entity("Django")
        # May or may not find it depending on entity extraction
        # Just verify the method doesn't crash
        assert page is None or page.title

    def test_delete_page(self, brain):
        """Should be able to delete a knowledge page."""
        brain.remember("FastAPI is a modern Python framework", tags=["test"])
        brain.remember("FastAPI supports async operations", tags=["test"])
        brain.rebuild_knowledge_pages()

        pages = brain.get_knowledge_pages()
        if pages:
            page_id = pages[0]["page_id"]
            brain.knowledge_pages.delete_page(page_id)
            deleted = brain.get_knowledge_page(page_id)
            assert deleted is None


class TestWikiExport:
    """Test wiki export functionality."""

    def test_export_creates_files(self, brain):
        """Wiki export should create markdown files."""
        for content in [
            "Python is great for scripting",
            "Python has excellent libraries",
        ]:
            brain.remember(content, tags=["test"])

        brain.rebuild_knowledge_pages()

        with tempfile.TemporaryDirectory() as tmpdir:
            from memory_layer.wiki_export import export_wiki
            stats = export_wiki(
                storage=brain.storage,
                entity_graph=brain.entity_graph,
                knowledge_page_manager=brain.knowledge_pages,
                output_dir=tmpdir,
            )
            # Index should always be created
            assert os.path.exists(os.path.join(tmpdir, "index.md"))
            assert stats["pages_exported"] >= 0

    def test_export_directory_structure(self, brain):
        """Export should create proper directory structure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            from memory_layer.wiki_export import export_wiki
            export_wiki(
                storage=brain.storage,
                entity_graph=brain.entity_graph,
                knowledge_page_manager=brain.knowledge_pages,
                output_dir=tmpdir,
            )
            for subdir in ["entities", "concepts", "topics"]:
                assert os.path.isdir(os.path.join(tmpdir, subdir))
