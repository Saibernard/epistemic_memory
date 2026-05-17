"""Shared fixtures for Memory Layer tests."""

import os
import tempfile
import pytest

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


@pytest.fixture
def tmp_db(tmp_path):
    """Return a temporary database path that is cleaned up after the test."""
    return str(tmp_path / "test_memory.db")


@pytest.fixture
def brain(tmp_db):
    """Create a MemoryManager instance using the fallback (hash) embeddings."""
    from memory_layer import MemoryManager
    return MemoryManager(db_path=tmp_db, embedding_mode="local")


@pytest.fixture
def storage(tmp_db):
    """Create a bare MemoryStorage for low-level tests."""
    from memory_layer.storage import MemoryStorage
    return MemoryStorage(db_path=tmp_db)
