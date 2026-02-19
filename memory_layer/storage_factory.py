"""
Storage Factory — creates the right storage backend based on config.

Reads the [storage] section from config.ini (or env vars) and returns
an instance that satisfies the StorageBackend protocol.

Usage:
    from memory_layer.storage_factory import create_storage

    storage = create_storage()            # uses config
    storage = create_storage("sqlite")    # explicit override
"""

import os
from typing import Optional

from .storage_protocol import StorageBackend


def create_storage(
    backend: Optional[str] = None,
    *,
    # SQLite
    sqlite_path: Optional[str] = None,
    # Postgres
    postgres_url: Optional[str] = None,
    # DynamoDB
    aws_region: Optional[str] = None,
    dynamodb_table: Optional[str] = None,
) -> StorageBackend:
    """
    Create and return a storage backend instance.

    Priority:
        1. Explicit `backend` argument
        2. MEMORY_STORAGE_BACKEND env var
        3. Config file [storage] backend
        4. Default: "sqlite"
    """
    if backend is None:
        backend = os.environ.get("MEMORY_STORAGE_BACKEND")

    if backend is None:
        try:
            from .config import load_config
            cfg = load_config()
            backend = cfg.get("storage", "backend", fallback="sqlite")
        except Exception:
            backend = "sqlite"

    backend = backend.lower().strip()

    if backend == "sqlite":
        return _create_sqlite(sqlite_path)
    elif backend == "postgres":
        return _create_postgres(postgres_url)
    elif backend == "dynamodb":
        return _create_dynamodb(aws_region, dynamodb_table)
    else:
        raise ValueError(
            f"Unknown storage backend: '{backend}'. "
            f"Supported: sqlite, postgres, dynamodb"
        )


def _create_sqlite(path: Optional[str] = None) -> StorageBackend:
    if path is None:
        path = os.environ.get("MEMORY_DB_PATH")
    if path is None:
        try:
            from .config import get_db_path
            path = get_db_path()
        except Exception:
            path = "memory.db"

    from .storage import MemoryStorage
    return MemoryStorage(db_path=path)


def _create_postgres(url: Optional[str] = None) -> StorageBackend:
    if url is None:
        url = os.environ.get("MEMORY_POSTGRES_URL")
    if url is None:
        try:
            from .config import load_config
            cfg = load_config()
            url = cfg.get("storage", "postgres_url", fallback=None)
        except Exception:
            pass
    if not url:
        raise ValueError(
            "Postgres backend requires a connection URL.\n"
            "Set MEMORY_POSTGRES_URL or add postgres_url to [storage] in config.ini.\n"
            "Example: postgresql://user:pass@localhost:5432/memory"
        )

    from .storage_postgres import PostgresStorage
    return PostgresStorage(database_url=url)


def _create_dynamodb(
    region: Optional[str] = None,
    table: Optional[str] = None,
) -> StorageBackend:
    if region is None:
        region = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION")
    if table is None:
        table = os.environ.get("MEMORY_DYNAMODB_TABLE")

    if region is None or table is None:
        try:
            from .config import load_config
            cfg = load_config()
            region = region or cfg.get("storage", "aws_region", fallback=None)
            table = table or cfg.get("storage", "dynamodb_table", fallback=None)
        except Exception:
            pass

    if not region:
        raise ValueError(
            "DynamoDB backend requires an AWS region.\n"
            "Set AWS_REGION or add aws_region to [storage] in config.ini."
        )
    if not table:
        raise ValueError(
            "DynamoDB backend requires a table name.\n"
            "Set MEMORY_DYNAMODB_TABLE or add dynamodb_table to [storage] in config.ini."
        )

    from .storage_dynamo import DynamoStorage
    return DynamoStorage(region=region, table_name=table)
