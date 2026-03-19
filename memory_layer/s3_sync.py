"""
S3 FAISS Index Sync — push/pull FAISS index files to/from S3.

When using a shared backend (DynamoDB, Postgres), FAISS indices are local
files. This module lets multiple servers share the same index via S3:

    sync = S3IndexSync(bucket="my-memory-layer", prefix="indices/")

    # On startup — pull latest index from S3
    sync.pull("memory_mem_idx.faiss")
    sync.pull("memory_pass_idx.faiss")

    # After writes — push updated index to S3
    sync.push("memory_mem_idx.faiss")

Requires: pip install boto3
"""

import os
import time
from pathlib import Path
from typing import Optional


class S3IndexSync:
    """Push/pull FAISS index files to/from S3."""

    def __init__(
        self,
        bucket: str,
        prefix: str = "faiss-indices/",
        region: Optional[str] = None,
        local_dir: Optional[str] = None,
    ):
        try:
            import boto3
        except ImportError:
            raise ImportError(
                "S3 sync requires boto3.\n"
                "Install it: pip install boto3"
            )

        self._bucket = bucket
        self._prefix = prefix.rstrip("/") + "/"
        self._s3 = boto3.client("s3", region_name=region)

        if local_dir is None:
            try:
                from .config import get_home_dir
                local_dir = str(get_home_dir())
            except Exception:
                local_dir = "."
        self._local_dir = local_dir

    def _s3_key(self, filename: str) -> str:
        return self._prefix + os.path.basename(filename)

    def _local_path(self, filename: str) -> str:
        return os.path.join(self._local_dir, os.path.basename(filename))

    def push(self, filename: str) -> bool:
        """Upload a local FAISS index file to S3."""
        local = self._local_path(filename)
        if not os.path.exists(local):
            return False
        key = self._s3_key(filename)
        self._s3.upload_file(local, self._bucket, key)
        return True

    def pull(self, filename: str) -> bool:
        """Download a FAISS index file from S3 to local dir."""
        key = self._s3_key(filename)
        local = self._local_path(filename)
        try:
            self._s3.download_file(self._bucket, key, local)
            return True
        except self._s3.exceptions.ClientError:
            return False

    def push_all(self, base_name: str = "memory"):
        """Push both memory and passage FAISS indices."""
        self.push(f"{base_name}_mem_idx.faiss")
        self.push(f"{base_name}_mem_idx.json")
        self.push(f"{base_name}_pass_idx.faiss")
        self.push(f"{base_name}_pass_idx.json")

    def pull_all(self, base_name: str = "memory"):
        """Pull both memory and passage FAISS indices."""
        self.pull(f"{base_name}_mem_idx.faiss")
        self.pull(f"{base_name}_mem_idx.json")
        self.pull(f"{base_name}_pass_idx.faiss")
        self.pull(f"{base_name}_pass_idx.json")

    def exists(self, filename: str) -> bool:
        """Check if a file exists in S3."""
        key = self._s3_key(filename)
        try:
            self._s3.head_object(Bucket=self._bucket, Key=key)
            return True
        except Exception:
            return False
