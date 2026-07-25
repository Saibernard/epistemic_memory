"""
Warm daemon management + thin HTTP client for the Memory Layer.

The efficiency problem this solves: the MCP server, the Claude Code
hooks, and the CLI each used to spin up their own Python process and
load the embedding model (seconds of CPU per call), each holding a
private in-RAM FAISS copy over the same database.

The fix: ONE long-lived local daemon (the existing FastAPI server,
``memory-layer serve``) keeps the model and index hot; every other
entry point becomes a thin HTTP client with millisecond latency.  If
the daemon is not running (and cannot be started), callers fall back
to their old in-process path — nothing ever breaks, it's just slower.

Security posture: the daemon binds 127.0.0.1 only.  Set
``MEMORY_API_KEY`` to also require a key on localhost.

Stdlib-only on the client side (urllib) so hooks stay dependency-free.
"""

import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional

_HEALTH_TIMEOUT = 0.4        # fast probe — don't stall hooks
_DEFAULT_TIMEOUT = 6.0
_SPAWN_WAIT_SECS = 40.0      # first spawn loads the embedding model; a
                             # restart may also overlap the old daemon's
                             # graceful shutdown holding the port


class DaemonUnavailable(Exception):
    """Raised by MemoryClient when the daemon can't serve the request."""


def _home_dir() -> str:
    return os.path.join(os.path.expanduser("~"), ".memory-layer")


def daemon_url() -> str:
    override = os.environ.get("MEMORY_DAEMON_URL", "").strip()
    if override:
        return override.rstrip("/")
    host, port = "127.0.0.1", "8484"
    try:
        from .config import load_config
        cfg = load_config()
        host = cfg.get("server", "host", fallback=host) or host
        port = cfg.get("server", "port", fallback=port) or port
    except Exception:
        pass
    if host in ("0.0.0.0", "::"):
        host = "127.0.0.1"
    return f"http://{host}:{port}"


def _request(
    method: str,
    path: str,
    payload: Optional[dict] = None,
    timeout: float = _DEFAULT_TIMEOUT,
    base_url: Optional[str] = None,
) -> Any:
    url = (base_url or daemon_url()) + path
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(
        url, data=data, method=method,
        headers={"Content-Type": "application/json"},
    )
    api_key = os.environ.get("MEMORY_API_KEY", "").strip()
    if api_key:
        req.add_header("x-api-key", api_key)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read()
            return json.loads(body) if body else None
    except (urllib.error.URLError, urllib.error.HTTPError, OSError,
            json.JSONDecodeError, TimeoutError) as exc:
        raise DaemonUnavailable(str(exc)) from exc


def daemon_matches_db(db_path: str) -> bool:
    """
    True when the daemon would serve the SAME database this process
    targets.  The daemon always serves the configured default DB; if the
    caller overrides MEMORY_DB_PATH to somewhere else (tests, secondary
    stores), routing to the daemon would silently read/write the wrong
    data.  An explicit MEMORY_DAEMON_URL overrides the check — the
    operator is telling us which daemon to trust.
    """
    if os.environ.get("MEMORY_DAEMON_URL", "").strip():
        return True
    try:
        from .config import get_db_path
        default_db = get_db_path()
    except Exception:
        default_db = os.path.join(_home_dir(), "memory.db")
    try:
        return os.path.realpath(db_path) == os.path.realpath(default_db)
    except OSError:
        return False


def is_daemon_up(base_url: Optional[str] = None) -> bool:
    try:
        health = _request(
            "GET", "/health", timeout=_HEALTH_TIMEOUT, base_url=base_url,
        )
        return isinstance(health, dict)
    except DaemonUnavailable:
        return False


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError):
        # PermissionError means the pid exists but isn't ours
        return True if isinstance(sys.exc_info()[1], PermissionError) else False
    except OSError:
        return False


def ensure_daemon(wait: bool = False) -> Optional[str]:
    """
    Return the daemon base URL, spawning it in the background if needed.

    ``wait=False`` (default) fires-and-forgets the spawn and returns None
    when the daemon isn't up yet — callers use their fallback this time
    and get the warm path on the next call.  ``wait=True`` blocks until
    healthy (or gives up after _SPAWN_WAIT_SECS).
    """
    url = daemon_url()
    if is_daemon_up(url):
        return url

    home = _home_dir()
    lock_path = os.path.join(home, "daemon.lock")
    try:
        os.makedirs(home, exist_ok=True)
        # Stale-lock cleanup: a lock whose pid is gone is from a crash
        if os.path.exists(lock_path):
            try:
                with open(lock_path) as fh:
                    old_pid = int(fh.read().strip() or "0")
                if old_pid and _pid_alive(old_pid):
                    # Someone is already starting/running the daemon
                    return _wait_up(url) if wait else None
            except (ValueError, OSError):
                pass
            try:
                os.unlink(lock_path)
            except OSError:
                pass

        fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return _wait_up(url) if wait else None
    except OSError:
        return None

    try:
        log_path = os.path.join(home, "daemon.log")
        log = open(log_path, "ab")
        proc = subprocess.Popen(
            [sys.executable, "-m", "memory_layer", "serve"],
            stdout=log, stderr=log,
            stdin=subprocess.DEVNULL,
            start_new_session=True,     # survives the parent hook/MCP exit
            cwd=os.path.expanduser("~"),
        )
        os.write(fd, str(proc.pid).encode())
    except Exception:
        try:
            os.unlink(lock_path)
        except OSError:
            pass
        return None
    finally:
        os.close(fd)

    return _wait_up(url) if wait else None


def _wait_up(url: str) -> Optional[str]:
    deadline = time.time() + _SPAWN_WAIT_SECS
    while time.time() < deadline:
        if is_daemon_up(url):
            return url
        time.sleep(0.4)
    return None


class MemoryClient:
    """Thin client over the warm daemon.  Raises DaemonUnavailable so
    callers can fall back to an in-process MemoryManager."""

    def __init__(self, base_url: Optional[str] = None):
        self.base_url = base_url or daemon_url()

    def is_up(self) -> bool:
        return is_daemon_up(self.base_url)

    def remember(
        self,
        content: str,
        importance: float = 0.5,
        tags: Optional[List[str]] = None,
        namespace: str = "default",
        memory_type: str = "semantic",
    ) -> Dict:
        return _request("POST", "/remember", {
            "content": content,
            "importance": importance,
            "tags": tags or [],
            "namespace": namespace,
            "memory_type": memory_type,
        }, timeout=15.0, base_url=self.base_url)

    def recall(
        self,
        query: str,
        top_k: int = 5,
        namespace: str = "default",
        min_strength: float = 0.05,
        tags: Optional[List[str]] = None,
        include_history: bool = False,
    ) -> List[Dict]:
        payload: Dict[str, Any] = {
            "query": query,
            "top_k": top_k,
            "namespace": namespace,
            "min_strength": min_strength,
            "include_history": include_history,
        }
        if tags:
            payload["tags"] = tags
        return _request(
            "POST", "/recall", payload, timeout=8.0, base_url=self.base_url,
        ) or []

    def record_episode(
        self,
        user_message: str,
        assistant_response: str,
        importance: float = 0.4,
        tags: Optional[List[str]] = None,
        namespace: str = "default",
    ) -> Dict:
        return _request("POST", "/episode", {
            "user_message": user_message,
            "assistant_response": assistant_response,
            "importance": importance,
            "tags": tags or [],
            "namespace": namespace,
        }, timeout=20.0, base_url=self.base_url)

    def health(self) -> Dict:
        return _request(
            "GET", "/health", timeout=2.0, base_url=self.base_url,
        )

    def stats(self) -> Dict:
        return _request(
            "GET", "/stats", timeout=4.0, base_url=self.base_url,
        )
