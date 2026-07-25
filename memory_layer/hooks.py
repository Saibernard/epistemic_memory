#!/usr/bin/env python3
"""
Claude Code lifecycle hooks for the Memory Layer.

Two entry points, wired via ~/.claude/settings.json:

  python3 -m memory_layer.hooks session-start
      SessionStart hook.  Reads this project's memories (plus shared
      global ones) STRAIGHT from SQLite — no embedding model, no
      MemoryManager — and injects a compact "memory brief" into the new
      session's context.  Claude starts already knowing the project.

  python3 -m memory_layer.hooks record-turn
      Stop hook (async).  Parses the turn transcript, extracts the user
      request + what Claude did, and records it as an episode in the
      project's namespace.  The memory layer's consolidation later
      compresses episodes into durable semantic facts — so memory
      accumulates with zero manual "remember this" prompts.

  python3 -m memory_layer.hooks prompt-recall
      UserPromptSubmit hook.  Semantically recalls memories relevant to
      the prompt the user JUST typed via the warm daemon (~50ms) and
      injects them — targeted context, every prompt, not just a static
      session brief.  Skips silently when the daemon isn't up.

All hooks prefer the warm daemon (millisecond latency) and fall back to
an in-process MemoryManager where sensible.  Every path is best-effort:
any failure exits 0 silently — a memory hook must never break a coding
session.
"""

import hashlib
import json
import os
import re
import sqlite3
import sys
import time

MAX_BRIEF_ITEMS = 12
MIN_TURN_CHARS = 40          # skip trivial turns ("ok", "thanks")
MAX_FIELD_CHARS = 1200       # cap stored episode fields

GLOBAL_NAMESPACE = "global"


# ── namespace derivation (keep in sync with memory_layer/mcp.py) ──────

def _derive_project_namespace() -> str:
    override = os.environ.get("MEMORY_NAMESPACE", "").strip()
    if override:
        return override
    cwd = os.getcwd()
    home = os.path.expanduser("~")
    if cwd in ("/", home, ""):
        return GLOBAL_NAMESPACE
    base = re.sub(
        r"[^a-z0-9_-]+", "-", os.path.basename(cwd).lower()
    ).strip("-") or "project"
    digest = hashlib.sha1(cwd.encode("utf-8", "replace")).hexdigest()[:6]
    return f"{base}-{digest}"


def _db_path() -> str:
    return os.environ.get(
        "MEMORY_DB_PATH",
        os.path.join(os.path.expanduser("~"), ".memory-layer", "memory.db"),
    )


# ── SessionStart: inject memory brief ─────────────────────────────────

def session_start() -> int:
    path = _db_path()
    if not os.path.exists(path):
        return 0
    ns = _derive_project_namespace()

    try:
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=2.0)
        rows = conn.execute(
            "SELECT namespace, memory_type, content, importance "
            "FROM memories "
            "WHERE is_active = 1 AND is_current = 1 "
            "AND namespace IN (?, ?) "
            "ORDER BY importance DESC, last_accessed DESC LIMIT ?",
            (ns, GLOBAL_NAMESPACE, MAX_BRIEF_ITEMS),
        ).fetchall()
        conn.close()
    except sqlite3.Error:
        return 0

    if not rows:
        return 0

    proj = [r for r in rows if r[0] != GLOBAL_NAMESPACE]
    glob = [r for r in rows if r[0] == GLOBAL_NAMESPACE]

    lines = ["# Memory Layer — persistent memory for this project", ""]
    if proj:
        lines.append("## This project (remembered from earlier sessions)")
        for _, mtype, content, _ in proj:
            lines.append(f"- {content.strip()[:300]}")
        lines.append("")
    if glob:
        lines.append("## About the user (global)")
        for _, mtype, content, _ in glob:
            lines.append(f"- {content.strip()[:300]}")
        lines.append("")
    lines.append(
        "(Use the memory_recall tool for anything deeper; use "
        "memory_remember to store new lasting facts or decisions.)"
    )

    print(json.dumps({
        "hookSpecificOutput": {
            "hookEventName": "SessionStart",
            "additionalContext": "\n".join(lines),
        }
    }))

    # Pre-warm: fire-and-forget daemon spawn so prompt-recall and
    # record-turn get the millisecond path from the first prompt on.
    try:
        from .daemon import ensure_daemon
        ensure_daemon(wait=False)
    except Exception:
        pass
    return 0


# ── UserPromptSubmit: inject memories relevant to THIS prompt ─────────

def prompt_recall() -> int:
    try:
        payload = json.load(sys.stdin)
    except Exception:
        return 0
    prompt = str(payload.get("prompt", "") or "").strip()
    # Too short to carry meaning, or a slash-command
    if len(prompt) < 15 or prompt.startswith("/"):
        return 0

    ns = _derive_project_namespace()
    try:
        from .daemon import (
            MemoryClient, DaemonUnavailable, ensure_daemon, daemon_matches_db,
        )
        if not daemon_matches_db(_db_path()):
            return 0
        client = MemoryClient()
        try:
            rows = client.recall(
                prompt[:1000], top_k=5, namespace=ns, min_strength=0.05,
            )
            if ns != GLOBAL_NAMESPACE:
                seen = {r.get("memory", {}).get("id") for r in rows}
                for r in client.recall(
                    prompt[:1000], top_k=3, namespace=GLOBAL_NAMESPACE,
                    min_strength=0.05,
                ):
                    if r.get("memory", {}).get("id") not in seen:
                        rows.append(r)
        except DaemonUnavailable:
            # Never load the model synchronously here — that would add
            # seconds to every prompt.  Warm it for next time instead.
            ensure_daemon(wait=False)
            return 0
    except Exception:
        return 0

    # Keep only genuinely relevant hits — injecting weak matches on
    # every prompt is noise the model then has to ignore.
    relevant = [
        r for r in rows
        if (r.get("composite_score") or 0) >= 0.35
    ][:5]
    if not relevant:
        return 0

    lines = ["Relevant memories for this request (from Memory Layer):"]
    for r in relevant:
        content = (r.get("memory", {}).get("content") or "").strip()
        if content:
            lines.append(f"- {content[:280]}")
    if len(lines) == 1:
        return 0

    print(json.dumps({
        "hookSpecificOutput": {
            "hookEventName": "UserPromptSubmit",
            "additionalContext": "\n".join(lines),
        }
    }))
    return 0


# ── Stop: record the turn as an episode ───────────────────────────────

def _extract_text(message) -> str:
    """Pull plain text out of a transcript message (string or blocks)."""
    if isinstance(message, str):
        return message
    if isinstance(message, dict):
        content = message.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    parts.append(block.get("text", ""))
            return "\n".join(parts)
    return ""


def record_turn() -> int:
    try:
        payload = json.load(sys.stdin)
    except Exception:
        return 0
    transcript_path = payload.get("transcript_path", "")
    if not transcript_path or not os.path.exists(transcript_path):
        return 0

    last_user, last_assistant = "", ""
    try:
        with open(transcript_path, errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                etype = entry.get("type")
                text = _extract_text(entry.get("message", entry))
                if not text.strip():
                    continue
                if etype == "user":
                    # A new user message starts a new turn
                    last_user, last_assistant = text, ""
                elif etype == "assistant":
                    last_assistant = text
    except OSError:
        return 0

    if len(last_user.strip()) < MIN_TURN_CHARS:
        return 0
    if not last_assistant.strip():
        return 0
    # Skip harness-injected pseudo-user content: system reminders, task
    # notifications, slash-command output, and skill/command instruction
    # payloads are not things the user said — capturing them poisons
    # recall with framework text.
    stripped = last_user.lstrip()
    if stripped.startswith(("<system-reminder", "<task-notification",
                            "<local-command", "<command-name",
                            "#", "Caveat:")):
        return 0
    if "<command-name>" in last_user or "<system-reminder>" in last_user:
        return 0

    ns = _derive_project_namespace()
    user_text = last_user.strip()[:MAX_FIELD_CHARS]
    assistant_text = last_assistant.strip()[:MAX_FIELD_CHARS]

    # Fast path: warm daemon (~90ms).  Spawns it for next time if down.
    try:
        from .daemon import (
            MemoryClient, DaemonUnavailable, ensure_daemon, daemon_matches_db,
        )
        if daemon_matches_db(_db_path()):
            try:
                MemoryClient().record_episode(
                    user_message=user_text,
                    assistant_response=assistant_text,
                    importance=0.4,
                    tags=["auto_captured"],
                    namespace=ns,
                )
                return 0
            except DaemonUnavailable:
                ensure_daemon(wait=False)
    except Exception:
        pass

    # Fallback: in-process (async hook, so the model load doesn't block
    # the session — it's just background CPU).
    try:
        import contextlib
        import io
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            from memory_layer import MemoryManager
            brain = MemoryManager(db_path=_db_path())
            brain.record_episode(
                user_message=user_text,
                assistant_response=assistant_text,
                importance=0.4,
                tags=["auto_captured"],
                namespace=ns,
            )
            brain.shutdown()
    except Exception:
        return 0
    return 0


def main() -> int:
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "session-start":
        return session_start()
    if cmd == "record-turn":
        return record_turn()
    if cmd == "prompt-recall":
        return prompt_recall()
    print("usage: python3 -m memory_layer.hooks "
          "{session-start|record-turn|prompt-recall}",
          file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
