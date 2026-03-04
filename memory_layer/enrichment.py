"""
Local LLM Enrichment Pipeline for the Memory Layer.

Runs a lightweight local model (via Ollama or llama-cpp-python) over
each ingested memory chunk to produce an enriched, self-contained
version.  Enrichment includes:

  1. Pronoun / entity resolution  ("he" → "Bernard")
  2. Context injection            (adds surrounding-session context)
  3. Preference extraction         (tags like preference:dark-mode)
  4. Temporal grounding            ("yesterday" → "2026-03-08")

The enriched text is stored in `memory.metadata["enriched_content"]`
and indexed by both FTS5 and the vector store so that retrieval
quality improves without touching the raw content.

Works entirely offline — zero API calls if a local model is used.
Falls back to a regex-based lightweight enricher when no LLM is
available, so the pipeline never blocks ingestion.
"""

from __future__ import annotations

import os
import re
import time
import json
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, Protocol


class LLMBackend(Protocol):
    """Minimal interface any LLM backend must satisfy."""
    def generate(self, prompt: str, max_tokens: int = 512) -> str: ...


_ENRICHMENT_SYSTEM = """You are a memory enrichment engine. Given a raw memory chunk and surrounding context, rewrite the chunk so it is fully self-contained.

Rules:
1. Resolve ALL pronouns (he, she, it, they, that, this) and ambiguous references (the project, that framework, etc.) to their actual entity names using the surrounding context
2. If the chunk mentions relative dates (yesterday, last week, etc.), convert to absolute dates using today's date: {today}
3. Extract any user preferences and append them as [preference: X] tags at the end
4. Keep the rewritten text concise — under 200 words
5. Do NOT add information not present or inferable from the input
6. Return ONLY the enriched text, nothing else
7. If surrounding chunks mention specific names, projects, tools, or topics that the current chunk references implicitly, resolve those references explicitly"""

_ENRICHMENT_USER = """Context (recent conversation):
{context}

Raw memory chunk:
{chunk}

Rewrite the chunk to be self-contained:"""


class OllamaBackend:
    """Ollama REST API backend for local LLM inference."""

    def __init__(self, model: str = "phi3:mini", base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url.rstrip("/")

    def generate(self, prompt: str, max_tokens: int = 512) -> str:
        import urllib.request
        import urllib.error

        payload = json.dumps({
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"num_predict": max_tokens, "temperature": 0.1},
        }).encode()

        req = urllib.request.Request(
            f"{self.base_url}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = json.loads(resp.read())
                return data.get("response", "").strip()
        except (urllib.error.URLError, OSError):
            raise ConnectionError(f"Cannot reach Ollama at {self.base_url}")


class OpenAICompatibleBackend:
    """OpenAI-compatible API backend (works with vLLM, LM Studio, etc.)."""

    def __init__(self, base_url: str, model: str, api_key: str = "not-needed"):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key

    def generate(self, prompt: str, max_tokens: int = 512) -> str:
        import urllib.request

        payload = json.dumps({
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.1,
        }).encode()

        req = urllib.request.Request(
            f"{self.base_url}/v1/chat/completions",
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read())
            return data["choices"][0]["message"]["content"].strip()


class GeminiBackend:
    """Google Gemini API backend for LLM generation."""

    def __init__(self, api_key: str, model: str = "gemini-2.5-flash"):
        self.api_key = api_key
        self.model = model

    def generate(self, prompt: str, max_tokens: int = 512) -> str:
        import urllib.request

        payload = json.dumps({
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "maxOutputTokens": max_tokens,
                "temperature": 0.2,
            },
        }).encode()

        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.model}:generateContent?key={self.api_key}"
        )
        req = urllib.request.Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=90) as resp:
            data = json.loads(resp.read())
            return (
                data["candidates"][0]["content"]["parts"][0]["text"].strip()
            )


def _resolve_relative_dates(text: str) -> str:
    """Best-effort replacement of relative dates with absolute ones."""
    today = datetime.now()
    replacements = {
        r"\byesterday\b": (today - timedelta(days=1)).strftime("%Y-%m-%d"),
        r"\btoday\b": today.strftime("%Y-%m-%d"),
        r"\btomorrow\b": (today + timedelta(days=1)).strftime("%Y-%m-%d"),
        r"\blast week\b": f"week of {(today - timedelta(weeks=1)).strftime('%Y-%m-%d')}",
        r"\blast month\b": f"{(today.replace(day=1) - timedelta(days=1)).strftime('%B %Y')}",
        r"\blast year\b": str(today.year - 1),
    }
    for pattern, replacement in replacements.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return text


def _extract_preferences_regex(text: str) -> List[str]:
    """Regex-based preference extraction fallback."""
    prefs: List[str] = []
    patterns = [
        r"(?:i |user |they )?(?:prefer|like|love|enjoy|use|favor|choose)\s+(.{3,60}?)(?:\.|,|$)",
        r"(?:my |their )?(?:favorite|favourite|preferred)\s+(?:\w+\s+)?(?:is|are)\s+(.{3,40}?)(?:\.|,|$)",
        r"(?:always|usually|typically)\s+(?:use|go with|pick|choose)\s+(.{3,40}?)(?:\.|,|$)",
    ]
    for pat in patterns:
        for m in re.finditer(pat, text, re.IGNORECASE):
            pref = m.group(1).strip().rstrip(".")
            if pref and len(pref) > 2:
                prefs.append(pref)
    return prefs[:5]


def _lightweight_enrich(chunk: str, context: str = "") -> str:
    """
    Regex-based enrichment fallback when no LLM is available.
    Resolves relative dates and extracts obvious preferences.
    """
    enriched = _resolve_relative_dates(chunk)
    prefs = _extract_preferences_regex(enriched)
    if prefs:
        tags = " ".join(f"[preference: {p}]" for p in prefs)
        enriched = f"{enriched} {tags}"
    return enriched


class EnrichmentPipeline:
    """
    Enrichment pipeline that produces self-contained memory chunks.

    Falls back gracefully:
      1. Local LLM (Ollama / llama-cpp / OpenAI-compat) → full enrichment
      2. Regex-only fallback → date resolution + preference extraction
    """

    def __init__(
        self,
        backend: str = "auto",
        model: str = "phi3:mini",
        base_url: str = "http://localhost:11434",
        api_key: str = "",
    ):
        self._llm: Optional[LLMBackend] = None
        self._backend_name = "regex"

        if backend == "none":
            return

        if backend in ("auto", "ollama"):
            try:
                b = OllamaBackend(model=model, base_url=base_url)
                b.generate("test", max_tokens=1)
                self._llm = b
                self._backend_name = f"ollama:{model}"
                return
            except Exception:
                if backend == "ollama":
                    print(f"  ! Ollama not reachable at {base_url}, falling back to regex enrichment")

        if backend in ("auto", "openai_compat"):
            compat_url = os.environ.get("ENRICHMENT_API_URL", base_url)
            compat_model = os.environ.get("ENRICHMENT_MODEL", model)
            compat_key = os.environ.get("ENRICHMENT_API_KEY", api_key or "not-needed")
            if compat_url and compat_url != "http://localhost:11434":
                try:
                    b = OpenAICompatibleBackend(
                        base_url=compat_url, model=compat_model, api_key=compat_key
                    )
                    b.generate("test", max_tokens=1)
                    self._llm = b
                    self._backend_name = f"openai_compat:{compat_model}"
                    return
                except Exception:
                    pass

        if backend in ("auto", "gemini", "regex"):
            google_key = os.environ.get("GOOGLE_API_KEY", "")
            if google_key:
                gemini_model = os.environ.get(
                    "ENRICHMENT_GEMINI_MODEL", "gemini-2.5-flash"
                )
                try:
                    b = GeminiBackend(api_key=google_key, model=gemini_model)
                    self._llm = b
                    self._backend_name = f"gemini:{gemini_model}"
                    return
                except Exception:
                    pass

    @property
    def backend_name(self) -> str:
        return self._backend_name

    @property
    def has_llm(self) -> bool:
        return self._llm is not None

    def generate(self, prompt: str, max_tokens: int = 512) -> Optional[str]:
        """Public interface to the underlying LLM. Returns None if no LLM."""
        if self._llm is None:
            return None
        try:
            result = self._llm.generate(prompt, max_tokens=max_tokens)
            return result if result and len(result.strip()) > 0 else None
        except Exception as e:
            print(f"  ! EnrichmentPipeline.generate failed ({self._backend_name}): {e}")
            return None

    def enrich(
        self,
        chunk: str,
        context: str = "",
        max_tokens: int = 512,
    ) -> str:
        """
        Enrich a single memory chunk.

        Returns the enriched text (always returns something — never fails).
        """
        if not chunk or not chunk.strip():
            return chunk

        if self._llm is None:
            return _lightweight_enrich(chunk, context)

        today_str = datetime.now().strftime("%Y-%m-%d")
        prompt = (
            _ENRICHMENT_SYSTEM.format(today=today_str)
            + "\n\n"
            + _ENRICHMENT_USER.format(context=context or "(no context)", chunk=chunk)
        )

        try:
            result = self._llm.generate(prompt, max_tokens=max_tokens)
            if result and len(result) > 10:
                return result
        except Exception:
            pass

        return _lightweight_enrich(chunk, context)

    def enrich_batch(
        self,
        chunks: List[str],
        context: str = "",
    ) -> List[str]:
        """Enrich multiple chunks sequentially."""
        return [self.enrich(c, context=context) for c in chunks]
