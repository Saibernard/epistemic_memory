"""
LLM Fact Extraction for the Memory Layer.

Two variants:
    LLMFactExtractor  — OpenAI API (gpt-4o-mini default). Best quality.
    LocalFactExtractor — Routes through any enrichment LLM (Ollama/local).
                         Simpler prompts, tolerant parsing. No API key needed.

Example:
    Input:  "Bernard prefers dark mode, uses Python 3.11, and cofounded Wolf AI."
    Output: [
        {"content": "User prefers dark mode", "importance": 0.7, "tags": ["preference", "ui"]},
        {"content": "User uses Python 3.11", "importance": 0.6, "tags": ["preference", "python"]},
        {"content": "User is cofounder of Wolf AI", "importance": 0.8, "tags": ["identity", "work"]},
    ]
"""

from __future__ import annotations

import os
import json
import re
import time
from typing import List, Dict, Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .enrichment import EnrichmentPipeline


_MAX_RETRIES = 3
_RETRY_DELAY = 1.0

# Minimum content length to trigger extraction (short content is already atomic)
MIN_EXTRACT_LENGTH = 80

# Maximum content length to send to LLM in one call (avoid timeout on huge inputs)
MAX_EXTRACT_CHUNK = 3000

# Maximum total facts to extract from a single remember() call
MAX_FACTS_PER_CALL = 30

# System prompt for fact extraction
_SYSTEM_PROMPT = """You are a fact extraction engine for a memory system. Given user content, extract individual atomic facts along with temporal and relationship information.

Rules:
1. Each fact must be a single, self-contained statement
2. Preserve specifics: names, numbers, dates, versions, preferences
3. Do NOT add information that isn't in the original content
4. Do NOT extract trivial or obvious facts
5. Assign importance 0.0-1.0: identity/preferences=0.7-0.9, technical=0.5-0.7, context=0.3-0.5
6. Add relevant tags from: [preference, identity, technical, project, decision, workflow, tool, language, framework, personal, work, opinion]
7. For conversations: extract WHO said/did WHAT, WHEN, and any key decisions or preferences
8. Keep each fact under 120 characters when possible
9. Return at most 15 facts — focus on the most important ones
10. Extract any dates or time references mentioned (both document dates and event dates)
11. Extract entity relationships as triples: subject -> relation_type -> object

Respond ONLY with a JSON object with these keys:
- "facts": array of objects with keys: "content" (string), "importance" (0.0-1.0), "tags" (array of strings)
- "event_dates": array of objects with keys: "date" (ISO 8601 string or null), "type" (string like "event", "deadline", "appointment"), "description" (string)
- "relationships": array of objects with keys: "subject" (string), "relation" (string like WORKS_AT, PREFERS, LIVES_IN, USES, STUDIES, KNOWS, CREATED, SWITCHED_TO), "object" (string), "temporal" (string or null, e.g. "since 2024", "as of March")

If the content is already a single atomic fact, return it as the only element in the facts array.
If no dates or relationships are found, return empty arrays for those fields."""

_USER_PROMPT_TEMPLATE = """Extract facts from this content:

{content}"""


class LLMFactExtractor:
    """
    Extracts structured facts from raw content using OpenAI GPT models.

    Usage:
        extractor = LLMFactExtractor(api_key="sk-...")
        facts = extractor.extract("Bernard prefers dark mode and uses Python 3.11")
        # Returns: [{"content": "...", "importance": 0.7, "tags": [...]}, ...]
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
    ):
        self.model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self._client = None

        if not self.api_key:
            raise ValueError(
                "OpenAI API key required for LLM extraction. "
                "Set OPENAI_API_KEY env var or pass api_key="
            )

        self._init_client()

    def _init_client(self):
        """Initialize the OpenAI client."""
        try:
            from openai import OpenAI
            self._client = OpenAI(api_key=self.api_key)
            print(f"  + LLM fact extraction enabled: {self.model}")
        except ImportError:
            raise ImportError(
                "openai package required for LLM extraction. "
                "Install with: pip install openai"
            )

    def extract(self, content: str) -> List[Dict[str, Any]]:
        """
        Extract atomic facts from content.

        For large content (> MAX_EXTRACT_CHUNK chars), the text is split into
        chunks and each chunk is extracted separately, then results are merged
        and capped at MAX_FACTS_PER_CALL.

        Args:
            content: Raw text to extract facts from

        Returns:
            List of dicts with keys: content, importance, tags
            Returns a single-element list with the original content if
            extraction fails or content is too short.
        """
        # Short content is likely already atomic
        if len(content.strip()) < MIN_EXTRACT_LENGTH:
            return [{"content": content.strip(), "importance": 0.7, "tags": []}]

        # For large content, chunk and extract per chunk
        if len(content) > MAX_EXTRACT_CHUNK:
            return self._chunked_extract(content)

        return self._extract_single(content)

    def _chunked_extract(self, content: str) -> List[Dict[str, Any]]:
        """Split large content into chunks, extract facts from each, merge."""
        chunks = []
        start = 0
        overlap = 200
        while start < len(content):
            end = start + MAX_EXTRACT_CHUNK
            chunk = content[start:end]
            if chunk.strip():
                chunks.append(chunk.strip())
            if end >= len(content):
                break
            start += MAX_EXTRACT_CHUNK - overlap

        all_facts = []
        seen_content = set()
        for chunk in chunks:
            facts = self._extract_single(chunk)
            for fact in facts:
                # Deduplicate by content similarity
                normalized = fact["content"].lower().strip()
                if normalized not in seen_content:
                    seen_content.add(normalized)
                    all_facts.append(fact)
            if len(all_facts) >= MAX_FACTS_PER_CALL:
                break

        # Sort by importance and cap
        all_facts.sort(key=lambda f: f.get("importance", 0.5), reverse=True)
        return all_facts[:MAX_FACTS_PER_CALL] if all_facts else [
            {"content": content[:500].strip(), "importance": 0.7, "tags": []}
        ]

    def _extract_single(self, content: str) -> List[Dict[str, Any]]:
        """Extract facts from a single chunk of content."""
        for attempt in range(_MAX_RETRIES):
            try:
                response = self._client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": _SYSTEM_PROMPT},
                        {"role": "user", "content": _USER_PROMPT_TEMPLATE.format(content=content)},
                    ],
                    temperature=0.1,  # Low temp for consistent extraction
                    max_tokens=2000,
                    response_format={"type": "json_object"},
                )

                raw = response.choices[0].message.content.strip()
                parsed = json.loads(raw)

                # Handle both {"facts": [...]} and [...] formats
                if isinstance(parsed, dict):
                    facts = parsed.get("facts", parsed.get("results", []))
                    if not facts:
                        for v in parsed.values():
                            if isinstance(v, list):
                                facts = v
                                break
                    extracted_event_dates = parsed.get("event_dates", [])
                    extracted_relationships = parsed.get("relationships", [])
                elif isinstance(parsed, list):
                    facts = parsed
                    extracted_event_dates = []
                    extracted_relationships = []
                else:
                    facts = []
                    extracted_event_dates = []
                    extracted_relationships = []

                # Validate and normalize facts
                validated = []
                for fact in facts:
                    if not isinstance(fact, dict) or "content" not in fact:
                        continue
                    validated.append({
                        "content": str(fact["content"]).strip(),
                        "importance": float(fact.get("importance", 0.7)),
                        "tags": [str(t) for t in fact.get("tags", [])],
                    })

                # Attach extracted temporal and relationship data
                if validated:
                    validated[0]["_event_dates"] = [
                        ed for ed in extracted_event_dates
                        if isinstance(ed, dict)
                    ]
                    validated[0]["_relationships"] = [
                        r for r in extracted_relationships
                        if isinstance(r, dict) and "subject" in r and "relation" in r and "object" in r
                    ]
                    return validated

                # If extraction returned nothing useful, fall through to default
                break

            except json.JSONDecodeError:
                if attempt < _MAX_RETRIES - 1:
                    time.sleep(_RETRY_DELAY)
                    continue
                break
            except Exception as e:
                if attempt < _MAX_RETRIES - 1:
                    wait = _RETRY_DELAY * (2 ** attempt)
                    print(f"  ! LLM extract retry {attempt + 1}/{_MAX_RETRIES}: {e}")
                    time.sleep(wait)
                else:
                    print(f"  ! LLM extraction failed, storing raw content: {e}")
                    break

        # Fallback: return original content as a single fact
        return [{"content": content[:500].strip(), "importance": 0.7, "tags": []}]

    def should_extract(self, content: str) -> bool:
        """Check if content is worth extracting (long enough, multi-fact)."""
        if len(content.strip()) < MIN_EXTRACT_LENGTH:
            return False
        indicators = content.count(". ") + content.count("\n") + content.count("- ")
        return indicators >= 1


# ── Local variant prompts ─────────────────────────────────────────

_LOCAL_EXTRACT_PROMPT = """Extract individual facts from this content. Return one fact per line, prefixed with a dash.
Each fact must be a single self-contained statement. Preserve specific names, numbers, dates, and preferences.
Only extract what is explicitly stated — do not add information.

Content:
{content}

Facts:"""

_LOCAL_EXTRACT_JSON_PROMPT = """Extract individual atomic facts from this content. Return ONLY a JSON array of objects.
Each object has: "content" (the fact as a string), "importance" (0.0 to 1.0), "tags" (array of keyword strings).

Importance guide: identity/preferences = 0.7-0.9, technical = 0.5-0.7, context = 0.3-0.5
Tag options: preference, identity, technical, project, decision, workflow, tool, language, framework, personal, work

Content:
{content}

JSON:"""

_TAG_KEYWORDS = {
    "prefer": "preference", "like": "preference", "use": "tool",
    "python": "language", "javascript": "language", "java": "language",
    "react": "framework", "vue": "framework", "django": "framework",
    "work": "work", "job": "work", "company": "work", "team": "work",
    "project": "project", "build": "project", "create": "project",
    "name": "identity", "live": "personal", "born": "personal",
    "decide": "decision", "switch": "decision", "chose": "decision",
}


class LocalFactExtractor:
    """
    Extracts facts from content using a local LLM via the enrichment pipeline.

    Tries structured JSON extraction first, falls back to line-by-line parsing.
    Designed for Ollama models (Qwen, Phi, Llama, Mistral, etc.).
    """

    def __init__(self, enrichment: "EnrichmentPipeline"):
        self.enrichment = enrichment
        if enrichment and enrichment.has_llm:
            print(f"  + LLM fact extraction enabled: local ({enrichment.backend_name})")

    def extract(self, content: str) -> List[Dict[str, Any]]:
        """Extract atomic facts from content via local LLM."""
        if len(content.strip()) < MIN_EXTRACT_LENGTH:
            return [{"content": content.strip(), "importance": 0.7, "tags": []}]

        if not self.enrichment or not self.enrichment.has_llm:
            return [{"content": content[:500].strip(), "importance": 0.7, "tags": []}]

        if len(content) > MAX_EXTRACT_CHUNK:
            return self._chunked_extract(content)

        return self._extract_single(content)

    def _chunked_extract(self, content: str) -> List[Dict[str, Any]]:
        """Split large content and extract per chunk."""
        chunks = []
        start = 0
        overlap = 200
        while start < len(content):
            end = start + MAX_EXTRACT_CHUNK
            chunk = content[start:end]
            if chunk.strip():
                chunks.append(chunk.strip())
            if end >= len(content):
                break
            start += MAX_EXTRACT_CHUNK - overlap

        all_facts = []
        seen = set()
        for chunk in chunks:
            for fact in self._extract_single(chunk):
                key = fact["content"].lower().strip()
                if key not in seen:
                    seen.add(key)
                    all_facts.append(fact)
            if len(all_facts) >= MAX_FACTS_PER_CALL:
                break

        all_facts.sort(key=lambda f: f.get("importance", 0.5), reverse=True)
        return all_facts[:MAX_FACTS_PER_CALL] if all_facts else [
            {"content": content[:500].strip(), "importance": 0.7, "tags": []}
        ]

    def _extract_single(self, content: str) -> List[Dict[str, Any]]:
        """Extract facts from one chunk, trying JSON first then line-based."""
        # Strategy 1: Try structured JSON extraction
        facts = self._try_json_extract(content)
        if facts and len(facts) > 0:
            return facts

        # Strategy 2: Fall back to simple line-based extraction
        facts = self._try_line_extract(content)
        if facts and len(facts) > 0:
            return facts

        return [{"content": content[:500].strip(), "importance": 0.7, "tags": []}]

    def _try_json_extract(self, content: str) -> List[Dict[str, Any]]:
        """Attempt structured JSON extraction."""
        prompt = _LOCAL_EXTRACT_JSON_PROMPT.format(content=content[:MAX_EXTRACT_CHUNK])
        raw = self.enrichment.generate(prompt, max_tokens=1500)
        if not raw:
            return []

        parsed = self._parse_json_array(raw)
        if not parsed:
            return []

        validated = []
        for item in parsed:
            if not isinstance(item, dict) or "content" not in item:
                continue
            c = str(item["content"]).strip()
            if len(c) < 5:
                continue
            validated.append({
                "content": c,
                "importance": float(item.get("importance", 0.7)),
                "tags": [str(t) for t in item.get("tags", [])] if isinstance(item.get("tags"), list) else [],
            })
        return validated[:15]

    def _try_line_extract(self, content: str) -> List[Dict[str, Any]]:
        """Fall back to line-by-line extraction with simple prompts."""
        prompt = _LOCAL_EXTRACT_PROMPT.format(content=content[:MAX_EXTRACT_CHUNK])
        raw = self.enrichment.generate(prompt, max_tokens=1000)
        if not raw:
            return []

        facts = []
        for line in raw.strip().split("\n"):
            line = line.strip().lstrip("-•*0123456789.) ")
            if len(line) < 10:
                continue
            if line.lower().startswith(("here", "the ", "note", "fact")):
                continue
            tags = self._auto_tag(line)
            importance = self._estimate_importance(line)
            facts.append({
                "content": line[:300],
                "importance": importance,
                "tags": tags,
            })

        return facts[:15]

    @staticmethod
    def _parse_json_array(text: str) -> Optional[list]:
        """Parse a JSON array from local LLM output with tolerant fallbacks."""
        if not text:
            return None
        text = text.strip()

        # Direct parse
        try:
            result = json.loads(text)
            if isinstance(result, list):
                return result
            if isinstance(result, dict):
                for v in result.values():
                    if isinstance(v, list):
                        return v
        except json.JSONDecodeError:
            pass

        # Extract from code fence
        fence = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
        if fence:
            try:
                result = json.loads(fence.group(1).strip())
                if isinstance(result, list):
                    return result
            except json.JSONDecodeError:
                pass

        # Find first [...] block
        start = text.find("[")
        end = text.rfind("]") + 1
        if start >= 0 and end > start:
            try:
                return json.loads(text[start:end])
            except json.JSONDecodeError:
                pass

        return None

    @staticmethod
    def _auto_tag(text: str) -> List[str]:
        """Infer tags from keyword presence."""
        tags = set()
        lower = text.lower()
        for keyword, tag in _TAG_KEYWORDS.items():
            if keyword in lower:
                tags.add(tag)
        return list(tags)[:3]

    @staticmethod
    def _estimate_importance(text: str) -> float:
        """Heuristic importance scoring."""
        lower = text.lower()
        if any(w in lower for w in ("name is", "i am", "my name", "cofounder", "founder", "ceo")):
            return 0.9
        if any(w in lower for w in ("prefer", "favorite", "always", "never", "love", "hate")):
            return 0.8
        if any(w in lower for w in ("use", "work", "build", "project", "team")):
            return 0.7
        if any(w in lower for w in ("version", "framework", "language", "tool")):
            return 0.6
        return 0.5

    def should_extract(self, content: str) -> bool:
        """Check if content is worth extracting."""
        if len(content.strip()) < MIN_EXTRACT_LENGTH:
            return False
        if not self.enrichment or not self.enrichment.has_llm:
            return False
        indicators = content.count(". ") + content.count("\n") + content.count("- ")
        return indicators >= 1
