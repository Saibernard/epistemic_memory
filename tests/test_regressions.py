"""Regression tests for the 2026-07 stabilization pass.

Each test pins one of the fixed lifecycle bugs:
  1. Supersede/FAISS sync invariant (resurrection via rebuild)
  2. Namespace write isolation (cross-tenant supersession)
  3. Distinct-fact identifier guard (supersession cascade)
  4. correct_memory resurrection / self-supersede destruction
  5. Decay write-back compounding
  6. Batch (extraction) store path dedup
  7. MCP per-project namespace derivation
"""

import time

import pytest

from memory_layer.core import _identifier_conflict


def _needs_real_model(brain):
    if getattr(brain.embeddings, "_using_fallback", False):
        pytest.skip("needs the real embedding model (semantic similarity)")


def _tokens(text):
    import re
    return set(re.findall(r"[a-z0-9_]+", text.lower()))


# ── 1. Supersede / FAISS invariant ─────────────────────────────────────

def test_supersede_keeps_faiss_synced_and_health_ok(brain):
    _needs_real_model(brain)
    brain.remember("The CEO of Initech is John Smith")
    brain.remember("The CEO of Initech is now Jane Doe")

    db_count = brain.storage.count_active_memories_with_embeddings()
    assert db_count == brain.memory_index.size

    report = brain.health_check()
    assert report.get("faiss_synced") is True
    assert not any("FAISS out of sync" in i for i in report.get("issues", []))


def test_rebuild_does_not_resurrect_superseded(brain):
    _needs_real_model(brain)
    m1 = brain.remember("The project deadline is March 3rd")
    old_id = (m1[0] if isinstance(m1, list) else m1).id
    brain.remember("The project deadline is now April 7th")

    brain._rebuild_faiss_indices()
    assert old_id not in brain.memory_index._id_to_idx


def test_reaffirmed_fact_not_hijacked_by_ghost(brain):
    _needs_real_model(brain)
    m1 = brain.remember("The database engine is MariaDB")
    old_id = (m1[0] if isinstance(m1, list) else m1).id
    brain.remember("The database engine is now SQLite")
    brain._rebuild_faiss_indices()

    m3 = brain.remember("The database engine is MariaDB")
    m3 = m3[0] if isinstance(m3, list) else m3
    # The dead (superseded) memory must not swallow the re-affirmation
    assert m3.is_current is True
    assert m3.id != old_id


# ── 2. Namespace write isolation ───────────────────────────────────────

def test_cross_namespace_update_does_not_supersede(brain):
    _needs_real_model(brain)
    ma = brain.remember("Team lead is Alice Johnson", namespace="proj_a")
    ma = ma[0] if isinstance(ma, list) else ma
    brain.remember("Team lead is now Bob Cratchit", namespace="proj_b")

    fresh = brain.storage.get_memory(ma.id)
    assert fresh.is_current is True
    assert fresh.epistemic_status != "contradicted"
    assert "superseded_by" not in fresh.metadata


def test_recall_associations_stay_in_namespace(brain):
    brain.remember("Postgres tuning notes for this service", namespace="ns1")
    brain.remember("Postgres tuning guide chapter two", namespace="ns2")
    results = brain.recall("postgres tuning", namespace="ns1", top_k=5)
    for r in results:
        assert r.memory.namespace == "ns1"
        for assoc_id in r.associations:
            assoc = brain.storage.get_memory(assoc_id)
            assert assoc is None or assoc.namespace == "ns1"


# ── 3. Distinct-fact identifier guard ─────────────────────────────────

def test_identifier_conflict_blocks_distinct_tickets():
    assert _identifier_conflict(
        "Note 468: the ml team decided option 6 for ticket T-1468",
        "Note 469: the infra team decided option 0 for ticket T-1469",
    ) is True


def test_identifier_conflict_blocks_same_template_instances():
    # Identical template, several changed numbers → different instances
    assert _identifier_conflict(
        "Note 3: the api team decided option 3 for ticket T-1003",
        "Note 0: the api team decided option 0 for ticket T-1000",
    ) is True


def test_identifier_conflict_blocks_same_head_word_ids():
    # "ticket <id>" on both sides → instances of the same kind of thing
    assert _identifier_conflict(
        "Bug in ticket T-1468 about the navbar",
        "Bug in ticket T-1469 about dark mode",
    ) is True


def test_identifier_conflict_allows_pure_value_update():
    assert _identifier_conflict(
        "The service uses Postgres 15",
        "The service uses Postgres 16",
    ) is False


def test_identifier_conflict_allows_date_update():
    # Changed date value (different preceding words) → genuine update
    assert _identifier_conflict(
        "The project deadline is now April 7th",
        "The project deadline is March 3rd",
    ) is False


def test_identifier_conflict_ignores_no_identifier_updates():
    assert _identifier_conflict(
        "User's favorite language is now Go, not Rust",
        "User's favorite language is Rust",
    ) is False


def test_no_supersession_cascade_across_distinct_facts(brain):
    _needs_real_model(brain)
    ids = []
    for i in range(6):
        team = ["api", "db", "ui"][i % 3]
        m = brain.remember(
            f"Note {i}: the {team} team decided option {i % 4} "
            f"for ticket T-{1000 + i}"
        )
        ids.append((m[0] if isinstance(m, list) else m).id)
    current = [
        brain.storage.get_memory(mid).is_current for mid in ids
    ]
    # Previously a chain of false supersessions killed most of these
    assert sum(current) >= 5


# ── 4. correct_memory ──────────────────────────────────────────────────

def test_correct_memory_marks_old_noncurrent(brain):
    _needs_real_model(brain)
    m = brain.remember("The API rate limit is 100 requests per minute")
    m = m[0] if isinstance(m, list) else m
    new = brain.correct_memory(
        m.id, "The API rate limit is 250 requests per minute",
        reason="limit raised",
    )
    assert new is not None and new.id != m.id
    assert new.epistemic_status == "verified"
    assert new.confidence == pytest.approx(0.9)

    old = brain.storage.get_memory(m.id)
    assert old.is_current is False
    assert old.metadata.get("superseded_by") == new.id
    assert m.id not in brain.memory_index._id_to_idx


def test_correct_memory_near_identical_is_verification_not_destruction(brain):
    _needs_real_model(brain)
    m = brain.remember("the deploy pipeline runs on github actions")
    m = m[0] if isinstance(m, list) else m
    # "Correcting" to essentially identical content must not destroy it
    new = brain.correct_memory(
        m.id, "The deploy pipeline runs on GitHub Actions",
    )
    assert new is not None
    fresh = brain.storage.get_memory(new.id)
    assert fresh.is_current is True
    assert fresh.epistemic_status == "verified"
    assert fresh.metadata.get("superseded_by") != fresh.id


# ── 5. Decay: no compounding write-back ───────────────────────────────

def test_decay_pass_does_not_rewrite_strength(brain):
    m = brain.remember("A fact that will sit unused for a while")
    m = m[0] if isinstance(m, list) else m
    # Backdate so decay is measurable but far above the floor
    m.last_accessed = time.time() - 3 * 86400
    brain.storage.update_memory(m)

    brain.run_decay()
    s1 = brain.storage.get_memory(m.id).strength
    brain.run_decay()
    s2 = brain.storage.get_memory(m.id).strength

    # Stored strength stays anchored at last_accessed semantics; repeated
    # passes must not compound (previously each pass shrank it again).
    assert s1 == pytest.approx(m.strength)
    assert s2 == pytest.approx(s1)


# ── 6. Batch (extraction) path dedup ──────────────────────────────────

def test_batch_store_deduplicates_repeated_facts(brain):
    facts = [{"content": "User works at Globex Corporation", "tags": []}]
    from memory_layer.models import MemoryType
    brain._batch_store_extracted_facts(
        [dict(f) for f in facts], "profile text", MemoryType.SEMANTIC, 0.6,
    )
    before = brain.get_stats().total_memories
    brain._batch_store_extracted_facts(
        [dict(f) for f in facts], "profile text again", MemoryType.SEMANTIC, 0.6,
    )
    after = brain.get_stats().total_memories
    assert after == before  # reinforced, not duplicated


# ── 7. MCP per-project namespace ──────────────────────────────────────

def test_mcp_namespace_env_override(monkeypatch):
    from memory_layer import mcp as mcp_mod
    monkeypatch.setenv("MEMORY_NAMESPACE", "custom-space")
    assert mcp_mod._derive_project_namespace() == "custom-space"


def test_mcp_namespace_derives_from_cwd(monkeypatch, tmp_path):
    from memory_layer import mcp as mcp_mod
    monkeypatch.delenv("MEMORY_NAMESPACE", raising=False)
    proj = tmp_path / "My Frontend_App"
    proj.mkdir()
    monkeypatch.chdir(proj)
    ns = mcp_mod._derive_project_namespace()
    assert ns.startswith("my-frontend_app-")
    assert ns != "global"
    # Deterministic across calls
    assert mcp_mod._derive_project_namespace() == ns


def test_mcp_namespace_home_is_global(monkeypatch):
    import os
    from memory_layer import mcp as mcp_mod
    monkeypatch.delenv("MEMORY_NAMESPACE", raising=False)
    monkeypatch.chdir(os.path.expanduser("~"))
    assert mcp_mod._derive_project_namespace() == "global"
