#!/usr/bin/env python3
"""
🧠 Memory Layer Demo - See it in action!

This demo shows how the memory system works:
1. Storing different types of memories
2. Recalling by meaning (semantic search)
3. Spaced repetition (memories strengthen with use)
4. Automatic association linking
5. Memory consolidation
6. Contradiction detection

Run: python demo.py
"""

from memory_layer import MemoryManager, MemoryType


def main():
    print("=" * 60)
    print("  🧠 MEMORY LAYER DEMO")
    print("  Biologically-Inspired Memory for AI")
    print("=" * 60)
    print()

    # Initialize with a fresh database for the demo
    brain = MemoryManager(db_path="demo_memory.db")

    # ── 1. Store Memories ───────────────────────────
    print("━" * 60)
    print("  📝 STEP 1: Storing memories...")
    print("━" * 60)

    m1 = brain.remember(
        "User's name is Alice and she is a software engineer",
        importance=0.9,
        tags=["identity", "user_info"],
    )
    print(f"  ✓ Stored: '{m1.content[:50]}...' (importance={m1.importance})")

    m2 = brain.remember(
        "User prefers Python over JavaScript for backend work",
        importance=0.7,
        tags=["preference", "programming"],
    )
    print(f"  ✓ Stored: '{m2.content[:50]}...' (importance={m2.importance})")

    m3 = brain.remember(
        "User likes dark mode in all applications",
        tags=["preference", "ui"],
    )
    print(f"  ✓ Stored: '{m3.content[:50]}...' (importance={m3.importance})")

    m4 = brain.remember(
        "Important: User is allergic to peanuts - never suggest peanut recipes",
        importance=0.5,  # Will be auto-boosted due to "important" keyword
        tags=["health", "critical"],
    )
    print(f"  ✓ Stored: '{m4.content[:50]}...' (importance={m4.importance:.2f} - auto-boosted!)")

    # ── 2. Record Episodes ──────────────────────────
    print()
    print("━" * 60)
    print("  💬 STEP 2: Recording interaction episodes...")
    print("━" * 60)

    brain.record_episode(
        user_message="How do I sort a list in Python?",
        assistant_response="Use sorted() for a new list or list.sort() for in-place sorting.",
        feedback="positive",
        tags=["python", "sorting"],
    )
    print("  ✓ Episode recorded: Python sorting (positive feedback)")

    brain.record_episode(
        user_message="What's the best way to handle errors in Python?",
        assistant_response="Use try/except blocks with specific exception types.",
        feedback="positive",
        tags=["python", "error_handling"],
    )
    print("  ✓ Episode recorded: Python error handling (positive feedback)")

    brain.record_episode(
        user_message="Can you help me with Python decorators?",
        assistant_response="Decorators wrap functions. Use @decorator syntax.",
        tags=["python", "decorators"],
    )
    print("  ✓ Episode recorded: Python decorators")

    # ── 3. Learn a Procedure ────────────────────────
    print()
    print("━" * 60)
    print("  📋 STEP 3: Learning a procedure...")
    print("━" * 60)

    proc = brain.learn_procedure(
        name="Deploy to Production",
        description="Alice's deployment workflow for her Python app",
        steps=[
            "Run pytest to ensure all tests pass",
            "Build Docker image with new tag",
            "Push to staging and run smoke tests",
            "If staging is green, deploy to production",
            "Monitor logs for 15 minutes after deploy",
        ],
        tags=["devops", "deployment"],
    )
    print(f"  ✓ Learned procedure: '{proc.metadata['name']}'")
    print(f"    Steps: {len(proc.metadata['steps'])}")

    # ── 4. Semantic Recall ──────────────────────────
    print()
    print("━" * 60)
    print("  🔍 STEP 4: Recalling by meaning (not exact text)...")
    print("━" * 60)

    print()
    print('  Query: "What programming language does the user prefer?"')
    results = brain.recall("What programming language does the user prefer?")
    for i, r in enumerate(results[:3]):
        print(f"    #{i+1} [{r.relevance_score:.3f}] {r.memory.content[:70]}...")
        print(f"        type={r.memory.memory_type.value}, strength={r.effective_strength:.2f}")

    print()
    print('  Query: "food allergies"')
    results = brain.recall("food allergies")
    for i, r in enumerate(results[:3]):
        print(f"    #{i+1} [{r.relevance_score:.3f}] {r.memory.content[:70]}...")

    print()
    print('  Query: "how to release software"')
    results = brain.recall("how to release software")
    for i, r in enumerate(results[:3]):
        print(f"    #{i+1} [{r.relevance_score:.3f}] {r.memory.content[:70]}...")

    # ── 5. Spaced Repetition ────────────────────────
    print()
    print("━" * 60)
    print("  🔄 STEP 5: Spaced repetition in action...")
    print("━" * 60)

    mem = brain.storage.get_memory(m2.id)
    print(f"  Before reinforcement:")
    print(f"    Content: '{mem.content[:50]}...'")
    print(f"    Strength: {mem.strength:.3f}, Access count: {mem.access_count}")

    brain.reinforce_memory(m2.id, boost=0.3)
    brain.reinforce_memory(m2.id, boost=0.3)
    brain.reinforce_memory(m2.id, boost=0.3)

    mem = brain.storage.get_memory(m2.id)
    print(f"  After 3 reinforcements:")
    print(f"    Strength: {mem.strength:.3f}, Access count: {mem.access_count}")
    print(f"    → This memory will now decay MUCH slower!")

    # ── 6. Contradiction Detection ──────────────────
    print()
    print("━" * 60)
    print("  ⚡ STEP 6: Contradiction detection...")
    print("━" * 60)

    contradicting = brain.remember(
        "User prefers JavaScript over Python for backend work",
        tags=["preference", "programming"],
    )
    if contradicting.metadata.get("contradicts"):
        print(f"  ⚠ Contradiction detected!")
        print(f"    New: '{contradicting.content[:60]}...'")
        print(f"    Contradicts {len(contradicting.metadata['contradicts'])} existing memory(ies)")
    else:
        print(f"  Stored (no contradiction flagged with fallback embeddings)")

    # ── 7. Memory Correction ────────────────────────
    print()
    print("━" * 60)
    print("  ✏️  STEP 7: Correcting a memory...")
    print("━" * 60)

    corrected = brain.correct_memory(
        memory_id=m3.id,
        new_content="User likes light mode during the day and dark mode at night",
        reason="User clarified their preference",
    )
    print(f"  Old: '{m3.content}'")
    print(f"  New: '{corrected.content}'")
    print(f"  Importance boosted: {m3.importance:.2f} → {corrected.importance:.2f}")

    # ── 8. Stats ────────────────────────────────────
    print()
    print("━" * 60)
    print("  📊 STEP 8: Memory system statistics")
    print("━" * 60)

    stats = brain.get_stats()
    print(f"  Total memories:     {stats.total_memories}")
    print(f"    Episodic:         {stats.episodic_count}")
    print(f"    Semantic:         {stats.semantic_count}")
    print(f"    Procedural:       {stats.procedural_count}")
    print(f"  Association links:  {stats.total_links}")
    print(f"  Working memory:     {stats.working_memory_size} items")
    print(f"  Avg strength:       {stats.avg_strength:.3f}")
    print(f"  Avg importance:     {stats.avg_importance:.3f}")
    print(f"  Consolidations:     {stats.consolidation_count}")

    # ── 9. Consolidation ────────────────────────────
    print()
    print("━" * 60)
    print("  🌙 STEP 9: Memory consolidation (like sleep)...")
    print("━" * 60)

    consolidation_stats = brain.consolidate()
    print(f"  Episodes analyzed:        {consolidation_stats['episodes_analyzed']}")
    print(f"  Clusters found:           {consolidation_stats['clusters_found']}")
    print(f"  Semantic memories created: {consolidation_stats['semantic_memories_created']}")
    print(f"  Links created:            {consolidation_stats['links_created']}")

    if consolidation_stats["semantic_memories_created"] > 0:
        semantic_memories = brain.storage.get_all_memories(memory_type=MemoryType.SEMANTIC)
        for sm in semantic_memories:
            print(f"\n  📚 Consolidated knowledge:")
            print(f"     {sm.content[:200]}...")

    # ── Done ────────────────────────────────────────
    print()
    print("=" * 60)
    print("  ✅ Demo complete!")
    print()
    print("  The memory persists in: demo_memory.db")
    print("  Run the server with:    python run.py --db demo_memory.db")
    print("  API docs at:            http://localhost:8484/docs")
    print("=" * 60)

    # Clean up demo db
    import os
    os.remove("demo_memory.db")
    print("\n  (Demo database cleaned up)")


if __name__ == "__main__":
    main()
