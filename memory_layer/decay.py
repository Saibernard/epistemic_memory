"""
Decay & Reinforcement Engine for the Memory Layer.

Implements biologically-inspired memory dynamics:

1. FORGETTING CURVE (Ebbinghaus, 1885):
   Memories naturally weaken over time if not recalled.
   retention = e^(-t / stability)

2. SPACED REPETITION EFFECT:
   Each recall strengthens the memory AND slows future decay.
   More recalls → more stable memory → slower decay.

3. IMPORTANCE WEIGHTING:
   Important memories decay much slower than trivial ones.

4. TYPE-AWARE DECAY:
   Different memory types decay at different rates:
   - Procedural (how-to knowledge): 5× slower — learned skills persist
   - Semantic (facts): 3× slower — knowledge is durable
   - Episodic (events): baseline — specific events fade fastest

The result: memories that are frequently needed stay vivid,
while unused memories gradually fade - just like human memory.
"""

import math
import time
from typing import Dict, List

from .models import Memory, MemoryType
from .storage import MemoryStorage

TYPE_STABILITY_MULTIPLIERS: Dict[str, float] = {
    "procedural": 5.0,
    "semantic": 3.0,
    "episodic": 1.0,
}

LEVEL_STABILITY_MULTIPLIERS: Dict[int, float] = {
    0: 1.0,    # raw facts — baseline
    1: 3.0,    # patterns — more stable
    2: 10.0,   # traits/preferences — very stable
    3: 50.0,   # identity/values — near permanent
}

DECAY_BATCH_SIZE = 500


class DecayEngine:
    """
    Manages memory strength dynamics using the Ebbinghaus forgetting curve
    enhanced with spaced repetition, importance weighting, and per-type
    stability rates.

    Decay Formula:
        type_mult = TYPE_STABILITY_MULTIPLIERS[memory_type]
        stability = base_stability × type_mult × (1 + importance × imp_mult) × ln(2 + access_count)
        current_strength = stored_strength × e^(-time_elapsed / stability)

    Reinforcement:
        On recall: strength += boost × (1 - strength)  [diminishing returns]
        access_count += 1 (slows future decay)
        last_accessed = now (resets decay clock)
    """

    def __init__(
        self,
        base_stability: float = 86400.0,
        importance_multiplier: float = 3.0,
        reinforcement_boost: float = 0.2,
        min_strength: float = 0.01,
    ):
        self.base_stability = base_stability
        self.importance_multiplier = importance_multiplier
        self.reinforcement_boost = reinforcement_boost
        self.min_strength = min_strength

    def compute_current_strength(self, memory: Memory) -> float:
        """
        Compute the current effective strength of a memory after decay.

        Procedural memories decay ~5× slower, semantic ~3× slower
        than episodic, reflecting how human memory works — you forget
        what happened last Tuesday long before you forget how to ride
        a bike or that Paris is in France.
        """
        time_elapsed = time.time() - memory.last_accessed

        if time_elapsed <= 0:
            return memory.strength

        type_mult = TYPE_STABILITY_MULTIPLIERS.get(
            memory.memory_type.value if hasattr(memory.memory_type, 'value')
            else str(memory.memory_type),
            1.0,
        )
        # Phase 2A: Higher abstraction levels decay much slower
        level = getattr(memory, "abstraction_level", 0) or 0
        level_mult = LEVEL_STABILITY_MULTIPLIERS.get(level, 1.0)

        importance_factor = 1.0 + (memory.importance * self.importance_multiplier)
        access_factor = math.log(2 + memory.access_count)

        stability = (
            self.base_stability * type_mult * level_mult
            * importance_factor * access_factor
        )

        decay = math.exp(-time_elapsed / stability)
        current_strength = memory.strength * decay

        return max(self.min_strength, current_strength)

    def reinforce(self, memory: Memory, boost: float = None) -> Memory:
        """
        Reinforce a memory — called when it's recalled or used.

        Implements the spacing effect:
        - Strength increases (with diminishing returns near 1.0)
        - Access count increases (making future decay slower)
        - Last accessed time resets (restarting the decay clock)
        """
        if boost is None:
            boost = self.reinforcement_boost

        current = self.compute_current_strength(memory)
        memory.strength = min(1.0, current + boost * (1.0 - current))
        memory.access_count += 1
        memory.last_accessed = time.time()

        return memory

    def apply_decay_to_all(self, storage: MemoryStorage) -> Dict:
        """
        Apply decay to all active memories in batches to control memory
        usage at scale.

        Truly forgotten memories (strength < min_strength) are deactivated.
        """
        stats = {
            "processed": 0,
            "decayed": 0,
            "forgotten": 0,
            "stable": 0,
        }

        for mtype in MemoryType:
            memories = storage.get_all_memories(
                active_only=True, memory_type=mtype,
            )
            for i in range(0, len(memories), DECAY_BATCH_SIZE):
                batch = memories[i: i + DECAY_BATCH_SIZE]
                self._process_batch(batch, storage, stats)

        return stats

    def _process_batch(
        self, batch: List[Memory], storage: MemoryStorage, stats: Dict
    ):
        for memory in batch:
            current_strength = self.compute_current_strength(memory)
            stats["processed"] += 1

            if current_strength <= self.min_strength:
                storage.deactivate_memory(memory.id)
                stats["forgotten"] += 1
            elif current_strength < memory.strength * 0.99:
                memory.strength = current_strength
                storage.update_memory(memory)
                stats["decayed"] += 1
            else:
                stats["stable"] += 1

    def preview_decay(self, storage: MemoryStorage, hours_ahead: float = 24.0) -> Dict:
        """
        Preview what would be forgotten in the next N hours without
        actually applying decay. Useful for dashboards.
        """
        future_time = time.time() + hours_ahead * 3600
        will_forget: List[str] = []
        will_weaken: List[str] = []

        for memory in storage.get_all_memories(active_only=True):
            time_elapsed = future_time - memory.last_accessed
            if time_elapsed <= 0:
                continue

            type_mult = TYPE_STABILITY_MULTIPLIERS.get(
                memory.memory_type.value, 1.0,
            )
            importance_factor = 1.0 + (memory.importance * self.importance_multiplier)
            access_factor = math.log(2 + memory.access_count)
            stability = self.base_stability * type_mult * importance_factor * access_factor

            projected = memory.strength * math.exp(-time_elapsed / stability)
            if projected <= self.min_strength:
                will_forget.append(memory.id)
            elif projected < memory.strength * 0.8:
                will_weaken.append(memory.id)

        return {
            "hours_ahead": hours_ahead,
            "will_forget": len(will_forget),
            "will_weaken": len(will_weaken),
            "forget_ids": will_forget[:20],
            "weaken_ids": will_weaken[:20],
        }
