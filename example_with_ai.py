#!/usr/bin/env python3
"""
🧠 How ANY AI uses the Memory Layer

This shows the exact pattern. The memory layer sits BETWEEN
the user and the AI. It's middleware.

WITHOUT memory:
    User → AI → Response (forgets everything next conversation)

WITH memory layer:
    User → [1. Recall relevant memories]
         → [2. Inject memories into AI prompt]  
         → AI → Response
         → [3. Store this interaction in memory]
         → NEXT TIME: AI knows everything from before!

This example works with OpenAI, but the pattern is the SAME
for Claude, Llama, Mistral, Gemini, or any AI.
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import warnings
warnings.filterwarnings("ignore")

from memory_layer import MemoryManager


# ─── Initialize memory (once, reuse forever) ────────────
brain = MemoryManager(db_path="ai_memory.db")


def ai_with_memory(user_message: str) -> str:
    """
    This is the function that gives ANY AI memory.
    
    Step 1: Recall relevant memories
    Step 2: Build a prompt with memory context
    Step 3: Send to AI (OpenAI/Claude/Local LLM/anything)
    Step 4: Store interaction in memory for next time
    """

    # ━━━ STEP 1: Recall relevant memories ━━━
    memories = brain.recall(user_message, top_k=5, min_strength=0.1)
    
    memory_context = ""
    if memories:
        memory_context = "Here is what you remember about this user from past conversations:\n"
        for r in memories:
            memory_context += f"  - {r.memory.content} (confidence: {r.relevance_score:.0%})\n"
        memory_context += "\nUse this knowledge to give a personalized response.\n\n"

    # ━━━ STEP 2: Build the prompt with memory ━━━
    system_prompt = f"""You are a helpful AI assistant with persistent memory.
You remember things about the user across conversations.

{memory_context}"""

    full_prompt = f"{system_prompt}\nUser: {user_message}\nAssistant:"

    # ━━━ STEP 3: Send to ANY AI ━━━
    # 
    # OPTION A: OpenAI
    # from openai import OpenAI
    # client = OpenAI()
    # response = client.chat.completions.create(
    #     model="gpt-4",
    #     messages=[
    #         {"role": "system", "content": system_prompt},
    #         {"role": "user", "content": user_message}
    #     ]
    # )
    # ai_response = response.choices[0].message.content
    #
    # OPTION B: Claude  
    # import anthropic
    # client = anthropic.Anthropic()
    # response = client.messages.create(
    #     model="claude-3-5-sonnet-20241022",
    #     system=system_prompt,
    #     messages=[{"role": "user", "content": user_message}]
    # )
    # ai_response = response.content[0].text
    #
    # OPTION C: Local LLM (ollama)
    # import requests
    # response = requests.post("http://localhost:11434/api/generate", json={
    #     "model": "llama3",
    #     "prompt": full_prompt,
    # })
    # ai_response = response.json()["response"]

    # For this demo, we simulate the AI response:
    ai_response = simulate_ai_response(user_message, memories)

    # ━━━ STEP 4: Store in memory for next time ━━━
    brain.record_episode(
        user_message=user_message,
        assistant_response=ai_response,
        feedback="positive",
    )

    return ai_response


def simulate_ai_response(user_message: str, memories) -> str:
    """
    Simulates what an AI would say using the memory context.
    In real usage, replace this with an actual AI API call.
    """
    if memories:
        # The AI would use the memories to personalize
        top_memory = memories[0].memory.content
        return f"Based on what I remember ({top_memory}), here's my response to your question."
    else:
        return "I don't have any prior context about you yet, but I'm happy to help!"


# ─── Demo: Simulate 3 separate "conversations" ────────────

def main():
    print()
    print("=" * 60)
    print("  🧠 HOW AI USES THE MEMORY LAYER")
    print("  Simulating 3 separate conversations over time")
    print("=" * 60)

    # ── Conversation 1: First meeting ──
    print()
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("  📅 CONVERSATION 1 (Day 1 - First meeting)")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

    # User tells the AI about themselves
    brain.remember("User's name is Sai", importance=0.9)
    brain.remember("User is building an AI memory system", importance=0.8)
    brain.remember("User prefers Python for backend development", importance=0.7)
    brain.remember("User is based in India", importance=0.6)
    
    response = ai_with_memory("Hi! Can you help me with my project?")
    print(f"  User:  Hi! Can you help me with my project?")
    print(f"  AI:    {response}")

    # ── Conversation 2: Next day ──
    print()
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("  📅 CONVERSATION 2 (Day 2 - Comes back)")
    print("  The AI was \"restarted\" but memory persists!")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    
    # Simulate restarting by creating a new brain from same DB
    brain2 = MemoryManager(db_path="ai_memory.db")
    
    # What does the AI remember?
    results = brain2.recall("What do I know about this user?", top_k=5)
    print(f"\n  🧠 What the AI remembers from yesterday:")
    for r in results:
        print(f"     • {r.memory.content}")
    
    response = ai_with_memory("What programming language should I use for my API?")
    print(f"\n  User:  What programming language should I use for my API?")
    print(f"  AI:    {response}")
    print(f"  (AI remembered the user prefers Python!)")

    # ── Conversation 3: A week later ──
    print()
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("  📅 CONVERSATION 3 (Day 7 - A week later)")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    
    response = ai_with_memory("I need help with deployment")
    print(f"  User:  I need help with deployment")
    print(f"  AI:    {response}")
    print(f"  (AI still remembers everything from Day 1!)")

    # ── Show the magic ──
    print()
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("  🔍 UNDER THE HOOD: What the AI's prompt looks like")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    
    memories = brain.recall("deployment help", top_k=3)
    print()
    print('  ┌─────────────────────────────────────────────┐')
    print('  │ SYSTEM PROMPT (sent to AI every time):      │')
    print('  │                                             │')
    print('  │ You are a helpful AI assistant with memory.  │')
    print('  │                                             │')
    print('  │ What you remember about this user:          │')
    for r in memories:
        content = r.memory.content[:45]
        print(f'  │   • {content:45s}│')
    print('  │                                             │')
    print('  │ Use this to personalize your response.       │')
    print('  └─────────────────────────────────────────────┘')
    print('  ┌─────────────────────────────────────────────┐')
    print('  │ USER: I need help with deployment           │')
    print('  └─────────────────────────────────────────────┘')
    print()
    print("  ^ THIS is how the AI \"remembers\" — the memory layer")
    print("    injects past knowledge into every prompt automatically.")
    print()

    # ── The pattern ──
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("  📐 THE PATTERN (works with ANY AI)")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print()
    print("  User sends message")
    print("       │")
    print("       ▼")
    print("  ┌─────────────────────────┐")
    print("  │ 1. brain.recall(message) │ ← Find relevant memories")
    print("  └───────────┬─────────────┘")
    print("              │")
    print("              ▼")
    print("  ┌─────────────────────────┐")
    print("  │ 2. Build prompt:        │ ← Inject memories into prompt")
    print("  │    system = memories    │")
    print("  │    user = message       │")
    print("  └───────────┬─────────────┘")
    print("              │")
    print("              ▼")
    print("  ┌─────────────────────────┐")
    print("  │ 3. Send to AI           │ ← OpenAI / Claude / Llama / any")
    print("  └───────────┬─────────────┘")
    print("              │")
    print("              ▼")
    print("  ┌─────────────────────────┐")
    print("  │ 4. brain.record_episode │ ← Store for next time")
    print("  └───────────┬─────────────┘")
    print("              │")
    print("              ▼")
    print("  Response to user (AI now has memory!)")
    print()

    # Cleanup
    os.remove("ai_memory.db")
    print("  ✅ Demo complete!\n")


if __name__ == "__main__":
    main()
