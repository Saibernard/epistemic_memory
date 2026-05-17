"""Tests for agent framework integration adapters."""

import pytest


class TestLangChainAdapter:
    """Test LangChain memory adapter."""

    def test_load_memory_variables(self, brain):
        from memory_layer.integrations import LangChainMemory
        mem = LangChainMemory(manager=brain, namespace="test-lc")
        brain.remember("User prefers Python", namespace="test-lc")

        result = mem.load_memory_variables({"input": "What language?"})
        assert "history" in result
        assert isinstance(result["history"], str)

    def test_save_context(self, brain):
        from memory_layer.integrations import LangChainMemory
        mem = LangChainMemory(manager=brain, namespace="test-lc")
        mem.save_context(
            {"input": "What is Python?"},
            {"output": "Python is a programming language"},
        )
        # Should not raise

    def test_memory_variables_property(self, brain):
        from memory_layer.integrations import LangChainMemory
        mem = LangChainMemory(manager=brain)
        assert mem.memory_variables == ["history"]

    def test_clear(self, brain):
        from memory_layer.integrations import LangChainMemory
        mem = LangChainMemory(manager=brain)
        mem.clear()  # Should not raise


class TestCrewAIAdapter:
    """Test CrewAI memory adapter."""

    def test_save_and_search(self, brain):
        from memory_layer.integrations import CrewAIMemory
        mem = CrewAIMemory(manager=brain, namespace="test-crew")
        mid = mem.save("The project uses FastAPI", agent_name="researcher")
        assert mid

        results = mem.search("What framework?")
        assert isinstance(results, list)

    def test_get_context(self, brain):
        from memory_layer.integrations import CrewAIMemory
        mem = CrewAIMemory(manager=brain, namespace="test-crew")
        mem.save("Important project detail")
        context = mem.get_context("project details")
        assert isinstance(context, str)

    def test_save_task_result(self, brain):
        from memory_layer.integrations import CrewAIMemory
        mem = CrewAIMemory(manager=brain, namespace="test-crew")
        mid = mem.save_task_result(
            task_description="Research competitors",
            result="Found 3 competitors",
            agent_name="analyst",
        )
        assert mid


class TestLlamaIndexAdapter:
    """Test LlamaIndex memory adapter."""

    def test_put_and_get(self, brain):
        from memory_layer.integrations import LlamaIndexMemory
        mem = LlamaIndexMemory(manager=brain, namespace="test-li")
        mid = mem.put({"role": "user", "content": "I need help with React"})
        assert mid

        results = mem.get("React help")
        assert isinstance(results, list)

    def test_get_all(self, brain):
        from memory_layer.integrations import LlamaIndexMemory
        mem = LlamaIndexMemory(manager=brain, namespace="test-li")
        mem.put({"role": "user", "content": "Test memory"})
        all_mems = mem.get_all()
        assert isinstance(all_mems, list)
        assert len(all_mems) >= 1

    def test_to_string(self, brain):
        from memory_layer.integrations import LlamaIndexMemory
        mem = LlamaIndexMemory(manager=brain, namespace="test-li")
        mem.put({"role": "user", "content": "String format test"})
        s = mem.to_string("format")
        assert isinstance(s, str)


class TestOpenAIAdapter:
    """Test OpenAI thread memory adapter."""

    def test_save_and_get_context(self, brain):
        from memory_layer.integrations import OpenAIThreadMemory
        mem = OpenAIThreadMemory(manager=brain)
        mid = mem.save_message(
            thread_id="thread_123",
            role="user",
            content="I want to learn Python",
        )
        assert mid

        context = mem.get_context("thread_123", "Python")
        assert isinstance(context, list)

    def test_system_prompt_context(self, brain):
        from memory_layer.integrations import OpenAIThreadMemory
        mem = OpenAIThreadMemory(manager=brain)
        mem.save_message("thread_456", "user", "I use VS Code")
        prompt = mem.get_system_prompt_context("thread_456", "editor preferences")
        assert isinstance(prompt, str)


class TestAutoGenAdapter:
    """Test AutoGen memory adapter."""

    def test_add_and_search(self, brain):
        from memory_layer.integrations import AutoGenMemory
        mem = AutoGenMemory(manager=brain, agent_name="coder")
        mid = mem.add("User prefers TypeScript for frontend")
        assert mid

        results = mem.search("frontend language preference")
        assert isinstance(results, list)

    def test_conversation_save(self, brain):
        from memory_layer.integrations import AutoGenMemory
        mem = AutoGenMemory(manager=brain, agent_name="planner")
        ids = mem.save_conversation([
            {"role": "user", "content": "Plan a REST API"},
            {"role": "assistant", "content": "Here's the plan..."},
        ])
        assert len(ids) == 2


class TestVercelAIAdapter:
    """Test Vercel AI SDK memory adapter."""

    def test_save_and_get_context(self, brain):
        from memory_layer.integrations import VercelAIMemory
        mem = VercelAIMemory(manager=brain, namespace="test-vercel")
        mid = mem.save("User location is San Francisco")
        assert mid

        context = mem.get_context("Where is the user?")
        assert isinstance(context, str)

    def test_get_messages(self, brain):
        from memory_layer.integrations import VercelAIMemory
        mem = VercelAIMemory(manager=brain, namespace="test-vercel")
        mem.save("Test message", role="system")
        msgs = mem.get_messages("test")
        assert isinstance(msgs, list)

    def test_save_interaction(self, brain):
        from memory_layer.integrations import VercelAIMemory
        mem = VercelAIMemory(manager=brain, namespace="test-vercel")
        mem.save_interaction("Hello", "Hi there!")
        # Should not raise
