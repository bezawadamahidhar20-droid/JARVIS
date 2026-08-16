"""Conversation memory tests."""


from brain.memory import ConversationMemory


def test_add_and_history():
    mem = ConversationMemory(max_turns=6, max_chars=3000)
    mem.add_user_message("hello")
    mem.add_assistant_message("hi there")
    history = mem.get_history()
    assert len(history) == 2
    assert history[0] == {"role": "user", "content": "hello"}
    assert history[1] == {"role": "assistant", "content": "hi there"}


def test_empty_messages_ignored():
    mem = ConversationMemory(max_turns=6, max_chars=3000)
    mem.add_user_message("")
    mem.add_assistant_message("   ")
    assert len(mem) == 0


def test_turn_limit_drops_oldest_pair():
    mem = ConversationMemory(max_turns=2, max_chars=3000)
    for i in range(4):
        mem.add_user_message(f"u{i}")
        mem.add_assistant_message(f"a{i}")
    # max_turns=2 -> at most 4 messages
    assert len(mem) <= 4
    history = mem.get_history()
    # Oldest pair (u0/a0) must be gone.
    texts = [m["content"] for m in history]
    assert "u0" not in texts
    assert "u3" in texts


def test_char_limit_trims_old_turns():
    mem = ConversationMemory(max_turns=6, max_chars=100)
    mem.add_user_message("x" * 40)
    mem.add_assistant_message("y" * 40)
    mem.add_user_message("z" * 40)
    mem.add_assistant_message("w" * 40)
    ctx = mem.get_trimmed_context("SYS")
    # First message is always the system prompt.
    assert ctx[0] == {"role": "system", "content": "SYS"}
    total = sum(len(m["content"]) for m in ctx)
    assert total <= 100
    # The newest turn must never be dropped.
    assert ctx[-1]["content"] == "w" * 40


def test_clear():
    mem = ConversationMemory(max_turns=6, max_chars=3000)
    mem.add_user_message("hello")
    mem.clear()
    assert len(mem) == 0
