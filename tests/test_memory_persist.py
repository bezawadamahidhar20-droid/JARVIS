"""Issue 5 — persistent conversation memory.

History must survive a component restart, and a missing/corrupt memory
file must never crash startup (it is backed up and recreated).
"""

import json

from brain.memory import ConversationMemory


def test_history_survives_restart(tmp_path):
    path = str(tmp_path / "conv.json")
    mem = ConversationMemory(max_turns=6, max_chars=3000, persist_path=path)
    mem.add_user_message("who is elon musk")
    mem.add_assistant_message("He runs several companies.")

    # A brand-new component reading the same file must see the history.
    fresh = ConversationMemory(max_turns=6, max_chars=3000, persist_path=path)
    history = fresh.get_history()
    assert len(history) == 2
    assert history[0] == {"role": "user", "content": "who is elon musk"}
    assert history[1]["role"] == "assistant"


def test_missing_file_starts_empty(tmp_path):
    path = str(tmp_path / "does-not-exist.json")
    mem = ConversationMemory(max_turns=6, max_chars=3000, persist_path=path)
    assert len(mem) == 0
    mem.add_user_message("hi")
    # File is created on first save.
    assert json.loads((tmp_path / "does-not-exist.json").read_text())[0]["content"] == "hi"


def test_corrupt_file_starts_safely(tmp_path):
    path = tmp_path / "conv.json"
    path.write_text("{ this is not valid json !!!", encoding="utf-8")

    mem = ConversationMemory(max_turns=6, max_chars=3000, persist_path=str(path))
    assert len(mem) == 0  # fresh start, no crash

    # The corrupt file was backed up and the working file recreated
    # with a clean (empty) conversation.
    backups = list(tmp_path.glob("conv.corrupt-*.json"))
    assert backups, "corrupt file should be backed up, not lost"
    assert path.is_file()
    assert json.loads(path.read_text(encoding="utf-8")) == []


def test_garbage_json_types_are_filtered(tmp_path):
    path = str(tmp_path / "conv.json")
    (tmp_path / "conv.json").write_text(
        json.dumps([
            {"role": "user", "content": "keep me"},
            {"role": "admin", "content": "drop me"},
            {"role": "assistant", "content": 42},
            "not a dict",
        ]),
        encoding="utf-8",
    )
    mem = ConversationMemory(max_turns=6, max_chars=3000, persist_path=path)
    history = mem.get_history()
    assert len(history) == 1
    assert history[0]["content"] == "keep me"


def test_clear_is_persisted(tmp_path):
    path = str(tmp_path / "conv.json")
    mem = ConversationMemory(max_turns=6, max_chars=3000, persist_path=path)
    mem.add_user_message("hello")
    mem.clear()

    fresh = ConversationMemory(max_turns=6, max_chars=3000, persist_path=path)
    assert len(fresh) == 0


def test_persistence_disabled_with_empty_path():
    mem = ConversationMemory(max_turns=6, max_chars=3000, persist_path="")
    mem.add_user_message("hello")
    assert mem._persist_path is None
