"""
brain/memory.py — Conversation memory for JARVIS
Maintains rolling window of conversation turns so
follow-up questions work correctly.

Example:
    User: "Who is Elon Musk?"
    User: "What companies does he own?"
    JARVIS knows "he" = Elon Musk from history.

Persistence: the conversation is saved to a JSON file (MEMORY_FILE,
"data/conversation.json" by default) after every change and reloaded on
startup, so context survives restarts. The file is gitignored and
never stores secrets (only the conversation turns themselves).
"""

import json
import os
import shutil
import tempfile
import threading
import time
from pathlib import Path
from typing import List, Dict, Optional, Callable

from config import jarvis_config, memory_config
from utils.logger import get_logger

logger = get_logger("memory")

__all__ = [
    "ConversationMemory",
]

# Message type
Message = Dict[str, str]

# Hard safety caps on loaded messages: a crafted/corrupt conversation
# file must never inject an oversized system prompt or an unbounded
# message list into the LLM context.
_MAX_MESSAGE_CHARS = 10_000
_MAX_MESSAGE_COUNT = 500
_MAX_FILE_BYTES = 1_000_000  # 1 MB
_VALID_ROLES = ("user", "assistant")


# Repository root (parent of brain/), so the default MEMORY_FILE
# resolves correctly regardless of the current working directory.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _resolve_path(cfg_file: str) -> str:
    """Resolve a configured memory file relative to the project root."""
    p = Path(cfg_file)
    if p.is_absolute():
        return str(p)
    return str(_PROJECT_ROOT / p)


class ConversationMemory:
    """
    Stores conversation as a list of message dicts.
    Sends full history to Ollama on every request
    so follow-up questions work correctly.
    """

    def __init__(
        self,
        max_turns: int | None = None,
        max_chars: int | None = None,
        persist_path: str | None = None,
        on_save_failure: Optional[Callable[[str], None]] = None,
    ):
        """
        Args:
            max_turns: Max number of user+assistant pairs to keep.
                       Older turns dropped when limit exceeded.
                       Defaults to the .env MEMORY_MAX_TURNS value.
            max_chars: Max total characters of history sent to the
                       model per request. Older turns dropped (in
                       pairs) when the total exceeds this.
                       Defaults to the .env MEMORY_MAX_CHARS value.
            persist_path: Optional JSON file to load/save history.
                       None = use MEMORY_FILE from .env (empty string
                       disables persistence entirely).
            on_save_failure: Optional callback invoked when a save has
                       failed (e.g. disk full) so main.py can warn the
                       user via TTS instead of silently losing context.
        """
        # Explicit constructor arguments always win over .env defaults.
        cfg_turns = jarvis_config.MEMORY_MAX_TURNS
        cfg_chars = jarvis_config.MEMORY_MAX_CHARS
        cfg_persist = memory_config.PERSIST
        cfg_file = memory_config.FILE

        self.max_turns = max_turns if max_turns is not None else cfg_turns
        self.max_chars = max_chars if max_chars is not None else cfg_chars

        if persist_path is not None:
            self._persist_path: str | None = persist_path or None
        elif cfg_persist and cfg_file:
            self._persist_path = _resolve_path(cfg_file)
        else:
            self._persist_path = None

        self._lock = threading.RLock()
        self._messages: List[Message] = []
        # Disk-full protection: consecutive save failures are counted so
        # the caller can be warned and we never hammer a full disk.
        self._save_failures = 0
        self._on_save_failure = on_save_failure
        self._load()
        logger.info(
            f"Memory initialized (max {self.max_turns} turns, "
            f"{self.max_chars} chars, "
            f"persist={self._persist_path or 'off'})."
        )

    def add_user_message(self, content: str) -> None:
        """Add a user message to history."""
        if not content or not content.strip():
            return
        with self._lock:
            self._messages.append({
                "role": "user",
                "content": content.strip()
            })
            self._enforce_limit()
        self._save()

    def add_assistant_message(self, content: str) -> None:
        """Add an assistant (JARVIS) response to history."""
        if not content or not content.strip():
            return
        with self._lock:
            self._messages.append({
                "role": "assistant",
                "content": content.strip()
            })
            self._enforce_limit()
        self._save()

    def pop_last(self) -> Message | None:
        """
        Remove and return the most recent message (used to roll back a
        failed interaction). Returns None when empty. Thread-safe.
        """
        with self._lock:
            if not self._messages:
                return None
            return self._messages.pop()

    def get_history(self) -> List[Message]:
        """Return a snapshot of the conversation history (thread-safe)."""
        with self._lock:
            return list(self._messages)

    def get_context_for_ollama(
        self, system_prompt: str
    ) -> List[Message]:
        """
        Build full message list for Ollama.
        Format: [system, user, assistant, user, assistant, ...]

        Delegate to get_trimmed_context() so every caller gets the
        character-trimmed history automatically.
        """
        return self.get_trimmed_context(system_prompt)

    def get_trimmed_context(
        self, system_prompt: str
    ) -> List[Message]:
        """
        Build the message list for Ollama, dropping the oldest turns
        (in pairs) when the total character count exceeds max_chars.

        Returns a new list, so callers can safely modify it (e.g.
        appending the current user message).
        """
        with self._lock:
            history = list(self._messages)

        messages: List[Message] = [
            {"role": "system", "content": system_prompt}
        ]
        messages.extend(history)

        if self.max_chars > 0:
            total = sum(len(m["content"]) for m in messages)
            # Never drop the system message or the last (current) turn.
            while total > self.max_chars and len(messages) > 3:
                removed = messages.pop(1)          # oldest user message
                total -= len(removed["content"])
                if len(messages) > 2 and messages[1]["role"] == "assistant":
                    removed = messages.pop(1)      # its assistant reply
                    total -= len(removed["content"])
                logger.debug(
                    "Dropped oldest turn from trimmed context."
                )

        return messages

    def clear(self) -> None:
        """Clear all conversation history (persisted immediately)."""
        with self._lock:
            self._messages.clear()
        self._save()
        logger.info("Conversation memory cleared.")

    # ── Persistence ───────────────────────────────────────────

    def _save(self) -> None:
        """Write history to disk atomically. Never raises.

        Disk-full protection: the free space is checked before writing;
        on any failure the failure counter is bumped and the configured
        callback is invoked so the user can be warned (context would be
        lost on restart otherwise).
        """
        if not self._persist_path:
            return
        try:
            path = Path(self._persist_path)
            path.parent.mkdir(parents=True, exist_ok=True)

            # Disk-full guard: refuse to write when < 50 MB free, so a
            # full disk produces a clear warning instead of a silent
            # failure (or a huge temp file that never renames).
            try:
                free_bytes = shutil.disk_usage(path.parent).free
                if free_bytes < 50 * 1024 * 1024:
                    raise OSError(
                        f"low disk space on {path.parent} "
                        f"({free_bytes // (1024 * 1024)} MB free)"
                    )
            except OSError as e:
                raise OSError(f"cannot save conversation memory: {e}") from e

            # Atomic write: temp file in the same directory, then rename.
            fd, tmp = tempfile.mkstemp(
                prefix=".conversation_", suffix=".tmp",
                dir=str(path.parent),
            )
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    json.dump(self._messages, f, ensure_ascii=False, indent=2)
                os.replace(tmp, str(path))
            except Exception:
                # Clean up the temp file if the write failed.
                try:
                    os.unlink(tmp)
                except OSError:
                    pass
                raise
            # Success — reset the failure streak.
            if self._save_failures:
                self._save_failures = 0
        except Exception as e:
            self._save_failures += 1
            logger.warning(
                f"Could not persist conversation memory (attempt "
                f"{self._save_failures}): {e}"
            )
            if self._on_save_failure is not None:
                try:
                    self._on_save_failure(
                        "I could not save our conversation to disk. "
                        "It will be lost if I restart."
                    )
                except Exception:
                    pass

    @property
    def save_failure_count(self) -> int:
        """Number of consecutive failed saves (0 = healthy)."""
        return self._save_failures

    def _load(self) -> None:
        """Load history from disk. Missing/corrupt files are handled
        safely — a corrupt file is backed up and memory starts fresh."""
        if not self._persist_path:
            return
        path = Path(self._persist_path)
        if not path.is_file():
            return  # nothing to load — first run
        # Size cap: refuse to trust a giant/crafted conversation file.
        try:
            if path.stat().st_size > _MAX_FILE_BYTES:
                logger.warning(
                    f"Conversation memory file is {path.stat().st_size} "
                    f"bytes (> {_MAX_FILE_BYTES} cap) — treating as corrupt."
                )
                self._messages = []
                self._backup_corrupt(path, "file too large")
                return
        except OSError:
            return
        try:
            with open(path, encoding="utf-8") as f:
                raw = json.load(f)
            # Validate the schema: a list of {role, content} messages
            # with strict role/content checks and size caps, so a
            # crafted file can never inject system prompts or force an
            # unbounded context.
            messages = []
            if isinstance(raw, list):
                for item in raw[: _MAX_MESSAGE_COUNT]:
                    if not isinstance(item, dict):
                        continue
                    role = item.get("role")
                    content = item.get("content")
                    if role in _VALID_ROLES and isinstance(content, str):
                        content = content.strip()
                        if len(content) > _MAX_MESSAGE_CHARS:
                            content = content[:_MAX_MESSAGE_CHARS]
                        messages.append({"role": role, "content": content})
            self._messages = messages
            self._enforce_limit()
            if messages:
                logger.info(
                    f"Loaded {len(messages)} messages from conversation "
                    f"memory ({path})."
                )
        except (json.JSONDecodeError, OSError, ValueError) as e:
            # Corrupt file: keep it for inspection, start fresh, and
            # recreate the working file so the next run is clean.
            self._messages = []
            self._backup_corrupt(path, str(e))

    def _backup_corrupt(self, path: Path, reason: str) -> None:
        """Move a corrupt conversation file aside and start fresh."""
        try:
            backup = path.with_name(
                f"{path.stem}.corrupt-{int(time.time())}{path.suffix}"
            )
            os.replace(str(path), str(backup))
            logger.warning(
                f"Conversation memory file was corrupt ({reason}); "
                f"backed up to {backup.name} and starting fresh."
            )
            self._save()
        except OSError:
            logger.warning(
                f"Conversation memory file was corrupt ({reason}); "
                "starting fresh."
            )

    def _enforce_limit(self) -> None:
        """
        Remove oldest turn when memory exceeds max_turns.
        Always removes in pairs (user + assistant) to keep
        history coherent. Caller must hold ``self._lock``.
        """
        max_messages = self.max_turns * 2
        while len(self._messages) > max_messages:
            # Remove oldest pair
            self._messages.pop(0)
            if self._messages:
                self._messages.pop(0)
            logger.debug("Dropped oldest turn from memory.")

    def __len__(self) -> int:
        with self._lock:
            return len(self._messages)

    def __repr__(self) -> str:
        turns = len(self._messages) // 2
        return (
            f"ConversationMemory("
            f"turns={turns}, max={self.max_turns})"
        )