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
import tempfile
import time
from pathlib import Path
from typing import List, Dict

from utils.logger import get_logger

logger = get_logger("memory")

# Message type
Message = Dict[str, str]


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
        """
        # Import here to avoid circular import issues. Explicit
        # constructor arguments always win over .env defaults.
        try:
            from config import jarvis_config, memory_config

            cfg_turns = jarvis_config.MEMORY_MAX_TURNS
            cfg_chars = jarvis_config.MEMORY_MAX_CHARS
            cfg_persist = memory_config.PERSIST
            cfg_file = memory_config.FILE
        except Exception:
            cfg_turns = 6
            cfg_chars = 3000
            cfg_persist = True
            cfg_file = "data/conversation.json"

        self.max_turns = max_turns if max_turns is not None else cfg_turns
        self.max_chars = max_chars if max_chars is not None else cfg_chars

        if persist_path is not None:
            self._persist_path: str | None = persist_path or None
        elif cfg_persist and cfg_file:
            self._persist_path = _resolve_path(cfg_file)
        else:
            self._persist_path = None

        self._messages: List[Message] = []
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
        self._messages.append({
            "role": "assistant",
            "content": content.strip()
        })
        self._enforce_limit()
        self._save()

    def get_history(self) -> List[Message]:
        """Return full conversation history."""
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
        messages: List[Message] = [
            {"role": "system", "content": system_prompt}
        ]
        messages.extend(self._messages)

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
        self._messages.clear()
        self._save()
        logger.info("Conversation memory cleared.")

    # ── Persistence ───────────────────────────────────────────

    def _save(self) -> None:
        """Write history to disk atomically. Never raises."""
        if not self._persist_path:
            return
        try:
            path = Path(self._persist_path)
            path.parent.mkdir(parents=True, exist_ok=True)
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
        except Exception as e:
            logger.warning(
                f"Could not persist conversation memory: {e}"
            )

    def _load(self) -> None:
        """Load history from disk. Missing/corrupt files are handled
        safely — a corrupt file is backed up and memory starts fresh."""
        if not self._persist_path:
            return
        path = Path(self._persist_path)
        if not path.is_file():
            return  # nothing to load — first run
        try:
            with open(path, encoding="utf-8") as f:
                raw = json.load(f)
            messages = []
            for item in raw:
                if not isinstance(item, dict):
                    continue
                role = item.get("role")
                content = item.get("content")
                if role in ("user", "assistant") and isinstance(content, str):
                    messages.append({"role": role, "content": content.strip()})
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
            try:
                backup = path.with_name(
                    f"{path.stem}.corrupt-{int(time.time())}{path.suffix}"
                )
                os.replace(str(path), str(backup))
                logger.warning(
                    f"Conversation memory file was corrupt ({e}); "
                    f"backed up to {backup.name} and starting fresh."
                )
                self._save()
            except OSError:
                logger.warning(
                    f"Conversation memory file was corrupt ({e}); "
                    "starting fresh."
                )

    def _enforce_limit(self) -> None:
        """
        Remove oldest turn when memory exceeds max_turns.
        Always removes in pairs (user + assistant) to keep
        history coherent.
        """
        max_messages = self.max_turns * 2
        while len(self._messages) > max_messages:
            # Remove oldest pair
            self._messages.pop(0)
            if self._messages:
                self._messages.pop(0)
            logger.debug("Dropped oldest turn from memory.")

    def __len__(self) -> int:
        return len(self._messages)

    def __repr__(self) -> str:
        turns = len(self._messages) // 2
        return (
            f"ConversationMemory("
            f"turns={turns}, max={self.max_turns})"
        )