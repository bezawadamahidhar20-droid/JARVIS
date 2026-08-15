"""
brain/memory.py — Conversation memory for JARVIS
Maintains rolling window of conversation turns so
follow-up questions work correctly.

Example:
    User: "Who is Elon Musk?"
    User: "What companies does he own?"
    JARVIS knows "he" = Elon Musk from history.
"""

from typing import List, Dict
from utils.logger import get_logger

logger = get_logger("memory")

# Message type
Message = Dict[str, str]


class ConversationMemory:
    """
    Stores conversation as a list of message dicts.
    Sends full history to Ollama on every request
    so follow-up questions work correctly.
    """

    def __init__(self, max_turns: int = 6, max_chars: int = 3000):
        """
        Args:
            max_turns: Max number of user+assistant pairs to keep.
                       Older turns dropped when limit exceeded.
            max_chars: Max total characters of history sent to the
                       model per request. Older turns dropped (in
                       pairs) when the total exceeds this.
        """
        # Import here to avoid circular import issues
        try:
            from config import jarvis_config
            self.max_turns = jarvis_config.MEMORY_MAX_TURNS
            self.max_chars = jarvis_config.MEMORY_MAX_CHARS
        except Exception:
            self.max_turns = max_turns
            self.max_chars = max_chars

        self._messages: List[Message] = []
        logger.info(
            f"Memory initialized (max {self.max_turns} turns, "
            f"{self.max_chars} chars)."
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

    def add_assistant_message(self, content: str) -> None:
        """Add an assistant (JARVIS) response to history."""
        if not content or not content.strip():
            return
        self._messages.append({
            "role": "assistant",
            "content": content.strip()
        })
        self._enforce_limit()

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
        """Clear all conversation history."""
        self._messages.clear()
        logger.info("Conversation memory cleared.")

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