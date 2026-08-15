"""Rolling conversation memory for JARVIS.

WHY THIS EXISTS
---------------
Ollama is stateless per request: without memory, the follow-up question
"who is Elon Musk?" -> "what does he own?" fails because "he" has nothing
to point at. This class keeps the conversation as a list of OpenAI-style
message dicts and hands the WHOLE window to Ollama on every call.

The window is bounded: once ``max_turns`` full user+assistant pairs are
stored, the oldest pair is dropped first (FIFO) so prompt size stays
predictable no matter how long the session runs.
"""

import config


class ConversationMemory:
    """In-memory, rolling conversation history."""

    def __init__(self, max_turns: int = config.MEMORY_MAX_TURNS) -> None:
        self.max_turns: int = max_turns
        # List of {"role": "user"|"assistant", "content": str}.
        self._messages: list[dict[str, str]] = []

    def add_user_message(self, content: str) -> None:
        """Record what the user said (empty/whitespace input is ignored)."""
        text = (content or "").strip()
        if text:
            self._messages.append({"role": "user", "content": text})

    def add_assistant_message(self, content: str) -> None:
        """Record JARVIS's reply, then trim the window to the max turns."""
        text = (content or "").strip()
        if text:
            self._messages.append({"role": "assistant", "content": text})
            self._trim()

    def _trim(self) -> None:
        """Drop the oldest full turns once ``max_turns`` pairs are exceeded.

        A "turn" is a user message followed by its assistant reply. We trim
        only after an assistant message is added, so the history is always
        well-formed here (alternating user/assistant, starting with user).
        """
        if self.max_turns <= 0:
            return
        user_count = sum(1 for m in self._messages if m["role"] == "user")
        while user_count > self.max_turns:
            # Locate the oldest user message and delete it together with the
            # following message (its assistant reply).
            for i, message in enumerate(self._messages):
                if message["role"] == "user":
                    del self._messages[i : i + 2]
                    break
            user_count -= 1

    def get_context_for_ollama(self, system_prompt: str) -> list[dict[str, str]]:
        """Return the full messages list, with the system prompt first.

        This is what gets POSTed to Ollama's ``/api/chat`` endpoint, so the
        model always sees the current prompt plus all recent context.
        """
        return [{"role": "system", "content": system_prompt}] + list(self._messages)

    def clear(self) -> None:
        """Forget everything (used by the "clear memory" command)."""
        self._messages.clear()

    def __len__(self) -> int:
        """Number of stored messages (for tests / diagnostics)."""
        return len(self._messages)