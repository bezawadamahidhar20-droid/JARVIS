"""Collect real user/assistant turns into a ShareGPT-format dataset.

Every completed JARVIS exchange is appended to a JSONL file. The format
matches what unsloth / axolotl expect for fine-tuning:

    {"conversations": [{"from": "human", "value": ...},
                       {"from": "gpt", "value": ...}],
     "system": ...}

Run tools/prepare_dataset.py afterwards to deduplicate, filter and
convert these raw logs into a training JSON.
"""

import json
import os

from utils import logger

# Responses that indicate a failed exchange and must never be trained on.
SKIP_RESPONSES = (
    "Ollama is not running, so I cannot answer that right now.",
    "The Qwen model is not installed on this Ollama.",
    "Ollama returned an error while answering.",
    "Sorry, something went wrong on my end.",
    "Sorry, I could not complete that command.",
)


class ConversationDataset:
    """Append real exchanges to a ShareGPT-format JSONL file."""

    def __init__(self, path: str, system_prompt: str = "") -> None:
        self.path = path
        self.system_prompt = system_prompt

    def record(self, user_text: str, assistant_text: str) -> None:
        """Append one turn. Silently drops empty or failed exchanges."""
        user_text = (user_text or "").strip()
        assistant_text = (assistant_text or "").strip()
        if not user_text or not assistant_text:
            return
        if assistant_text in SKIP_RESPONSES:
            return

        entry = {
            "conversations": [
                {"from": "human", "value": user_text},
                {"from": "gpt", "value": assistant_text},
            ],
        }
        if self.system_prompt:
            entry["system"] = self.system_prompt

        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        try:
            with open(self.path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except OSError as exc:
            logger.warning(f"Could not write dataset entry: {exc}")

    def load(self) -> list:
        """Read all recorded turns (empty list if the file is missing)."""
        entries: list = []
        if not os.path.isfile(self.path):
            return entries
        try:
            with open(self.path, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        try:
                            entries.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
        except OSError as exc:
            logger.warning(f"Could not read dataset: {exc}")
        return entries

    @staticmethod
    def conversation_pairs(entries: list) -> list:
        """Extract (user, assistant) string pairs from raw entries."""
        pairs = []
        for entry in entries:
            convs = entry.get("conversations", [])
            if len(convs) >= 2:
                user = convs[0].get("value", "")
                assistant = convs[1].get("value", "")
                if user and assistant:
                    pairs.append((user, assistant))
        return pairs