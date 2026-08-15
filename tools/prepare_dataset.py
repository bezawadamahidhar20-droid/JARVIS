"""Prepare raw conversation logs for fine-tuning.

Reads the JSONL written by JARVIS (utils/dataset.py), deduplicates,
drops failed exchanges, and writes a ShareGPT-format JSON file ready
for unsloth / axolotl / LLM-Foundry training.

Usage:
    python tools/prepare_dataset.py [--input data/conversations.jsonl]
                                    [--output data/dataset.json]
                                    [--min-pairs 20]
"""

import argparse
import json
import os
import sys
from typing import List, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config  # noqa: E402
from utils.dataset import ConversationDataset  # noqa: E402


def dedupe(pairs: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
    """Drop exact duplicate (user, assistant) exchanges."""
    seen = set()
    out = []
    for user, assistant in pairs:
        key = (user.lower(), assistant.lower())
        if key in seen:
            continue
        seen.add(key)
        out.append((user, assistant))
    return out


def to_sharegpt(pairs: List[Tuple[str, str]], system_prompt: str) -> list:
    entries = []
    for user, assistant in pairs:
        entries.append(
            {
                "conversations": [
                    {"from": "human", "value": user},
                    {"from": "gpt", "value": assistant},
                ],
                "system": system_prompt,
            }
        )
    return entries


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=config.DATASET_PATH)
    parser.add_argument("--output", default="data/dataset.json")
    parser.add_argument("--min-pairs", type=int, default=20,
                        help="abort if fewer real pairs than this")
    args = parser.parse_args()

    if not os.path.isfile(args.input):
        print(f"[x] No dataset log at {args.input} — run JARVIS first.")
        return 1

    raw = ConversationDataset(args.input).load()
    pairs = dedupe(ConversationDataset.conversation_pairs(raw))
    print(f"[i] {len(raw)} raw entries, {len(pairs)} unique usable pairs")

    if len(pairs) < args.min_pairs:
        print(f"[!] Only {len(pairs)} pairs; want >= {args.min_pairs} for a "
              "meaningful fine-tune. Keep talking to JARVIS.")
        return 1

    entries = to_sharegpt(pairs, config.SYSTEM_PROMPT)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as fh:
        json.dump(entries, fh, ensure_ascii=False, indent=2)
    print(f"[+] Wrote {len(entries)} examples to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())