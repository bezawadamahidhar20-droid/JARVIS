"""
brain/text_utils.py — shared text helpers for LLM providers.

Sentence splitting is used by the streaming paths of both Ollama and
Groq so the first finished sentence can be handed to TTS while the rest
of the reply is still generating.

Rules:
  * A sentence ends at ``.`` ``!`` ``?`` (followed by whitespace or
    end-of-text) or at a newline.
  * Common abbreviations ("Dr.", "e.g.", "vs.", month names, single
    initials like "J. Smith") do NOT end a sentence.
  * Punctuation-only leftovers ("...") are never emitted as speech.
  * A line with no sentence-ending punctuation is emitted whole, so
    newline-formatted replies never stall TTS.
"""

import re

# A sentence ends at . ! ? followed by whitespace or end-of-text.
_SENTENCE_END_RE = re.compile(r"[.!?](?=\s|$)")
# Punctuation-only text (ellipsis, stray marks) is not speech.
_PUNCT_ONLY_RE = re.compile(r"^[^a-z0-9]*$", re.IGNORECASE)

# Abbreviations that must not be treated as sentence ends. Matched as a
# whole word before the punctuation (e.g. "Dr." or "e.g.").
_ABBREVIATIONS = frozenset({
    "dr", "mr", "mrs", "ms", "mx", "st", "vs", "etc", "e.g", "i.e",
    "approx", "dept", "est", "fig", "jr", "sr", "no", "vol", "mt",
    "ft", "in", "sec", "min", "hr", "jan", "feb", "mar", "apr", "jun",
    "jul", "aug", "sep", "sept", "oct", "nov", "dec", "prof", "rev",
    "capt", "gov", "gen", "col", "lt", "sgt", "al", "inc", "ltd",
    "co", "corp", "u.s", "u.k", "a.m", "p.m",
})


def _is_abbreviation(word: str) -> bool:
    """True when *word* (the token before the punctuation) is a known
    abbreviation or a single-letter initial ("J. Smith")."""
    w = (word or "").strip().lower().rstrip(".")
    if not w:
        return False
    if len(w) == 1 and w.isalpha():
        return True  # single-letter initial
    return w in _ABBREVIATIONS


def _split_punctuation(text: str) -> tuple[list[str], str]:
    """
    Pull complete sentences out of *text*, honouring abbreviations.

    Returns (sentences, remainder). The remainder is the trailing text
    with no sentence-ending punctuation yet (it may still grow).

    ``pos`` is the scan position (it advances past abbreviation
    periods too) while ``emitted_upto`` only advances when a real
    sentence is emitted — so abbreviation text is never lost from the
    remainder.
    """
    sentences: list[str] = []
    pos = 0              # scan position (advances past abbreviation periods too)
    segment_start = 0    # start of the sentence currently being built
    while True:
        m = _SENTENCE_END_RE.search(text, pos)
        if not m:
            break
        head = text[:m.start()]
        words = head.split()
        if words and _is_abbreviation(words[-1]):
            # Abbreviation ("Dr.", "e.g.") — not a sentence end; skip
            # past the period and keep scanning. The abbreviation stays
            # part of the current segment.
            pos = m.end()
            continue
        end = m.end()
        sentence = text[segment_start:end].strip()
        if sentence and not _PUNCT_ONLY_RE.match(sentence):
            sentences.append(sentence)
            segment_start = end
        pos = end
    return sentences, text[segment_start:]


def split_into_sentences(buffer: str) -> tuple[list[str], str]:
    """
    Split a streaming text buffer into complete sentences.

    Newlines are hard sentence breaks: each complete line is emitted
    (punctuation-split where possible), and the final unterminated
    line is kept as the remainder for the next chunk.

    Returns (sentences, remainder).
    """
    sentences: list[str] = []
    lines = buffer.split("\n")
    last = lines.pop() if lines else ""
    for line in lines:
        if not line.strip():
            continue
        done, _ = _split_punctuation(line)
        if done:
            sentences.extend(done)
        else:
            # Complete line with no punctuation — still a full sentence.
            sentences.append(line.strip())
    done, remainder = _split_punctuation(last)
    sentences.extend(done)
    return sentences, remainder


def clean_response(text: str) -> str:
    """
    Strip markdown / markup artifacts that sound bad when spoken aloud.
    """
    text = re.sub(r"<[^>]+>", "", text)      # XML-style tags
    text = re.sub(r"\*+", "", text)          # bold / italic markers
    text = re.sub(r"^#+\s*", "", text, flags=re.MULTILINE)  # headers
    text = re.sub(r"`+", "", text)           # inline code markers
    text = re.sub(r"\n{3,}", "\n\n", text)   # collapse blank runs
    return text.strip()
