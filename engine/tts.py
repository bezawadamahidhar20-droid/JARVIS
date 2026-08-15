"""Text-to-speech engine built on pyttsx3 (Windows SAPI5).

``speak()`` NEVER raises: if the TTS engine is broken or re-initialisation
fails, the text is still printed to the console and JARVIS keeps running.
Markdown artefacts from the model (``**``, ``##``, backticks, link
syntax) are stripped before speaking so the audio stays clean.
"""

import re

from utils import logger

# Strip common markdown artefacts that Qwen3 may sprinkle into replies.
# Order matters: links first ("[text](url)" -> "text"), then the symbols.
_MARKDOWN_LINK_RE = re.compile(r"\[([^\]]*)\]\([^)]*\)")
_MARKDOWN_SYMBOL_RE = re.compile(r"[*_`#>~]|(\{\d+:\w+\})")


def clean_for_speech(text: str) -> str:
    """Return *text* with markdown noise removed, ready to be read aloud."""
    text = _MARKDOWN_LINK_RE.sub(r"\1", text)
    text = _MARKDOWN_SYMBOL_RE.sub("", text)
    # Collapse stray double-spaces / leading space from removed symbols.
    return " ".join(text.split())


class TTSEngine:
    """Wraps pyttsx3 with safe re-initialisation and console fallback."""

    def __init__(self, rate: int = 185) -> None:
        self.rate: int = rate
        self._engine = None  # lazily created; rebuilt on RuntimeError

    def _ensure_engine(self) -> None:
        """Create the pyttsx3 engine if it doesn't exist yet."""
        if self._engine is not None:
            return
        import pyttsx3

        engine = pyttsx3.init()
        engine.setProperty("rate", self.rate)
        # Prefer a female voice (closer to JARVIS's tone) when one exists.
        try:
            for voice in engine.getProperty("voices"):
                name = (voice.name or "").lower()
                vid = (voice.id or "").lower()
                if "female" in name or "zira" in vid:
                    engine.setProperty("voice", voice.id)
                    break
        except Exception:
            pass  # voice selection is a nicety, never a hard requirement
        self._engine = engine

    def speak(self, text: str) -> None:
        """Speak *text* aloud; always prints it to the console too."""
        text = (text or "").strip()
        if not text:
            return

        # The console always shows the raw reply; only the audio gets cleaned.
        print(f"[JARVIS] {text}")
        clean = clean_for_speech(text)

        try:
            self._ensure_engine()
            self._engine.say(clean)
            self._engine.runAndWait()
        except RuntimeError:
            # The engine can die (e.g. audio driver hiccup). Rebuild it once;
            # if that still fails, fall back to console-only output.
            logger.warning("TTS engine failed; reinitialising.")
            self._engine = None
            try:
                self._ensure_engine()
                self._engine.say(clean)
                self._engine.runAndWait()
            except Exception as exc:
                logger.error(f"TTS unavailable after retry: {exc}")
        except Exception as exc:
            logger.error(f"TTS unavailable: {exc}")