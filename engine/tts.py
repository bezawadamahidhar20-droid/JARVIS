"""Text-to-speech engine built on pyttsx3 (Windows SAPI5).

Speed model:
  * speak() is NON-BLOCKING. Sentences are pushed onto a queue and a
    daemon worker thread speaks them one at a time. The main thread can
    keep generating the next sentence while the current one plays.
  * speak_blocking() queues text and waits until it is done. Used only
    for farewells and anything that must complete before the process
    exits.
  * wait() drains the queue — used before re-arming the microphone so
    JARVIS never hears its own voice (echo).

Robustness:
  * speak() NEVER raises: if the TTS engine is broken or re-initialisation
    fails, the text is still printed to the console and JARVIS keeps running.
  * The pyttsx3 engine is created on the worker thread (it must not be
    shared across threads), and stop() is called from the caller thread
    so it can interrupt a runAndWait() that the worker is blocked in.
  * Markdown artefacts from the model are stripped before speaking.
"""

import queue
import re
import threading

from utils.logger import get_logger

logger = get_logger("tts")

# Strip common markdown artefacts that Qwen3 may sprinkle into replies.
# Order matters: links first ("[text](url)" -> "text"), then the symbols.
_MARKDOWN_LINK_RE = re.compile(r"\[([^\]]*)\]\([^)]*\)")
_MARKDOWN_SYMBOL_RE = re.compile(r"[*_`#>~]|(\{\d+:\w+\})")

# Sentinel used to interrupt a speaking utterance between queue items.
_STOP = "__STOP__"


def clean_for_speech(text: str) -> str:
    """Return *text* with markdown noise removed, ready to be read aloud."""
    text = _MARKDOWN_LINK_RE.sub(r"\1", text)
    text = _MARKDOWN_SYMBOL_RE.sub("", text)
    # Collapse stray double-spaces / leading space from removed symbols.
    return " ".join(text.split())


class TTSEngine:
    """
    Wraps pyttsx3 with a queue + daemon worker so ``speak()`` never
    blocks the caller.
    """

    def __init__(self, rate: int = 200) -> None:
        self.rate: int = rate
        self._engine = None  # lazily created on the worker thread
        self._queue: "queue.Queue" = queue.Queue()

        self._worker = threading.Thread(
            target=self._worker_loop,
            name="jarvis-tts",
            daemon=True,
        )
        self._worker.start()

    # ── Engine lifecycle ──────────────────────────────────────

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

    def _say(self, clean: str) -> None:
        """Speak one already-cleaned string (blocking)."""
        self._ensure_engine()
        self._engine.say(clean)
        self._engine.runAndWait()

    # ── Worker thread ─────────────────────────────────────────

    def _worker_loop(self) -> None:
        """Consume the queue forever, speaking one sentence at a time."""
        while True:
            text, done_event = self._queue.get()

            if text is None:
                # wait() sentinel — nothing to say, just signal completion.
                if done_event is not None:
                    done_event.set()
                continue
            if text == _STOP:
                continue

            try:
                self._say(text)
            except RuntimeError:
                # The engine can die (e.g. audio driver hiccup). Rebuild it
                # once; if that still fails, fall back to console-only.
                logger.warning("TTS engine failed; reinitialising.")
                self._engine = None
                try:
                    self._say(text)
                except Exception as exc:
                    logger.error(f"TTS unavailable after retry: {exc}")
            except Exception as exc:
                logger.error(f"TTS unavailable: {exc}")

            if done_event is not None:
                done_event.set()

    # ── Public API ────────────────────────────────────────────

    def speak(self, text: str) -> None:
        """
        Speak *text* aloud WITHOUT blocking the caller.
        Always prints it to the console too.
        """
        text = (text or "").strip()
        if not text:
            return

        # The console always shows the raw reply; only the audio gets cleaned.
        print(f"[JARVIS] {text}")
        self._queue.put((clean_for_speech(text), None))

    def speak_blocking(self, text: str) -> None:
        """
        Queue *text* and block until it has finished speaking.
        Use for farewells or anything that must complete before exit.
        """
        text = (text or "").strip()
        if not text:
            return

        print(f"[JARVIS] {text}")
        done_event = threading.Event()
        self._queue.put((clean_for_speech(text), done_event))
        done_event.wait()

    def stop(self) -> None:
        """
        Clear queued (unspoken) sentences and interrupt the sentence
        currently being spoken. Best-effort: if the underlying engine
        cannot be interrupted, the current sentence simply finishes.
        """
        # Drop everything still waiting to be spoken.
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break

        # engine.stop() purges the current utterance (SAPI5). It is called
        # from THIS thread so it can interrupt a runAndWait() the worker is
        # blocked inside. Safe to call before the engine exists.
        engine = self._engine
        if engine is not None:
            try:
                engine.stop()
            except Exception:
                pass

    def wait(self) -> None:
        """
        Block until all currently queued speech has finished.
        Called before re-arming the microphone to avoid JARVIS hearing
        its own voice (echo).
        """
        done_event = threading.Event()
        self._queue.put((None, done_event))
        done_event.wait()