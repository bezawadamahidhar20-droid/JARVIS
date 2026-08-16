"""Text-to-speech engine.

Primary backend: **Piper** (local neural TTS, ONNX voice). The voice
model is loaded ONCE and reused for every sentence. Speech is
synthesized in memory (no temp WAV files) and played through
sounddevice.

Fallback backend: **pyttsx3** (Windows SAPI5) — select it with
TTS_ENGINE=pyttsx3 in .env.

Speed / robustness model (kept from the old engine):
  * speak() is NON-BLOCKING — sentences are pushed onto a queue and a
    daemon worker thread speaks them one at a time.
  * speak_blocking() queues text and waits until it is done (farewells).
  * wait() drains the queue — used before re-arming the microphone so
    JARVIS never hears its own voice (echo).
  * speak() NEVER raises: if the TTS engine is broken, the text is
    still printed to the console and JARVIS keeps running.
"""

import queue
import re
import threading
import time

import numpy as np

from config import tts_config
from utils.logger import get_logger

logger = get_logger("tts")

__all__ = [
    "TTSEngine",
    "PiperBackend",
    "Pyttsx3Backend",
    "clean_for_speech",
]

# ── Config (config.py is always import-safe; no local fallbacks) ─────────────
ENGINE = tts_config.ENGINE
VOICE_NAME = tts_config.VOICE
VOICE_PATH = tts_config.VOICE_PATH
RATE = tts_config.RATE

# Strip common markdown artefacts the model may sprinkle into replies.
# Order matters: links first ("[text](url)" -> "text"), then the symbols.
_MARKDOWN_LINK_RE = re.compile(r"\[([^\]]*)\]\([^)]*\)")
_MARKDOWN_SYMBOL_RE = re.compile(r"[*_`#>~]|(\{\d+:\w+\})")

# Sentinel used to interrupt a speaking utterance between queue items.
# A dedicated singleton class (not a plain string) so identity (is)
# comparison is used — a user typing "__STOP__" in --text mode could
# otherwise be mistaken for the stop signal by an == check.
class _StopSentinel:
    """Singleton marker for queue items that should interrupt speech."""

    _instance: "_StopSentinel | None" = None

    def __new__(cls) -> "_StopSentinel":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:  # pragma: no cover - debugging only
        return "<_StopSentinel>"


_STOP = _StopSentinel()


def clean_for_speech(text: str) -> str:
    """Return *text* with markdown noise removed, ready to be read aloud."""
    text = _MARKDOWN_LINK_RE.sub(r"\1", text)
    text = _MARKDOWN_SYMBOL_RE.sub("", text)
    # Collapse stray double-spaces / leading space from removed symbols.
    return " ".join(text.split())


def _resolve_voice_path() -> str | None:
    """Return the path to the Piper ONNX voice, or None if missing."""
    if VOICE_PATH:
        return VOICE_PATH
    from pathlib import Path

    candidate = Path(__file__).parent.parent / "voices" / f"{VOICE_NAME}.onnx"
    if candidate.is_file():
        return str(candidate)
    return None


class PiperBackend:
    """Piper synthesis + sounddevice playback. Voice loaded once."""

    def __init__(self) -> None:
        self._voice = None
        self.rate = 22050

    def load(self) -> bool:
        """Load the Piper voice model (idempotent)."""
        if self._voice is not None:
            return True
        path = _resolve_voice_path()
        if not path:
            logger.warning(
                f"Piper voice '{VOICE_NAME}.onnx' not found in voices/. "
                "Download it with: python -m piper.download_voices "
                f"{VOICE_NAME} --download_dir voices"
            )
            return False
        try:
            from piper import PiperVoice

            self._voice = PiperVoice.load(path)
            self.rate = self._voice.config.sample_rate
            logger.info(f"Piper voice loaded: {path}")
            return True
        except Exception as e:
            logger.error(f"Failed to load Piper voice: {e}")
            self._voice = None
            return False

    def synthesize(self, text: str) -> np.ndarray:
        """
        Synthesize *text* into float32 mono audio (in memory).
        """
        if self._voice is None:
            self.load()
        if self._voice is None:
            raise RuntimeError("Piper voice not available")

        chunks = [
            c.audio_float_array
            for c in self._voice.synthesize(text)
        ]
        if not chunks:
            return np.array([], dtype=np.float32)
        return np.concatenate(chunks)

    def play(self, audio: np.ndarray) -> None:
        """Play *audio* (float32, self.rate) and block until done."""
        if audio is None or audio.size == 0:
            return
        import sounddevice as sd

        sd.play(audio, self.rate)
        sd.wait()

    def interrupt(self) -> None:
        """Stop any audio currently playing (non-blocking)."""
        try:
            import sounddevice as sd

            sd.stop()
        except Exception as e:
            logger.debug(f"TTS interrupt failed: {e}")


class Pyttsx3Backend:
    """pyttsx3 (Windows SAPI5) fallback backend."""

    def __init__(self, rate: int = 200) -> None:
        self.rate = rate
        self._engine = None

    def load(self) -> bool:
        try:
            import pyttsx3

            self._engine = pyttsx3.init()
            self._engine.setProperty("rate", self.rate)
            return True
        except Exception as e:
            logger.error(f"Failed to init pyttsx3: {e}")
            self._engine = None
            return False

    def synthesize(self, text: str) -> np.ndarray:
        # pyttsx3 speaks directly; nothing to synthesize here.
        return np.array([], dtype=np.float32)

    def play(self, audio: np.ndarray) -> None:
        # Audio handled by the engine's own runAndWait().
        raise NotImplementedError


class TTSEngine:
    """
    Wraps the selected TTS backend with a queue + daemon worker so
    ``speak()`` never blocks the caller.
    """

    def __init__(
        self,
        engine: str | None = None,
        rate: int = RATE,
        backend=None,
    ) -> None:
        self.engine_name = (engine or ENGINE).lower()
        self.rate = rate
        # ``backend`` is injected by tests to avoid real audio.
        self._backend = backend if backend is not None else self._make_backend()
        self._queue: queue.Queue = queue.Queue()

        # Generation counter: every stop() bumps it, and any utterance
        # queued under an older generation is skipped by the worker, so
        # cancelled speech can never play after an interrupt.
        self._generation = 0
        self._gen_lock = threading.Lock()

        self._worker = threading.Thread(
            target=self._worker_loop,
            name="jarvis-tts",
            daemon=True,
        )
        self._worker.start()

    def _make_backend(self):
        if self.engine_name == "pyttsx3":
            return Pyttsx3Backend(rate=self.rate)
        return PiperBackend()

    # ── Engine lifecycle ──────────────────────────────────────

    def load(self) -> bool:
        """Load the voice model once at startup."""
        return self._backend.load()

    def _speak_backend(self, clean: str) -> None:
        """Speak one already-cleaned string (blocking)."""
        backend = self._backend

        if self.engine_name == "pyttsx3":
            if backend._engine is None:  # noqa: SLF001
                backend.load()
            if backend._engine is None:
                raise RuntimeError("pyttsx3 unavailable")
            backend._engine.say(clean)  # noqa: SLF001
            backend._engine.runAndWait()  # noqa: SLF001
            return

        # Piper path: synthesize in memory, then play.
        t0 = time.perf_counter()
        audio = backend.synthesize(clean)
        t1 = time.perf_counter()
        backend.play(audio)
        t2 = time.perf_counter()
        logger.debug(
            f"[timing] TTS synth {(t1 - t0) * 1000:.0f}ms | "
            f"play {(t2 - t1) * 1000:.0f}ms | "
            f"{len(clean)} chars"
        )

    # ── Worker thread ─────────────────────────────────────────

    def _worker_loop(self) -> None:
        """Consume the queue forever, speaking one sentence at a time."""
        while True:
            text, done_event, generation = self._queue.get()

            if text is None:
                # wait() sentinel — nothing to say, just signal completion.
                if done_event is not None:
                    done_event.set()
                continue
            if text is _STOP:
                if done_event is not None:
                    done_event.set()
                continue

            # A stop() happened after this item was queued — skip it.
            if generation != self._current_generation():
                if done_event is not None:
                    done_event.set()
                continue

            try:
                self._speak_backend(text)
            except RuntimeError:
                # The engine can die (e.g. audio driver hiccup). Rebuild
                # once; if that still fails, fall back to console-only.
                logger.warning("TTS engine failed; reinitialising.")
                self._backend = self._make_backend()
                self._backend.load()
                try:
                    self._speak_backend(text)
                except Exception as exc:
                    logger.error(f"TTS unavailable after retry: {exc}")
            except Exception as exc:
                logger.error(f"TTS unavailable: {exc}")

            if done_event is not None:
                done_event.set()

    def _current_generation(self) -> int:
        with self._gen_lock:
            return self._generation

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
        self._queue.put((clean_for_speech(text), None, self._current_generation()))

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
        self._queue.put(
            (clean_for_speech(text), done_event, self._current_generation())
        )
        done_event.wait()

    def stop(self) -> None:
        """
        Interrupt speech: drop queued (unspoken) sentences, cancel the
        sentence currently being spoken, and unblock any waiters.

        A generation bump guarantees nothing queued before the stop can
        ever be spoken afterwards (even if the worker already picked it
        up). pyttsx3 (Windows SAPI) cannot be interrupted mid-sentence
        — its current sentence simply finishes, then it stops.
        """
        with self._gen_lock:
            self._generation += 1

        # Drop everything still waiting to be spoken, signalling any
        # speak_blocking() waiters so they never hang.
        while True:
            try:
                _text, done_event, _gen = self._queue.get_nowait()
            except queue.Empty:
                break
            if done_event is not None:
                done_event.set()

        # Interrupt active playback immediately (Piper path).
        if self.engine_name != "pyttsx3":
            interrupt = getattr(self._backend, "interrupt", None)
            if interrupt is not None:
                try:
                    interrupt()
                except Exception:
                    pass

    def wait(self) -> None:
        """
        Block until all currently queued speech has finished.
        Called before re-arming the microphone to avoid JARVIS hearing
        its own voice (echo).
        """
        done_event = threading.Event()
        self._queue.put((None, done_event, self._current_generation()))
        done_event.wait()
