"""
Text-to-speech engine.
 
[FIX m4] Replaced string _STOP sentinel with singleton class using
         identity comparison (is) to avoid user input collision.
[FIX m5] Added __all__ exports.
[FIX m1] Removed try/except config fallbacks - using direct imports.
"""
 
import queue
import re
import threading
import time
 
import numpy as np
 
from config import tts_config
from utils.logger import get_logger
 
__all__ = [
    "TTSEngine",
    "PiperBackend",
    "Pyttsx3Backend",
    "clean_for_speech",
]
 
logger = get_logger("tts")
 
# Config values - direct import
ENGINE = tts_config.ENGINE
VOICE_NAME = tts_config.VOICE
VOICE_PATH = tts_config.VOICE_PATH
RATE = tts_config.RATE
 
# Strip markdown from replies
_MARKDOWN_LINK_RE = re.compile(r"\[([^\]]*)\]\([^)]*\)")
_MARKDOWN_SYMBOL_RE = re.compile(r"[*_`#>~]|(\{\d+:\w+\})")
 
 
# [FIX m4] Singleton sentinel class instead of string
class _StopSentinel:
    """Singleton sentinel for stop signal. Uses identity comparison."""
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __repr__(self):
        return "<STOP>"
 
 
_STOP = _StopSentinel()
 
 
def clean_for_speech(text: str) -> str:
    """Return text with markdown noise removed."""
    text = _MARKDOWN_LINK_RE.sub(r"\1", text)
    text = _MARKDOWN_SYMBOL_RE.sub("", text)
    return " ".join(text.split())
 
 
def _resolve_voice_path() -> str | None:
    """Return the path to the Piper ONNX voice."""
    if VOICE_PATH:
        return VOICE_PATH
    from pathlib import Path
    candidate = Path(__file__).parent.parent / "voices" / f"{VOICE_NAME}.onnx"
    if candidate.is_file():
        return str(candidate)
    return None
 
 
class PiperBackend:
    """Piper synthesis + sounddevice playback."""
 
    def __init__(self) -> None:
        self._voice = None
        self.rate = 22050
 
    def load(self) -> bool:
        if self._voice is not None:
            return True
        path = _resolve_voice_path()
        if not path:
            logger.warning(
                f"Piper voice '{VOICE_NAME}.onnx' not found in voices/. "
                f"Download it with: python -m piper.download_voices {VOICE_NAME}"
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
        """Synthesize text into float32 mono audio."""
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
        """Play audio and block until done."""
        if audio is None or audio.size == 0:
            return
        import sounddevice as sd
        sd.play(audio, self.rate)
        sd.wait()
 
    def interrupt(self) -> None:
        """Stop any audio currently playing."""
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
        return np.array([], dtype=np.float32)
 
    def play(self, audio: np.ndarray) -> None:
        raise NotImplementedError
 
 
class TTSEngine:
    """Wraps TTS backend with queue + daemon worker."""
 
    def __init__(
        self,
        engine: str | None = None,
        rate: int = RATE,
        backend=None,
    ) -> None:
        self.engine_name = (engine or ENGINE).lower()
        self.rate = rate
        self._backend = backend if backend is not None else self._make_backend()
        self._queue: queue.Queue = queue.Queue()
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
 
    def load(self) -> bool:
        return self._backend.load()
 
    def _speak_backend(self, clean: str) -> None:
        """Speak one cleaned string (blocking)."""
        backend = self._backend
 
        if self.engine_name == "pyttsx3":
            if backend._engine is None:
                backend.load()
            if backend._engine is None:
                raise RuntimeError("pyttsx3 unavailable")
            backend._engine.say(clean)
            backend._engine.runAndWait()
            return
 
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
 
    def _worker_loop(self) -> None:
        """Consume the queue, speaking one sentence at a time."""
        while True:
            item = self._queue.get()
            text, done_event, generation = item
 
            if text is None:
                if done_event is not None:
                    done_event.set()
                continue
            
            # [FIX m4] Use identity comparison for sentinel
            if text is _STOP:
                if done_event is not None:
                    done_event.set()
                continue
 
            with self._gen_lock:
                if generation < self._generation:
                    if done_event is not None:
                        done_event.set()
                    continue
 
            try:
                clean = clean_for_speech(text)
                if clean:
                    self._speak_backend(clean)
            except Exception as e:
                logger.error(f"TTS failed: {e}")
                print(f"[JARVIS] {text}")
 
            if done_event is not None:
                done_event.set()
 
    def speak(self, text: str) -> None:
        """Queue text for speaking (non-blocking)."""
        if not text or not text.strip():
            return
        with self._gen_lock:
            gen = self._generation
        self._queue.put((text, None, gen))
 
    def speak_blocking(self, text: str) -> None:
        """Queue text and wait until spoken."""
        if not text or not text.strip():
            return
        done = threading.Event()
        with self._gen_lock:
            gen = self._generation
        self._queue.put((text, done, gen))
        done.wait()
 
    def stop(self) -> None:
        """Interrupt current speech and clear queue."""
        with self._gen_lock:
            self._generation += 1
 
        # Clear queue
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
 
        # Interrupt active playback, but only if the backend supports it
        # (e.g. pyttsx3 has no mid-sentence interrupt).
        interrupt = getattr(self._backend, "interrupt", None)
        if callable(interrupt):
            try:
                interrupt()
            except Exception:
                pass
 
    def wait(self) -> None:
        """Wait for queue to drain."""
        done = threading.Event()
        self._queue.put((None, done, 0))
        done.wait()
 