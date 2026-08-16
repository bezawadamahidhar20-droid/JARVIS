"""
engine/stt.py — Speech-to-Text engine (Faster-Whisper, fully local)

Pipeline:
    AdaptiveVAD captures one phrase (int16 @ 16 kHz)
        -> converted to float32
        -> transcribed by a locally-loaded Faster-Whisper model

Design rules:
  * The Whisper model is loaded ONCE (``load()``) and reused for every
    utterance — never re-initialized per sentence.
  * Nothing here needs the internet; transcription is 100% on-device.
  * Empty / silent / too-short audio returns None instead of crashing.
  * Any failure is logged and recovered from; the caller keeps looping.
"""

import threading
import time

import numpy as np

from config import jarvis_config, stt_config, whisper_config
from engine.vad import AdaptiveVAD
from utils.logger import get_logger

logger = get_logger("stt")

__all__ = [
    "STTEngine",
]

# ── Config (config.py is always import-safe; no local fallbacks) ─────────────
SAMPLE_RATE = stt_config.SAMPLE_RATE
WHISPER_MODEL = whisper_config.MODEL
WHISPER_COMPUTE_TYPE = whisper_config.COMPUTE_TYPE
WHISPER_DEVICE = whisper_config.DEVICE
WHISPER_LANGUAGE = whisper_config.LANGUAGE
WHISPER_BEAM_SIZE = whisper_config.BEAM_SIZE
STT_STREAM = jarvis_config.STT_STREAM
ENABLE_WAKE_WORD = jarvis_config.ENABLE_WAKE_WORD
WAKE_WORD = jarvis_config.WAKE_WORD


# Compute type used when CUDA is selected but the config still says the
# CPU-only "int8" (ctranslate2 does not support int8 kernels on CUDA).
_CUDA_COMPUTE_TYPE = "float16"


def _detect_device(preferred: str = WHISPER_DEVICE) -> str:
    """
    Resolve the Whisper device: "cpu", "cuda", or "auto".

    "auto" (the default) uses CUDA when the installed ctranslate2 has a
    CUDA-capable GPU, and falls back to CPU otherwise — so machines
    without NVIDIA hardware never crash and never need CUDA installed.
    """
    if preferred != "auto":
        return preferred
    try:
        import ctranslate2

        if ctranslate2.get_cuda_device_count() > 0:
            logger.info("CUDA detected — using GPU for Whisper.")
            return "cuda"
    except Exception as e:
        logger.debug(f"CUDA detection skipped: {e}")
    return "cpu"


class STTEngine:
    """
    Speech-to-text using the VAD recorder + a local Faster-Whisper model.
    """

    def __init__(self, mic_manager=None):
        # mic_manager kept for compatibility; the VAD queries
        # sounddevice directly.
        self.sample_rate = SAMPLE_RATE
        self.language = WHISPER_LANGUAGE
        self.model = None
        self.model_name = WHISPER_MODEL
        self.compute_type = WHISPER_COMPUTE_TYPE
        self.device = WHISPER_DEVICE
        self.vad = AdaptiveVAD(sample_rate=SAMPLE_RATE)
        self._loaded = False

        # Streaming STT (windowed partials while recording) — only when
        # enabled in .env (extra CPU on CPU-only machines).
        self._streaming = None
        if STT_STREAM:
            try:
                from engine.streaming_stt import StreamingListener

                self._streaming = StreamingListener(self.vad, self)
            except Exception as e:
                logger.warning(f"Streaming STT unavailable: {e}")

        # Optional wake-word detector (openwakeword) — graceful
        # fallback to always-on VAD when the package is missing.
        self.wake_word = None
        if ENABLE_WAKE_WORD:
            try:
                from engine.wakeword import WakeWordDetector

                self.wake_word = WakeWordDetector(
                    wake_word=WAKE_WORD,
                    sample_rate=SAMPLE_RATE,
                    input_device=stt_config.INPUT_DEVICE,
                )
            except Exception as e:
                logger.warning(f"Wake-word detector unavailable: {e}")

    # ── Model lifecycle ───────────────────────────────────────

    def load(self) -> bool:
        """
        Load the Faster-Whisper model once at startup.

        With WHISPER_DEVICE=auto (default) the best available device is
        chosen automatically; if a CUDA load fails for any reason
        (missing cuDNN, driver, out-of-memory) the model is retried on
        CPU so JARVIS never crashes on GPU-less machines.

        Returns True on success. Safe to call repeatedly (idempotent).
        """
        if self._loaded and self.model is not None:
            return True
        try:
            t0 = time.perf_counter()
            from faster_whisper import WhisperModel

            device = _detect_device(self.device)
            compute = self.compute_type
            if device == "cuda" and compute == "int8":
                # int8 is a CPU-only compute type; CUDA needs float16.
                logger.info(
                    "CUDA selected — switching compute type "
                    f"'{compute}' -> '{_CUDA_COMPUTE_TYPE}'."
                )
                compute = _CUDA_COMPUTE_TYPE

            logger.info(
                f"Loading Faster-Whisper model '{self.model_name}' "
                f"({device}/{compute})..."
            )
            try:
                self.model = WhisperModel(
                    self.model_name,
                    device=device,
                    compute_type=compute,
                )
            except Exception as e:
                if device != "cuda":
                    raise
                # CUDA load failed — fall back to CPU, never crash.
                logger.warning(
                    f"CUDA load failed ({e}); falling back to CPU."
                )
                device = "cpu"
                compute = "int8"
                self.model = WhisperModel(
                    self.model_name,
                    device=device,
                    compute_type=compute,
                )

            # Keep the effective device/compute for describe() and logs.
            self.device = device
            self.compute_type = compute
            self._loaded = True
            logger.info(
                f"Whisper model ready in {time.perf_counter() - t0:.1f}s "
                f"({self.device}/{self.compute_type})."
            )
            return True
        except ImportError:
            logger.error(
                "faster-whisper is not installed. Run: "
                "pip install faster-whisper"
            )
        except Exception as e:
            logger.error(
                f"Failed to load Whisper model '{self.model_name}': {e}. "
                f"Run: pip install faster-whisper and ensure the model "
                f"is downloadable (WHISPER_MODEL={self.model_name})."
            )
        self.model = None
        return False

    def unload(self) -> None:
        """Release the model (used during shutdown)."""
        self.model = None
        self._loaded = False

    # ── Public API ────────────────────────────────────────────

    def listen(self) -> str | None:
        """
        Record one phrase from the microphone and transcribe it.

        Returns:
            str  : recognized text (lowercase)
            None : nothing heard, or recognition/transcription failed
        """
        t0 = time.perf_counter()
        audio = self.vad.record_phrase()
        if audio is None:
            logger.debug("No speech captured.")
            return None

        t1 = time.perf_counter()
        text = self.transcribe(audio)
        t2 = time.perf_counter()

        logger.debug(
            f"[timing] record {(t1 - t0) * 1000:.0f}ms | "
            f"transcribe {(t2 - t1) * 1000:.0f}ms"
        )

        if text:
            logger.info(f"Recognized: '{text}'")
            print(f"[You said: {text}]")
        else:
            # Silence is not an error — just keep listening. The VAD
            # already printed "[Listening... speak now]".
            logger.debug("No transcription returned.")
            print("[No speech detected — listening...]")

        return text

    def transcribe(self, audio: np.ndarray) -> str | None:
        """
        Transcribe int16 audio with the loaded Whisper model.

        Returns lowercase text, or None on empty input / failure.
        """
        if audio is None or audio.size == 0:
            logger.debug("Empty audio — nothing to transcribe.")
            return None
        if self.model is None:
            logger.error("Whisper model not loaded; call load() first.")
            return None

        try:
            # Whisper expects float32 mono in [-1, 1].
            samples = audio.astype(np.float32) / 32768.0
            segments, _info = self.model.transcribe(
                samples,
                language=self.language,
                beam_size=WHISPER_BEAM_SIZE,
                condition_on_previous_text=False,
                vad_filter=False,
            )
            text = " ".join(
                seg.text.strip() for seg in segments if seg.text
            ).strip()
            if not text:
                return None
            return text.lower()
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            return None

    # ── Streaming STT ─────────────────────────────────────────

    def stream_listen(self):
        """
        Blocking streaming listen: yields (partial, is_final) tuples.

        Partial transcriptions arrive every ~3 seconds while the user
        is still speaking; the final tuple carries the full phrase.
        Requires STT_STREAM=true; otherwise raises RuntimeError.
        """
        if self._streaming is None:
            raise RuntimeError(
                "Streaming STT not enabled (set STT_STREAM=true)."
            )
        yield from self._streaming.listen_stream()

    async def astream_listen(self):
        """Async streaming listen: same tuples, but recording and
        transcription run on a worker thread so the event loop stays
        responsive."""
        import asyncio

        queue: "asyncio.Queue" = asyncio.Queue()

        def _produce() -> None:
            try:
                for partial, is_final in self.stream_listen():
                    queue.put_nowait((partial, is_final))
            except Exception as e:
                logger.error(f"Streaming listen failed: {e}")
                try:
                    queue.put_nowait((None, True))
                except Exception:
                    pass

        threading.Thread(
            target=_produce, name="jarvis-stream-listen", daemon=True
        ).start()

        while True:
            partial, is_final = await queue.get()
            yield partial, is_final
            if is_final:
                break

    # ── Diagnostics ───────────────────────────────────────────

    def describe(self) -> str:
        """Short human-readable status for `jarvis --doctor`."""
        if self._loaded and self.model is not None:
            return (
                f"OK ({self.model_name}, "
                f"{self.device}/{self.compute_type})"
            )
        return (
            f"NOT LOADED ({self.model_name})"
        )
