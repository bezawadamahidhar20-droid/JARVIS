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

import time

import numpy as np

from engine.vad import AdaptiveVAD
from utils.logger import get_logger

logger = get_logger("stt")

# ── Load config safely ────────────────────────────────────────
try:
    from config import stt_config, whisper_config

    SAMPLE_RATE = stt_config.SAMPLE_RATE
    WHISPER_MODEL = whisper_config.MODEL
    WHISPER_COMPUTE_TYPE = whisper_config.COMPUTE_TYPE
    WHISPER_DEVICE = whisper_config.DEVICE
    WHISPER_LANGUAGE = whisper_config.LANGUAGE
    WHISPER_BEAM_SIZE = whisper_config.BEAM_SIZE
except Exception:
    SAMPLE_RATE = 16000
    WHISPER_MODEL = "base"
    WHISPER_COMPUTE_TYPE = "int8"
    WHISPER_DEVICE = "cpu"
    WHISPER_LANGUAGE = "en"
    WHISPER_BEAM_SIZE = 1


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

    # ── Model lifecycle ───────────────────────────────────────

    def load(self) -> bool:
        """
        Load the Faster-Whisper model once at startup.

        Returns True on success. Safe to call repeatedly (idempotent).
        """
        if self._loaded and self.model is not None:
            return True
        try:
            t0 = time.perf_counter()
            from faster_whisper import WhisperModel

            logger.info(
                f"Loading Faster-Whisper model '{self.model_name}' "
                f"({self.device}/{self.compute_type})..."
            )
            self.model = WhisperModel(
                self.model_name,
                device=self.device,
                compute_type=self.compute_type,
            )
            self._loaded = True
            logger.info(
                f"Whisper model ready in {time.perf_counter() - t0:.1f}s"
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
