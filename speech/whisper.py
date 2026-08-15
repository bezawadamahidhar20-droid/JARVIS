"""Faster-Whisper transcription wrapper (in-memory, no disk writes)."""

from typing import Any

import numpy as np
from faster_whisper import WhisperModel

from utils import logger


class WhisperEngine:
    """Loads a Whisper model once and transcribes numpy audio arrays."""

    def __init__(self, model_size: str = "base", device: str = "cpu",
                 compute_type: str = "int8", language: str = "en",
                 beam_size: int = 1):
        self.language = language
        self.beam_size = beam_size
        logger.info(f"Loading Whisper model '{model_size}' ({device}/{compute_type})...")
        self.model: Any = WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type,
        )
        logger.ok(f"Whisper '{model_size}' ready")

    def transcribe(self, audio: np.ndarray) -> str:
        """audio: float32 mono numpy array (16 kHz). Returns transcript string."""
        segments, _info = self.model.transcribe(
            audio,
            language=self.language,
            beam_size=self.beam_size,
            vad_filter=True,
            condition_on_previous_text=False,
        )
        parts = [seg.text.strip() for seg in segments]
        return " ".join(parts).strip()