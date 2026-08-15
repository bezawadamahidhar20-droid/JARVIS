"""Faster-Whisper transcription wrapper (in-memory, no disk writes).

Type annotations follow PEP 526 / PEP 484 throughout so that mypy / pyright
can provide full IDE support without ``Any`` escapes in the public API.
"""

from typing import Any

import numpy as np
from faster_whisper import WhisperModel

from utils import logger


class WhisperEngine:
    """Loads a Faster-Whisper model once and transcribes numpy audio arrays.

    Parameters
    ----------
    model_size:
        Whisper model variant to load, e.g. ``"tiny.en"``, ``"base"``,
        ``"small"``.  Downloaded automatically from Hugging Face on first use.
    device:
        ``"cpu"`` or ``"cuda"``.
    compute_type:
        Quantisation level, e.g. ``"int8"``, ``"float16"``.
    language:
        BCP-47 language code for the transcription (``"en"`` for English).
    beam_size:
        Beam search width.  ``1`` is equivalent to greedy decoding and is the
        fastest option on CPU.
    """

    def __init__(
        self,
        model_size: str = "base",
        device: str = "cpu",
        compute_type: str = "int8",
        language: str = "en",
        beam_size: int = 1,
    ) -> None:
        self.language: str = language
        self.beam_size: int = beam_size

        logger.info(
            f"Loading Whisper model '{model_size}' ({device}/{compute_type})..."
        )
        # WhisperModel is typed as ``Any`` because faster-whisper ships no
        # py.typed marker or stub package; every public attribute access is
        # safe at runtime.
        self.model: Any = WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type,
        )
        logger.ok(f"Whisper '{model_size}' ready")

    def transcribe(self, audio: np.ndarray) -> str:
        """Transcribe a mono float32 audio array captured at 16 kHz.

        Parameters
        ----------
        audio:
            A 1-D ``float32`` numpy array of PCM samples at 16 000 Hz.
            Produced by ``audio.microphone.SpeechRecorder.capture_speech()``.

        Returns
        -------
        str
            The full transcript string with segment texts joined by spaces.
            Returns an empty string if no speech was detected.
        """
        segments, _info = self.model.transcribe(
            audio,
            language=self.language,
            beam_size=self.beam_size,
            vad_filter=True,
            condition_on_previous_text=False,
        )
        parts: list[str] = [seg.text.strip() for seg in segments]
        return " ".join(parts).strip()
