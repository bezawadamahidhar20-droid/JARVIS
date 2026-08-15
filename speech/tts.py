"""Local neural text-to-speech using Piper (ONNX, fully offline)."""


import numpy as np
import sounddevice as sd
from piper import PiperVoice

import config
from utils import logger


class TTSEngine:
    """Wraps Piper's voice synthesis and plays audio via sounddevice.

    Speaks directly from in-memory float audio — no temp WAV files.
    Swapping engines later means replacing the internals here; nothing
    else in JARVIS needs to change.
    """

    def __init__(self, voice_path: str = config.TTS_VOICE_PATH):
        self.voice_path = voice_path
        self.voice: PiperVoice | None = None
        self.ready: bool = False

    def initialize(self) -> None:
        if self.ready:
            return
        logger.status("Initializing TTS...")
        try:
            self.voice = PiperVoice.load(self.voice_path)
        except FileNotFoundError:
            logger.error(
                "TTS voice not found. Download it once with:\n"
                "  python -m piper.download_voices en_US-lessac-medium"
            )
            raise
        except Exception as exc:
            logger.error(f"TTS initialization failed: {exc}")
            raise
        self.ready = True
        logger.ok("TTS ready")
        logger.info(f"TTS voice: {self.voice.config.espeak_voice} ({self.voice.config.sample_rate} Hz)")

    def _synth_to_array(self, text: str) -> tuple[np.ndarray | None, int]:
        chunks = [ch.audio_float_array for ch in self.voice.synthesize(text)]
        if not chunks:
            return None, self.voice.config.sample_rate
        return np.concatenate(chunks), self.voice.config.sample_rate

    def speak(self, text: str) -> float:
        """Synthesizes and speaks `text`. Returns duration in seconds.

        Never raises: TTS failures are logged, not fatal.
        """
        text = text.strip()
        if not text:
            logger.warning("TTS: empty text, nothing to speak.")
            return 0.0

        self.initialize()

        start = logger.tick()
        try:
            logger.status("TTS: Speaking...")
            audio, sample_rate = self._synth_to_array(text)
            if audio is None or audio.size == 0:
                logger.error("TTS: no audio produced.")
                return 0.0
            sd.play(audio, sample_rate)
            sd.wait()
        except Exception as exc:
            logger.error(f"TTS failed: {exc}")
            return 0.0

        elapsed = logger.tick() - start
        logger.report("TTS", elapsed)
        return elapsed