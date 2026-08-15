"""Local neural text-to-speech using Piper (ONNX, fully offline).

Post-TTS microphone cooldown
------------------------------
After ``sd.wait()`` returns the speaker cone is still moving and room reverb
has not fully decayed.  Without a short hold-off the VAD on the next loop
iteration can pick up JARVIS's own voice tail and feed phantom speech into
Whisper.

The cooldown is applied by the *caller* (``main.py::mic_cooldown()``) rather
than inside ``speak()`` so the duration is configurable in one place
(``config.POST_TTS_COOLDOWN_S``) and can be set to 0.0 for tests or
headphone-only setups where echo is not an issue.
"""

import numpy as np
import sounddevice as sd
from piper import PiperVoice

import config
from utils import logger


class TTSEngine:
    """Wraps Piper's voice synthesis and plays audio via sounddevice.

    Speaks directly from in-memory float32 audio — no temp WAV files are
    written during normal operation.  Swapping TTS engines later means
    replacing the internals here; nothing else in JARVIS needs to change.

    Parameters
    ----------
    voice_path:
        Path to the ``.onnx`` Piper voice file (the matching ``.json`` config
        must live alongside it).  Defaults to ``config.TTS_VOICE_PATH``.
    """

    def __init__(self, voice_path: str = config.TTS_VOICE_PATH) -> None:
        self.voice_path: str = voice_path
        self.voice: PiperVoice | None = None
        self.ready: bool = False

    def initialize(self) -> None:
        """Load the Piper voice model (idempotent — safe to call multiple times).

        Raises
        ------
        FileNotFoundError
            If the ``.onnx`` voice file is missing.
        Exception
            Any other Piper initialisation failure.
        """
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
        logger.info(
            f"TTS voice: {self.voice.config.espeak_voice} "
            f"({self.voice.config.sample_rate} Hz)"
        )

    def _synth_to_array(self, text: str) -> tuple[np.ndarray | None, int]:
        """Synthesise *text* into a float32 numpy array.

        Returns ``(audio_array, sample_rate)``.  If Piper produces no chunks
        ``audio_array`` is ``None``.
        """
        assert self.voice is not None, "Call initialize() before _synth_to_array()"
        chunks = [ch.audio_float_array for ch in self.voice.synthesize(text)]
        if not chunks:
            return None, self.voice.config.sample_rate
        return np.concatenate(chunks), self.voice.config.sample_rate

    def speak(self, text: str) -> float:
        """Synthesise and play *text* through the default output device.

        Returns the wall-clock duration of the TTS stage (synthesis + playback)
        in seconds.  Caller is responsible for the post-TTS mic cooldown
        (``main.py::mic_cooldown()``) — see module docstring.

        This method never raises; TTS failures are logged and the return value
        is ``0.0`` so the main loop can continue.

        Parameters
        ----------
        text:
            The string to speak.  Leading/trailing whitespace is stripped.
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
            # ── post-TTS cooldown lives in main.py::mic_cooldown() ──────────
            # After sd.wait() the speaker tail / room reverb has NOT yet decayed.
            # The 0.75 s hold-off in mic_cooldown() prevents JARVIS from
            # re-arming the VAD while its own voice is still audible.
        except Exception as exc:
            logger.error(f"TTS failed: {exc}")
            return 0.0

        elapsed: float = logger.tick() - start
        logger.report("TTS", elapsed)
        return elapsed
