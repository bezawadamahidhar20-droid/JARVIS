"""
engine/wakeword.py — optional \"hey JARVIS\" wake-word detection.

The always-on microphone is the biggest privacy concern of a voice
assistant. With wake-word detection enabled (ENABLE_WAKE_WORD=true),
JARVIS only records an utterance *after* the wake phrase is heard —
idle CPU drops and nothing is transcribed while you are not talking.

Implementation uses `openwakeword` (MIT, runs on CPU). The model file
downloads on first use. If the package or model is unavailable, the
detector reports ``available() == False`` and the orchestrator falls
back to the normal always-on VAD behavior — the assistant never stops
working because of a missing optional dependency.
"""

import threading

import numpy as np
import sounddevice as sd

from utils.logger import get_logger

logger = get_logger("wakeword")


class WakeWordDetector:
    """Listens for a wake phrase on the mic (openwakeword)."""

    def __init__(
        self,
        wake_word: str = "hey jarvis",
        sample_rate: int = 16000,
        threshold: float = 0.5,
        chunk_duration: float = 0.05,
        input_device: int | None = None,
    ) -> None:
        self.wake_word = (wake_word or "hey jarvis").strip().lower()
        self.sample_rate = sample_rate
        self.threshold = float(threshold)
        self.chunk_size = max(1, int(sample_rate * chunk_duration))
        self.input_device = input_device
        self._model = None
        self._load_error: Exception | None = None

    @property
    def available(self) -> bool:
        """True when the openwakeword model is loaded."""
        return self._model is not None

    @property
    def load_error(self) -> Exception | None:
        return self._load_error

    def load(self) -> bool:
        """Load the openwakeword model (idempotent). False when the
        optional package/model is unavailable — caller falls back."""
        if self._model is not None:
            return True
        try:
            from openwakeword.model import Model

            self._model = Model(wakeword_models=[self.wake_word])
            logger.info(f"Wake-word model loaded: {self.wake_word}")
            return True
        except Exception as e:
            self._model = None
            self._load_error = e
            logger.warning(
                "Wake-word detection unavailable (%s); falling back to "
                "always-on VAD. Install with: pip install openwakeword",
                e,
            )
            return False

    def _score(self, prediction: dict) -> float:
        """Extract the wake-word score from an openwakeword prediction."""
        if not prediction:
            return 0.0
        key = self.wake_word.replace(" ", "_")
        score = prediction.get(self.wake_word, prediction.get(key, 0.0))
        if score:
            return float(score)
        # Unknown model key: take the strongest signal as a fallback so
        # arbitrary model names still work.
        return max(float(v) for v in prediction.values())

    def wait_for_wake_word(self, timeout: float = 60.0) -> bool:
        """Blocking: listen until the wake word is heard or *timeout*.

        Returns True when the wake phrase was detected. Safe to call
        repeatedly (a fresh listen per invocation).
        """
        if not self.load():
            return False

        done = threading.Event()
        result = {"heard": False}

        def _callback(indata, frames, time_info, status) -> None:  # noqa: ARG001
            if done.is_set():
                return
            audio = np.asarray(indata, dtype=np.float32).reshape(-1)
            try:
                prediction = self._model.predict(audio)
                if self._score(prediction) > self.threshold:
                    result["heard"] = True
                    done.set()
            except Exception as e:  # pragma: no cover - defensive
                logger.debug(f"Wake-word scoring failed: {e}")

        try:
            with sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype="float32",
                blocksize=self.chunk_size,
                device=self.input_device,
                callback=_callback,
            ):
                done.wait(timeout=timeout)
        except Exception as e:
            logger.warning(f"Wake-word listening failed: {e}")
            return False
        return result["heard"]
