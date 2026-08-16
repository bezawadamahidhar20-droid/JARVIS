"""
engine/microphone.py — Microphone manager

Built on sounddevice (PortAudio). Detects the default input device,
reports useful diagnostics, and never crashes when a microphone is
missing or disconnected — it simply reports itself unavailable and the
main loop keeps running (text mode / commands still work).
"""

import sounddevice as sd

from utils.logger import get_logger

logger = get_logger("microphone")

try:
    from config import stt_config

    INPUT_DEVICE = stt_config.INPUT_DEVICE
except Exception:
    INPUT_DEVICE = None


class MicrophoneManager:
    """
    Lightweight microphone manager using sounddevice.
    No PyAudio required. No cffi buffer issues.
    """

    def __init__(self, input_device: int | None = INPUT_DEVICE):
        self.input_device = input_device
        self.device_name: str = "Unknown"
        self._available = False
        self._initialize()

    def _initialize(self) -> None:
        """Check if a microphone is available and record its name."""
        try:
            sd.query_devices()
            default_input = self.input_device
            if default_input is None:
                default_input = sd.default.device[0]

            if default_input is None:
                logger.error(
                    "No default input device found. "
                    "Check Windows Sound settings."
                )
                self._available = False
                return

            device_info = sd.query_devices(default_input)
            self.device_name = device_info.get("name", "Unknown")
            logger.info(f"Microphone found: {self.device_name}")
            self._available = True

        except Exception as e:
            logger.error(f"Microphone initialization failed: {e}")
            self._available = False

    def is_available(self) -> bool:
        return self._available

    def describe(self) -> str:
        """Short human-readable status for `jarvis --doctor`."""
        if self._available:
            return f"OK ({self.device_name})"
        return "UNAVAILABLE — no input device detected"

    # ── Compatibility helpers ─────────────────────────────────

    def get_recognizer(self):
        """Kept for compatibility with old code."""
        import speech_recognition as sr

        return sr.Recognizer()

    def get_microphone(self):
        """Not used in sounddevice mode."""
        return None
