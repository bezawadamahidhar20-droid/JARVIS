"""
engine/microphone.py — Microphone manager
Simplified for Python 3.14 + sounddevice compatibility.
"""

import sounddevice as sd
from utils.logger import get_logger

logger = get_logger("microphone")


class MicrophoneManager:
    """
    Lightweight microphone manager using sounddevice.
    No PyAudio required. No cffi buffer issues.
    """

    def __init__(self):
        self._available = False
        self._initialize()

    def _initialize(self) -> None:
        """Check if a microphone is available."""
        try:
            devices = sd.query_devices()
            default_input = sd.default.device[0]

            if default_input is None:
                logger.error("No default input device found.")
                return

            device_info = sd.query_devices(default_input)
            logger.info(
                f"Microphone found: "
                f"{device_info.get('name', 'Unknown')}"
            )
            self._available = True

        except Exception as e:
            logger.error(f"Microphone initialization failed: {e}")
            self._available = False

    def is_available(self) -> bool:
        return self._available

    def get_recognizer(self):
        """Kept for compatibility with old code."""
        import speech_recognition as sr
        return sr.Recognizer()

    def get_microphone(self):
        """Not used in sounddevice mode."""
        return None