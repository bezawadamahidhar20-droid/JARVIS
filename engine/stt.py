"""Speech-to-text engine built on SpeechRecognition.

``listen()`` is the single entry point and it NEVER raises — every failure
(silence, unintelligible audio, recognition service down, mic dropped) is
logged and returns ``None`` so the main loop can simply continue.
"""

import speech_recognition as sr

import config
from engine.microphone import MicrophoneManager
from utils import logger


class STTEngine:
    """Listens through the microphone and returns recognized text."""

    def __init__(
        self,
        mic_manager: MicrophoneManager,
        language: str = config.STT_LANGUAGE,
        timeout: int = config.STT_TIMEOUT,
    ) -> None:
        self.mic_manager = mic_manager
        self.language = language
        self.timeout = timeout
        self.recognizer = sr.Recognizer()
        # The energy threshold is tuned by calibrate(); the defaults below
        # are a sane starting point before calibration runs.
        self.recognizer.energy_threshold = 300
        self.recognizer.dynamic_energy_threshold = True

    def calibrate(self, duration: float = 2.0) -> None:
        """Measure ambient noise so the mic ignores background hum."""
        self.mic_manager.calibrate(self.recognizer, duration=duration)

    def listen(self) -> str | None:
        """Record one utterance and return its text (or ``None`` on any failure)."""
        if not self.mic_manager.available or self.mic_manager.source is None:
            return None

        # ── Capture ───────────────────────────────────────────────────────────
        try:
            with self.mic_manager.source as source:
                audio = self.recognizer.listen(
                    source,
                    timeout=self.timeout,
                    phrase_time_limit=None,
                )
        except sr.WaitTimeoutError:
            # No speech within timeout — normal when the room is quiet.
            return None
        except OSError as exc:
            logger.error(f"Microphone error: {exc}")
            return None
        except Exception as exc:
            logger.error(f"Listening failed: {exc}")
            return None

        # ── Recognition ───────────────────────────────────────────────────────
        try:
            text = self.recognizer.recognize_google(audio, language=self.language)
        except sr.UnknownValueError:
            logger.warning("Heard audio but could not understand it.")
            return None
        except sr.RequestError as exc:
            logger.error(f"Speech recognition service error: {exc}")
            return None
        except Exception as exc:
            logger.error(f"Recognition failed: {exc}")
            return None

        text = text.strip()
        return text if text else None