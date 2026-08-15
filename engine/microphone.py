"""Microphone management for JARVIS.

WHY A CUSTOM AUDIO SOURCE
-------------------------
SpeechRecognition's built-in ``sr.Microphone`` requires PyAudio, which has
no prebuilt wheel for recent Python releases on Windows (3.13/3.14) and
fails to compile from source without a full MSVC+PortAudio toolchain.

This module therefore ships a tiny ``sr.AudioSource`` backed by
``sounddevice`` (already installed, bundles its own PortAudio DLLs). If
PyAudio happens to be present we still prefer it; otherwise we fall back
to the sounddevice source automatically.

The manager NEVER raises during construction: if no input device exists it
simply reports ``available == False`` so the caller can disable voice mode
gracefully instead of crashing.
"""

import speech_recognition as sr

from utils import logger

# Audio captured as 16-bit mono PCM at 16 kHz — what Google's recognizer
# expects and a small, fast stream for a local assistant.
SAMPLE_RATE = 16000
SAMPLE_WIDTH = 2  # bytes per sample for int16
CHUNK = 1024      # frames read per buffer


class _StreamAdapter:
    """Makes sounddevice's RawInputStream look like a PyAudio stream.

    SpeechRecognition's ``listen()`` / ``adjust_for_ambient_noise()`` call
    ``source.stream.read(num_frames)`` and expect raw PCM bytes back. The
    sounddevice stream returns ``(numpy_array, overflowed)``, so we adapt.
    """

    def __init__(self, raw_stream) -> None:
        self._raw = raw_stream

    def read(self, num_frames: int) -> bytes:
        data, _overflowed = self._raw.read(num_frames)
        return data.tobytes()

    def stop(self) -> None:
        self._raw.stop()

    def close(self) -> None:
        self._raw.close()


class SoundDeviceSource(sr.AudioSource):
    """An ``sr.AudioSource`` implementation backed by sounddevice."""

    def __init__(
        self,
        sample_rate: int = SAMPLE_RATE,
        chunk: int = CHUNK,
        device_index: int | None = None,
    ) -> None:
        self.SAMPLE_RATE: int = sample_rate
        self.SAMPLE_WIDTH: int = SAMPLE_WIDTH
        self.CHUNK: int = chunk
        self._device_index: int | None = device_index
        self.stream = None  # filled by __enter__
        self.audio_open: bool = False

    def __enter__(self) -> "SoundDeviceSource":
        if self.audio_open:
            raise ValueError("Audio source is already open.")
        import sounddevice as sd

        raw = sd.RawInputStream(
            samplerate=self.SAMPLE_RATE,
            channels=1,
            dtype="int16",
            blocksize=self.CHUNK,
            device=self._device_index,
        )
        raw.start()
        self.stream = _StreamAdapter(raw)
        self.audio_open = True
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        if self.stream is not None:
            try:
                self.stream.stop()
            finally:
                self.stream.close()
        self.stream = None
        self.audio_open = False
        return False  # never swallow exceptions


class MicrophoneManager:
    """Owns the microphone source and its availability status."""

    def __init__(self) -> None:
        self._source = None
        self.available: bool = False
        self._init_source()

    def _init_source(self) -> None:
        """Pick the best available audio backend and probe it."""
        # 1. Prefer the official PyAudio-backed source when installed.
        try:
            import pyaudio  # noqa: F401  (imported only as an availability probe)

            self._source = sr.Microphone()
            self.available = True
            logger.ok("Microphone ready (PyAudio backend)")
            return
        except Exception:
            logger.info("PyAudio not available; using the sounddevice backend.")

        # 2. Fall back to our sounddevice source. Opening and closing the
        #    stream immediately verifies a real input device exists.
        try:
            self._source = SoundDeviceSource()
            with self._source:
                pass  # probe: enter/exit opens & closes the stream
            self.available = True
            logger.ok("Microphone ready (sounddevice backend)")
        except Exception as exc:
            logger.error(f"No microphone available: {exc}")
            self._source = None
            self.available = False

    def calibrate(self, recognizer: sr.Recognizer, duration: float = 2.0) -> None:
        """Calibrate the energy threshold against ambient noise.

        Must be called once before listening so background hum doesn't
        register as speech. Failures are logged, never fatal.
        """
        if not self.available or self._source is None:
            return
        try:
            with self._source as source:
                recognizer.adjust_for_ambient_noise(source, duration=duration)
            logger.ok("Ambient noise calibrated")
        except Exception as exc:
            logger.warning(f"Ambient calibration failed: {exc}")

    @property
    def source(self):
        """The active ``sr.AudioSource`` (or None when unavailable)."""
        return self._source

    def is_available(self) -> bool:
        return self.available