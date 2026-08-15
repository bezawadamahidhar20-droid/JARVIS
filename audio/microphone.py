"""Streaming microphone capture with automatic speech start/stop."""

import threading

import numpy as np
import sounddevice as sd

from audio.vad import AdaptiveVAD
from utils import logger


class SpeechRecorder:
    """Captures one speech utterance using a callback stream.

    Recording starts after `min_speech_frames` consistent speech frames
    and stops after `silence_frames` consistent non-speech frames, or at
    a hard cap. No audio is written to disk.
    """

    def __init__(self, sample_rate: int = 16000, channels: int = 1,
                 dtype: str = "float32", frame_ms: int = 30,
                 min_speech_ms: int = 250, silence_ms: int = 900,
                 max_record_ms: int = 15000, input_device: int | None = None,
                 calibration_ms: int = 500, flush_ms: int = 150,
                 initial_threshold: float = 0.012, multiplier: float = 3.0,
                 alpha: float = 0.15, settle_frames: int = 10,
                 max_settle_ms: int = 2000):
        self.sample_rate = sample_rate
        self.channels = channels
        self.dtype = dtype
        self.input_device = input_device

        self.frame_ms = frame_ms
        self.blocksize = int(sample_rate * frame_ms / 1000)
        self.min_speech_frames = max(1, int(min_speech_ms / frame_ms))
        self.silence_frames = max(1, int(silence_ms / frame_ms))
        self.max_frames = int(max_record_ms / frame_ms)
        self.flush_frames = max(0, int(flush_ms / frame_ms))

        # Post-open settling: after the initial flush, discard frames until
        # ~settle_frames consecutive non-speech frames confirm the device
        # buffer has drained, so stale audio can never arm the VAD.
        self.settle_frames = max(1, int(settle_frames))
        self._max_settle_frames = max(1, int(max_settle_ms / frame_ms))

        self.vad = AdaptiveVAD(
            initial_threshold=initial_threshold,
            multiplier=multiplier,
            alpha=alpha,
            calibration_frames=max(1, int(calibration_ms / frame_ms)),
        )

        self._buffer: list[np.ndarray] = []
        self._recent: list[np.ndarray] = []
        self._speech_frames = 0
        self._silence_frames = 0
        self._max_reached = False
        self._state = "idle"
        self._settle_count = 0
        self._settle_elapsed = 0
        self._flush_remaining = 0
        self._finished = threading.Event()
        self._lock = threading.Lock()

    @property
    def display_name(self) -> str:
        try:
            idx = sd.default.device[0] if self.input_device is None else self.input_device
            if isinstance(idx, int):
                return sd.query_devices(idx)["name"]
        except Exception:
            pass
        return str(self.input_device)

    def _callback(self, indata, frames, time_info, status) -> None:
        chunk = indata[:, 0].copy()

        with self._lock:
            if self._flush_remaining > 0:
                self._flush_remaining -= 1
                return

            rms = float(np.sqrt(np.mean(chunk * chunk)))
            speech = self.vad.is_speech(rms)

            if self._state == "settling":
                if not self.vad.ready:
                    self._settle_elapsed += 1
                    if self._settle_elapsed >= self._max_settle_frames:
                        self._state = "idle"
                    return
                if speech:
                    self._settle_count = 0
                else:
                    self._settle_count += 1
                self._settle_elapsed += 1
                if self._settle_count >= self.settle_frames \
                        or self._settle_elapsed >= self._max_settle_frames:
                    self._state = "idle"
                return

            if self._state == "idle":
                self._recent.append(chunk)
                if len(self._recent) > self.min_speech_frames:
                    self._recent.pop(0)
                if speech:
                    self._speech_frames += 1
                    if self._speech_frames >= self.min_speech_frames:
                        self._state = "speaking"
                        self._buffer = list(self._recent)
                        self._speech_frames = 0
                        self._silence_frames = 0
                else:
                    self._speech_frames = 0
                return

            self._buffer.append(chunk)
            if speech:
                self._silence_frames = 0
            else:
                self._silence_frames += 1

            if self._silence_frames >= self.silence_frames:
                self._finished.set()
            elif len(self._buffer) >= self.max_frames:
                self._max_reached = True
                self._finished.set()

    def capture_speech(self) -> np.ndarray:
        """Records until trailing silence (or hard cap); returns float32 mono."""
        self._buffer = []
        self._recent = []
        self._speech_frames = 0
        self._silence_frames = 0
        self._max_reached = False
        self.vad.reset()
        self._state = "settling"
        self._settle_count = 0
        self._settle_elapsed = 0
        self._flush_remaining = self.flush_frames
        self._finished.clear()

        # Safety bound: the state machine can only set _finished while in the
        # "speaking" arm. If the device is muted/dead (no frame ever crosses
        # the VAD threshold) the wait would otherwise block forever. The bound
        # covers the longest legitimate capture the machine can produce:
        # flush + quiet-verify settling + max recording, plus a small margin.
        wait_secs = (
            (self.flush_frames + self._max_settle_frames + self.max_frames)
            * self.frame_ms / 1000.0
            + 1.0
        )

        with sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype=self.dtype,
            device=self.input_device,
            blocksize=self.blocksize,
            callback=self._callback,
        ):
            self._finished.wait(timeout=wait_secs)

        if not self._finished.is_set():
            logger.warning(
                f"Capture wait timed out after {wait_secs:.1f} s with no speech "
                "detected (microphone muted/silent or signal below VAD threshold)."
            )

        with self._lock:
            audio = (
                np.concatenate(self._buffer)
                if self._buffer
                else np.zeros(0, dtype=np.float32)
            )
        return audio