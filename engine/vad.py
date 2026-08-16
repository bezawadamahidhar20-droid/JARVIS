"""
engine/vad.py — Adaptive Voice Activity Detection

Records one phrase from the microphone using an *adaptive* RMS
threshold:

  * At startup the VAD samples a few hundred ms of ambient noise and
    derives a speech threshold from it (threshold = max(min_threshold,
    ambient_rms * multiplier)). This adapts to quiet and noisy rooms.
  * Speech is detected chunk-by-chunk; recording stops after a
    configured silence duration, or after the phrase length cap.
  * All timing/threshold knobs live in config.py (VAD_* settings).

Event-driven capture (no polling): record_phrase() uses a sounddevice
*callback* stream. PortAudio invokes the callback as soon as audio
arrives, the callback updates the shared state and signals a
threading.Event when the phrase is done, and the calling thread simply
blocks on that event. There is no sleep-based polling loop, so CPU
usage is negligible while waiting for speech and speech onset is
reacted to immediately.

The VAD only *captures* audio — transcription happens elsewhere
(engine/stt.py). No audio hardware is touched at import time, so this
module is unit-testable with synthetic arrays.
"""

import threading

import numpy as np
import sounddevice as sd

from config import stt_config, vad_config
from utils.logger import get_logger

logger = get_logger("vad")

__all__ = [
    "AdaptiveVAD",
    "rms",
    "derive_threshold",
]

# ── Config (config.py is always import-safe; no local fallbacks) ─────────────
SAMPLE_RATE = stt_config.SAMPLE_RATE
INPUT_DEVICE = stt_config.INPUT_DEVICE
TIMEOUT = stt_config.TIMEOUT
PHRASE_LIMIT = stt_config.PHRASE_LIMIT
CALIBRATE_SECONDS = vad_config.CALIBRATE_SECONDS
THRESHOLD_MULTIPLIER = vad_config.THRESHOLD_MULTIPLIER
MIN_THRESHOLD = vad_config.MIN_THRESHOLD
FIXED_THRESHOLD = vad_config.FIXED_THRESHOLD
SILENCE_DURATION = vad_config.SILENCE_DURATION
CHUNK_DURATION = vad_config.CHUNK_DURATION
MIN_SPEECH_CHUNKS = vad_config.MIN_SPEECH_CHUNKS
VERBOSE = vad_config.VERBOSE


def rms(chunk: np.ndarray) -> float:
    """Volume level (RMS) of an audio chunk."""
    if chunk is None or chunk.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(chunk.astype(np.float32) ** 2)))


def derive_threshold(ambient_rms: float) -> float:
    """
    Map measured ambient noise to a speech threshold.

    Quiet rooms floor at MIN_THRESHOLD; noisy rooms scale with the
    ambient level so speech is still detected above the noise floor.
    """
    return max(MIN_THRESHOLD, ambient_rms * THRESHOLD_MULTIPLIER)


class AdaptiveVAD:
    """
    Calibrates ambient noise and records a single phrase.

    Captured audio is int16 mono at SAMPLE_RATE — the format
    faster-whisper expects after a float conversion.
    """

    def __init__(
        self,
        sample_rate: int = SAMPLE_RATE,
        input_device: int | None = INPUT_DEVICE,
        channels: int = 1,
        dtype: str = "int16",
    ) -> None:
        self.sample_rate = sample_rate
        self.input_device = input_device
        self.channels = channels
        self.dtype = dtype
        self.threshold = FIXED_THRESHOLD
        self.calibrated = False
        # Live levels exposed for dashboards / tests.
        self.live_rms: float = 0.0
        self.live_speech: bool = False

    # ── Calibration ───────────────────────────────────────────

    def calibrate(self) -> None:
        """
        Sample ambient noise and derive the speech threshold.

        Never raises: on any failure the fixed threshold stays in
        effect so JARVIS keeps working.
        """
        seconds = CALIBRATE_SECONDS
        if seconds <= 0:
            self.calibrated = False
            return
        try:
            chunk_size = int(self.sample_rate * CHUNK_DURATION)
            samples = []
            with sd.InputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype=self.dtype,
                blocksize=chunk_size,
                device=self.input_device,
            ) as stream:
                n = int(seconds / CHUNK_DURATION)
                for _ in range(max(1, n)):
                    raw, _ = stream.read(chunk_size)
                    chunk = np.frombuffer(bytes(raw), dtype=np.int16)
                    samples.append(rms(chunk))

            if samples:
                ambient = float(np.median(samples))
                self.threshold = derive_threshold(ambient)
                self.calibrated = True
                logger.debug(
                    f"VAD calibrated: ambient RMS {ambient:.0f} -> "
                    f"threshold {self.threshold:.0f}"
                )
        except Exception as e:
            self.calibrated = False
            logger.warning(
                f"VAD calibration skipped ({e}); using fixed "
                f"threshold {self.threshold:.0f}."
            )

    # ── Recording ─────────────────────────────────────────────

    def record_phrase(self) -> np.ndarray | None:
        """
        Record until the user stops speaking (event-driven).

        The PortAudio callback thread feeds audio chunks into shared
        state and sets ``done`` the moment the phrase finishes (silence
        run, phrase cap, or the no-speech timeout). The calling thread
        blocks on that event — no polling loop, no busy waiting.

        Returns:
            np.ndarray : int16 mono audio, or None on timeout/error.
        """
        chunk_size = int(self.sample_rate * CHUNK_DURATION)
        max_silence = max(1, int(SILENCE_DURATION / CHUNK_DURATION))
        max_total = int(PHRASE_LIMIT / CHUNK_DURATION)
        max_wait = int(TIMEOUT / CHUNK_DURATION)

        lock = threading.Lock()
        chunks: list[np.ndarray] = []
        done = threading.Event()
        state = {
            "wait_count": 0,
            "speaking": False,
            "silence_count": 0,
            "reason": "timeout",  # set to "speech_ended" when phrase done
            "finished": False,
        }

        def _callback(indata, frames, time_info, status) -> None:  # noqa: ARG001
            # Copy! PortAudio reuses the buffer after the callback.
            chunk = np.asarray(indata, dtype=np.int16).reshape(-1).copy()
            level = rms(chunk)
            self.live_rms = level

            with lock:
                if not state["speaking"]:
                    state["wait_count"] += 1
                    if state["wait_count"] > max_wait:
                        state["reason"] = "timeout"
                        state["finished"] = True
                        done.set()
                        return
                    if level > self.threshold:
                        state["speaking"] = True
                        self.live_speech = True
                        chunks.append(chunk)
                        if VERBOSE:
                            print("[Speech detected...]")
                    return

                chunks.append(chunk)
                if level < self.threshold:
                    state["silence_count"] += 1
                else:
                    state["silence_count"] = 0

                if state["silence_count"] >= max_silence or len(chunks) >= max_total:
                    state["reason"] = "speech_ended"
                    state["finished"] = True
                    done.set()

        try:
            if VERBOSE:
                print("\n[Listening... speak now]")

            with sd.InputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype=self.dtype,
                blocksize=chunk_size,
                device=self.input_device,
                callback=_callback,
            ) as stream:
                # Block (event-driven) until the callback signals the
                # phrase is done. The +1s margin covers the case where
                # the stream never delivers data (dead device).
                done.wait(timeout=max_wait * CHUNK_DURATION + 1.0)
                try:
                    stream.stop()
                except Exception:
                    pass
        except sd.PortAudioError as e:
            logger.error(f"PortAudio error while recording: {e}")
            return None
        except Exception as e:
            logger.error(f"Recording error: {e}")
            return None
        finally:
            self.live_speech = False

        with lock:
            if not state["finished"] or state["reason"] != "speech_ended":
                logger.debug("No speech within timeout; giving up.")
                return None  # nobody spoke / dead device
            captured = chunks

        if len(captured) < MIN_SPEECH_CHUNKS:
            logger.debug("Speech too short, ignoring as noise blip.")
            return None

        if VERBOSE:
            print("[Processing...]")
        return np.concatenate(captured, axis=0)
