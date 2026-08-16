"""
engine/barge_in.py — interrupt TTS the moment the user starts talking.

True barge-in requires the microphone to run *concurrently* with TTS
playback. While JARVIS is speaking, :class:`BargeInMonitor` runs a
sounddevice callback stream that watches for speech; the moment enough
loud chunks are heard it fires ``on_speech`` (the orchestrator calls
``tts.stop()``), so a long answer can be cut off mid-sentence and the
next utterance captured immediately.

Design rules:
  * The monitor only *observes* — it never records or transcribes.
  * ``on_speech`` is invoked from the audio callback thread; the
    callback (e.g. TTSEngine.stop) must be thread-safe.
  * Any failure (no mic, busy device) degrades to no-op: the assistant
    simply keeps its normal wait-then-listen behavior.
  * Opt-in (ENABLE_BARGE_IN=false by default): without acoustic echo
    cancellation the monitor can be tripped by JARVIS's own voice.
"""

import threading

import numpy as np
import sounddevice as sd

from utils.logger import get_logger

logger = get_logger("barge_in")


class BargeInMonitor:
    """Listens on the mic while JARVIS is speaking and fires
    ``on_speech`` when the user starts talking."""

    def __init__(
        self,
        sample_rate: int = 16000,
        threshold: float = 500.0,
        chunks_to_confirm: int = 3,
        chunk_duration: float = 0.05,
        input_device: int | None = None,
    ) -> None:
        self.sample_rate = sample_rate
        # int16-scale RMS threshold (like the VAD's); the float32 audio
        # is scaled up inside the callback before comparison.
        self.threshold = max(1.0, float(threshold))
        self.chunks_to_confirm = max(1, int(chunks_to_confirm))
        self.chunk_size = int(sample_rate * chunk_duration)
        self.input_device = input_device
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._on_speech = None

    def start(self, on_speech) -> bool:
        """Begin monitoring. Returns True when the monitor is running."""
        self._on_speech = on_speech
        if self._thread is not None and self._thread.is_alive():
            return True
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, name="jarvis-barge-in", daemon=True
        )
        self._thread.start()
        return True

    def stop(self) -> None:
        """Stop monitoring and join the monitor thread."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self._thread = None

    @property
    def active(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def _run(self) -> None:
        speech_chunks = 0

        def _callback(indata, frames, time_info, status) -> None:  # noqa: ARG001
            nonlocal speech_chunks
            if self._stop.is_set():
                return
            chunk = np.asarray(indata, dtype=np.float32).reshape(-1)
            # sounddevice delivers float32 in [-1, 1]; scale the RMS to
            # the int16 convention so the threshold matches the VAD's
            # (vad.threshold is in int16 units, e.g. ~500).
            level = (
                float(np.sqrt(np.mean(chunk ** 2))) * 32767.0
                if chunk.size else 0.0
            )
            if level > self.threshold:
                speech_chunks += 1
                if speech_chunks >= self.chunks_to_confirm:
                    self._stop.set()
                    try:
                        if self._on_speech is not None:
                            self._on_speech()
                    except Exception as e:  # pragma: no cover - defensive
                        logger.debug(f"Barge-in callback failed: {e}")
            else:
                speech_chunks = 0

        try:
            with sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype="float32",
                blocksize=self.chunk_size,
                device=self.input_device,
                callback=_callback,
            ):
                # The callback sets _stop on speech; wait() also lets
                # stop() from the orchestrator end the loop promptly.
                while not self._stop.wait(0.05):
                    pass
        except Exception as e:
            logger.warning(f"Barge-in monitor unavailable: {e}")
