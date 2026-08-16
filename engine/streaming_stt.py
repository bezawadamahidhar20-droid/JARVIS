"""
engine/streaming_stt.py — windowed partial transcription while recording.

The classic VAD waits for the full phrase (up to STT_PHRASE_LIMIT
seconds) to end before transcribing anything. :class:`StreamingListener`
additionally hands each completed window of speech to the STT engine for
a *partial* transcription *while the user is still talking*, so the
orchestrator can react early (e.g. detect \"stop speaking\" instantly)
instead of waiting for the phrase to end.

Perceived-latency win: on a typical 3-5 second utterance the final
transcription is identical to the non-streaming path, but partials
arrive every ~3 seconds during recording, so instant intents are
actioned ~2-4 seconds sooner.

Recording uses blocking reads (like VAD calibration) so the generator
can yield partials mid-phrase; the VAD threshold/silence model is
reused. The whole generator is blocking — call it from a worker thread
or via the STTEngine.async wrapper.
"""

import numpy as np
import sounddevice as sd

from engine.vad import rms
from utils.logger import get_logger

logger = get_logger("streaming_stt")

# ── Load config safely ────────────────────────────────────────
try:
    from config import stt_config, vad_config

    SAMPLE_RATE = stt_config.SAMPLE_RATE
    INPUT_DEVICE = stt_config.INPUT_DEVICE
    TIMEOUT = stt_config.TIMEOUT
    PHRASE_LIMIT = stt_config.PHRASE_LIMIT
    MIN_THRESHOLD = vad_config.MIN_THRESHOLD
    FIXED_THRESHOLD = vad_config.FIXED_THRESHOLD
    SILENCE_DURATION = vad_config.SILENCE_DURATION
    CHUNK_DURATION = vad_config.CHUNK_DURATION
    MIN_SPEECH_CHUNKS = vad_config.MIN_SPEECH_CHUNKS
    STT_STREAM_WINDOW = 3.0
except Exception:
    SAMPLE_RATE = 16000
    INPUT_DEVICE = None
    TIMEOUT = 5
    PHRASE_LIMIT = 10
    MIN_THRESHOLD = 120.0
    FIXED_THRESHOLD = 500.0
    SILENCE_DURATION = 0.7
    CHUNK_DURATION = 0.05
    MIN_SPEECH_CHUNKS = 3
    STT_STREAM_WINDOW = 3.0


class StreamingListener:
    """Records one phrase and yields (partial, is_final) transcriptions.

    The generator blocks (call it from a worker thread or via
    ``STTEngine.astream_listen``); partials arrive while recording is
    still in progress.
    """

    def __init__(
        self,
        vad,
        stt,
        window_seconds: float | None = None,
    ) -> None:
        self.vad = vad
        self.stt = stt
        self.window_seconds = max(
            1.0, float(window_seconds or STT_STREAM_WINDOW)
        )

    def listen_stream(self):
        """
        Yields (partial_text, is_final) tuples.

        * Every ``window_seconds`` of continuous speech, the audio of
          the newest window is transcribed and yielded as a partial.
        * When the phrase ends (silence run, phrase cap), the *full*
          buffer is transcribed and yielded with is_final=True.
        * On timeout / too-short / error yields a single (None, True).
        """
        chunk_size = int(SAMPLE_RATE * CHUNK_DURATION)
        max_silence = max(1, int(SILENCE_DURATION / CHUNK_DURATION))
        max_total = int(PHRASE_LIMIT / CHUNK_DURATION)
        max_wait = int(TIMEOUT / CHUNK_DURATION)
        window_chunks = max(1, int(self.window_seconds / CHUNK_DURATION))
        threshold = max(MIN_THRESHOLD, self.vad.threshold)

        try:
            with sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype="int16",
                blocksize=chunk_size,
                device=INPUT_DEVICE,
            ) as stream:
                captured: list[np.ndarray] = []
                wait_count = 0
                speaking = False
                silence_count = 0

                while True:
                    if not speaking and wait_count >= max_wait:
                        break  # nobody spoke within the timeout
                    try:
                        raw, _ = stream.read(chunk_size)
                    except Exception as e:  # pragma: no cover - defensive
                        logger.error(f"Streaming read failed: {e}")
                        break
                    chunk = np.frombuffer(bytes(raw), dtype=np.int16)
                    level = rms(chunk)

                    if not speaking:
                        wait_count += 1
                        if level > threshold:
                            speaking = True
                            captured.append(chunk)
                        continue

                    captured.append(chunk)
                    if level < threshold:
                        silence_count += 1
                        if silence_count >= max_silence:
                            break
                    else:
                        silence_count = 0

                    if len(captured) >= max_total:
                        break

                    # Window boundary -> partial transcription NOW,
                    # while the user is still talking.
                    if len(captured) % window_chunks == 0:
                        window_audio = np.concatenate(
                            captured[-window_chunks:], axis=0
                        )
                        partial = self.stt.transcribe(window_audio)
                        if partial:
                            yield partial, False
        except Exception as e:
            logger.error(f"Streaming recording error: {e}")
            yield None, True
            return

        if not speaking or len(captured) < MIN_SPEECH_CHUNKS:
            yield None, True
            return

        full_audio = np.concatenate(captured, axis=0)
        final = self.stt.transcribe(full_audio)
        yield final, True
