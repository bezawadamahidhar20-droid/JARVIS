"""Test the new auto start/stop microphone capture."""

import numpy as np
from scipy.io.wavfile import write

import config
from audio.microphone import SpeechRecorder
from utils import logger


def main():
    recorder = SpeechRecorder(
        sample_rate=config.SAMPLE_RATE,
        channels=config.CHANNELS,
        input_device=config.INPUT_DEVICE,
        frame_ms=config.FRAME_MS,
        min_speech_ms=config.MIN_SPEECH_MS,
        silence_ms=config.SILENCE_MS,
        max_record_ms=config.MAX_RECORD_MS,
        initial_threshold=config.INITIAL_RMS_THRESHOLD,
        multiplier=config.NOISE_MULTIPLIER,
        alpha=config.NOISE_EMA_ALPHA,
        calibration_ms=config.CALIBRATION_MS,
        flush_ms=config.STREAM_FLUSH_MS,
    )

    logger.info(f"Input device: {recorder.display_name}")
    logger.status("Listening... speak now, then pause. No ENTER key needed.")

    start = logger.tick()
    audio = recorder.capture_speech()
    logger.report("CAPTURE", start)

    duration = len(audio) / config.SAMPLE_RATE
    peak = float(np.max(np.abs(audio))) if len(audio) else 0.0
    logger.ok(f"Captured {duration:.2f} sec of speech, peak={peak:.4f}")

    if peak > 0:
        int16 = (audio / peak * 32767 * 0.9).astype(np.int16)
    else:
        int16 = np.zeros(0, dtype=np.int16)
    write("captured_test.wav", config.SAMPLE_RATE, int16)
    logger.info("Saved preview to captured_test.wav")

    if len(audio) == 0:
        logger.warning("No speech detected.")


if __name__ == "__main__":
    main()