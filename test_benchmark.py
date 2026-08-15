"""Benchmark tiny.en vs base on the same audio to pick the best model.

Usage:  .\\.venv\\Scripts\\python.exe test_benchmark.py [audio_file.wav]
Default file: test_audio.wav
"""

import sys

import numpy as np
from scipy.io.wavfile import read
from scipy.signal import resample_poly

import config
from speech.whisper import WhisperEngine
from utils import logger


def load_audio(path, target_sr=16000):
    sample_rate, data = read(path)
    if data.ndim > 1:
        data = data.mean(axis=1)
    if data.dtype == np.int16:
        data = data.astype(np.float32) / 32768.0
    elif data.dtype == np.uint8:
        data = (data.astype(np.float32) - 128.0) / 128.0
    if sample_rate != target_sr:
        data = resample_poly(data, target_sr, sample_rate).astype(np.float32)
    return data


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "test_audio.wav"
    audio = load_audio(path)
    logger.info(f"Loaded {path}: {len(audio) / 16000:.2f} sec")

    for size in ("tiny.en", "base"):
        engine = WhisperEngine(
            model_size=size,
            device=config.WHISPER_DEVICE,
            compute_type=config.WHISPER_COMPUTE_TYPE,
            language=config.WHISPER_LANGUAGE,
            beam_size=config.WHISPER_BEAM_SIZE,
        )
        start = logger.tick()
        text = engine.transcribe(audio)
        logger.report(f"WHISPER {size}", start)
        print(f"[TEXT] {text or '(empty)'}\n")


if __name__ == "__main__":
    main()