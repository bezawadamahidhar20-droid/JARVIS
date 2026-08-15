"""STEP 4 — Capture-quality test: one fresh sentence -> stats -> timeline -> both models.

Usage:
  .\\.venv\\Scripts\\python.exe test_capture_quality.py "Open Notepad."
  .\\.venv\\Scripts\\python.exe test_capture_quality.py "What time is it?"
"""

import re
import sys
import time

import numpy as np
from scipy.io.wavfile import write

import config
from audio.microphone import SpeechRecorder
from speech.whisper import WhisperEngine
from utils import logger


def energy_timeline(audio, sample_rate, frame_ms=100):
    win = int(sample_rate * frame_ms / 1000)
    n = len(audio) // win
    windows = [float(np.sqrt(np.mean(audio[i * win:(i + 1) * win] ** 2))) for i in range(n)]
    peak = max(windows) or 1e-12
    print(f"\n=== Energy timeline (100 ms windows, peak={peak:.4f}) ===")
    for i, w in enumerate(windows[:120]):
        bar = "#" * int(round(w / peak * 20))
        print(f"  {i * frame_ms:5d} ms | {w:.5f} | {bar}")


def words(text):
    return " ".join(re.sub(r"[^a-z ]", "", text.strip().lower()).split())


def main():
    sentence = " ".join(sys.argv[1:]).strip() or "Open Notepad."
    print(f"\nTARGET: {sentence}\n")

    recorder = SpeechRecorder(
        sample_rate=config.SAMPLE_RATE,
        channels=config.CHANNELS,
        input_device=config.INPUT_DEVICE,
        frame_ms=config.FRAME_MS,
        min_speech_ms=config.MIN_SPEECH_MS,
        silence_ms=config.SILENCE_MS,
        max_record_ms=config.MAX_RECORD_MS,
        calibration_ms=config.CALIBRATION_MS,
        flush_ms=config.STREAM_FLUSH_MS,
        initial_threshold=config.INITIAL_RMS_THRESHOLD,
        multiplier=config.NOISE_MULTIPLIER,
        alpha=config.NOISE_EMA_ALPHA,
    )

    print("Listen... speak the sentence now, then pause.\n")
    audio = recorder.capture_speech()
    if len(audio) == 0:
        logger.error("No speech captured.")
        return

    dur = len(audio) / config.SAMPLE_RATE
    rms = float(np.sqrt(np.mean(audio ** 2)))
    peak = float(np.max(np.abs(audio)))
    print(f"Duration: {dur:.3f} sec")
    print(f"RMS:      {rms:.4f}")
    print(f"Peak:     {peak:+.4f}")
    print(f"Clipping: {int(np.sum(np.abs(audio) >= 0.999))} samples")
    energy_timeline(audio, config.SAMPLE_RATE)

    int16 = (np.clip(audio, -1, 1) * 32767).astype(np.int16)
    write("captured_quality.wav", config.SAMPLE_RATE, int16)
    print("\nSaved captured_quality.wav")

    tiny = WhisperEngine("tiny.en", device=config.WHISPER_DEVICE,
                         compute_type=config.WHISPER_COMPUTE_TYPE,
                         language=config.WHISPER_LANGUAGE, beam_size=config.WHISPER_BEAM_SIZE)
    base = WhisperEngine("base", device=config.WHISPER_DEVICE,
                         compute_type=config.WHISPER_COMPUTE_TYPE,
                         language=config.WHISPER_LANGUAGE, beam_size=config.WHISPER_BEAM_SIZE)

    print("\nEXPECTED:", sentence)
    for name, eng in (("tiny.en", tiny), ("base", base)):
        start = time.perf_counter()
        text = eng.transcribe(audio)
        dt = time.perf_counter() - start
        print(f"{name}: {dt:.2f} sec -> {text or '(empty)'}")

    matches = {n: None for n in ("tiny.en", "base")}
    results = {}
    for name, eng in (("tiny.en", tiny), ("base", base)):
        start = time.perf_counter()
        text = eng.transcribe(audio)
        results[name] = (time.perf_counter() - start, text)
    for name, (dt, text) in results.items():
        exp = words(sentence)
        got = words(text)
        if got == exp:
            verdict = "EXACT MATCH"
        elif exp in got or got in exp:
            verdict = "MINOR TRANSCRIPTION DIFFERENCE"
        elif set(got) & set(exp.split()):
            verdict = "INCORRECT WORDING (partial)"
        else:
            verdict = "COMMAND-SAFETY FAILURE (no overlap)"
        print(f"  {name}: {dt:.2f} sec | {verdict}")


if __name__ == "__main__":
    main()