"""Real-voice Whisper benchmark: tiny.en vs base on fresh VAD captures.

Speak each listed sentence into the mic; the recorder waits for you to
start and stops automatically when you pause. The exact same captured
audio is then transcribed by both models and timed.

Usage:  .\\.venv\\Scripts\\python.exe test_whisper_benchmark.py
"""

import time

import config
from audio.microphone import SpeechRecorder
from speech.whisper import WhisperEngine
from utils import logger

SENTENCES = (
    "Hello JARVIS, how are you?",
    "What is Python?",
    "Open Notepad",
    "What time is it?",
    "Explain recursion in simple words.",
)

BANNER = (
    "================================\n"
    "        WHISPER BENCHMARK\n"
    "================================\n"
)


def main():
    print(BANNER)

    whisper_tiny = WhisperEngine(
        model_size="tiny.en",
        device=config.WHISPER_DEVICE,
        compute_type=config.WHISPER_COMPUTE_TYPE,
        language=config.WHISPER_LANGUAGE,
        beam_size=config.WHISPER_BEAM_SIZE,
    )
    whisper_base = WhisperEngine(
        model_size="base",
        device=config.WHISPER_DEVICE,
        compute_type=config.WHISPER_COMPUTE_TYPE,
        language=config.WHISPER_LANGUAGE,
        beam_size=config.WHISPER_BEAM_SIZE,
    )

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

    results = []

    for i, sentence in enumerate(SENTENCES, start=1):
        print(f"Test {i}")
        print(f"Speech: {sentence}\n")
        print("Listen... speak now, then pause.\n")

        audio = recorder.capture_speech()
        duration = len(audio) / config.SAMPLE_RATE
        print(f"[captured {duration:.2f} sec]\n")

        if len(audio) == 0:
            logger.warning("No speech captured for this test — skipping.\n")
            continue

        row = {"sentence": sentence}

        start = time.perf_counter()
        text_tiny = whisper_tiny.transcribe(audio)
        time_tiny = time.perf_counter() - start

        start = time.perf_counter()
        text_base = whisper_base.transcribe(audio)
        time_base = time.perf_counter() - start

        row["tiny"] = (time_tiny, text_tiny)
        row["base"] = (time_base, text_base)
        results.append(row)

        print("tiny.en:")
        print(f"Time: {time_tiny:.2f} sec")
        print(f"Text: {text_tiny or '(empty)'}\n")
        print("base:")
        print(f"Time: {time_base:.2f} sec")
        print(f"Text: {text_base or '(empty)'}")
        print("----------------------------------------\n")

    print("================================\n          RESULTS\n================================\n")

    if not results:
        logger.error("No valid captures recorded.")
        return

    tiny_times = [r["tiny"][0] for r in results]
    base_times = [r["base"][0] for r in results]

    print(f"tiny.en average: {sum(tiny_times) / len(tiny_times):.2f} sec")
    print(f"base average:    {sum(base_times) / len(base_times):.2f} sec\n")

    print("Accuracy is judged by you — compare the transcripts above "
          "against the sentences you spoke.\n")
    for i, r in enumerate(results, start=1):
        print(f"Test {i}: \"{r['sentence']}\"")
        print(f"  tiny.en: {r['tiny'][1]!r}")
        print(f"  base:    {r['base'][1]!r}")


if __name__ == "__main__":
    main()