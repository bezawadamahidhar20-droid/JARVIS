"""Audio capture diagnostic: verify the mic only records the current sentence.

Uses the EXACT same VAD recorder path as main.py. Records ONE fresh
utterance, saves it, prints stats + an energy timeline, then transcribes
the saved WAV with both tiny.en and base. No commands are executed.

Usage:
  .\\.venv\\Scripts\\python.exe test_diagnostic_capture.py [device_index]
Example:
  .\\.venv\\Scripts\\python.exe test_diagnostic_capture.py 1
"""

import sys

import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write

import config
from audio.microphone import SpeechRecorder
from speech.whisper import WhisperEngine
from utils import logger

DIAG_SENTENCE = "This is a clean microphone test. I am speaking only this sentence."

BANNER = (
    "================================\n"
    "    AUDIO CAPTURE DIAGNOSTIC\n"
    "================================\n"
)


def list_input_devices():
    print("=== Available input devices ===")
    for i, d in enumerate(sd.query_devices()):
        if d["max_input_channels"] > 0:
            print(f"  IN [{i}] {d['name']} | ch={d['max_input_channels']} | sr={d['default_samplerate']}")
    di = sd.default.device[0]
    if isinstance(di, int):
        info = sd.query_devices(di)
        print(f"\n=== CURRENT DEFAULT INPUT ===")
        print(f"Name:        {info['name']}")
        print(f"Channels:    {info['max_input_channels']}")
        print(f"Sample rate: {info['default_samplerate']}")
    print()


def energy_timeline(audio, sample_rate, frame_ms=100):
    """Per-window RMS to reveal stale/leftover audio or a long tail."""
    win = int(sample_rate * frame_ms / 1000)
    n = len(audio) // win
    windows = [float(np.sqrt(np.mean(audio[i * win:(i + 1) * win] ** 2))) for i in range(n)]
    if not windows:
        return
    peak = max(windows) or 1e-12
    print(f"=== Energy timeline (100 ms windows, peak={peak:.4f}) ===")
    for i, w in enumerate(windows[:200]):
        bar = "#" * int(round(w / peak * 20))
        print(f"  {i * frame_ms:5d} ms | {w:.5f} | {bar}")
    print()


def analyze(audio, sample_rate):
    n = len(audio)
    dur = n / sample_rate
    amps = np.abs(audio)
    peak = float(np.max(amps)) if n else 0.0
    rms = float(np.sqrt(np.mean(audio ** 2))) if n else 0.0
    clipped = int(np.sum(amps >= 0.999)) if n else 0

    print("=== Capture statistics ===")
    print(f"Sample rate:  {sample_rate}")
    print(f"Samples:      {n}")
    print(f"Duration:     {dur:.2f} sec")
    print(f"Min amplitude:{float(np.min(audio)):+.4f}" if n else "Min: n/a")
    print(f"Max amplitude:{peak:+.4f}")
    print(f"RMS:          {rms:.4f}")
    print(f"Clipping:     {'YES (' + str(clipped) + ' samples)' if clipped else 'no'}")

    rms_threshold = max(config.INITIAL_RMS_THRESHOLD,
                        (float(np.sqrt(np.mean(audio[:max(1, int(0.5 * sample_rate))] ** 2)))
                         if n > 0.5 * sample_rate else config.INITIAL_RMS_THRESHOLD) * config.NOISE_MULTIPLIER)
    print(f"Est. VAD threshold: {rms_threshold:.4f}")
    print()


def main():
    print(BANNER)
    list_input_devices()

    device = None
    if len(sys.argv) > 1:
        try:
            device = int(sys.argv[1])
            logger.info(f"Using input device index {device}")
        except ValueError:
            logger.error("Device index must be an integer.")
            return
    else:
        print(f"Using default input device index {sd.default.device[0]}")

    recorder = SpeechRecorder(
        sample_rate=config.SAMPLE_RATE,
        channels=config.CHANNELS,
        input_device=device,
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

    print()
    print(f"Say this EXACTLY: \"{DIAG_SENTENCE}\"")
    print("Listen... speak now, then pause.\n")

    audio = recorder.capture_speech()

    if len(audio) == 0:
        logger.error("Nothing captured.")
        return

    analyze(audio, config.SAMPLE_RATE)
    energy_timeline(audio, config.SAMPLE_RATE)

    int16 = (np.clip(audio, -1, 1) * 32767).astype(np.int16) if np.max(np.abs(audio)) > 0 else np.zeros(0, np.int16)
    write("diagnostic_capture.wav", config.SAMPLE_RATE, int16)
    logger.ok("Saved to diagnostic_capture.wav")

    print("\nPress Enter to PLAY BACK the captured audio so you can hear "
          "exactly what was recorded (then answer comes).")
    input()
    sd.play(audio, config.SAMPLE_RATE)
    sd.wait()
    print("[playback finished]\n")

    tiny = WhisperEngine("tiny.en", device=config.WHISPER_DEVICE,
                         compute_type=config.WHISPER_COMPUTE_TYPE,
                         language=config.WHISPER_LANGUAGE, beam_size=config.WHISPER_BEAM_SIZE)
    base = WhisperEngine("base", device=config.WHISPER_DEVICE,
                         compute_type=config.WHISPER_COMPUTE_TYPE,
                         language=config.WHISPER_LANGUAGE, beam_size=config.WHISPER_BEAM_SIZE)

    import time
    for name, engine in (("tiny.en", tiny), ("base", base)):
        start = time.perf_counter()
        text = engine.transcribe(audio)
        dt = time.perf_counter() - start
        print(f"{name}: {dt:.2f} sec -> {text or '(empty)'}")

    logger.ok("Diagnostic complete. Listen to the playback — does it contain "
              "ONLY your sentence?")


if __name__ == "__main__":
    main()