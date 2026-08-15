"""Channel-level microphone diagnostic: one sentence, all 4 channels.

Uses the SAME VAD parameters, flush, blocksize, and sample rate as the
production path, but opens a 4-channel stream so channel 0 drives the VAD
(exactly like production's `indata[:, 0]`) while channels 0-3 are buffered
in parallel. Prints per-channel RMS/peak and the current mono mix.

Usage:
  .\\.venv\\Scripts\\python.exe test_channel_analysis.py
"""

import threading

import numpy as np
import sounddevice as sd

import config
from audio.vad import AdaptiveVAD
from utils import logger

FRAME_MS = config.FRAME_MS
BLOCKSIZE = int(config.SAMPLE_RATE * FRAME_MS / 1000)
MIN_SPEECH_FRAMES = max(1, int(config.MIN_SPEECH_MS / FRAME_MS))
SILENCE_FRAMES = max(1, int(config.SILENCE_MS / FRAME_MS))
FLUSH_FRAMES = max(0, int(config.STREAM_FLUSH_MS / FRAME_MS))
MAX_FRAMES = int(config.MAX_RECORD_MS / FRAME_MS)
N_CHANNELS = 4


def analyze(label, data):
    if len(data) == 0:
        print(f"{label}: EMPTY")
        return
    rms = float(np.sqrt(np.mean(data * data)))
    peak = float(np.max(np.abs(data)))
    print(f"{label}:")
    print(f"  RMS:  {rms:.6f}")
    print(f"  Peak: {peak:.6f}")


def main():
    print("=== CHANNEL ANALYSIS ===")
    print(f"Device: {sd.query_devices(sd.default.device[0])['name']}")
    print(f"Sample rate: {config.SAMPLE_RATE} Hz | channels: {N_CHANNELS}")
    print("Speak ONE sentence now, then pause.\n")

    vad = AdaptiveVAD(
        initial_threshold=config.INITIAL_RMS_THRESHOLD,
        multiplier=config.NOISE_MULTIPLIER,
        alpha=config.NOISE_EMA_ALPHA,
        calibration_frames=max(1, int(config.CALIBRATION_MS / FRAME_MS)),
    )

    buffers = [[] for _ in range(N_CHANNELS)]
    recent = [[] for _ in range(N_CHANNELS)]
    speech_count = 0
    silence_count = 0
    flush_remaining = FLUSH_FRAMES
    state = "idle"
    finished = threading.Event()

    def callback(indata, frames, time_info, status):
        nonlocal speech_count, silence_count, flush_remaining, state
        if flush_remaining > 0:
            flush_remaining -= 1
            return

        ch = [indata[:, i].copy() for i in range(N_CHANNELS)]
        rms_ch0 = float(np.sqrt(np.mean(ch[0] * ch[0])))
        speech = vad.is_speech(rms_ch0)

        if state == "idle":
            for i in range(N_CHANNELS):
                recent[i].append(ch[i])
                if len(recent[i]) > MIN_SPEECH_FRAMES:
                    recent[i].pop(0)
            if speech:
                speech_count += 1
                if speech_count >= MIN_SPEECH_FRAMES:
                    state = "speaking"
                    for i in range(N_CHANNELS):
                        buffers[i] = list(recent[i])
                    speech_count = 0
                    silence_count = 0
            else:
                speech_count = 0
            return

        for i in range(N_CHANNELS):
            buffers[i].append(ch[i])
        if speech:
            silence_count = 0
        else:
            silence_count += 1

        if silence_count >= SILENCE_FRAMES:
            finished.set()
        elif len(buffers[0]) >= MAX_FRAMES:
            finished.set()

    with sd.InputStream(
        samplerate=config.SAMPLE_RATE,
        channels=N_CHANNELS,
        dtype="float32",
        device=config.INPUT_DEVICE,
        blocksize=BLOCKSIZE,
        callback=callback,
    ):
        finished.wait()

    print("CHANNEL ANALYSIS\n")
    mixes = []
    for i in range(N_CHANNELS):
        data = np.concatenate(buffers[i]) if buffers[i] else np.zeros(0, np.float32)
        mixes.append(data)
        analyze(f"Channel {i}", data)

    # Production mono mix = channel 0 (what main.py actually transcribes)
    analyze("Current mono mix (channel 0)", mixes[0])

    # Reference: naive mean of all 4 channels (for information only)
    if all(len(m) for m in mixes):
        n = min(len(m) for m in mixes)
        naive = np.mean(np.stack([m[:n] for m in mixes], axis=0), axis=0)
        analyze("Naive 4-ch mean (reference)", naive)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nCancelled.")