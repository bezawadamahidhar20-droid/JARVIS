import sounddevice as sd
import numpy as np

SAMPLE_RATE = 16000
DURATION = 5

print("🎤 Speak normally for 5 seconds...")
print("Watch the microphone level below.\n")

audio = sd.rec(
    int(DURATION * SAMPLE_RATE),
    samplerate=SAMPLE_RATE,
    channels=1,
    dtype="float32"
)

sd.wait()

audio = audio.flatten()

peak = np.max(np.abs(audio))
rms = np.sqrt(np.mean(audio ** 2))

print("\n📊 Microphone results")
print("----------------------")
print(f"Peak level: {peak:.6f}")
print(f"RMS level:  {rms:.6f}")

if peak < 0.001:
    print("\n❌ Almost no sound detected.")
    print("Your microphone/input device is probably wrong or muted.")
elif peak < 0.01:
    print("\n⚠️ Very low microphone level.")
    print("Your microphone is detected, but the input is extremely quiet.")
else:
    print("\n✅ Microphone is receiving sound.")