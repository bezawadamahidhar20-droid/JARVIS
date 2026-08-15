import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write

SAMPLE_RATE = 16000
DURATION = 5

print("🎤 Speak LOUDLY and clearly for 5 seconds...")
print("Say: Hello JARVIS, this is a microphone test.")

audio = sd.rec(
    int(DURATION * SAMPLE_RATE),
    samplerate=SAMPLE_RATE,
    channels=1,
    dtype="float32"
)

sd.wait()

# Normalize the recording
peak = np.max(np.abs(audio))

if peak > 0:
    audio = audio / peak * 0.9

# Convert to 16-bit PCM
audio_int16 = (audio * 32767).astype(np.int16)

write("test_audio.wav", SAMPLE_RATE, audio_int16)

print("✅ Recording complete!")
print(f"Original peak: {peak:.6f}")
print("Saved as: test_audio.wav")