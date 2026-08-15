from faster_whisper import WhisperModel

AUDIO_FILE = "test_audio.wav"

print("🧠 Loading Whisper model...")

model = WhisperModel(
    "base",
    device="cpu",
    compute_type="int8"
)

print("🎤 Transcribing audio...")

segments, info = model.transcribe(
    AUDIO_FILE,
    beam_size=5
)

text = ""

for segment in segments:
    text += segment.text

print("\n📝 You said:")
print(text.strip())

print("\n✅ Transcription complete!")