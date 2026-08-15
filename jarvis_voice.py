import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
from faster_whisper import WhisperModel
import requests
import time

SAMPLE_RATE = 16000
DURATION = 4

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen3:8b"

print("🧠 Loading Whisper model...")

whisper = WhisperModel(
    "base",
    device="cpu",
    compute_type="int8"
)

print("✅ Whisper ready")
print("🤖 JARVIS is ready!\n")


def record_audio():
    print("🎤 Listening...")

    audio = sd.rec(
        int(DURATION * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32"
    )

    sd.wait()

    audio = audio.flatten()

    # Normalize quiet microphone input
    peak = np.max(np.abs(audio))

    if peak > 0:
        audio = audio / peak * 0.9

    audio_int16 = (audio * 32767).astype(np.int16)

    write("voice_input.wav", SAMPLE_RATE, audio_int16)

    return "voice_input.wav"


def transcribe_audio(audio_file):
    print("🧠 Understanding...")

    segments, info = whisper.transcribe(
        audio_file,
        beam_size=1,
        language="en",
        vad_filter=True,
        condition_on_previous_text=False
    )

    text = " ".join(
        segment.text.strip()
        for segment in segments
    ).strip()

    return text


def ask_jarvis(text):
    print("🤖 JARVIS is thinking...")

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": OLLAMA_MODEL,
            "prompt": text,
            "stream": False
        },
        timeout=120
    )

    response.raise_for_status()

    return response.json()["response"]


while True:

    try:
        input("Press ENTER to speak...")

        start = time.time()

        audio_file = record_audio()

        text = transcribe_audio(audio_file)

        transcription_time = time.time() - start

        if not text:
            print("❌ I couldn't understand you.\n")
            continue

        print(f"\n👤 You: {text}")
        print(f"⏱️ Speech processing: {transcription_time:.2f} seconds")

        if text.lower() in [
            "exit",
            "quit",
            "stop",
            "goodbye"
        ]:
            print("🤖 JARVIS: Goodbye.")
            break

        answer = ask_jarvis(text)

        print(f"\n🤖 JARVIS: {answer}\n")

    except KeyboardInterrupt:
        print("\n👋 JARVIS stopped.")
        break

    except Exception as e:
        print(f"\n❌ Error: {e}\n")