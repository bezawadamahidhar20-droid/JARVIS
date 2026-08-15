"""Central configuration for JARVIS."""

# ---- Audio capture ----
SAMPLE_RATE = 16000
CHANNELS = 1
DTYPE = "float32"
INPUT_DEVICE = None  # None = system default input; or device index/name

# ---- Voice activity detection ----
FRAME_MS = 30
INITIAL_RMS_THRESHOLD = 0.012
NOISE_MULTIPLIER = 3.0
NOISE_EMA_ALPHA = 0.15
CALIBRATION_MS = 500
MIN_SPEECH_MS = 250
SILENCE_MS = 900
MAX_RECORD_MS = 15000
STREAM_FLUSH_MS = 150  # discard this much audio right after opening the input stream

# ---- Whisper ----
WHISPER_MODEL = "base"  # try "tiny.en" vs "base" in Step 2 benchmark
WHISPER_DEVICE = "cpu"
WHISPER_COMPUTE_TYPE = "int8"
WHISPER_LANGUAGE = "en"
WHISPER_BEAM_SIZE = 1

# ---- Ollama ----
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen3:8b"
OLLAMA_TIMEOUT = 120
OLLAMA_NUM_PREDICT = 120  # cap generation length for voice-friendly replies
OLLAMA_KEEP_ALIVE = "30m"  # keep model resident between turns

# ---- Text-to-speech ----
TTS_VOICE_PATH = "voices/en_US-lessac-medium.onnx"

# ---- JARVIS personality ----
SYSTEM_PROMPT = (
    "You are JARVIS, a concise and helpful desktop voice assistant. "
    "Answer in one to two short sentences, spoken aloud. "
    "Give direct answers; avoid long explanations unless the user "
    "explicitly asks for detail."
)