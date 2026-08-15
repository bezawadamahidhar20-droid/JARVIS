"""Central configuration for JARVIS.

Edit values here to tune audio capture, VAD, Whisper, Ollama, and TTS.
No code changes are needed for common adjustments — everything is wired
through this single module.
"""

# ── Audio capture ─────────────────────────────────────────────────────────────
SAMPLE_RATE: int = 16000
CHANNELS: int = 1
DTYPE: str = "float32"
INPUT_DEVICE = None  # None = system default input; or a device index / name

# ── Voice activity detection ──────────────────────────────────────────────────
FRAME_MS: int = 30                # Audio frame duration fed to the VAD (ms)
INITIAL_RMS_THRESHOLD: float = 0.012   # Absolute floor — nothing quieter is ever speech
NOISE_MULTIPLIER: float = 3.0         # speech = noise_estimate × this
NOISE_EMA_ALPHA: float = 0.15         # EMA smoothing for the noise estimate
CALIBRATION_MS: int = 500             # How long to calibrate before detecting speech
MIN_SPEECH_MS: int = 250              # Minimum consecutive speech before arming
SILENCE_MS: int = 900                 # Trailing silence before stopping capture
MAX_RECORD_MS: int = 15000            # Hard cap per utterance (15 s)
STREAM_FLUSH_MS: int = 150            # Discard this much audio right after stream open

# ── Whisper ───────────────────────────────────────────────────────────────────
WHISPER_MODEL: str = "base"           # "tiny.en" is faster; "base" is more accurate
WHISPER_DEVICE: str = "cpu"
WHISPER_COMPUTE_TYPE: str = "int8"    # int8 quantisation keeps CPU latency tolerable
WHISPER_LANGUAGE: str = "en"
WHISPER_BEAM_SIZE: int = 1            # Greedy decoding — fastest on CPU

# ── Ollama / Qwen3 ────────────────────────────────────────────────────────────
OLLAMA_URL: str = "http://localhost:11434/api/generate"
OLLAMA_MODEL: str = "qwen3:8b"
OLLAMA_TIMEOUT: int = 120             # Seconds to wait for a generation response
OLLAMA_NUM_PREDICT: int = 120         # Cap generation length for voice-friendly replies
OLLAMA_KEEP_ALIVE: str = "30m"        # Keep model resident between turns
OLLAMA_MEMORY_TURNS: int = 5          # Sliding window: last N user/assistant pairs
                                      # sent as context on every generate() call.
                                      # Set to 0 to disable multi-turn memory.

# ── Text-to-speech ────────────────────────────────────────────────────────────
TTS_VOICE_PATH: str = "voices/en_US-lessac-medium.onnx"

# Post-TTS microphone cooldown.
# After sd.wait() returns the speaker tail / room reverb has not decayed yet.
# This 0.75 s hold-off prevents the VAD from picking up JARVIS's own voice on
# the next loop iteration and feeding phantom speech into Whisper.
# Set to 0.0 for headphone-only setups or when echo cancellation is external.
POST_TTS_COOLDOWN_S: float = 0.75

# ── JARVIS personality ────────────────────────────────────────────────────────
SYSTEM_PROMPT: str = (
    "You are JARVIS, a concise and helpful desktop voice assistant. "
    "Answer in one to two short sentences, spoken aloud. "
    "Give direct answers; avoid long explanations unless the user "
    "explicitly asks for detail."
)

# ── Conversation dataset (fine-tuning) ────────────────────────────────────────
DATASET_PATH: str = "data/conversations.jsonl"  # raw ShareGPT-format log of real turns
