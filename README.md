# JARVIS — Local AI Voice Assistant

A voice assistant that runs **100% locally**: your microphone captures
speech, [Ollama](https://ollama.com) + **llama3.2:3b** answer questions, and
your speakers reply. No cloud AI, no paid API keys, no monthly fees.

```
   ╔════════════════════════════════════════════════════════════╗
   ║  microphone → speech-to-text → intent router → llama3.2:3b  ║
   ║  llama3.2 / commands → text-to-speech → speakers            ║
   ╚════════════════════════════════════════════════════════════╝
```

## Key features

- **llama3.2:3b brain** — ~3× faster than the old Qwen3 8B on CPU.
- **Streaming replies** — the first sentence is spoken while the rest of the
  answer is still generating (`ask_stream()` + non-blocking TTS queue).
- **Instant greetings** — 24 canned replies ("hello", "who are you", "good
  night", …) answered locally with zero AI round-trip.
- **Background model warm-up** — the model is pre-loaded into RAM at startup,
  so the first question has no cold-start delay.
- **Conversation memory** — follow-ups like *"Who is Elon Musk?"* →
  *"What companies does he own?"* just work.
- **No PyAudio, no FLAC** — microphone capture uses `sounddevice`, and speech
  recognition sends raw PCM straight to Google's HTTP API. Works on Python
  3.14 Windows 11 (where PyAudio and the bundled `flac.exe` both fail).
- **AI is the default route** — anything the router doesn't recognise as a
  command goes to the model, never an "I don't understand" dead-end.

## Project structure

```
JARVIS/
├── main.py                  # JARVIS class + main loop (voice & --text modes)
├── config.py                # typed safe getters, loads all settings from .env
├── .env                     # YOUR settings (never commit this)
├── .env.example             # template with defaults & comments
├── requirements.txt
├── engine/                  # audio I/O
│   ├── microphone.py        # sounddevice mic source (no PyAudio)
│   ├── stt.py               # sounddevice capture → Google STT via HTTP (raw PCM)
│   └── tts.py               # pyttsx3 queue + daemon worker (non-blocking)
├── brain/                   # AI brain
│   ├── ollama_client.py     # llama3.2:3b via Ollama /api/chat (streaming + timing)
│   ├── memory.py            # rolling memory + character trimming (follow-ups!)
│   └── router.py            # COMMAND / AI_QUESTION / FAST_RESPONSE / EXIT / CLEAR_MEMORY
├── commands/
│   ├── registry.py          # dispatches local commands
│   ├── system_commands.py   # open apps/websites, screenshots
│   └── time_commands.py     # time & date
└── utils/
    └── logger.py            # console + file logging
```

## Requirements

- **Windows 10/11**
- **Python 3.10–3.14** — download from [python.org](https://python.org)
- **Ollama** (local model server) — see below

---

## Installation (Windows)

### 1. Create a virtual environment

```powershell
cd JARVIS
python -m venv .venv
.\.venv\Scripts\activate
```

### 2. Install Python dependencies

```powershell
pip install -r requirements.txt
```

No PyAudio needed. The microphone uses `sounddevice` (bundles its own
PortAudio DLLs), and speech recognition hits Google's HTTP API directly, so
nothing extra has to compile on Python 3.14.

### 3. Configure your settings

```powershell
Copy-Item .env.example .env
notepad .env
```

Edit anything you like (owner name, TTS rate, model, …). All values have
working defaults, so this step is optional.

### 4. Install and start Ollama

1. Download & install **Ollama** from <https://ollama.com/download>.
2. Install the brain:

```powershell
ollama pull llama3.2:3b
```

3. Make sure it's running — the Ollama app runs it automatically, or:

```powershell
ollama serve
```

Verify it's up: open <http://localhost:11434> in a browser (you'll see
"Ollama is running").

### 5. Run JARVIS

Voice mode (needs a working microphone):

```powershell
python main.py
```

Text mode (no mic needed — great for testing):

```powershell
python main.py --text
```

---

## Using JARVIS

| You say | What happens |
|---|---|
| "Hello JARVIS" / "Who are you" | **Instant canned reply** — no AI call |
| "What is Python?" | AI explains it |
| "Who is Elon Musk?" → "What companies does he own?" | **AI, with memory** — "he" resolves correctly |
| "Tell me a joke" | AI tells a joke |
| "What time is it?" / "What's the date?" | Local clock / calendar |
| "Open YouTube" / "Go to GitHub" | Opens the website in your browser |
| "Open notepad" / "Open calculator" | Launches the app |
| "Take a screenshot" | Saves a PNG into `outputs/` |
| "Clear memory" | Forgets the conversation |
| "Goodbye" / "Exit" | Gracefully shuts down |

**Anything** not in the command list goes to llama3.2:3b — that's the point.
General questions, follow-ups, trivia, coding help… it's all answered.

---

## Speed tuning

The stack is tuned for low latency on CPU:

| Feature | What it does |
|---|---|
| `OLLAMA_STREAM=true` | Tokens arrive as generated; first sentence is spoken while the rest renders |
| `OLLAMA_KEEP_ALIVE=30m` | Model stays loaded in RAM — no cold reload between questions |
| `OLLAMA_NUM_PREDICT=150` | Caps answers at ~3 sentences, no rambling |
| `OLLAMA_NUM_CTX=2048` | Smaller context window = faster prompt processing |
| `MEMORY_MAX_TURNS=6` + `MEMORY_MAX_CHARS=3000` | Trims old turns so prompts stay short |
| `ENABLE_FAST_RESPONSES=true` | Greetings answered instantly from a local table |
| `ENABLE_WARMUP=true` | Background thread pre-loads the model at startup |
| `STT_SILENCE_DURATION=0.7` | Faster phrase cut-off than the old 1.2 s |

Timing is logged for every interaction as
`[timing] first token 0.8s | total 3.2s | 47 tokens`.

---

## Configuration reference (`.env`)

| Key | Default | Meaning |
|---|---|---|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Your local Ollama server |
| `OLLAMA_MODEL` | `llama3.2:3b` | Model used as the brain |
| `OLLAMA_TIMEOUT` | `120` | Seconds to wait for a reply |
| `OLLAMA_TEMPERATURE` | `0.7` | Creativity (0 strict → 1 wild) |
| `OLLAMA_STREAM` | `true` | Stream tokens for instant first sentence |
| `OLLAMA_NUM_PREDICT` | `150` | Max response length (tokens) |
| `OLLAMA_NUM_CTX` | `2048` | Context window size |
| `OLLAMA_KEEP_ALIVE` | `30m` | How long the model stays loaded in RAM |
| `OLLAMA_NUM_GPU` | `99` | GPU offload (ignored on CPU-only boxes) |
| `STT_LANGUAGE` | `en-US` | Speech-recognition language |
| `STT_TIMEOUT` | `5` | Seconds to wait for speech to start |
| `STT_PHRASE_LIMIT` | `10` | Max phrase length (seconds) |
| `STT_SILENCE_DURATION` | `0.7` | Seconds of silence that ends a phrase |
| `STT_CHUNK_DURATION` | `0.05` | Audio chunk length (seconds) |
| `GOOGLE_STT_KEY` | *(built-in)* | Google speech API key |
| `TTS_RATE` | `200` | Speaking rate (words/min) |
| `JARVIS_NAME` | `JARVIS` | Assistant name |
| `JARVIS_OWNER` | `Sir` | What JARVIS calls you |
| `MEMORY_MAX_TURNS` | `6` | How many turns of context to remember |
| `MEMORY_MAX_CHARS` | `3000` | Max characters of history per request |
| `ENABLE_FAST_RESPONSES` | `true` | Instant canned greeting replies |
| `ENABLE_WARMUP` | `true` | Pre-load model at startup |
| `LOG_LEVEL` | `INFO` | Logging verbosity |

---

## Troubleshooting

**"Cannot reach Ollama at http://localhost:11434"**
- Ollama isn't running. Start the Ollama app or run `ollama serve`.
- Check <http://localhost:11434> in your browser.

**"Model 'llama3.2:3b' is not installed"**
- Run `ollama pull llama3.2:3b`.

**The first AI answer feels slow**
- That's the model cold-starting into RAM on the very first run after a
  reboot. The background warm-up and `OLLAMA_KEEP_ALIVE=30m` keep every
  later question fast. Raise `OLLAMA_TIMEOUT` if it ever times out.

**"No microphone found"**
- Check your mic is plugged in and not disabled in Windows Sound settings.
- Run `python main.py --text` to test without a mic.

**Speech isn't being recognized / JARVIS hears nothing**
- Check the mic isn't muted (Windows Sound → Input).
- JARVIS only listens once it has finished speaking (echo guard), so keep
  the room quiet while it talks.

**No sound from JARVIS**
- Check your default output device. pyttsx3 uses Windows SAPI5 voices.
- JARVIS still prints replies to the console, so the loop keeps working.

**The AI answers are full of markdown (`**`, `##`, backticks)**
- They shouldn't be — the system prompt forbids markdown and the TTS layer
  strips it anyway. If you still see it, the reply you heard is the clean
  version; the console shows the raw model output.

---

## License

MIT — see `LICENSE`.