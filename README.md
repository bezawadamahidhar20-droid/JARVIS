# JARVIS — Local AI Voice Assistant

A voice assistant that runs **100% locally**: your microphone captures
speech, [Ollama](https://ollama.com) + **Qwen3 8B** answer questions, and
your speakers reply. No cloud AI, no API keys, no monthly fees.

```
   ╔══════════════════════════════════════════════════════════╗
   ║  microphone → speech-to-text → intent router → Qwen3    ║
   ║  Qwen3 / commands → text-to-speech → speakers            ║
   ╚══════════════════════════════════════════════════════════╝
```

## What was fixed

The previous version had four root problems — all solved here:

| Problem | Fix |
|---|---|
| General questions hit an "I don't understand" branch | **AI is now the default** for any input the router doesn't recognise as a command (`brain/router.py`) |
| No conversation memory — follow-ups broke | **Rolling message window** sent with every Ollama call (`brain/memory.py`) |
| Crashed when Ollama was offline / mic failed | **try/except everywhere** with graceful spoken fallbacks |
| Secrets & settings hardcoded | Everything moved to a **`.env` file** via `python-dotenv` |

## Project structure

```
JARVIS/
├── main.py                  # JARVIS class + main loop (voice & --text modes)
├── config.py                # loads all settings from .env
├── .env                     # YOUR settings (never commit this)
├── .env.example             # template with defaults & comments
├── requirements.txt
├── engine/                  # audio I/O
│   ├── microphone.py        # mic source (PyAudio or sounddevice fallback)
│   ├── stt.py               # SpeechRecognition → text (never crashes)
│   └── tts.py               # pyttsx3 → speech (markdown cleaned)
├── brain/                   # AI brain
│   ├── ollama_client.py     # Qwen3 via Ollama /api/chat + error handling
│   ├── memory.py            # rolling conversation memory (follow-ups!)
│   └── router.py            # COMMAND / AI_QUESTION / EXIT / CLEAR_MEMORY
├── commands/
│   ├── registry.py          # dispatches local commands
│   ├── system_commands.py   # open apps/websites, screenshots
│   └── time_commands.py     # time & date
└── utils/
    └── logger.py            # coloured console logging
```

## Requirements

- **Windows 10/11**
- **Python 3.10+** — download from [python.org](https://python.org)
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

**PyAudio** (optional): `SpeechRecognition` needs a mic backend. If you are
on Python 3.10–3.12 and want PyAudio, install it with:

```powershell
pipwin install pyaudio
```

> **On Python 3.13+/3.14** PyAudio has no wheel and won't compile without a
> full MSVC + PortAudio toolchain. **Don't worry** — JARVIS automatically
> falls back to a `sounddevice`-based mic backend (bundled with its own
> PortAudio), so voice mode just works.

### 3. Configure your settings

```powershell
Copy-Item .env.example .env
notepad .env
```

Edit anything you like (owner name, TTS rate, timeout, …). All values have
working defaults, so this step is optional.

### 4. Install and start Ollama

1. Download & install **Ollama** from <https://ollama.com/download>.
2. Install the Qwen3 8B brain:

```powershell
ollama pull qwen3:8b
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
| "What is Python?" | AI explains it |
| "Who is Elon Musk?" → "What companies does he own?" | **AI, with memory** — "he" resolves correctly |
| "Tell me a joke" | AI tells a joke |
| "What time is it?" / "What's the date?" | Local clock / calendar |
| "Open YouTube" / "Go to GitHub" | Opens the website in your browser |
| "Open notepad" / "Open calculator" | Launches the app |
| "Take a screenshot" | Saves a PNG into `outputs/` |
| "Clear memory" | Forgets the conversation |
| "Goodbye" / "Exit" | Gracefully shuts down |

**Anything** not in the command list goes to Qwen3 — that's the point.
General questions, follow-ups, trivia, coding help… it's all answered.

---

## Configuration reference (`.env`)

| Key | Default | Meaning |
|---|---|---|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Your local Ollama server |
| `OLLAMA_MODEL` | `qwen3:8b` | Model used as the brain |
| `OLLAMA_TIMEOUT` | `60` | Seconds to wait for a reply |
| `OLLAMA_TEMPERATURE` | `0.7` | Creativity (0 strict → 1 wild) |
| `STT_LANGUAGE` | `en-US` | Speech-recognition language |
| `STT_TIMEOUT` | `5` | Seconds to wait for speech to start |
| `TTS_RATE` | `185` | Speaking rate (words/min) |
| `JARVIS_OWNER` | `Sir` | What JARVIS calls you |
| `MEMORY_MAX_TURNS` | `20` | How many turns of context to remember |

---

## Troubleshooting

**"Cannot reach Ollama at http://localhost:11434"**
- Ollama isn't running. Start the Ollama app or run `ollama serve`.
- Check <http://localhost:11434> in your browser.

**"Model 'qwen3:8b' is not installed"**
- Run `ollama pull qwen3:8b`.

**The first AI answer takes ~30 seconds**
- That's Qwen3 cold-starting into RAM. Raise `OLLAMA_TIMEOUT` in `.env` if
  it ever times out. Later answers are fast.

**"No microphone found"**
- Check your mic is plugged in and not disabled in Windows Sound settings.
- Run `python main.py --text` to test without a mic.

**Speech isn't being recognized / JARVIS hears nothing**
- Run JARVIS once; ambient calibration happens at startup. Keep the room
  quiet for the first 2 seconds.
- Check the mic isn't muted (Windows Sound → Input).

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