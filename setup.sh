#!/usr/bin/env bash
# =============================================================================
#  JARVIS - one-command setup for POSIX (Linux/macOS)
#
#  NOTE: JARVIS's command router launches Windows apps (os.startfile,
#  chrome.exe, cmd.exe), so the assistant itself is Windows-focused. This
#  script only provisions the Python environment, the Ollama model, and the
#  Piper voice so the core modules can be developed/tested elsewhere.
#
#  Usage:
#    bash setup.sh
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")"

step() { printf "\n==> %s\n" "$1"; }

# --- 1. Prerequisites -------------------------------------------------------
if ! command -v python3 >/dev/null 2>&1; then
    echo "[ERROR] python3 not found. Install Python 3.10+ first." >&2
    exit 1
fi

if ! command -v ollama >/dev/null 2>&1; then
    echo "[WARN] ollama not found on PATH. Conversational answers will be disabled."
fi

# --- 2. Virtual environment --------------------------------------------------
step "Creating virtual environment..."
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
else
    echo "[i] .venv already exists, skipping creation."
fi

PYTHON=".venv/bin/python"
PIP=".venv/bin/pip"
[ -x "$PYTHON" ] || PYTHON=".venv/Scripts/python.exe"
[ -x "$PIP" ] || PIP=".venv/Scripts/pip.exe"

if [ ! -x "$PYTHON" ]; then
    echo "[ERROR] Virtual environment Python not found." >&2
    exit 1
fi

# --- 3. Dependencies ----------------------------------------------------------
step "Installing pinned dependencies from requirements.txt..."
"$PIP" install -r requirements.txt

# --- 4. Ollama model (optional) ------------------------------------------------
if command -v ollama >/dev/null 2>&1; then
    step "Pulling Qwen3 model into Ollama (qwen3:8b)..."
    ollama pull qwen3:8b || echo "[WARN] ollama pull failed." >&2
else
    echo "[i] Ollama not installed - skipping model pull."
fi

# --- 5. Piper voice files ------------------------------------------------------
step "Downloading Piper voice (en_US-lessac-medium)..."
mkdir -p voices
"$PYTHON" -m piper.download_voices --download-dir voices en_US-lessac-medium

if [ ! -f "voices/en_US-lessac-medium.onnx" ]; then
    echo "[WARN] Voice file not found at voices/en_US-lessac-medium.onnx - verify the download." >&2
fi

# --- Done ---------------------------------------------------------------------
echo ""
echo "Setup complete!"
echo ""
echo "Run JARVIS (Windows) with:"
echo "    .venv\\Scripts\\python.exe main.py"
echo ""