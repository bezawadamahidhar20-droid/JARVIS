#!/usr/bin/env bash
# setup.sh — one-command JARVIS setup for Linux / macOS / WSL
# Usage:  bash setup.sh
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

PYTHON_MIN_MAJOR=3
PYTHON_MIN_MINOR=10

# ── Colour helpers ────────────────────────────────────────────────────────────
RED='\033[0;31m'; GRN='\033[0;32m'; YLW='\033[1;33m'; RST='\033[0m'
info()  { echo -e "${GRN}[+]${RST} $*"; }
warn()  { echo -e "${YLW}[!]${RST} $*"; }
error() { echo -e "${RED}[x]${RST} $*" >&2; exit 1; }

echo ""
echo "=============================="
echo "   JARVIS Setup"
echo "=============================="
echo ""

# ── 1. Locate a suitable Python ───────────────────────────────────────────────
info "Checking Python version..."
PYTHON=""
for cmd in python3 python; do
    if command -v "$cmd" &>/dev/null; then
        VER=$("$cmd" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null || true)
        MAJOR=$(echo "$VER" | cut -d. -f1)
        MINOR=$(echo "$VER" | cut -d. -f2)
        if [ "${MAJOR:-0}" -ge "$PYTHON_MIN_MAJOR" ] && [ "${MINOR:-0}" -ge "$PYTHON_MIN_MINOR" ]; then
            PYTHON="$cmd"
            info "Found Python $VER at $(command -v "$cmd")"
            break
        fi
    fi
done

[ -z "$PYTHON" ] && error "Python ${PYTHON_MIN_MAJOR}.${PYTHON_MIN_MINOR}+ not found. Install it first: https://www.python.org/downloads/"

# ── 2. Create virtual environment ─────────────────────────────────────────────
VENV_DIR=".venv"
if [ -d "$VENV_DIR" ]; then
    warn "Virtual environment '$VENV_DIR' already exists — skipping creation."
else
    info "Creating virtual environment in '$VENV_DIR'..."
    "$PYTHON" -m venv "$VENV_DIR"
fi

# ── 3. Activate venv ──────────────────────────────────────────────────────────
# shellcheck source=/dev/null
source "$VENV_DIR/bin/activate"
info "Virtual environment activated."

# ── 4. Upgrade pip ────────────────────────────────────────────────────────────
info "Upgrading pip..."
pip install --quiet --upgrade pip

# ── 5. Install pinned dependencies ────────────────────────────────────────────
info "Installing dependencies from requirements.txt..."
pip install --quiet -r requirements.txt
info "All dependencies installed."

# ── 6. Check Ollama ───────────────────────────────────────────────────────────
echo ""
if command -v ollama &>/dev/null; then
    info "Ollama found: $(ollama --version 2>/dev/null | head -1)"
    warn "If you haven't pulled the model yet, run:  ollama pull qwen3:8b"
else
    warn "Ollama not found."
    warn "Install it from https://ollama.com/download, then run:  ollama pull qwen3:8b"
    warn "JARVIS will start without Ollama; only conversational answers will be unavailable."
fi

# ── 7. Piper voice reminder ───────────────────────────────────────────────────
echo ""
if [ -f "voices/en_US-lessac-medium.onnx" ]; then
    info "Piper voice already present at voices/en_US-lessac-medium.onnx"
else
    warn "Piper voice not found. Download it once with:"
    echo "      python -m piper.download_voices en_US-lessac-medium"
fi

# ── 8. Quick microphone sanity check ─────────────────────────────────────────
echo ""
info "Running microphone level check..."
if python test_microphone_level.py 2>/dev/null; then
    info "Microphone OK."
else
    warn "Microphone check failed or test script missing. Run manually:"
    warn "    python test_microphone_level.py"
fi

echo ""
echo "=============================="
echo "   Setup complete!"
echo "=============================="
echo ""
echo "  Activate the venv :  source .venv/bin/activate"
echo "  Start JARVIS      :  python main.py"
echo ""
