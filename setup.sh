#!/usr/bin/env bash
# setup.sh — one-command JARVIS setup for Linux / macOS / WSL
# Usage:  bash setup.sh
#
# Installs dependencies into .venv and puts the `jarvis` command on
# your PATH so you can start JARVIS from any directory.
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

PYTHON_MIN_MAJOR=3
PYTHON_MIN_MINOR=10

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

# ── 2. Repository root ────────────────────────────────────────────────────────
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[ -f "$ROOT/main.py" ] || error "main.py not found. Run setup.sh from the JARVIS repository root."
cd "$ROOT"

# ── 3. Create virtual environment ─────────────────────────────────────────────
if [ -x "$ROOT/.venv/bin/python" ]; then
    warn "Virtual environment already exists — skipping creation."
else
    info "Creating virtual environment in .venv..."
    "$PYTHON" -m venv .venv
fi
# shellcheck source=/dev/null
source "$ROOT/.venv/bin/activate"

# ── 4. Upgrade pip + install dependencies ─────────────────────────────────────
info "Upgrading pip..."
pip install --quiet --upgrade pip
info "Installing dependencies from requirements.txt (this can take a while)..."
pip install --quiet -r requirements.txt
info "All dependencies installed."

# ── 5. Install the `jarvis` command into PATH ─────────────────────────────────
info "Installing the jarvis command..."
chmod +x "$ROOT/jarvis"

INSTALL_DIR=""
for dir in "$HOME/.local/bin" "$HOME/bin" "/usr/local/bin"; do
    if [ -d "$dir" ] && echo "$PATH" | grep -q "$dir"; then
        INSTALL_DIR="$dir"
        break
    fi
done

if [ -n "$INSTALL_DIR" ]; then
    cp "$ROOT/jarvis" "$INSTALL_DIR/jarvis"
    info "Installed jarvis -> $INSTALL_DIR/jarvis"
else
    warn "No directory on your PATH found to install into."
    warn "Add this to your shell profile and re-run setup.sh, or use:"
    warn "    ln -s $ROOT/jarvis /usr/local/bin/jarvis"
fi

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

# ── Done ──────────────────────────────────────────────────────────────────────
echo ""
echo "=============================="
echo "   Setup complete!"
echo "=============================="
echo ""
echo "  Open a NEW terminal and type:" 
echo "      jarvis"
echo ""
echo "  Other commands:  jarvis --doctor | --version | --text | --benchmark"
echo ""
