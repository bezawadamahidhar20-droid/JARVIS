#Requires -Version 5.1
<#
.SYNOPSIS
    One-command JARVIS setup for Windows (PowerShell 5.1+).

.DESCRIPTION
    1. Verifies Python 3.10+ is on PATH.
    2. Creates a .venv virtual environment (skips if it exists).
    3. Upgrades pip inside the venv.
    4. Installs all pinned dependencies from requirements.txt.
    5. Reports Ollama and Piper voice status.
    6. Runs the microphone level check.

.EXAMPLE
    # Open PowerShell in the repo root and run:
    .\setup.ps1

    # If script execution is blocked:
    Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
    .\setup.ps1
#>

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$PYTHON_MIN_MAJOR = 3
$PYTHON_MIN_MINOR = 10

# ── Colour helpers ────────────────────────────────────────────────────────────
function Write-Info  { param($Msg) Write-Host "[+] $Msg" -ForegroundColor Green }
function Write-Warn  { param($Msg) Write-Host "[!] $Msg" -ForegroundColor Yellow }
function Write-Err   { param($Msg) Write-Host "[x] $Msg" -ForegroundColor Red; exit 1 }

Write-Host ""
Write-Host "==============================" -ForegroundColor Cyan
Write-Host "   JARVIS Setup"              -ForegroundColor Cyan
Write-Host "==============================" -ForegroundColor Cyan
Write-Host ""

# ── 1. Verify Python ─────────────────────────────────────────────────────────
Write-Info "Checking Python version..."

$pythonCmd = $null
foreach ($cmd in @('python', 'python3', 'py')) {
    try {
        $verStr = & $cmd -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>$null
        if ($verStr -match '^(\d+)\.(\d+)') {
            $major = [int]$Matches[1]
            $minor = [int]$Matches[2]
            if ($major -ge $PYTHON_MIN_MAJOR -and $minor -ge $PYTHON_MIN_MINOR) {
                $pythonCmd = $cmd
                Write-Info "Found Python $verStr at $(Get-Command $cmd | Select-Object -ExpandProperty Source)"
                break
            }
        }
    } catch { }
}

if (-not $pythonCmd) {
    Write-Err "Python ${PYTHON_MIN_MAJOR}.${PYTHON_MIN_MINOR}+ not found on PATH.`n  Download from: https://www.python.org/downloads/`n  Enable 'Add python.exe to PATH' during installation."
}

# ── 2. Create virtual environment ─────────────────────────────────────────────
$venvDir = '.venv'
if (Test-Path $venvDir) {
    Write-Warn "Virtual environment '$venvDir' already exists — skipping creation."
} else {
    Write-Info "Creating virtual environment in '$venvDir'..."
    & $pythonCmd -m venv $venvDir
}

# ── 3. Resolve venv python/pip ───────────────────────────────────────────────
$venvPython = Join-Path $venvDir 'Scripts\python.exe'
$venvPip    = Join-Path $venvDir 'Scripts\pip.exe'

if (-not (Test-Path $venvPython)) {
    Write-Err "venv creation failed — '$venvPython' not found."
}

Write-Info "Virtual environment ready."

# ── 4. Upgrade pip ────────────────────────────────────────────────────────────
Write-Info "Upgrading pip..."
& $venvPip install --quiet --upgrade pip

# ── 5. Install pinned dependencies ────────────────────────────────────────────
if (-not (Test-Path 'requirements.txt')) {
    Write-Err "requirements.txt not found. Is this the JARVIS repo root?"
}

Write-Info "Installing dependencies from requirements.txt..."
& $venvPip install --quiet -r requirements.txt
Write-Info "All dependencies installed."

# ── 6. Ollama check ───────────────────────────────────────────────────────────
Write-Host ""
$ollamaExe = Get-Command ollama -ErrorAction SilentlyContinue
if ($ollamaExe) {
    $ollamaVer = & ollama --version 2>$null | Select-Object -First 1
    Write-Info "Ollama found: $ollamaVer"
    Write-Warn "If you haven't pulled the model yet, run:  ollama pull qwen3:8b"
} else {
    Write-Warn "Ollama not found."
    Write-Warn "Download from https://ollama.com/download, install it, then run:"
    Write-Warn "    ollama pull qwen3:8b"
    Write-Warn "JARVIS will start without Ollama; only conversational answers will be unavailable."
}

# ── 7. Piper voice reminder ───────────────────────────────────────────────────
Write-Host ""
$voicePath = 'voices\en_US-lessac-medium.onnx'
if (Test-Path $voicePath) {
    Write-Info "Piper voice already present at $voicePath"
} else {
    Write-Warn "Piper voice not found. Download it once (inside the activated venv) with:"
    Write-Host "      $venvPython -m piper.download_voices en_US-lessac-medium" -ForegroundColor White
}

# ── 8. Microphone sanity check ────────────────────────────────────────────────
Write-Host ""
Write-Info "Running microphone level check..."
if (Test-Path 'test_microphone_level.py') {
    try {
        & $venvPython test_microphone_level.py
        Write-Info "Microphone OK."
    } catch {
        Write-Warn "Microphone check encountered an error. Run manually:"
        Write-Warn "    $venvPython test_microphone_level.py"
    }
} else {
    Write-Warn "test_microphone_level.py not found — skipping mic check."
}

# ── Done ──────────────────────────────────────────────────────────────────────
Write-Host ""
Write-Host "==============================" -ForegroundColor Cyan
Write-Host "   Setup complete!"            -ForegroundColor Cyan
Write-Host "==============================" -ForegroundColor Cyan
Write-Host ""
Write-Host "  Activate the venv :  .\.venv\Scripts\Activate.ps1"  -ForegroundColor White
Write-Host "  Start JARVIS      :  python main.py"                 -ForegroundColor White
Write-Host ""
