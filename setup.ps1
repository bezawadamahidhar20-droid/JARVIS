# =============================================================================
#  JARVIS - one-command setup for Windows (PowerShell)
#
#  Creates a virtual environment, installs pinned dependencies, pulls the
#  Qwen3 model into Ollama, and downloads the Piper voice files.
#
#  Usage:
#    powershell -ExecutionPolicy Bypass -File setup.ps1
# =============================================================================

$ErrorActionPreference = "Stop"

function Write-Step($msg) {
    Write-Host ""
    Write-Host "==> $msg" -ForegroundColor Cyan
}

function Test-Command($name) {
    return [bool](Get-Command $name -ErrorAction SilentlyContinue)
}

Set-Location $PSScriptRoot

# ----------------------------------------------------------------------------
# 1. Prerequisites
# ----------------------------------------------------------------------------
if (-not (Test-Command python)) {
    Write-Host "[ERROR] Python not found on PATH. Install Python 3.10+ first: https://www.python.org/downloads/" -ForegroundColor Red
    exit 1
}

if (-not (Test-Command ollama)) {
    Write-Host "[ERROR] Ollama not found on PATH. Install it first: https://ollama.com/download" -ForegroundColor Red
    Write-Host "        JARVIS still runs without Ollama, but conversational answers will be disabled."
    Write-Host "        Install Ollama later and re-run this script (or just: ollama pull qwen3:8b)." -ForegroundColor Yellow
}

# ----------------------------------------------------------------------------
# 2. Virtual environment
# ----------------------------------------------------------------------------
Write-Step "Creating virtual environment..."
if (-not (Test-Path ".venv")) {
    python -m venv .venv
} else {
    Write-Host "[i] .venv already exists, skipping creation."
}

$python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
$pip = Join-Path $PSScriptRoot ".venv\Scripts\pip.exe"

if (-not (Test-Path $python)) {
    Write-Host "[ERROR] Virtual environment Python not found at $python" -ForegroundColor Red
    exit 1
}

# ----------------------------------------------------------------------------
# 3. Dependencies
# ----------------------------------------------------------------------------
Write-Step "Installing pinned dependencies from requirements.txt..."
& $pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] pip install failed." -ForegroundColor Red
    exit 1
}

# ----------------------------------------------------------------------------
# 4. Ollama model (optional but recommended)
# ----------------------------------------------------------------------------
if (Test-Command ollama) {
    Write-Step "Pulling Qwen3 model into Ollama (qwen3:8b)..."
    & ollama pull qwen3:8b
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[WARN] ollama pull failed. Conversational answers will be unavailable." -ForegroundColor Yellow
    }
} else {
    Write-Host "[i] Ollama not installed - skipping model pull. Run 'ollama pull qwen3:8b' later." -ForegroundColor Yellow
}

# ----------------------------------------------------------------------------
# 5. Piper voice files
# ----------------------------------------------------------------------------
Write-Step "Downloading Piper voice (en_US-lessac-medium)..."
if (-not (Test-Path "voices")) {
    New-Item -ItemType Directory -Path "voices" | Out-Null
}
& $python -m piper.download_voices --download-dir voices en_US-lessac-medium
if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] Piper voice download failed. TTS will be unavailable." -ForegroundColor Red
    exit 1
}

$voice = Join-Path $PSScriptRoot "voices\en_US-lessac-medium.onnx"
if (-not (Test-Path $voice)) {
    Write-Host "[WARN] Voice file not found at $voice - verify the download." -ForegroundColor Yellow
}

# ----------------------------------------------------------------------------
# 6. Verify the microphone
# ----------------------------------------------------------------------------
Write-Step "Verifying the microphone (speak for 5 seconds)..."
& $python test_microphone_level.py
if ($LASTEXITCODE -ne 0) {
    Write-Host "[WARN] Microphone check reported a problem - review the output above." -ForegroundColor Yellow
}

# ----------------------------------------------------------------------------
# Done
# ----------------------------------------------------------------------------
Write-Host ""
Write-Host "Setup complete!" -ForegroundColor Green
Write-Host ""
Write-Host "Run JARVIS with:" -ForegroundColor White
Write-Host "    .\.venv\Scripts\python.exe main.py" -ForegroundColor Cyan
Write-Host ""