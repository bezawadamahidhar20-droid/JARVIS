#Requires -Version 5.1
<#
.SYNOPSIS
    One-command JARVIS installation for Windows.
 
.DESCRIPTION
    [FIX m9] Fixed Python version check to accept Python 4.x
 
.NOTES
    After installation, open a NEW terminal window, then type: jarvis
#>
 
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
 
$PYTHON_MIN_MAJOR = 3
$PYTHON_MIN_MINOR = 10
 
function Write-Info  { param($Msg) Write-Host "[+] $Msg" -ForegroundColor Green }
function Write-Warn  { param($Msg) Write-Host "[!] $Msg" -ForegroundColor Yellow }
function Write-Err   { param($Msg) Write-Host "[x] $Msg" -ForegroundColor Red; exit 1 }
 
Write-Host ""
Write-Host "==============================" -ForegroundColor Cyan
Write-Host "   JARVIS Installer"           -ForegroundColor Cyan
Write-Host "==============================" -ForegroundColor Cyan
Write-Host ""
 
$RepoRoot = (Get-Location).Path
if (-not (Test-Path (Join-Path $RepoRoot 'main.py'))) {
    Write-Err "main.py not found here. Run this script from the JARVIS repository root."
}
 
# 1. Verify Python
Write-Info "Checking Python version..."
$pythonCmd = $null
foreach ($cmd in @('python', 'py', 'python3')) {
    try {
        $verStr = & $cmd -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>$null
        if ($verStr -match '^(\d+)\.(\d+)') {
            $major = [int]$Matches[1]
            $minor = [int]$Matches[2]
            # [FIX m9] Fixed version check: accept Python 4.x or Python 3.10+
            if ($major -gt $PYTHON_MIN_MAJOR -or ($major -eq $PYTHON_MIN_MAJOR -and $minor -ge $PYTHON_MIN_MINOR)) {
                $pythonCmd = $cmd
                Write-Info "Found Python $verStr"
                break
            }
        }
    } catch { }
}
if (-not $pythonCmd) {
    Write-Err "Python ${PYTHON_MIN_MAJOR}.${PYTHON_MIN_MINOR}+ not found on PATH.`n  Download from: https://www.python.org/downloads/ and enable 'Add python.exe to PATH'."
}
 
# 2. Create virtual environment
$venvDir = Join-Path $RepoRoot '.venv'
$venvPython = Join-Path $venvDir 'Scripts\python.exe'
$venvPip    = Join-Path $venvDir 'Scripts\pip.exe'
 
if (Test-Path $venvPython) {
    Write-Info "Virtual environment found at $venvDir"
} else {
    Write-Info "Creating virtual environment in .venv..."
    & $pythonCmd -m venv $venvDir
    if (-not (Test-Path $venvPython)) {
        Write-Err "venv creation failed - '$venvPython' not found."
    }
}
 
# 3. Upgrade pip + install dependencies
$req = Join-Path $RepoRoot 'requirements.txt'
if (-not (Test-Path $req)) {
    Write-Err "requirements.txt not found. Is this the JARVIS repo root?"
}
Write-Info "Upgrading pip..."
& $venvPip install --quiet --upgrade pip
Write-Info "Installing dependencies from requirements.txt (this can take a while)..."
& $venvPip install --quiet -r $req
Write-Info "All dependencies installed."
 
# 4. Install the jarvis command into user PATH
$binDir = Join-Path $env:LOCALAPPDATA 'JARVIS\bin'
$installFile = Join-Path $env:LOCALAPPDATA 'JARVIS\install.txt'
$cmdShim = Join-Path $RepoRoot 'jarvis.cmd'
 
Write-Info "Installing the jarvis command..."
New-Item -ItemType Directory -Force -Path $binDir | Out-Null
Copy-Item -Force $cmdShim (Join-Path $binDir 'jarvis.cmd')
New-Item -ItemType Directory -Force -Path (Split-Path $installFile) | Out-Null
Set-Content -Path $installFile -Value $RepoRoot -Encoding ascii
 
$installed = $false
 
try {
    $userPath = [Environment]::GetEnvironmentVariable('Path', 'User')
    if ($userPath -notlike "*$binDir*") {
        $newPath = if ([string]::IsNullOrEmpty($userPath)) { $binDir } else { "$userPath;$binDir" }
        [Environment]::SetEnvironmentVariable('Path', $newPath, 'User')
        Write-Info "Added $binDir to your user PATH."
        $installed = $true
    } else {
        Write-Info "User PATH already contains $binDir - nothing to change."
        $installed = $true
    }
} catch {
    Write-Warn "Could not modify the user PATH ($($_.Exception.Message))"
}
 
if (-not $installed) {
    $candidates = @(
        (Join-Path $env:USERPROFILE '.local\bin'),
        (Join-Path $env:APPDATA 'npm'),
        (Join-Path $env:LOCALAPPDATA 'Microsoft\WindowsApps')
    )
    foreach ($dir in $candidates) {
        if ([string]::IsNullOrEmpty($dir)) { continue }
        if (Test-Path $dir) {
            try {
                $target = Join-Path $dir 'jarvis.cmd'
                Copy-Item -Force $cmdShim $target
                Write-Info "Installed jarvis.cmd into $target (already on your PATH)."
                $installed = $true
                break
            } catch { }
        }
    }
}
 
if (-not $installed) {
    Write-Warn "Could not place the jarvis command automatically."
    Write-Warn "Copy jarvis.cmd into any directory on your PATH."
}
 
# 5. Download Piper voice if needed
$voicesDir = Join-Path $RepoRoot 'voices'
$voiceFile = Join-Path $voicesDir 'en_US-lessac-medium.onnx'
if (-not (Test-Path $voiceFile)) {
    Write-Info "Downloading Piper voice (en_US-lessac-medium)..."
    try {
        & $venvPython -m piper.download_voices en_US-lessac-medium
        Write-Info "Voice downloaded."
    } catch {
        Write-Warn "Could not download voice automatically. Run manually:"
        Write-Warn "  .\.venv\Scripts\python.exe -m piper.download_voices en_US-lessac-medium"
    }
}
 
# 6. Doctor check
Write-Host ""
Write-Info "Running jarvis --doctor to verify the installation..."
& $venvPython -X utf8 (Join-Path $RepoRoot 'jarvis_cli\__main__.py') --doctor
 
Write-Host ""
Write-Host "==============================" -ForegroundColor Cyan
Write-Host "   Installation complete!"     -ForegroundColor Cyan
Write-Host "==============================" -ForegroundColor Cyan
Write-Host ""
Write-Host "  Open a NEW terminal window (PowerShell or CMD) and type:" -ForegroundColor White
Write-Host "      jarvis" -ForegroundColor Cyan
Write-Host ""
 
