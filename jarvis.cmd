@echo off
rem ─────────────────────────────────────────────────────────────
rem  JARVIS launcher — run `jarvis` from ANY directory.
rem
rem  Resolves the repository root (JARVIS_HOME env var, or the
rem  install.txt written by install.ps1, or its own location) and
rem  runs the project's venv Python against jarvis_cli/__main__.py.
rem ─────────────────────────────────────────────────────────────
setlocal

set "JARVIS_ROOT="
if defined JARVIS_HOME set "JARVIS_ROOT=%JARVIS_HOME%"
if not defined JARVIS_ROOT if exist "%LOCALAPPDATA%\JARVIS\install.txt" set /p JARVIS_ROOT=<"%LOCALAPPDATA%\JARVIS\install.txt"
if not defined JARVIS_ROOT set "JARVIS_ROOT=%~dp0"

if not exist "%JARVIS_ROOT%\.venv\Scripts\python.exe" (
  echo [ERROR] JARVIS environment not found at "%JARVIS_ROOT%\.venv"
  echo         Open PowerShell in the JARVIS repository and run:
  echo             .\install.ps1
  exit /b 1
)

"%JARVIS_ROOT%\.venv\Scripts\python.exe" -X utf8 "%JARVIS_ROOT%\jarvis_cli\__main__.py" %*
exit /b %errorlevel%
