#Requires -Version 5.1
<#
.SYNOPSIS
    JARVIS setup for Windows. Thin wrapper around install.ps1.

.DESCRIPTION
    Creates the .venv, installs dependencies, and puts the `jarvis`
    command on your PATH so you can start JARVIS from any directory.

.EXAMPLE
    .\setup.ps1          # same as:  .\install.ps1
#>

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

Write-Host ""
Write-Host "JARVIS setup delegates to install.ps1..." -ForegroundColor Cyan
Write-Host ""

& (Join-Path $PSScriptRoot 'install.ps1')
exit $LASTEXITCODE
