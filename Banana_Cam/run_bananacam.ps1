#!/usr/bin/env pwsh
# Helper script to create venv, install OpenCV, and run BananaCam.py

Set-StrictMode -Version Latest

Push-Location $PSScriptRoot

# Create venv if it doesn't exist
if (-not (Test-Path ".\.venv")) {
    Write-Host "Creating virtual environment .venv..."
    python -m venv .venv
}

# Activate the venv
Write-Host "Activating virtual environment..."
. .\.venv\Scripts\Activate.ps1

Write-Host "Upgrading pip and installing dependencies..."
python -m pip install --upgrade pip
pip install opencv-python

Write-Host "Starting BananaCam.py... (press 'q' in the window to quit)"
python .\BananaCam.py

Pop-Location
