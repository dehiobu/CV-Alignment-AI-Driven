param(
    [string]$PythonVersion = "3.11",
    [switch]$Dev
)

Write-Host "=== CV Alignment Studio :: Environment Setup ===" -ForegroundColor Cyan

if (-not (Test-Path ".venv")) {
    Write-Host "Creating virtual environment (.venv) using Python $PythonVersion..." -ForegroundColor Yellow
    py -$PythonVersion -m venv .venv
}

Write-Host "Activating virtual environment..." -ForegroundColor Yellow
. ".\.venv\Scripts\Activate.ps1"

Write-Host "Upgrading pip/setuptools/wheel..." -ForegroundColor Yellow
python -m pip install --upgrade pip setuptools wheel

if ($Dev) {
    Write-Host "Installing development dependencies..." -ForegroundColor Yellow
    pip install -r requirements-dev.txt
} else {
    Write-Host "Installing runtime dependencies..." -ForegroundColor Yellow
    pip install -r requirements.txt
}

Write-Host "Environment setup complete." -ForegroundColor Green
