Write-Host "=== CV Alignment Studio :: Launch ===" -ForegroundColor Cyan

if (-not (Test-Path ".\.venv\Scripts\Activate.ps1")) {
    Write-Host "Virtual environment not found. Run scripts/setup.ps1 first." -ForegroundColor Red
    exit 1
}

. ".\.venv\Scripts\Activate.ps1"
streamlit run app.py
