# ==============================================================================
#  Roodio - Start Script (Windows PowerShell)
#  Run this after setup.ps1 is complete:
#    .\start.ps1
# ==============================================================================

$RepoRoot = $PSScriptRoot

Write-Host ''
Write-Host '========================================' -ForegroundColor Magenta
Write-Host '  Roodio - Starting Local Servers'        -ForegroundColor Magenta
Write-Host '========================================' -ForegroundColor Magenta

# ── Flask ML API ──────────────────────────────────────────────────────────────
$ApiDir     = Join-Path $RepoRoot 'machineLearning\api'
$VenvPython = Join-Path $ApiDir   'venv\Scripts\python.exe'

Write-Host "`n[1/2] Starting Flask ML API (port 7860)..." -ForegroundColor Cyan

if (-not (Test-Path $VenvPython)) {
    Write-Host "  [ERROR] venv not found at: $VenvPython" -ForegroundColor Red
    Write-Host "  Run: cd machineLearning\api && python -m venv venv && venv\Scripts\pip install -r requirements.txt" -ForegroundColor Yellow
} else {
    # Use EncodedCommand to avoid all quoting/escaping issues
    $FlaskCmd     = "Set-Location '$ApiDir'; Write-Host 'Flask ML API running at http://localhost:7860' -ForegroundColor Cyan; & '$VenvPython' app.py"
    $FlaskBytes   = [System.Text.Encoding]::Unicode.GetBytes($FlaskCmd)
    $FlaskEncoded = [Convert]::ToBase64String($FlaskBytes)
    Start-Process powershell -ArgumentList "-NoProfile", "-NoExit", "-EncodedCommand", $FlaskEncoded
}

Start-Sleep -Seconds 2

# ── Laravel Webapp ────────────────────────────────────────────────────────────
$WebAppDir = Join-Path $RepoRoot 'webApp'

Write-Host "[2/2] Starting Laravel Webapp (port 8000)..." -ForegroundColor Cyan

$LaravelCmd     = "Set-Location '$WebAppDir'; `$env:PATH = [System.Environment]::GetEnvironmentVariable('PATH','Machine') + ';' + [System.Environment]::GetEnvironmentVariable('PATH','User'); Write-Host 'Laravel Webapp running at http://localhost:8000' -ForegroundColor Green; php artisan serve --host=127.0.0.1 --port=8000"
$LaravelBytes   = [System.Text.Encoding]::Unicode.GetBytes($LaravelCmd)
$LaravelEncoded = [Convert]::ToBase64String($LaravelBytes)
Start-Process powershell -ArgumentList "-NoProfile", "-NoExit", "-EncodedCommand", $LaravelEncoded

Write-Host ''
Write-Host '  Two new terminal windows have been opened.' -ForegroundColor Green
Write-Host ''
Write-Host '  Open in browser:' -ForegroundColor Cyan
Write-Host '    Webapp   http://localhost:8000'
Write-Host '    ML API   http://localhost:7860'
Write-Host ''
Write-Host '  Press Ctrl+C in each terminal window to stop the servers.' -ForegroundColor Yellow
Write-Host ''
