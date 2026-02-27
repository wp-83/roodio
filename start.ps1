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

# Start Flask ML API in a new terminal window
Write-Host "`n[1/2] Starting Flask ML API (port 7860)..." -ForegroundColor Cyan
Start-Process powershell -ArgumentList "-NoExit", "-Command", `
    "cd '$RepoRoot\machineLearning\api'; `
     .\venv\Scripts\activate; `
     Write-Host 'Flask ML API running at http://localhost:7860' -ForegroundColor Cyan; `
     python app.py"

Start-Sleep -Seconds 2

# Start Laravel Webapp in a new terminal window
Write-Host "[2/2] Starting Laravel Webapp (port 8000)..." -ForegroundColor Cyan
Start-Process powershell -ArgumentList "-NoExit", "-Command", `
    "cd '$RepoRoot\webApp'; `
     Write-Host 'Laravel Webapp running at http://localhost:8000' -ForegroundColor Green; `
     php artisan serve --host=127.0.0.1 --port=8000"

Write-Host ''
Write-Host '  Two new terminal windows have been opened.' -ForegroundColor Green
Write-Host ''
Write-Host '  Open in browser:' -ForegroundColor Cyan
Write-Host '    Webapp   http://localhost:8000'
Write-Host '    ML API   http://localhost:7860'
Write-Host ''
Write-Host '  Press Ctrl+C in each terminal window to stop the servers.' -ForegroundColor Yellow
Write-Host ''
