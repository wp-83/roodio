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

# Refresh PATH from registry first
$MachinePath = [System.Environment]::GetEnvironmentVariable('PATH', 'Machine')
$UserPath    = [System.Environment]::GetEnvironmentVariable('PATH', 'User')
$env:PATH    = "$MachinePath;$UserPath"

# Auto-detect PHP executable (for Laragon / XAMPP users without PHP in PATH)
$PhpExe = 'php'
if (-not (Get-Command 'php' -ErrorAction SilentlyContinue)) {
    $CommonPhpPaths = @(
        'C:\laragon\bin\php\*\php.exe',
        'C:\xampp\php\php.exe',
        'C:\wamp64\bin\php\*\php.exe',
        'D:\laragon\bin\php\*\php.exe',
        'D:\xampp\php\php.exe'
    )
    foreach ($Pattern in $CommonPhpPaths) {
        $Found = Get-Item $Pattern -ErrorAction SilentlyContinue | Select-Object -First 1
        if ($Found) {
            $PhpExe = "$($Found.FullName)"
            Write-Host "  [INFO] PHP not in PATH. Using: $PhpExe" -ForegroundColor Yellow
            break
        }
    }
    if ($PhpExe -eq 'php') {
        Write-Host '' -ForegroundColor Red
        Write-Host '  [ERROR] PHP not found! Laravel cannot start.' -ForegroundColor Red
        Write-Host '  Fix options:' -ForegroundColor Yellow
        Write-Host '    1. Open Laragon, click Menu > PHP, and make sure PHP is enabled.' -ForegroundColor Yellow
        Write-Host '    2. Add your PHP folder to the system PATH:' -ForegroundColor Yellow
        Write-Host '       e.g. C:\laragon\bin\php\php8.2.x' -ForegroundColor Yellow
        Write-Host '    3. Or run manually: cd webApp && php artisan serve' -ForegroundColor Yellow
        Write-Host ''
        Read-Host 'Press Enter to exit'
        exit 1
    }
}

$LaravelCmd     = "Set-Location '$WebAppDir'; Write-Host 'Laravel Webapp running at http://localhost:8000' -ForegroundColor Green; & '$PhpExe' artisan serve --host=127.0.0.1 --port=8000"
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
