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

# ── 1. Flask ML API ───────────────────────────────────────────────────────────
$ApiDir     = Join-Path $RepoRoot 'machineLearning\api'
$VenvPython = Join-Path $ApiDir   'venv\Scripts\python.exe'

Write-Host "`n[1/3] Starting Flask ML API (port 7860)..." -ForegroundColor Cyan

if (-not (Test-Path $VenvPython)) {
    Write-Host "  [ERROR] Python virtual environment (venv) not found!" -ForegroundColor Red
    Write-Host "  Please run .\setup.ps1 first to install dependencies." -ForegroundColor Yellow
    exit 1
} else {
    $FlaskCmd     = "Set-Location '$ApiDir'; Write-Host 'Flask ML API running at http://localhost:7860' -ForegroundColor Cyan; & '$VenvPython' app.py"
    $FlaskBytes   = [System.Text.Encoding]::Unicode.GetBytes($FlaskCmd)
    $FlaskEncoded = [Convert]::ToBase64String($FlaskBytes)
    Start-Process powershell -ArgumentList "-NoProfile", "-NoExit", "-EncodedCommand", $FlaskEncoded
}

Start-Sleep -Seconds 1

# ── 2. Laravel Webapp ─────────────────────────────────────────────────────────
$WebAppDir = Join-Path $RepoRoot 'webApp'
$VendorDir = Join-Path $WebAppDir 'vendor'

Write-Host "[2/3] Starting Laravel Webapp (port 8000)..." -ForegroundColor Cyan

if (-not (Test-Path $VendorDir)) {
    Write-Host "  [ERROR] Laravel vendor folder not found!" -ForegroundColor Red
    Write-Host "  Please run .\setup.ps1 first to install composer dependencies." -ForegroundColor Yellow
    exit 1
}

# Refresh PATH from registry first
$MachinePath = [System.Environment]::GetEnvironmentVariable('PATH', 'Machine')
$UserPath    = [System.Environment]::GetEnvironmentVariable('PATH', 'User')
$env:PATH    = "$MachinePath;$UserPath"

# Auto-detect PHP
$PhpExe = 'php'
if (-not (Get-Command 'php' -ErrorAction SilentlyContinue)) {
    $CommonPhpPaths = @(
        'C:\laragon\bin\php\*\php.exe',
        'C:\xampp\php\php.exe',
        'D:\laragon\bin\php\*\php.exe',
        'D:\xampp\php\php.exe'
    )
    foreach ($Pattern in $CommonPhpPaths) {
        $Found = Get-Item $Pattern -ErrorAction SilentlyContinue | Select-Object -First 1
        if ($Found) {
            $PhpExe = "$($Found.FullName)"
            break
        }
    }
    if ($PhpExe -eq 'php') {
        Write-Host '' -ForegroundColor Red
        Write-Host '  [ERROR] PHP not found! Laravel cannot start.' -ForegroundColor Red
        Write-Host '  Please ensure PHP is running via Laragon/XAMPP or added to your system PATH.' -ForegroundColor Yellow
        exit 1
    }
}

$LaravelCmd     = "Set-Location '$WebAppDir'; Write-Host 'Laravel Webapp running at http://localhost:8000' -ForegroundColor Green; & '$PhpExe' artisan serve --host=127.0.0.1 --port=8000"
$LaravelBytes   = [System.Text.Encoding]::Unicode.GetBytes($LaravelCmd)
$LaravelEncoded = [Convert]::ToBase64String($LaravelBytes)
Start-Process powershell -ArgumentList "-NoProfile", "-NoExit", "-EncodedCommand", $LaravelEncoded

Start-Sleep -Seconds 1

# ── 3. Frontend Vite Server (Dev Mode) ────────────────────────────────────────
Write-Host "[3/3] Starting Vite Frontend Server..." -ForegroundColor Cyan

# Menjalankan npm run dev di background terminal agar perubahan file UI langsung terlihat
$ViteCmd     = "Set-Location '$WebAppDir'; Write-Host 'Vite HMR Server running...' -ForegroundColor Magenta; npm run dev"
$ViteBytes   = [System.Text.Encoding]::Unicode.GetBytes($ViteCmd)
$ViteEncoded = [Convert]::ToBase64String($ViteBytes)
Start-Process powershell -ArgumentList "-NoProfile", "-NoExit", "-EncodedCommand", $ViteEncoded

Write-Host ''
Write-Host '  Three new terminal windows have been opened.' -ForegroundColor Green
Write-Host ''
Write-Host '  Open in browser:' -ForegroundColor Cyan
Write-Host '    Webapp   http://localhost:8000'
Write-Host '    ML API   http://localhost:7860'
Write-Host ''
Write-Host '  Press Ctrl+C in each terminal window to stop the servers.' -ForegroundColor Yellow
Write-Host ''