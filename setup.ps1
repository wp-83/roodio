# ==============================================================================
#  Roodio - Setup Script (Windows PowerShell)
#  Run this in the repo root after git clone:
#    .\setup.ps1
# ==============================================================================

$ErrorActionPreference = 'Stop'
$RepoRoot = $PSScriptRoot

function Write-Step { param($msg); Write-Host "`n[SETUP] $msg" -ForegroundColor Cyan }
function Write-OK   { param($msg); Write-Host "[OK] $msg" -ForegroundColor Green }
function Write-Warn { param($msg); Write-Host "[WARN] $msg" -ForegroundColor Yellow }
function Write-Fail { param($msg); Write-Host "[ERROR] $msg" -ForegroundColor Red; exit 1 }

Write-Host ''
Write-Host '========================================' -ForegroundColor Magenta
Write-Host '  Roodio - Local Setup (Windows)'         -ForegroundColor Magenta
Write-Host '========================================' -ForegroundColor Magenta

# ==============================================================================
# 1. CHECK PREREQUISITES
# ==============================================================================
Write-Step 'Checking prerequisites...'

# Refresh PATH from registry
$MachinePath = [System.Environment]::GetEnvironmentVariable('PATH', 'Machine')
$UserPath    = [System.Environment]::GetEnvironmentVariable('PATH', 'User')
$env:PATH    = "$MachinePath;$UserPath"

# ── Auto-detect PHP (Prioritize Laragon) ──────────────────────────────────────
$PhpExe = 'php'

# Custom Logic: Force looking directly into Laragon/XAMPP active bins first
$ActivePhpPaths = @(
    'C:\laragon\bin\php\php-8.5*\php.exe',
    'C:\laragon\bin\php\php-8.4*\php.exe',
    'C:\laragon\bin\php\*\php.exe',
    'D:\laragon\bin\php\*\php.exe',
    'C:\xampp\php\php.exe',
    'C:\wamp64\bin\php\*\php.exe'
)

foreach ($p in $ActivePhpPaths) {
    $f = Get-Item $p -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if ($f) { 
        $PhpExe = "$($f.FullName)"
        
        # Inject this found path into session PATH to override system defaults
        $env:PATH = "$($f.DirectoryName);" + $env:PATH 
        break 
    }
}

if (-not (Get-Command 'php' -ErrorAction SilentlyContinue) -and ($PhpExe -eq 'php')) {
    Write-Host '  [ERROR] PHP not found.' -ForegroundColor Red
    Write-Host '  Install Laragon (https://laragon.org/download) or add PHP to PATH.' -ForegroundColor Yellow
    Write-Fail 'PHP is required. Please install it and re-run this script.'
}
Write-OK "PHP found: $PhpExe"


# ── Cek Ekstensi PHP Wajib ───────────────────────────────────────────────────
Write-Step 'Checking required PHP extensions...'
$RequiredExts = @('zip', 'mbstring', 'pdo_mysql', 'fileinfo', 'curl')
$MissingExts = @()

foreach ($ext in $RequiredExts) {
    $ExtCheck = & $PhpExe -r "echo extension_loaded('$ext') ? 'OK' : 'MISSING';"
    if ($ExtCheck -ne 'OK') {
        $MissingExts += $ext
    }
}

if ($MissingExts.Count -gt 0) {
    # Minta PHP mencari tahu lokasi php.ini yang sedang dipakai
    $IniPath = & $PhpExe -r "echo php_ini_loaded_file();"
    if ([string]::IsNullOrWhiteSpace($IniPath)) {
        $IniPath = "(Tidak ada php.ini yang dimuat. Buat dari php.ini-development di folder PHP Anda)"
    }

    Write-Host '  [ERROR] PHP is missing required extensions:' -ForegroundColor Red
    $MissingExts | ForEach-Object { Write-Host "    - $_" -ForegroundColor Red }
    
    Write-Host "`n  Silakan aktifkan ekstensi tersebut dengan mengedit file ini:" -ForegroundColor Yellow
    Write-Host "  -> $IniPath" -ForegroundColor Cyan
    Write-Host "  (Buka file tersebut, cari baris ';extension=...' dan hapus tanda ';' di depannya)`n" -ForegroundColor Yellow
    
    Write-Fail "Setup dihentikan karena konfigurasi PHP (php.ini) tidak lengkap."
}
Write-OK 'All required PHP extensions are active.'


# ── Auto-detect Composer ──────────────────────────────────────────────────────
$ComposerCmd = $null
$resolved = $null
if (Get-Command 'composer' -ErrorAction SilentlyContinue) {
    $ComposerCmd = { param($a) & composer @a }
} else {
    $ComposerPharCandidates = @(
        'C:\laragon\bin\composer\composer.phar',
        'D:\laragon\bin\composer\composer.phar',
        'C:\ProgramData\ComposerSetup\bin\composer.phar'
    )
    foreach ($p in $ComposerPharCandidates) {
        if (Test-Path $p) {
            $resolved = $p
            Write-Warn "composer not in PATH. Using fallback composer: $resolved"
            break
        }
    }
    
    if (-not $resolved) {
        Write-Step 'Composer not found. Attempting to download composer.phar...'
        try {
            Invoke-WebRequest -Uri "https://getcomposer.org/composer.phar" -OutFile (Join-Path $PSScriptRoot "composer.phar")
            $resolved = Join-Path $PSScriptRoot "composer.phar"
            Write-OK 'composer.phar downloaded successfully.'
        } catch {
            Write-Fail 'Failed to download composer.phar. Please install it manually from https://getcomposer.org.'
        }
    }
    $ComposerCmd = { param($a) & $PhpExe $resolved @a }
}
Write-OK 'Composer is ready.'


# ── Auto-detect Node & NPM ────────────────────────────────────────────────────
$NodeExe = 'node'
if (-not (Get-Command 'node' -ErrorAction SilentlyContinue)) {
    $NodeCandidates = @(
        'C:\laragon\bin\nodejs\*\node.exe',
        'D:\laragon\bin\nodejs\*\node.exe',
        'C:\Program Files\nodejs\node.exe',
        'D:\Program Files\nodejs\node.exe'
    )
    foreach ($p in $NodeCandidates) {
        $f = Get-Item $p -ErrorAction SilentlyContinue | Select-Object -First 1
        if ($f) { 
            $NodeExe = "$($f.FullName)"
            $env:PATH += ";$($f.DirectoryName)"
            break 
        }
    }
}

if (Get-Command $NodeExe -ErrorAction SilentlyContinue) {
    $NodeVer = & $NodeExe -v
    Write-Host "  [INFO] Detected Node version: $NodeVer. Roodio recommends Node v20.19+ or v22+" -ForegroundColor Cyan
}

# ── Check remaining tools ─────────────────────────────────────────────────────
$missing = @()
@('python','git') | ForEach-Object {
    if (-not (Get-Command $_ -ErrorAction SilentlyContinue)) { $missing += $_ }
}
if ($missing.Count -gt 0) {
    Write-Host '  Missing tools:' -ForegroundColor Red
    $missing | ForEach-Object { Write-Host "    - $_" -ForegroundColor Red }
    Write-Host '  Python  : https://python.org/downloads' -ForegroundColor Yellow
    Write-Host '  Git     : https://git-scm.com/download/win' -ForegroundColor Yellow
    Write-Fail 'Please install the missing tools above, then re-run this script.'
}
Write-OK 'All prerequisites found.'


# ==============================================================================
# 2. SETUP FLASK ML API
# ==============================================================================
Write-Step 'Setting up Flask ML API...'
$ApiDir = Join-Path $RepoRoot 'machineLearning\api'

Push-Location $ApiDir
    Write-Host '  Creating Python virtual environment...'
    python -m venv venv
    Write-Host '  Activating virtual environment & installing dependencies...'
    cmd /c ".\venv\Scripts\python.exe -m pip install --upgrade pip setuptools wheel"
    cmd /c ".\venv\Scripts\python.exe -m pip install -r requirements.txt"
    Write-OK 'Flask ML API dependencies installed in virtual environment.'
Pop-Location


# ==============================================================================
# 3. SETUP LARAVEL WEBAPP
# ==============================================================================
Write-Step 'Setting up Laravel Webapp...'
$WebDir = Join-Path $RepoRoot 'webApp'
Push-Location $WebDir

    # 3a. Composer install
    Write-Host '  Running composer install...'
    $PlatformCheck = Join-Path $WebDir 'vendor\composer\platform_check.php'
    if (Test-Path $PlatformCheck) {
        $PlatformContent = Get-Content $PlatformCheck -Raw
        if ($PlatformContent -match "'>= (\d+\.\d+\.\d+)'") {
            $RequiredVer = $Matches[1]
            $LocalVer    = (& $PhpExe -r 'echo PHP_VERSION;' 2>$null)
            if ($LocalVer -and ([version]$LocalVer -lt [version]$RequiredVer)) {
                Write-Warn "vendor/ was built for PHP $RequiredVer but you have PHP $LocalVer. Removing and reinstalling..."
                Remove-Item -Recurse -Force (Join-Path $WebDir 'vendor')
            }
        }
    }

    if ($ComposerCmd) {
        & $ComposerCmd @('install','--no-interaction')
    } else {
        & $PhpExe $resolved install --no-interaction
    }
    Write-OK 'Composer dependencies installed.'

    # 3b. Setup .env
    if (-not (Test-Path '.env')) {
        Copy-Item '.env.example' '.env'
        Write-OK '.env created from .env.example'
    } else {
        Write-Warn '.env already exists, skipping copy.'
    }

    # 3c. Point ROODIO_API_URL to local Flask server
    $envContent = Get-Content '.env' -Raw
    if ($envContent -notmatch 'ROODIO_API_URL=http://127.0.0.1:7860') {
        $envContent = $envContent -replace '(?m)^ROODIO_API_URL=.*$', 'ROODIO_API_URL=http://127.0.0.1:7860'
        if ($envContent -notmatch 'ROODIO_API_URL') {
            $envContent += "`nROODIO_API_URL=http://127.0.0.1:7860"
        }
        Set-Content '.env' $envContent
    }
    Write-OK 'ROODIO_API_URL set to http://127.0.0.1:7860'

    # 3d. Generate App Key
    & $PhpExe artisan key:generate --quiet
    Write-OK 'APP_KEY generated.'

    # 3e. Configure Database
    Write-Host ''
    Write-Host '  Database Configuration (MySQL):' -ForegroundColor Yellow
    Write-Host '  (Press Enter to use the default value)'
    $dbHost = Read-Host '    DB_HOST     [127.0.0.1]'
    $dbPort = Read-Host '    DB_PORT     [3306]'
    $dbName = Read-Host '    DB_DATABASE [roodio]'
    $dbUser = Read-Host '    DB_USERNAME [root]'
    $dbPass = Read-Host '    DB_PASSWORD []'

    if (-not $dbHost) { $dbHost = '127.0.0.1' }
    if (-not $dbPort) { $dbPort = '3306' }
    if (-not $dbName) { $dbName = 'roodio' }
    if (-not $dbUser) { $dbUser = 'root' }

    $envContent = Get-Content '.env' -Raw
    $envContent = $envContent -replace '(?m)^DB_HOST=.*$',     "DB_HOST=$dbHost"
    $envContent = $envContent -replace '(?m)^DB_PORT=.*$',     "DB_PORT=$dbPort"
    $envContent = $envContent -replace '(?m)^DB_DATABASE=.*$', "DB_DATABASE=$dbName"
    $envContent = $envContent -replace '(?m)^DB_USERNAME=.*$', "DB_USERNAME=$dbUser"
    $envContent = $envContent -replace '(?m)^DB_PASSWORD=.*$', "DB_PASSWORD=$dbPass"
    Set-Content '.env' $envContent
    Write-OK 'Database config saved to .env'

    # 3f. Auto-create database if it doesn't exist
    Write-Host '  Ensuring database exists...'
    $phpCode = @"
<?php
try {
    `$pdo = new PDO('mysql:host=$dbHost;port=$dbPort', '$dbUser', '$dbPass');
    `$pdo->exec("CREATE DATABASE IF NOT EXISTS $dbName");
} catch (Exception `$e) {}
"@
    Set-Content -Path "create_db.php" -Value $phpCode -Encoding ASCII
    & $PhpExe create_db.php 2>$null
    Remove-Item "create_db.php" -ErrorAction SilentlyContinue

    # 3g. Run Migrations & Seeders
    $migrationSuccess = $false
    while (-not $migrationSuccess) {
        Write-Host '  Running database migrations and seeders...'
        try {
            $migrationOutput = & $PhpExe artisan migrate:fresh --seed --force 2>&1
            if ($LASTEXITCODE -ne 0) {
                throw $migrationOutput
            }
            Write-OK 'Database migrated and seeded.'
            $migrationSuccess = $true
        } catch {
            Write-Warn 'Migration failed! This usually means MySQL is not running or the database credentials are wrong.'
            Write-Host "    Error detail: $_" -ForegroundColor Red
            Write-Host "  -> Please start your MySQL server (e.g. Laragon/XAMPP) if it is not running." -ForegroundColor Yellow
            
            $retry = Read-Host "  Do you want to retry the migration? (Y to retry / N to skip)"
            if ($retry -notmatch '^[Yy]$') {
                Write-Warn 'Skipping database migration. You MUST run "php artisan migrate:fresh --seed" manually later after fixing your database!'
                break
            }
        }
    }

    # 3h. Create storage symlink
    & $PhpExe artisan storage:link --quiet 2>$null
    Write-OK 'Storage symlink created (public/storage -> storage/app/public).'

    # 3i. Install Node dependencies & build assets
    Write-Host '  Installing Node.js dependencies...'
    npm install
    Write-Host '  Building frontend assets (Vite)...'
    npm run build
    Write-OK 'Frontend assets built.'

Pop-Location

# ==============================================================================
# DONE
# ==============================================================================
Write-Host ''
Write-Host '========================================' -ForegroundColor Green
Write-Host '  Setup complete!'                        -ForegroundColor Green
Write-Host '========================================' -ForegroundColor Green
Write-Host ''
Write-Host '  To start all servers, run:' -ForegroundColor Cyan
Write-Host '    .\start.ps1'
Write-Host ''