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

$missing = @()
@('php','composer','node','npm','python','git') | ForEach-Object {
    if (-not (Get-Command $_ -ErrorAction SilentlyContinue)) {
        $missing += $_
    }
}

if ($missing.Count -gt 0) {
    Write-Host ''
    Write-Host '  Missing prerequisites:' -ForegroundColor Red
    $missing | ForEach-Object { Write-Host "    - $_" -ForegroundColor Red }
    Write-Host ''
    Write-Host '  Installation guides (Windows):' -ForegroundColor Yellow
    Write-Host '    PHP + MySQL + Composer : Install Laragon  https://laragon.org/download'
    Write-Host '    Node.js                : https://nodejs.org'
    Write-Host '    Python                 : https://python.org/downloads'
    Write-Host '    Git                    : https://git-scm.com/download/win'
    Write-Host ''
    Write-Fail 'Please install the missing prerequisites above, then run this script again.'
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
    .\venv\Scripts\activate
    python -m pip install --upgrade pip
    python -m pip install -r requirements.txt
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
    composer install --no-interaction
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
    php artisan key:generate --quiet
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

    # 3f. Run Migrations & Seeders
    $migrationSuccess = $false
    while (-not $migrationSuccess) {
        Write-Host '  Running database migrations and seeders...'
        try {
            # Capture stdErr to detect real failures, stop on error
            $migrationOutput = php artisan migrate --seed --force 2>&1
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
                Write-Warn 'Skipping database migration. You MUST run "php artisan migrate --seed" manually later after fixing your database!'
                break
            }
        }
    }

    # 3g. Create storage symlink for local file uploads
    php artisan storage:link --quiet 2>$null
    Write-OK 'Storage symlink created (public/storage -> storage/app/public).'

    # 3h. Install Node dependencies & build assets
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
Write-Host '  Or manually in 2 separate terminals:' -ForegroundColor Cyan
Write-Host '    Terminal 1 (ML API) : cd machineLearning\api ; python app.py'
Write-Host '    Terminal 2 (Webapp) : cd webApp ; php artisan serve'
Write-Host ''
Write-Host '  Open in browser:' -ForegroundColor Cyan
Write-Host '    Webapp   http://localhost:8000'
Write-Host '    ML API   http://localhost:7860'
Write-Host ''
