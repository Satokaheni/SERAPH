# ============================================================
# SERAPH — Windows Setup Script (PowerShell)
# ============================================================
# Usage: Right-click PowerShell -> "Run as Administrator", then:
#   Set-ExecutionPolicy RemoteSigned -Scope CurrentUser
#   .\setup.ps1

$ErrorActionPreference = "Stop"

Write-Host ""
Write-Host "╔══════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║         SERAPH Setup (Windows)           ║" -ForegroundColor Cyan
Write-Host "╚══════════════════════════════════════════╝" -ForegroundColor Cyan
Write-Host ""

# ── Python version check ────────────────────────────────────
Write-Host "→ Checking Python version..." -ForegroundColor Yellow

$python = $null
foreach ($cmd in @("python", "python3")) {
    try {
        $ver = & $cmd --version 2>&1
        if ($ver -match "Python (\d+)\.(\d+)") {
            $major = [int]$Matches[1]
            $minor = [int]$Matches[2]
            if ($major -gt 3 -or ($major -eq 3 -and $minor -ge 10)) {
                $python = $cmd
                Write-Host "  ✓ Found $ver ($cmd)" -ForegroundColor Green
                break
            } else {
                Write-Host "  ✗ Found $ver but Python 3.10+ is required." -ForegroundColor Red
            }
        }
    } catch {
        continue
    }
}

if (-not $python) {
    Write-Host ""
    Write-Host "  Python 3.10+ not found. Download from https://www.python.org/downloads/" -ForegroundColor Red
    Write-Host "  Make sure to check 'Add Python to PATH' during installation." -ForegroundColor Red
    exit 1
}

# ── Virtual environment ──────────────────────────────────────
Write-Host "→ Setting up virtual environment..." -ForegroundColor Yellow

if (-not (Test-Path ".venv")) {
    & $python -m venv .venv
    Write-Host "  ✓ Created .venv" -ForegroundColor Green
} else {
    Write-Host "  ✓ .venv already exists" -ForegroundColor Green
}

# Activate
$activateScript = ".venv\Scripts\Activate.ps1"
if (-not (Test-Path $activateScript)) {
    Write-Host "  ✗ Could not find $activateScript" -ForegroundColor Red
    exit 1
}
& $activateScript
Write-Host "  ✓ Virtual environment activated" -ForegroundColor Green

# ── Dependencies ─────────────────────────────────────────────
Write-Host "→ Installing dependencies..." -ForegroundColor Yellow
& python -m pip install --upgrade pip --quiet
& pip install -r requirements.txt --quiet
Write-Host "  ✓ Dependencies installed" -ForegroundColor Green

# ── .env ─────────────────────────────────────────────────────
Write-Host "→ Checking .env file..." -ForegroundColor Yellow

if (-not (Test-Path ".env")) {
    Copy-Item ".env.example" ".env"
    Write-Host "  ✓ Created .env from .env.example" -ForegroundColor Green
    Write-Host "  ⚠  Open .env and set your ANTHROPIC_API_KEY before running." -ForegroundColor Yellow
} else {
    Write-Host "  ✓ .env already exists" -ForegroundColor Green
}

# ── Directories ───────────────────────────────────────────────
Write-Host "→ Creating project directories..." -ForegroundColor Yellow

$dirs = @(
    "data\raw",
    "data\meld",
    "data\empathetic_dialogues",
    "data\human_eval\annotations",
    "results\ablations",
    "results\checkpoints",
    "baselines\checkpoints\CEM",
    "baselines\checkpoints\MIME",
    "paper\tables",
    "paper\figures",
    "analysis",
    "baselines",
    "benchmarks",
    "metrics",
    "pipeline",
    "ablations"
)

foreach ($dir in $dirs) {
    if (-not (Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
    }
}
Write-Host "  ✓ Project directories created" -ForegroundColor Green

# ── Dataset check ─────────────────────────────────────────────
Write-Host "→ Checking datasets..." -ForegroundColor Yellow
try {
    & python data\prepare_datasets.py --verify
} catch {
    Write-Host "  (Dataset check skipped — run manually after setup)" -ForegroundColor Gray
}

# ── Done ──────────────────────────────────────────────────────
Write-Host ""
Write-Host "╔══════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║  Setup complete!                         ║" -ForegroundColor Cyan
Write-Host "╠══════════════════════════════════════════╣" -ForegroundColor Cyan
Write-Host "║  Next steps:                             ║" -ForegroundColor Cyan
Write-Host "║  1. Set ANTHROPIC_API_KEY in .env        ║" -ForegroundColor Cyan
Write-Host "║  2. Activate environment:                ║" -ForegroundColor Cyan
Write-Host "║     .venv\Scripts\Activate.ps1           ║" -ForegroundColor Cyan
Write-Host "║  3. Download datasets:                   ║" -ForegroundColor Cyan
Write-Host "║     python data\prepare_datasets.py --all║" -ForegroundColor Cyan
Write-Host "║  4. Run interactive mode:                ║" -ForegroundColor Cyan
Write-Host "║     python main.py                       ║" -ForegroundColor Cyan
Write-Host "║  5. Run benchmarks:                      ║" -ForegroundColor Cyan
Write-Host "║     python main.py --benchmark           ║" -ForegroundColor Cyan
Write-Host "╚══════════════════════════════════════════╝" -ForegroundColor Cyan
Write-Host ""
