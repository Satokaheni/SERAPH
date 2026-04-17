#!/usr/bin/env bash
# ============================================================
# SERAPH — Developer Setup Script
# ============================================================
# Works on: Linux (Lambda Labs, AWS, Ubuntu), macOS, Windows (Git Bash / WSL)
#
# Usage:
#   chmod +x setup.sh && bash setup.sh          # Linux/macOS
#   bash setup.sh                               # Windows Git Bash / WSL

set -e

echo ""
echo "╔══════════════════════════════════════════╗"
echo "║         SERAPH Setup                     ║"
echo "╚══════════════════════════════════════════╝"
echo ""

# ── Platform detection ───────────────────────────────────────
OS_TYPE="linux"
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" || "$OSTYPE" == "win32" ]]; then
  OS_TYPE="windows"
  VENV_ACTIVATE=".venv/Scripts/activate"
elif [[ "$OSTYPE" == "darwin"* ]]; then
  OS_TYPE="macos"
  VENV_ACTIVATE=".venv/bin/activate"
else
  VENV_ACTIVATE=".venv/bin/activate"
fi
echo "→ Platform: $OS_TYPE"

# ── GPU detection (Linux/Lambda) ─────────────────────────────
HAS_GPU=false
if command -v nvidia-smi &>/dev/null; then
  GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || true)
  if [ -n "$GPU_NAME" ]; then
    HAS_GPU=true
    echo "→ GPU detected: $GPU_NAME"
  fi
fi
if [ "$HAS_GPU" = false ]; then
  echo "→ No GPU detected — CPU mode (DialogueRNN checkpoint mode unavailable)"
fi

# ── Python version check ────────────────────────────────────
PY_VER=""
for CANDIDATE in python3.12 python3.11 python3.10 python3 python; do
  CMD=$(command -v "$CANDIDATE" 2>/dev/null) || continue
  VER=$("$CMD" --version 2>&1 || true)
  if echo "$VER" | grep -qE '^Python [0-9]+\.[0-9]+'; then
    PYTHON="$CMD"
    PY_VER=$(echo "$VER" | grep -oE '[0-9]+\.[0-9]+\.[0-9]+')
    break
  fi
done

if [ -z "${PY_VER:-}" ]; then
  echo "✗ Python not found. Install Python 3.10+ and ensure it's on your PATH."
  exit 1
fi

echo "→ Python: $PY_VER  ($PYTHON)"

MAJOR=$(echo "$PY_VER" | cut -d. -f1)
MINOR=$(echo "$PY_VER" | cut -d. -f2)
if [ "$MAJOR" -lt 3 ] || ([ "$MAJOR" -eq 3 ] && [ "$MINOR" -lt 10 ]); then
  echo "✗ Python 3.10+ required. Found $PY_VER"
  exit 1
fi

# ── Virtual environment ──────────────────────────────────────
if [ ! -d ".venv" ]; then
  echo "→ Creating virtual environment at .venv …"
  $PYTHON -m venv .venv
fi

source "$VENV_ACTIVATE"
echo "→ Virtual environment activated."

# ── Core dependencies ─────────────────────────────────────────
echo "→ Installing dependencies …"
python -m pip install --upgrade pip --quiet
python -m pip install -r requirements.txt --quiet
echo "  ✓ Core dependencies installed."

# ── PyTorch (optional — only for DialogueRNN checkpoint mode) ─
echo "→ Checking PyTorch …"
if python -c "import torch" &>/dev/null; then
  TORCH_VER=$(python -c "import torch; print(torch.__version__)" 2>/dev/null)
  echo "  ✓ PyTorch already installed: $TORCH_VER"
else
  if [ "$HAS_GPU" = true ]; then
    echo "  → GPU detected — installing PyTorch with CUDA 12.1 support …"
    python -m pip install torch --index-url https://download.pytorch.org/whl/cu121 --quiet
    echo "  ✓ PyTorch (CUDA) installed."
  else
    echo "  → No GPU — installing PyTorch CPU-only …"
    python -m pip install torch --index-url https://download.pytorch.org/whl/cpu --quiet
    echo "  ✓ PyTorch (CPU) installed."
  fi
fi

# ── .env ─────────────────────────────────────────────────────
if [ ! -f ".env" ]; then
  cp .env.example .env
  echo "  ✓ Created .env from .env.example"
  echo ""
  echo "  ⚠  Set your ANTHROPIC_API_KEY in .env before running the pipeline."
else
  echo "  ✓ .env already exists."
fi

# ── Directories ───────────────────────────────────────────────
mkdir -p data/raw data/iemocap data/meld data/empathetic_dialogues
mkdir -p data/human_eval/annotations
mkdir -p results/ablations results/checkpoints
mkdir -p baselines/checkpoints/CEM baselines/checkpoints/MIME
mkdir -p paper/tables paper/figures
echo "  ✓ Project directories created."

# ── File permissions (Linux/macOS) ───────────────────────────
if [ "$OS_TYPE" != "windows" ]; then
  chmod +x setup.sh 2>/dev/null || true
  echo "  ✓ File permissions set."
fi

# ── Dataset check ─────────────────────────────────────────────
echo ""
echo "→ Checking datasets …"
python prepare_datasets.py --verify 2>/dev/null || true

# ── CUDA verification (Lambda/GPU only) ──────────────────────
if [ "$HAS_GPU" = true ]; then
  echo ""
  echo "→ Verifying CUDA availability in PyTorch …"
  python -c "
import torch
available = torch.cuda.is_available()
if available:
    print(f'  ✓ CUDA available — device: {torch.cuda.get_device_name(0)}')
    print(f'  ✓ CUDA version: {torch.version.cuda}')
else:
    print('  ⚠  CUDA not available in PyTorch despite GPU detected.')
    print('     Re-install PyTorch with: pip install torch --index-url https://download.pytorch.org/whl/cu121')
" 2>/dev/null || echo "  ⚠  Could not verify CUDA."
fi

# ── Done ──────────────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════╗"
echo "║  Setup complete!                         ║"
echo "╠══════════════════════════════════════════╣"
echo "║  Next steps:                             ║"
echo "║  1. Set ANTHROPIC_API_KEY in .env        ║"
echo "║  2. Download datasets:                   ║"
echo "║     python prepare_datasets.py --all║"
echo "║  3. Run interactive mode:                ║"
echo "║     python main.py                       ║"
echo "║  4. Run benchmarks:                      ║"
echo "║     python main.py --benchmark           ║"
if [ "$HAS_GPU" = true ]; then
echo "║  5. Enable GPU checkpoint mode:          ║"
echo "║     Set CEM_MODE=checkpoint in .env      ║"
echo "║     Set DIALOGUE_RNN_MODE=checkpoint     ║"
fi
echo "╚══════════════════════════════════════════╝"
echo ""
