"""
SERAPH Path Resolver
======================
Single source of truth for all project paths.

All modules import from here instead of constructing paths with
hardcoded strings. This ensures the project works correctly
regardless of which directory you run scripts from, and is
fully compatible with Windows path separators.

Usage:
    from paths import PATHS
    data = PATHS.iemocap_data
    results = PATHS.results / "meld_full.json"
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


# Project root = directory containing this file (paths.py)
# This is reliable regardless of CWD on all platforms including Windows.
PROJECT_ROOT: Path = Path(__file__).parent.resolve()


@dataclass(frozen=True)
class ProjectPaths:
    """All project paths as absolute Path objects."""

    # ── Root ────────────────────────────────────────────────
    root:                   Path

    # ── Data ────────────────────────────────────────────────
    data:                   Path
    data_raw:               Path
    iemocap_data:           Path
    meld_data:              Path
    meld_train:             Path
    meld_dev:               Path
    meld_test:              Path
    ed_data:                Path
    ed_valid:               Path
    ed_train:               Path
    ed_test:                Path
    human_eval:             Path
    human_eval_annotations: Path
    human_eval_samples:     Path
    nrc_lexicon:            Path

    # ── Results ─────────────────────────────────────────────
    results:                Path
    results_ablations:      Path
    results_checkpoints:    Path

    # ── Baselines ───────────────────────────────────────────
    baselines_checkpoints:  Path
    cem_checkpoint:         Path
    mime_checkpoint:        Path
    drnn_checkpoint:        Path
    drnn_logreg:            Path

    # ── Paper ───────────────────────────────────────────────
    paper:                  Path
    paper_tables:           Path
    paper_figures:          Path

    @classmethod
    def build(cls, root: Path) -> "ProjectPaths":
        """Construct all paths from a root directory."""
        d = root / "data"
        return cls(
            root=root,

            # Data
            data=d,
            data_raw=d / "raw",
            iemocap_data=d / "iemocap" / "iemocap_processed.json",
            meld_data=d / "meld",
            meld_train=d / "meld" / "train_sent_emo.csv",
            meld_dev=d  / "meld" / "dev_sent_emo.csv",
            meld_test=d / "meld" / "test_sent_emo.csv",
            ed_data=d / "empathetic_dialogues",
            ed_valid=d / "empathetic_dialogues" / "valid.csv",
            ed_train=d / "empathetic_dialogues" / "train.csv",
            ed_test=d  / "empathetic_dialogues" / "test.csv",
            human_eval=d / "human_eval",
            human_eval_annotations=d / "human_eval" / "annotations",
            human_eval_samples=d / "human_eval" / "eval_samples.json",
            nrc_lexicon=d / "raw" / "NRC-Emotion-Lexicon-Wordlevel-v0.92.txt",

            # Results
            results=root / "results",
            results_ablations=root / "results" / "ablations",
            results_checkpoints=root / "results" / "checkpoints",

            # Baselines
            baselines_checkpoints=root / "baselines" / "checkpoints",
            cem_checkpoint=root  / "baselines" / "checkpoints" / "CEM",
            mime_checkpoint=root / "baselines" / "checkpoints" / "MIME",
            drnn_checkpoint=root / "baselines" / "checkpoints" / "dialogue_rnn.pt",
            drnn_logreg=root     / "baselines" / "checkpoints" / "dialogue_rnn_logreg.pkl",

            # Paper
            paper=root / "paper",
            paper_tables=root  / "paper" / "tables",
            paper_figures=root / "paper" / "figures",
        )

    def ensure_dirs(self) -> None:
        """Create all necessary output directories if they don't exist."""
        dirs = [
            self.data_raw,
            self.data / "iemocap",
            self.meld_data,
            self.ed_data,
            self.human_eval_annotations,
            self.results,
            self.results_ablations,
            self.baselines_checkpoints,
            self.paper_tables,
            self.paper_figures,
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)


# Singleton — import this everywhere
PATHS: ProjectPaths = ProjectPaths.build(PROJECT_ROOT)


# ---------------------------------------------------------------------------
# Device resolution — used by any module that loads PyTorch models
# ---------------------------------------------------------------------------

def get_torch_device() -> str:
    """
    Return the best available torch device string.
    Returns 'cuda' if a GPU is available, otherwise 'cpu'.
    Safe to call even if PyTorch is not installed (returns 'cpu').
    """
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"
