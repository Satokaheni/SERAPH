"""
SERAPH Configuration
====================
Central configuration for the SERAPH pipeline.
Controls model selection, cost strategy, API settings,
and experiment parameters.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# Load .env file explicitly from project root
load_dotenv(dotenv_path=Path(__file__).parent / ".env")

# ---------------------------------------------------------------------------
# Model Selection — cost-tiered strategy
# ---------------------------------------------------------------------------
# Stages 1 & 2 use Haiku (cheap, fast, adequate for signal extraction)
# Stages 3, 4, 5 use Sonnet (richer reasoning, required for self-simulation)

HAIKU_MODEL  = "claude-haiku-4-5-20251001"   # Stages 1 & 2
SONNET_MODEL = "claude-sonnet-4-6"            # Stages 3, 4, 5

# ---------------------------------------------------------------------------
# API Configuration
# ---------------------------------------------------------------------------

ANTHROPIC_API_KEY: str = os.environ.get("ANTHROPIC_API_KEY", "")

USE_PROMPT_CACHING: bool = True
USE_BATCH_API: bool = True  # flip to False for sequential/debug runs

MAX_TOKENS: dict[str, int] = {
    "stage1": 1024,   # increase from 512 — fixes truncation
    "stage2": 768,
    "stage3": 1536,
    "stage4": 1536,
    "stage5": 1024,
}

TEMPERATURE: dict[str, float] = {
    "stage1": 0.1,
    "stage2": 0.1,
    "stage3": 0.7,
    "stage4": 0.2,
    "stage5": 0.6,
}

# ---------------------------------------------------------------------------
# Experiment / Benchmark Settings
# ---------------------------------------------------------------------------

@dataclass
class ExperimentConfig:
    """Parameters for a benchmark run."""
    datasets: list[str] = field(
        default_factory=lambda: ["meld", "empathetic_dialogues"]
    )
    sample_limit: Optional[int] = None
    seed: int = 42
    results_dir: str = "results"
    human_eval_n: int = 50
    run_ablations: bool = True
    save_stage_outputs: bool = False


DEFAULT_EXPERIMENT = ExperimentConfig()

# ---------------------------------------------------------------------------
# Ablation Variant Keys
# ---------------------------------------------------------------------------

ABLATION_VARIANTS = {
    "full":           "Full pipeline — all 5 stages",
    "no_stage3":      "Remove Stage 3 (no self-simulation)",
    "no_stage4":      "Remove Stage 4 (no ethical gate)",
    "merged_1_2":     "Merge Stages 1+2 into a single classification step",
    "stage3_only":    "Self-simulation only, skip Stages 1+2",
    "random_emotion": "Random emotion label injected at Stage 2",
}

# ---------------------------------------------------------------------------
# Cost Estimation Helpers
# ---------------------------------------------------------------------------

COST_PER_MILLION = {
    HAIKU_MODEL:  {"input": 1.0,  "output": 5.0},
    SONNET_MODEL: {"input": 3.0,  "output": 15.0},
}

BATCH_DISCOUNT = 0.50
CACHE_DISCOUNT = 0.90


def estimate_cost(
    input_tokens: int,
    output_tokens: int,
    model: str,
    cached_input_tokens: int = 0,
    use_batch: bool = False,
) -> float:
    rates      = COST_PER_MILLION.get(model, {"input": 3.0, "output": 15.0})
    batch_mult = (1 - BATCH_DISCOUNT) if use_batch else 1.0
    non_cached = input_tokens - cached_input_tokens
    cost_input = (
        non_cached * rates["input"] / 1_000_000
        + cached_input_tokens * rates["input"] * (1 - CACHE_DISCOUNT) / 1_000_000
    )
    cost_output = output_tokens * rates["output"] / 1_000_000
    return (cost_input + cost_output) * batch_mult


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

LOG_LEVEL:  str = os.environ.get("SERAPH_LOG_LEVEL", "INFO")
LOG_FORMAT: str = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
