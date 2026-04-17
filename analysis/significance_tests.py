"""
SERAPH Significance Tests
===========================
Paired bootstrap resampling tests for all key comparisons in the paper.

Tests the null hypothesis: "System A and System B perform equally."
Rejects it if the observed difference is unlikely under random resampling.

The paired bootstrap is the standard significance test for NLP evaluation
(Efron & Tibshirani, 1993; Berg-Kirkpatrick et al., 2012).
It is preferred over t-tests for NLP metrics because:
  - Makes no distributional assumptions
  - Accounts for the paired nature of the data (same inputs, different systems)
  - Works correctly for non-normal metrics like F1 and empathy scores

Usage:
    # Run all key comparisons
    python analysis/significance_tests.py

    # Specific comparison
    python analysis/significance_tests.py --system-a full --system-b no_stage3 --dataset empathetic_dialogues

    # All comparisons, save LaTeX table
    python analysis/significance_tests.py --latex

Reference:
    Berg-Kirkpatrick, T., Burkett, D., & Klein, D. (2012).
    An empirical investigation of statistical significance in NLP.
    EMNLP 2012.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from paths import PATHS
logger = logging.getLogger(__name__)

# Number of bootstrap resamples — 10,000 is standard for NLP papers
N_BOOTSTRAP = 10_000
RANDOM_SEED = 42

# Significance thresholds
ALPHA_LEVELS = {0.05: "*", 0.01: "**", 0.001: "***"}


# ---------------------------------------------------------------------------
# Core bootstrap engine
# ---------------------------------------------------------------------------

@dataclass
class BootstrapResult:
    """Result of a single paired bootstrap test."""
    system_a:       str
    system_b:       str
    dataset:        str
    metric:         str
    mean_a:         float
    mean_b:         float
    difference:     float          # mean_b - mean_a (positive = B is better)
    p_value:        float
    ci_lower:       float          # 95% confidence interval on the difference
    ci_upper:       float
    n_samples:      int
    n_bootstrap:    int
    significant:    bool
    significance_marker: str       # '', '*', '**', '***'

    def __str__(self) -> str:
        direction = "B > A" if self.difference > 0 else "A > B"
        sig_str   = f"p={self.p_value:.4f}{self.significance_marker}"
        return (
            f"{self.system_a} vs {self.system_b} | {self.metric} | {self.dataset}\n"
            f"  A={self.mean_a:.4f}  B={self.mean_b:.4f}  Δ={self.difference:+.4f} ({direction})\n"
            f"  95% CI: [{self.ci_lower:+.4f}, {self.ci_upper:+.4f}]  {sig_str}"
        )

    def to_dict(self) -> dict:
        return {
            "system_a":           self.system_a,
            "system_b":           self.system_b,
            "dataset":            self.dataset,
            "metric":             self.metric,
            "mean_a":             round(self.mean_a, 4),
            "mean_b":             round(self.mean_b, 4),
            "difference":         round(self.difference, 4),
            "p_value":            round(self.p_value, 4),
            "ci_lower":           round(self.ci_lower, 4),
            "ci_upper":           round(self.ci_upper, 4),
            "n_samples":          self.n_samples,
            "n_bootstrap":        self.n_bootstrap,
            "significant":        self.significant,
            "significance_marker": self.significance_marker,
        }


def paired_bootstrap_test(
    scores_a: list[float],
    scores_b: list[float],
    n_bootstrap: int = N_BOOTSTRAP,
    seed: int = RANDOM_SEED,
) -> tuple[float, float, float, float]:
    """
    Paired bootstrap resampling significance test.

    Tests H0: mean(scores_a) == mean(scores_b)
    Returns p-value for a two-tailed test.

    Implementation follows Berg-Kirkpatrick et al. (2012):
      1. Compute observed difference δ* = mean(B) - mean(A)
      2. For each bootstrap sample:
         a. Sample n pairs with replacement
         b. Compute bootstrap difference δ_b
         c. Centre it: δ_b_centred = δ_b - δ*
      3. p-value = proportion of |δ_b_centred| >= |δ*|

    Args:
        scores_a: Per-sample scores for system A (e.g., empathy ratings)
        scores_b: Per-sample scores for system B
        n_bootstrap: Number of bootstrap resamples
        seed: Random seed for reproducibility

    Returns:
        (p_value, observed_difference, ci_lower, ci_upper)
    """
    assert len(scores_a) == len(scores_b), (
        f"Score arrays must be same length: {len(scores_a)} vs {len(scores_b)}"
    )
    assert len(scores_a) >= 10, "Need at least 10 paired samples for bootstrap test"

    rng = np.random.default_rng(seed)
    a   = np.array(scores_a, dtype=float)
    b   = np.array(scores_b, dtype=float)
    n   = len(a)

    observed_diff = float(np.mean(b) - np.mean(a))

    # Bootstrap resampling
    bootstrap_diffs = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        idx       = rng.integers(0, n, size=n)
        boot_diff = np.mean(b[idx]) - np.mean(a[idx])
        bootstrap_diffs[i] = boot_diff - observed_diff  # centre around observed

    # Two-tailed p-value
    p_value = float(np.mean(np.abs(bootstrap_diffs) >= abs(observed_diff)))

    # 95% confidence interval on the difference (not centred)
    raw_bootstrap = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        raw_bootstrap[i] = np.mean(b[idx]) - np.mean(a[idx])

    ci_lower = float(np.percentile(raw_bootstrap, 2.5))
    ci_upper = float(np.percentile(raw_bootstrap, 97.5))

    return p_value, observed_diff, ci_lower, ci_upper


def _get_significance_marker(p_value: float) -> tuple[bool, str]:
    """Return (is_significant, marker_string) for a p-value."""
    for threshold in sorted(ALPHA_LEVELS.keys()):
        if p_value < threshold:
            return True, ALPHA_LEVELS[threshold]
    return False, ""


# ---------------------------------------------------------------------------
# Result loader
# ---------------------------------------------------------------------------

def _load_per_sample_scores(
    results_path: Path,
    metric: str,
) -> Optional[list[float]]:
    """
    Load per-sample scores for a given metric from a results JSON file.

    For empathy metrics: loads from the per-result empathy_score field
    For weighted_f1: computes per-sample binary correctness (1 if correct, 0 if not)
    """
    if not results_path.exists():
        logger.warning("Results file not found: %s", results_path)
        return None

    with results_path.open(encoding="utf-8") as f:
        data = json.load(f)

    results = data.get("results", [])
    if not results:
        logger.warning("No results found in %s", results_path)
        return None

    if metric == "weighted_f1":
        # Per-sample: 1.0 if predicted == ground truth, 0.0 otherwise
        scores = []
        for r in results:
            correct = float(r.get("pred_emotion", "") == r.get("gt_emotion", ""))
            scores.append(correct)
        return scores

    # Empathy metrics — need to re-read from stored ratings if available,
    # or re-score responses on the fly
    empathy_metric_map = {
        "empathy_score":             "overall",
        "emotional_acknowledgment":  "emotional_acknowledgment",
        "perspective_taking":        "perspective_taking_quality",
        "tone_match":                "tone_match",
        "avoids_harmful":            "avoids_harmful_patterns",
        "meets_core_need":           "meets_core_need",
    }

    if metric not in empathy_metric_map:
        logger.error("Unknown metric: %s. Valid: %s", metric, list(empathy_metric_map.keys()))
        return None

    # Check if per-sample scores are stored
    field = empathy_metric_map[metric]
    scores = []
    for r in results:
        rating = r.get("empathy_rating", {})
        if field in rating:
            scores.append(float(rating[field]))

    if scores:
        return scores

    # If not stored, re-score using EmpathyScorer (adds API cost)
    logger.info(
        "Per-sample empathy ratings not found in %s — re-scoring %d responses...",
        results_path, len(results),
    )
    from metrics.empathy_scorer import EmpathyScorer
    scorer = EmpathyScorer()
    scores = []
    for r in results:
        response  = r.get("response") or r.get("seraph_response", "")
        utterance = r.get("utterance", "")
        gt        = r.get("gt_emotion", "")
        if response and utterance:
            rating = scorer.score_response(utterance, response, gt)
            val = getattr(rating, field, rating.overall)
            scores.append(float(val) if val is not None else 3.0)
        else:
            scores.append(3.0)  # neutral fallback for missing responses
    return scores


# ---------------------------------------------------------------------------
# Comparison runner
# ---------------------------------------------------------------------------

def run_comparison(
    system_a: str,
    system_b: str,
    dataset: str,
    metric: str,
    results_dir: Path,
    n_bootstrap: int = N_BOOTSTRAP,
) -> Optional[BootstrapResult]:
    """
    Run a single paired bootstrap comparison between two systems.

    Args:
        system_a: System A identifier (e.g., 'full', 'no_stage3', 'raw_claude')
        system_b: System B identifier
        dataset: Dataset name
        metric: Metric to compare ('weighted_f1', 'empathy_score', etc.)
        results_dir: Directory containing result JSON files
        n_bootstrap: Number of bootstrap resamples

    Returns:
        BootstrapResult or None if data not available
    """
    path_a = results_dir / f"{dataset}_{system_a}.json"
    path_b = results_dir / f"{dataset}_{system_b}.json"

    # Handle ablation results in subdirectory
    if not path_a.exists():
        path_a = results_dir / "ablations" / f"{dataset}_{system_a}.json"
    if not path_b.exists():
        path_b = results_dir / "ablations" / f"{dataset}_{system_b}.json"

    scores_a = _load_per_sample_scores(path_a, metric)
    scores_b = _load_per_sample_scores(path_b, metric)

    if scores_a is None or scores_b is None:
        return None

    # Align lengths (take the shorter)
    n = min(len(scores_a), len(scores_b))
    if n < 10:
        logger.warning("Too few paired samples (%d) for %s vs %s — skipping.", n, system_a, system_b)
        return None

    scores_a = scores_a[:n]
    scores_b = scores_b[:n]

    p_value, diff, ci_lo, ci_hi = paired_bootstrap_test(
        scores_a, scores_b, n_bootstrap=n_bootstrap
    )
    significant, marker = _get_significance_marker(p_value)

    result = BootstrapResult(
        system_a=system_a,
        system_b=system_b,
        dataset=dataset,
        metric=metric,
        mean_a=float(np.mean(scores_a)),
        mean_b=float(np.mean(scores_b)),
        difference=diff,
        p_value=p_value,
        ci_lower=ci_lo,
        ci_upper=ci_hi,
        n_samples=n,
        n_bootstrap=n_bootstrap,
        significant=significant,
        significance_marker=marker,
    )

    logger.info(str(result))
    return result


# ---------------------------------------------------------------------------
# Run all key paper comparisons
# ---------------------------------------------------------------------------

# The comparisons that matter for the paper
KEY_COMPARISONS = [
    # Primary claim: self-simulation helps
    ("full", "no_stage3",    "empathetic_dialogues", "empathy_score"),
    ("full", "no_stage3",    "meld",                 "empathy_score"),
    ("full", "no_stage3",    "iemocap",              "empathy_score"),

    # F1 classification comparisons
    ("full", "no_stage3",    "empathetic_dialogues", "weighted_f1"),
    ("full", "dialogue_rnn", "meld",                 "weighted_f1"),
    ("full", "dialogue_rnn", "iemocap",              "weighted_f1"),

    # SERAPH vs raw Claude
    ("full", "raw_claude",   "empathetic_dialogues", "empathy_score"),
    ("full", "raw_claude",   "meld",                 "empathy_score"),

    # SERAPH vs CEM/MIME
    ("full", "cem",          "empathetic_dialogues", "empathy_score"),
    ("full", "mime",         "empathetic_dialogues", "empathy_score"),

    # Ethical gate contribution
    ("full", "no_stage4",    "empathetic_dialogues", "empathy_score"),

    # Tone match dimension specifically (Stage 3's most direct effect)
    ("full", "no_stage3",    "empathetic_dialogues", "tone_match"),
    ("full", "no_stage3",    "empathetic_dialogues", "perspective_taking"),
]


def run_all_comparisons(
    results_dir: str = "results",
    output_dir: str = "results",
    n_bootstrap: int = N_BOOTSTRAP,
) -> list[BootstrapResult]:
    """
    Run all key comparisons and save results.

    Args:
        results_dir: Directory containing benchmark JSON files
        output_dir: Directory to save significance test results
        n_bootstrap: Bootstrap resamples (10000 recommended for paper)

    Returns:
        List of BootstrapResult objects
    """
    rdir = Path(results_dir)
    odir = Path(output_dir)
    odir.mkdir(parents=True, exist_ok=True)

    all_results = []

    print(f"\n{'═' * 70}")
    print(f"  SERAPH Significance Tests (n_bootstrap={n_bootstrap:,})")
    print(f"{'═' * 70}\n")

    for sys_a, sys_b, dataset, metric in KEY_COMPARISONS:
        result = run_comparison(sys_a, sys_b, dataset, metric, rdir, n_bootstrap)
        if result:
            all_results.append(result)
            _print_result(result)

    # Save all results
    out_path = odir / "significance_tests.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump([r.to_dict() for r in all_results], f, indent=2)
    logger.info("Significance test results saved to %s", out_path)

    _print_summary(all_results)
    return all_results


def _print_result(r: BootstrapResult) -> None:
    sig = r.significance_marker if r.significant else "n.s."
    direction = "↑" if r.difference > 0 else "↓"
    print(
        f"  {r.system_b:<15} vs {r.system_a:<12} | {r.dataset:<25} | {r.metric:<20} | "
        f"Δ={r.difference:+.4f}{direction}  p={r.p_value:.4f}  {sig}"
    )


def _print_summary(results: list[BootstrapResult]) -> None:
    sig   = [r for r in results if r.significant]
    insig = [r for r in results if not r.significant]

    print(f"\n{'─' * 70}")
    print(f"  Summary: {len(sig)}/{len(results)} comparisons significant")

    if sig:
        print(f"\n  Significant (p < 0.05):")
        for r in sig:
            print(f"    {r.system_b} > {r.system_a} | {r.dataset} | {r.metric} "
                  f"[p={r.p_value:.4f}{r.significance_marker}]")

    if insig:
        print(f"\n  Not significant:")
        for r in insig:
            print(f"    {r.system_b} vs {r.system_a} | {r.dataset} | {r.metric} "
                  f"[p={r.p_value:.4f}]")
    print()


# ---------------------------------------------------------------------------
# LaTeX significance table generator
# ---------------------------------------------------------------------------

def generate_significance_latex(
    results: list[BootstrapResult],
    output_dir: str = "paper/tables",
) -> str:
    """
    Generate a LaTeX table of significance test results for the paper.
    Markers (*, **, ***) are embedded in the main results tables
    via this output.
    """
    odir = Path(output_dir)
    odir.mkdir(parents=True, exist_ok=True)

    lines = []
    lines.append(r"% ── Significance Tests ─────────────────────────────────────────")
    lines.append(r"\begin{table}[t]")
    lines.append(r"  \centering")
    lines.append(r"  \caption{Paired bootstrap significance tests ($n=10{,}000$ resamples). "
                 r"$^*p<0.05$, $^{**}p<0.01$, $^{***}p<0.001$, n.s.\ = not significant. "
                 r"$\Delta$ = System B score minus System A score (positive = B better).}")
    lines.append(r"  \label{tab:significance}")
    lines.append(r"  \begin{tabular}{llllrrrl}")
    lines.append(r"    \toprule")
    lines.append(r"    \textbf{System A} & \textbf{System B} & \textbf{Dataset} "
                 r"& \textbf{Metric} & \textbf{A} & \textbf{B} "
                 r"& \textbf{$\Delta$} & \textbf{$p$-value} \\")
    lines.append(r"    \midrule")

    # Group by metric for readability
    by_metric: dict[str, list] = {}
    for r in results:
        by_metric.setdefault(r.metric, []).append(r)

    first = True
    for metric, metric_results in by_metric.items():
        if not first:
            lines.append(r"    \midrule")
        first = False

        for r in metric_results:
            sig_str = (
                f"${r.p_value:.4f}^{{{r.significance_marker}}}$"
                if r.significant
                else f"${r.p_value:.4f}$ (n.s.)"
            )
            delta_fmt = f"{r.difference:+.4f}"
            if r.significant and r.difference > 0:
                delta_fmt = r"\textbf{" + delta_fmt + "}"

            lines.append(
                f"    {r.system_a} & {r.system_b} & {r.dataset} & {r.metric} "
                f"& {r.mean_a:.4f} & {r.mean_b:.4f} & {delta_fmt} & {sig_str} \\\\"
            )

    lines.append(r"    \bottomrule")
    lines.append(r"  \end{tabular}")
    lines.append(r"\end{table}")

    latex = "\n".join(lines)
    out_path = odir / "table_significance.tex"
    out_path.write_text(latex, encoding="utf-8")
    logger.info("Significance table written to %s", out_path)
    return latex


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        level=logging.INFO,
    )

    parser = argparse.ArgumentParser(description="SERAPH Paired Bootstrap Significance Tests")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--output",      default="results")
    parser.add_argument("--n-bootstrap", type=int, default=N_BOOTSTRAP)
    parser.add_argument(
        "--system-a", default=None,
        help="System A for single comparison (e.g. 'full')"
    )
    parser.add_argument(
        "--system-b", default=None,
        help="System B for single comparison (e.g. 'no_stage3')"
    )
    parser.add_argument(
        "--dataset", default="empathetic_dialogues",
        choices=["iemocap", "meld", "empathetic_dialogues"],
    )
    parser.add_argument(
        "--metric", default="empathy_score",
        choices=["weighted_f1", "empathy_score", "tone_match",
                 "perspective_taking", "emotional_acknowledgment",
                 "avoids_harmful", "meets_core_need"],
    )
    parser.add_argument(
        "--latex", action="store_true",
        help="Generate LaTeX significance table"
    )
    args = parser.parse_args()

    if args.system_a and args.system_b:
        # Single comparison
        result = run_comparison(
            system_a=args.system_a,
            system_b=args.system_b,
            dataset=args.dataset,
            metric=args.metric,
            results_dir=Path(args.results_dir),
            n_bootstrap=args.n_bootstrap,
        )
        if result:
            print(f"\n{result}\n")
    else:
        # All key comparisons
        results = run_all_comparisons(
            results_dir=args.results_dir,
            output_dir=args.output,
            n_bootstrap=args.n_bootstrap,
        )
        if args.latex and results:
            generate_significance_latex(results, output_dir="paper/tables")
            print("LaTeX table written to paper/tables/table_significance.tex")


if __name__ == "__main__":
    main()
