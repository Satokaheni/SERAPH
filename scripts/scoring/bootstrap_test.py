"""
Experiment 1 — Paired Bootstrap Significance Test
====================================================
Tests whether SERAPH Full and CoT Empathy are statistically equivalent
on the 400-context comparison sample.

Reads aligned_data_4way.jsonl (which has per-record scores) or
results/empathy_comparison.csv if scores are already computed.

Method:
  Paired bootstrap resampling (n=10,000) on the per-record overall
  empathy score difference (SERAPH - CoT). Reports the two-tailed p-value.

Interpretation:
  p > 0.05  → cannot reject equivalence; claim statistical equivalence
  p < 0.05  → statistically significant difference; report direction honestly

Usage:
    python bootstrap_test.py
    python bootstrap_test.py --scores results/empathy_comparison.csv
    python bootstrap_test.py --jsonl aligned_data_4way.jsonl --n 10000
"""

from __future__ import annotations

import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path


def load_paired_scores(scores_path: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Load per-record overall scores for SERAPH Full and CoT Empathy.
    Returns (seraph_scores, cot_scores) as aligned numpy arrays.
    """
    df = pd.read_csv(scores_path)

    seraph = df[df["system"] == "SERAPH Full"].sort_values("sample_idx")
    cot    = df[df["system"] == "CoT Empathy"].sort_values("sample_idx")

    # Align on shared sample indices
    shared = set(seraph["sample_idx"]) & set(cot["sample_idx"])
    if len(shared) < len(seraph):
        print(f"Warning: {len(seraph) - len(shared)} records missing from one system — "
              f"using {len(shared)} shared records")

    seraph = seraph[seraph["sample_idx"].isin(shared)].set_index("sample_idx")
    cot    = cot[cot["sample_idx"].isin(shared)].set_index("sample_idx")
    seraph, cot = seraph.align(cot, join="inner")

    return seraph["overall"].values, cot["overall"].values


def paired_bootstrap_pvalue(
    a: np.ndarray,
    b: np.ndarray,
    n_resamples: int = 10_000,
    seed: int = 42,
) -> tuple[float, float, float]:
    """
    Paired bootstrap significance test.

    Tests H0: mean(a - b) = 0 against H1: mean(a - b) != 0.

    Method:
      1. Compute observed mean difference d_obs = mean(a - b)
      2. Centre the differences: d_centred = (a - b) - d_obs
      3. Resample d_centred with replacement n times, compute mean each time
      4. p-value = proportion of resampled means with |mean| >= |d_obs|

    Returns:
      (observed_diff, p_value, bootstrap_std)
    """
    rng  = np.random.default_rng(seed)
    diff = a - b
    d_obs = diff.mean()
    d_centred = diff - d_obs  # centre under H0

    boot_means = np.array([
        rng.choice(d_centred, size=len(d_centred), replace=True).mean()
        for _ in range(n_resamples)
    ])

    p_value = (np.abs(boot_means) >= np.abs(d_obs)).mean()
    return float(d_obs), float(p_value), float(boot_means.std())


def print_results(
    seraph: np.ndarray,
    cot: np.ndarray,
    d_obs: float,
    p_value: float,
    boot_std: float,
    n_resamples: int,
    dims_results: dict,
) -> None:
    print("\n" + "=" * 60)
    print("  PAIRED BOOTSTRAP TEST: SERAPH Full vs. CoT Empathy")
    print("=" * 60)
    print(f"  N records       : {len(seraph)}")
    print(f"  N resamples     : {n_resamples:,}")
    print(f"  SERAPH mean     : {seraph.mean():.4f} ± {seraph.std():.4f}")
    print(f"  CoT mean        : {cot.mean():.4f} ± {cot.std():.4f}")
    print(f"  Observed diff   : {d_obs:+.4f}  (SERAPH − CoT)")
    print(f"  Bootstrap std   : {boot_std:.4f}")
    print(f"  p-value (2-tail): {p_value:.4f}")
    print()

    if p_value > 0.05:
        print("  RESULT: p > 0.05 — cannot reject equivalence.")
        print("  ✓ Claim statistical equivalence between SERAPH Full and CoT Empathy.")
    elif d_obs > 0:
        print("  RESULT: p < 0.05 — SERAPH Full is significantly BETTER than CoT Empathy.")
        print("  → Report: SERAPH Full outperforms CoT Empathy (p={:.4f})".format(p_value))
    else:
        print("  RESULT: p < 0.05 — CoT Empathy is significantly BETTER than SERAPH Full.")
        print("  → Report: CoT Empathy outperforms SERAPH Full (p={:.4f})".format(p_value))

    print()
    print("  Per-dimension results:")
    print(f"  {'Dimension':<28} {'SERAPH':>8} {'CoT':>8} {'Diff':>8} {'p':>8}")
    print(f"  {'-'*56}")
    for dim, (s_mean, c_mean, d, p) in dims_results.items():
        sig = " *" if p < 0.05 else ""
        print(f"  {dim:<28} {s_mean:>8.4f} {c_mean:>8.4f} {d:>+8.4f} {p:>8.4f}{sig}")
    print()
    print("  (* p < 0.05)")
    print("=" * 60)

    # LaTeX-ready line for paper
    print("\n  LaTeX note for paper:")
    if p_value > 0.05:
        print(f"  The difference between \\seraph{{}} Full ({seraph.mean():.4f}) and "
              f"CoT Empathy ({cot.mean():.4f}) is not statistically significant "
              f"($p = {p_value:.3f}$, paired bootstrap, $n = 10{{,}}000$ resamples).")
    else:
        winner = "\\seraph{} Full" if d_obs > 0 else "CoT Empathy"
        loser  = "CoT Empathy" if d_obs > 0 else "\\seraph{} Full"
        print(f"  {winner} significantly outperforms {loser} "
              f"($p = {p_value:.3f}$, paired bootstrap, $n = 10{{,}}000$ resamples).")


def main() -> None:
    parser = argparse.ArgumentParser(description="Bootstrap test: SERAPH vs CoT Empathy")
    parser.add_argument("--scores", default="results/empathy_comparison.csv",
                        help="CSV from score_empathy.py with per-record scores")
    parser.add_argument("--n",      type=int, default=10_000,
                        help="Number of bootstrap resamples (default 10,000)")
    parser.add_argument("--seed",   type=int, default=42)
    args = parser.parse_args()

    if not Path(args.scores).exists():
        print(f"ERROR: {args.scores} not found.")
        print("Run score_empathy.py first, then rerun this script.")
        return

    df = pd.read_csv(args.scores)
    systems = df["system"].unique()
    if "SERAPH Full" not in systems or "CoT Empathy" not in systems:
        print(f"ERROR: Need both 'SERAPH Full' and 'CoT Empathy' in {args.scores}")
        print(f"Found: {list(systems)}")
        return

    seraph_scores, cot_scores = load_paired_scores(args.scores)
    d_obs, p_value, boot_std  = paired_bootstrap_pvalue(
        seraph_scores, cot_scores, args.n, args.seed
    )

    # Per-dimension tests
    dims = ["emotional_acknowledgment", "perspective_taking",
            "tone_match", "avoids_harm", "meets_core_need"]
    dims_results = {}
    seraph_df = df[df["system"] == "SERAPH Full"].sort_values("sample_idx")
    cot_df    = df[df["system"] == "CoT Empathy"].sort_values("sample_idx")
    shared    = set(seraph_df["sample_idx"]) & set(cot_df["sample_idx"])
    seraph_df = seraph_df[seraph_df["sample_idx"].isin(shared)].set_index("sample_idx")
    cot_df    = cot_df[cot_df["sample_idx"].isin(shared)].set_index("sample_idx")
    seraph_df, cot_df = seraph_df.align(cot_df, join="inner")

    for dim in dims:
        s_arr = seraph_df[dim].values
        c_arr = cot_df[dim].values
        d, p, _ = paired_bootstrap_pvalue(s_arr, c_arr, args.n, args.seed)
        dims_results[dim] = (s_arr.mean(), c_arr.mean(), d, p)

    print_results(seraph_scores, cot_scores, d_obs, p_value, boot_std,
                  args.n, dims_results)


if __name__ == "__main__":
    main()
