"""
Bootstrap Significance Test — Full ED Test Set
================================================
Paired bootstrap test (n=10,000) comparing CoT Empathy vs. SERAPH Full
on the full EmpatheticDialogues test set.

Observed values: CoT 4.0587, SERAPH Full 4.2721, Δ = 0.213

Input sources:
  CoT scores   : results/cot_ed_full_scores.csv  (from run_cot_ed_full.py)
  SERAPH scores: results/seraph_ed_full_scores.csv  (if pre-computed)
              OR re-scored on the fly from results/empathetic_dialogues_full.json

The script auto-detects which SERAPH source is available. If neither CSV
exists, it scores SERAPH Full responses using the same LLM-as-judge.

Usage:
    python bootstrap_ed_full.py
    python bootstrap_ed_full.py --cot results/cot_ed_full_scores.csv
    python bootstrap_ed_full.py --seraph results/seraph_ed_full_scores.csv
    python bootstrap_ed_full.py --rescore-seraph  # re-score from full results JSON
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import anthropic
from dotenv import load_dotenv

logging.basicConfig(
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("bootstrap_ed_full")

DIMS = ["emotional_acknowledgment", "perspective_taking",
        "tone_match", "avoids_harm", "meets_core_need"]

JUDGE_TEMPLATE = """Rate this empathic dialogue response on 5 dimensions (1-5 scale).

Scoring guide:
1 = Poor  2 = Below average  3 = Adequate  4 = Good  5 = Excellent

Definitions:
- emotional_acknowledgment: Does the response explicitly acknowledge the speaker's emotional state?
- perspective_taking: Does it demonstrate genuine first-person understanding of how the speaker feels?
- tone_match: Does the tone match the speaker's emotional register?
- avoids_harm: Does it avoid minimising, victim-blaming, or dysregulating content?
- meets_core_need: Does it address what the speaker most needs right now?

Return ONLY a JSON object with these exact keys (no markdown, no explanation):
{"emotional_acknowledgment": X, "perspective_taking": X, "tone_match": X, "avoids_harm": X, "meets_core_need": X}

where X is an integer 1-5.

Dialogue context:
CONTEXT_PLACEHOLDER

System response:
RESPONSE_PLACEHOLDER"""


# ── Scoring ───────────────────────────────────────────────────────────────────

def score_response(client: anthropic.Anthropic, context: str,
                   response: str) -> dict | None:
    for attempt in range(3):
        try:
            result = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=256,
                messages=[{"role": "user", "content":
                    JUDGE_TEMPLATE
                    .replace("CONTEXT_PLACEHOLDER", context.strip())
                    .replace("RESPONSE_PLACEHOLDER", response.strip())
                }],
            )
            text = result.content[0].text.strip()
            text = re.sub(r'^```(?:json)?\s*', '', text, flags=re.MULTILINE)
            text = re.sub(r'\s*```$',           '', text, flags=re.MULTILINE)
            scores = json.loads(text.strip())
            assert all(k in scores for k in DIMS)
            assert all(1 <= scores[k] <= 5 for k in DIMS)
            scores["overall"] = round(sum(scores[k] for k in DIMS) / len(DIMS), 4)
            return scores
        except Exception as e:
            if attempt < 2:
                time.sleep(2 ** attempt)
            else:
                return None


def score_seraph_from_json(
    full_json_path: str,
    output_csv: str,
    api_key: str,
) -> pd.DataFrame:
    """Score SERAPH Full responses from empathetic_dialogues_full.json."""
    with open(full_json_path, encoding="utf-8") as f:
        data = json.load(f)
    results = data["results"]

    client = anthropic.Anthropic(api_key=api_key)
    rows, errors = [], 0

    logger.info("Scoring %d SERAPH Full responses…", len(results))
    for i, rec in enumerate(results):
        if not rec.get("success") or rec.get("error"):
            continue
        scores = score_response(client, rec["utterance"], rec["response"])
        if scores is None:
            errors += 1
            scores = {d: float("nan") for d in DIMS + ["overall"]}
        rows.append({
            "system":     "SERAPH Full",
            "sample_idx": i,
            "context":    rec["utterance"][:200],
            "response":   rec["response"][:200],
            **scores,
        })
        time.sleep(0.5)
        if (i + 1) % 100 == 0:
            logger.info("Scored %d/%d (%d errors)", i + 1, len(results), errors)

    df = pd.DataFrame(rows)
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    logger.info("SERAPH Full scores saved → %s", output_csv)
    return df


# ── Bootstrap ─────────────────────────────────────────────────────────────────

def paired_bootstrap(
    a: np.ndarray,
    b: np.ndarray,
    n_resamples: int = 10_000,
    seed: int = 42,
) -> tuple[float, float, float]:
    """
    Paired bootstrap, H0: mean(a - b) = 0.
    Returns (observed_diff, p_value, bootstrap_std).
    """
    rng       = np.random.default_rng(seed)
    diff      = a - b
    d_obs     = diff.mean()
    centred   = diff - d_obs

    boot_means = np.array([
        rng.choice(centred, size=len(centred), replace=True).mean()
        for _ in range(n_resamples)
    ])

    p = (np.abs(boot_means) >= np.abs(d_obs)).mean()
    return float(d_obs), float(p), float(boot_means.std())


def align_scores(
    seraph_df: pd.DataFrame,
    cot_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Align on shared sample_idx, drop NaNs."""
    shared = set(seraph_df["sample_idx"]) & set(cot_df["sample_idx"])
    s = seraph_df[seraph_df["sample_idx"].isin(shared)].set_index("sample_idx")
    c = cot_df[cot_df["sample_idx"].isin(shared)].set_index("sample_idx")
    s, c = s.align(c, join="inner")

    # Drop rows where either has NaN overall
    valid = s["overall"].notna() & c["overall"].notna()
    return s[valid], c[valid]


# ── Reporting ─────────────────────────────────────────────────────────────────

def print_results(
    s: pd.DataFrame,
    c: pd.DataFrame,
    d_obs: float,
    p_value: float,
    boot_std: float,
    n_resamples: int,
    dim_results: dict,
) -> None:
    n = len(s)
    print("\n" + "=" * 65)
    print("  BOOTSTRAP TEST: SERAPH Full vs. CoT Empathy — Full ED Test Set")
    print("=" * 65)
    print(f"  N aligned records : {n}")
    print(f"  N resamples       : {n_resamples:,}")
    print(f"  SERAPH Full mean  : {s['overall'].mean():.4f} ± {s['overall'].std():.4f}")
    print(f"  CoT Empathy mean  : {c['overall'].mean():.4f} ± {c['overall'].std():.4f}")
    print(f"  Observed Δ        : {d_obs:+.4f}  (SERAPH − CoT)")
    print(f"  Bootstrap std     : {boot_std:.4f}")
    print(f"  p-value (2-tail)  : {p_value:.4f}")
    print()

    if p_value > 0.05:
        print("  RESULT: p > 0.05 — difference is NOT significant.")
        print("  ✓ Report statistical equivalence on the full test set.")
    elif d_obs > 0:
        print("  RESULT: p < 0.05 — SERAPH Full is significantly BETTER.")
        print(f"  → Report: SERAPH Full outperforms CoT Empathy (p = {p_value:.4f}).")
        print("  ✓ This strengthens the pipeline contribution claim.")
    else:
        print("  RESULT: p < 0.05 — CoT Empathy is significantly BETTER.")
        print(f"  → Report honestly: CoT Empathy outperforms SERAPH Full (p = {p_value:.4f}).")

    print()
    print(f"  {'Dimension':<28} {'SERAPH':>8} {'CoT':>8} {'Δ':>8} {'p':>8}")
    print(f"  {'-'*60}")
    for dim, (s_mean, c_mean, d, p) in dim_results.items():
        sig = " *" if p < 0.05 else ""
        print(f"  {dim:<28} {s_mean:>8.4f} {c_mean:>8.4f} {d:>+8.4f} {p:>8.4f}{sig}")
    print("  (* p < 0.05)")

    # LaTeX sentence ready to paste
    print("\n" + "-" * 65)
    print("  LaTeX sentence for paper:")
    s_mean = s["overall"].mean()
    c_mean = c["overall"].mean()
    if p_value > 0.05:
        print(
            f"  On the full EmpatheticDialogues test set, \\seraph{{}} Full "
            f"({s_mean:.4f}) and CoT Empathy ({c_mean:.4f}) do not differ "
            f"significantly ($\\Delta = {d_obs:+.4f}$, $p = {p_value:.3f}$, "
            f"paired bootstrap, $n = {n_resamples:,}$ resamples)."
        )
    else:
        winner = f"\\seraph{{}} Full ({s_mean:.4f})" if d_obs > 0 \
                 else f"CoT Empathy ({c_mean:.4f})"
        loser  = f"CoT Empathy ({c_mean:.4f})" if d_obs > 0 \
                 else f"\\seraph{{}} Full ({s_mean:.4f})"
        print(
            f"  {winner} significantly outperforms {loser} "
            f"($\\Delta = {abs(d_obs):.4f}$, $p = {p_value:.3f}$, "
            f"paired bootstrap, $n = {n_resamples:,}$ resamples)."
        )
    print("=" * 65)


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Bootstrap test: CoT Empathy vs SERAPH Full — full ED test set"
    )
    parser.add_argument("--cot",
                        default="results/cot_ed_full_scores.csv",
                        help="CoT Empathy scores CSV (from run_cot_ed_full.py)")
    parser.add_argument("--seraph",
                        default="results/seraph_ed_full_scores.csv",
                        help="SERAPH Full scores CSV (pre-computed)")
    parser.add_argument("--seraph-json",
                        default="results/empathetic_dialogues_full.json",
                        help="SERAPH Full results JSON (used if --seraph CSV not found)")
    parser.add_argument("--rescore-seraph", action="store_true",
                        help="Force re-scoring SERAPH from JSON even if CSV exists")
    parser.add_argument("--n",    type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    load_dotenv(".env")
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")

    # ── Load CoT scores ────────────────────────────────────────────────────
    if not Path(args.cot).exists():
        print(f"ERROR: CoT scores not found at {args.cot}")
        print("Run run_cot_ed_full.py first.")
        return
    cot_df = pd.read_csv(args.cot)
    logger.info("Loaded %d CoT records from %s", len(cot_df), args.cot)

    # ── Load or score SERAPH Full ──────────────────────────────────────────
    if Path(args.seraph).exists() and not args.rescore_seraph:
        seraph_df = pd.read_csv(args.seraph)
        logger.info("Loaded %d SERAPH records from %s", len(seraph_df), args.seraph)
    else:
        if not api_key:
            print("ERROR: ANTHROPIC_API_KEY not set — needed to score SERAPH responses.")
            return
        if not Path(args.seraph_json).exists():
            print(f"ERROR: SERAPH JSON not found at {args.seraph_json}")
            return
        logger.info("SERAPH CSV not found — scoring from %s", args.seraph_json)
        seraph_df = score_seraph_from_json(args.seraph_json, args.seraph, api_key)

    # ── Align and run bootstrap ────────────────────────────────────────────
    s, c = align_scores(seraph_df, cot_df)
    logger.info("Aligned on %d shared records", len(s))

    if len(s) < 100:
        print(f"WARNING: only {len(s)} aligned records — results may be unreliable.")

    d_obs, p_value, boot_std = paired_bootstrap(
        s["overall"].values, c["overall"].values, args.n, args.seed
    )

    # Per-dimension tests
    dim_results = {}
    for dim in DIMS:
        d, p, _ = paired_bootstrap(
            s[dim].values, c[dim].values, args.n, args.seed
        )
        dim_results[dim] = (s[dim].mean(), c[dim].mean(), d, p)

    print_results(s, c, d_obs, p_value, boot_std, args.n, dim_results)


if __name__ == "__main__":
    main()
