"""
Corrected Bootstrap: SERAPH Full vs. w/o Stage 3 — Full ED Test Set
=====================================================================
Scores w/o Stage 3 responses using the LLM-as-judge (batch API),
aligns with existing SERAPH Full scores, and runs the paired bootstrap.

This corrects the previously reported delta of -0.128 (which used a
3-dimension average) with the correct 5-dimension overall score.

Inputs:
  results/empathetic_dialogues_no_stage3.json  — w/o Stage 3 responses
  results/seraph_ed_full_scores.csv            — SERAPH Full per-record scores

Outputs:
  results/no3_ed_full_scores.csv    — w/o Stage 3 per-record scores
  bootstrap_full_vs_no3.txt         — bootstrap results + LaTeX sentences

Usage:
    python score_and_bootstrap_no3.py
    python score_and_bootstrap_no3.py --skip-scoring  # if CSV already exists
"""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import argparse
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Optional

import anthropic
import numpy as np
import pandas as pd
from dotenv import load_dotenv

from config import ANTHROPIC_API_KEY, USE_PROMPT_CACHING
from benchmarks.batch_runner import (
    submit_batch,
    poll_until_complete,
    _checkpoint_path,
    _load_checkpoint,
    _save_checkpoint,
)

logging.basicConfig(
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("bootstrap_no3")

DIMS = ["emotional_acknowledgment", "perspective_taking",
        "tone_match", "avoids_harm", "meets_core_need"]

JUDGE_SYSTEM = (
    "You are an expert evaluator of empathic dialogue responses. "
    "Return ONLY a JSON object with these exact keys, no markdown: "
    '{"emotional_acknowledgment": X, "perspective_taking": X, '
    '"tone_match": X, "avoids_harm": X, "meets_core_need": X} '
    "where X is an integer 1-5."
)

JUDGE_USER = """Rate this empathic dialogue response on 5 dimensions (1-5 scale).

1=Poor 2=Below average 3=Adequate 4=Good 5=Excellent

- emotional_acknowledgment: explicitly recognises the speaker's emotional state
- perspective_taking: genuine first-person understanding of how the speaker feels
- tone_match: tone matches the speaker's emotional register
- avoids_harm: avoids minimising, victim-blaming, or dysregulating content
- meets_core_need: addresses what the speaker most needs right now

Dialogue context:
{context}

System response:
{response}

Return ONLY the JSON object."""


# ── Alignment ─────────────────────────────────────────────────────────────────

def align_datasets(
    no3_json_path: str,
    seraph_csv_path: str,
) -> list[dict]:
    """
    Align w/o Stage 3 responses with SERAPH Full scores by utterance.
    Returns list of dicts with utterance, no3_response, seraph_overall, seraph_dims.
    """
    with open(no3_json_path, encoding="utf-8") as f:
        no3_data = json.load(f)
    no3_results = no3_data["results"]

    seraph_df = pd.read_csv(seraph_csv_path)
    seraph_idx = {
        row["context"][:100].strip(): row
        for _, row in seraph_df.iterrows()
    }

    pairs = []
    for j, rec in enumerate(no3_results):
        if not rec.get("success") or rec.get("error"):
            continue
        key = rec["utterance"][:100].strip()
        if key in seraph_idx:
            s = seraph_idx[key]
            pairs.append({
                "idx":            j,
                "utterance":      rec["utterance"],
                "no3_response":   rec["response"],
                "seraph_overall": float(s["overall"]),
                **{f"seraph_{d}": float(s[d]) for d in DIMS},
            })

    logger.info("Aligned %d pairs from %d no3 / %d seraph records",
                len(pairs), len(no3_results), len(seraph_df))
    return pairs


# ── Batch scoring ─────────────────────────────────────────────────────────────

def _build_judge_request(idx: int, utterance: str, response: str) -> dict:
    system = (
        [{"type": "text", "text": JUDGE_SYSTEM,
          "cache_control": {"type": "ephemeral"}}]
        if USE_PROMPT_CACHING else JUDGE_SYSTEM
    )
    return {
        "custom_id": f"no3_{idx:05d}",
        "params": {
            "model": "claude-haiku-4-5-20251001",
            "max_tokens": 256,
            "system": system,
            "messages": [{
                "role": "user",
                "content": JUDGE_USER.format(
                    context=utterance.strip(),
                    response=response.strip(),
                ),
            }],
        },
    }


def score_no3_batch(
    pairs: list[dict],
    client: anthropic.Anthropic,
    checkpoint_dir: Path,
) -> dict[int, dict]:
    """
    Score all w/o Stage 3 responses via batch API.
    Returns {idx: scores_dict}.
    """
    cp = _checkpoint_path(checkpoint_dir, 97)
    cached = _load_checkpoint(cp)

    if cached:
        logger.info("Using cached scoring results (%d records)", len(cached))
        results = cached
    else:
        requests = [
            _build_judge_request(p["idx"], p["utterance"], p["no3_response"])
            for p in pairs
        ]
        logger.info("Submitting %d judge requests (batch API)…", len(requests))
        batch_id = submit_batch(requests, client)
        results  = poll_until_complete(batch_id, client)
        _save_checkpoint(cp, results)

    scores_map: dict[int, dict] = {}
    errors = 0
    for custom_id, result in results.items():
        idx = int(custom_id.replace("no3_", ""))
        if not result.get("ok"):
            errors += 1
            continue
        data = result.get("data", {})
        raw  = result.get("raw", "")
        # Try to parse from raw if data is empty
        if not data and raw:
            try:
                raw_clean = re.sub(r'^```(?:json)?\s*', '', raw, flags=re.MULTILINE)
                raw_clean = re.sub(r'\s*```$', '', raw_clean, flags=re.MULTILINE)
                data = json.loads(raw_clean.strip())
            except Exception:
                errors += 1
                continue
        if all(k in data for k in DIMS) and all(1 <= data[k] <= 5 for k in DIMS):
            data["overall"] = round(sum(data[k] for k in DIMS) / len(DIMS), 4)
            scores_map[idx] = data
        else:
            errors += 1

    logger.info("Scoring complete — %d scored, %d errors", len(scores_map), errors)
    return scores_map


def build_scores_df(pairs: list[dict], scores_map: dict[int, dict]) -> pd.DataFrame:
    rows = []
    for p in pairs:
        sc = scores_map.get(p["idx"])
        if sc is None:
            continue
        rows.append({
            "system":     "SERAPH w/o Stage 3",
            "sample_idx": p["idx"],
            "context":    p["utterance"][:200],
            "response":   p["no3_response"][:200],
            **sc,
        })
    return pd.DataFrame(rows)


# ── Bootstrap ─────────────────────────────────────────────────────────────────

def paired_bootstrap(
    a: np.ndarray,
    b: np.ndarray,
    n_resamples: int = 10_000,
    seed: int = 42,
) -> tuple[float, float, float]:
    rng     = np.random.default_rng(seed)
    diff    = a - b
    d_obs   = diff.mean()
    centred = diff - d_obs
    boot    = np.array([
        rng.choice(centred, size=len(centred), replace=True).mean()
        for _ in range(n_resamples)
    ])
    p = (np.abs(boot) >= np.abs(d_obs)).mean()
    return float(d_obs), float(p), float(boot.std())


# ── Reporting ─────────────────────────────────────────────────────────────────

def print_and_save_results(
    seraph_df: pd.DataFrame,
    no3_df: pd.DataFrame,
    pairs: list[dict],
    n_resamples: int,
    output_path: str,
) -> None:
    # Align on shared idx
    no3_idx    = set(no3_df["sample_idx"])
    seraph_sub = pd.DataFrame([
        {"sample_idx": p["idx"], **{d: p[f"seraph_{d}"] for d in DIMS},
         "overall": p["seraph_overall"]}
        for p in pairs if p["idx"] in no3_idx
    ]).set_index("sample_idx")
    no3_sub = no3_df.set_index("sample_idx")
    seraph_sub, no3_sub = seraph_sub.align(no3_sub, join="inner")

    valid = seraph_sub["overall"].notna() & no3_sub["overall"].notna()
    seraph_sub = seraph_sub[valid]
    no3_sub    = no3_sub[valid]
    n = len(seraph_sub)

    d_obs, p_val, boot_std = paired_bootstrap(
        seraph_sub["overall"].values,
        no3_sub["overall"].values,
        n_resamples,
    )

    # Per-dimension
    dim_results = {}
    for dim in DIMS:
        d, p, _ = paired_bootstrap(
            seraph_sub[dim].values,
            no3_sub[dim].values,
            n_resamples,
        )
        dim_results[dim] = (
            seraph_sub[dim].mean(),
            no3_sub[dim].mean(),
            d, p
        )

    lines = []
    lines.append("=" * 65)
    lines.append("  CORRECTED BOOTSTRAP: SERAPH Full vs. w/o Stage 3")
    lines.append("  Full EmpatheticDialogues Test Set (5-dimension scorer)")
    lines.append("=" * 65)
    lines.append(f"  N aligned records  : {n}")
    lines.append(f"  N resamples        : {n_resamples:,}")
    lines.append(f"  SERAPH Full mean   : {seraph_sub['overall'].mean():.4f} "
                 f"± {seraph_sub['overall'].std():.4f}")
    lines.append(f"  w/o Stage 3 mean   : {no3_sub['overall'].mean():.4f} "
                 f"± {no3_sub['overall'].std():.4f}")
    lines.append(f"  Observed Δ         : {d_obs:+.4f}  (SERAPH Full − w/o Stage 3)")
    lines.append(f"  Bootstrap std      : {boot_std:.4f}")
    lines.append(f"  p-value (2-tail)   : {p_val:.4f}")
    lines.append("")

    if p_val < 0.001:
        sig = "p < 0.001"
    elif p_val < 0.01:
        sig = f"p = {p_val:.3f}"
    elif p_val < 0.05:
        sig = f"p = {p_val:.3f}"
    else:
        sig = f"p = {p_val:.3f} (NOT significant)"

    if p_val < 0.05:
        lines.append(f"  RESULT: Significant ({sig}) — Stage 3 effect confirmed.")
    else:
        lines.append(f"  RESULT: NOT significant ({sig}) — Stage 3 effect not confirmed.")

    lines.append("")
    lines.append(f"  {'Dimension':<28} {'SERAPH':>8} {'w/o S3':>8} "
                 f"{'Δ':>8} {'p':>8}")
    lines.append(f"  {'-'*60}")
    for dim, (s_m, n_m, d, p) in dim_results.items():
        sig_marker = " ***" if p < 0.001 else (" **" if p < 0.01 else
                     (" *" if p < 0.05 else ""))
        lines.append(f"  {dim:<28} {s_m:>8.4f} {n_m:>8.4f} "
                     f"{d:>+8.4f} {p:>8.4f}{sig_marker}")

    lines.append("")
    lines.append("-" * 65)
    lines.append("  CORRECTED TABLE VALUES:")
    lines.append(f"  SERAPH Full overall        : {seraph_sub['overall'].mean():.4f}")
    lines.append(f"  w/o Stage 3 overall        : {no3_sub['overall'].mean():.4f}")
    lines.append(f"  Delta                      : {d_obs:+.4f}")
    lines.append(f"  Previously reported delta  : -0.1277  (was 3-dim average — WRONG)")
    lines.append("")
    lines.append("  LaTeX sentences for paper:")

    s_mean  = seraph_sub["overall"].mean()
    n3_mean = no3_sub["overall"].mean()
    if p_val < 0.001:
        p_str = "p < 0.001"
    else:
        p_str = f"p = {p_val:.3f}"

    lines.append(
        f"  Removing \\stage{{3}} causes an empathy drop of "
        f"$\\Delta = {d_obs:.4f}$ ({s_mean:.4f} $\\to$ {n3_mean:.4f}), "
        f"significant at ${p_str}$ (paired bootstrap, "
        f"$n = {n_resamples:,}$ resamples)."
    )
    lines.append("")
    lines.append("  Updated table2_ablation row:")
    lines.append(
        f"  Full Pipeline  & 0.5211 & {s_mean:.4f} & [tone] & $+$0.0000 \\\\"
    )
    lines.append(
        f"  w/o Stage~3    & 0.5198 & {n3_mean:.4f} & [tone] & "
        f"${d_obs:.4f}$ \\\\"
    )
    lines.append("=" * 65)

    output = "\n".join(lines)
    print("\n" + output)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(output)
    logger.info("Results saved → %s", output_path)


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score w/o Stage 3 and run corrected bootstrap vs SERAPH Full"
    )
    parser.add_argument("--no3-json",
                        default="results/empathetic_dialogues_no_stage3.json")
    parser.add_argument("--seraph-csv",
                        default="results/seraph_ed_full_scores.csv")
    parser.add_argument("--no3-scores",
                        default="results/no3_ed_full_scores.csv",
                        help="Output CSV for w/o Stage 3 scores")
    parser.add_argument("--output",
                        default="results/bootstrap_full_vs_no3.txt")
    parser.add_argument("--checkpoint-dir",
                        default=".checkpoints/no3_scoring")
    parser.add_argument("--skip-scoring", action="store_true",
                        help="Load existing no3 scores CSV, skip batch API call")
    parser.add_argument("--n", type=int, default=10_000)
    args = parser.parse_args()

    load_dotenv(".env")
    api_key = os.environ.get("ANTHROPIC_API_KEY") or ANTHROPIC_API_KEY

    # Align datasets
    pairs = align_datasets(args.no3_json, args.seraph_csv)

    if args.skip_scoring and Path(args.no3_scores).exists():
        logger.info("Loading existing w/o Stage 3 scores from %s", args.no3_scores)
        no3_df = pd.read_csv(args.no3_scores)
    else:
        client = anthropic.Anthropic(api_key=api_key)
        cp_dir = Path(args.checkpoint_dir)
        cp_dir.mkdir(parents=True, exist_ok=True)

        logger.info("=== Scoring w/o Stage 3 responses (batch API) ===")
        scores_map = score_no3_batch(pairs, client, cp_dir)

        no3_df = build_scores_df(pairs, scores_map)
        no3_df.to_csv(args.no3_scores, index=False)
        logger.info("w/o Stage 3 scores saved → %s", args.no3_scores)

        means = no3_df[DIMS + ["overall"]].mean().round(4)
        logger.info("w/o Stage 3 means: %s", means.to_dict())

    logger.info("=== Running paired bootstrap (n=%d resamples) ===", args.n)
    print_and_save_results(
        pd.read_csv(args.seraph_csv), no3_df, pairs, args.n, args.output
    )


if __name__ == "__main__":
    main()
