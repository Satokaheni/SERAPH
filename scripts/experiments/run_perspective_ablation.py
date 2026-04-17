"""
Matched-Compute Ablation: First-Person vs. Third-Person Stage 3
================================================================
Tests whether SERAPH's empathy gains come from the *first-person perspective
frame specifically* or just from *having an additional reasoning step*.

Design:
  - Two conditions run on the same contexts, fully paired:
      first_person  — original Stage 3 (model adopts user's perspective)
      third_person  — matched-compute Stage 3 (analytical, never first-person)
  - Both conditions use identical inputs and identical Stage 5 generator
  - Token counts are compared to confirm compute is matched
  - Scored by the same 5-dimension LLM-as-judge protocol
  - Paired bootstrap (n=10,000) on overall + each dimension

Key design constraint:
  The third-person prompt is NOT a strawman. It covers the same psychological
  dimensions (emotion regulation, core needs, tone) at comparable depth.
  The ONLY difference is perspective frame. If first-person wins on
  Perspective-Taking specifically, that isolates the mechanism.

Input JSONL format (one record per line):
  {"context": "...", "emotion": "...", "appraisal": "..."}

  If 'appraisal' is not available, the script will generate it from
  the full SERAPH pipeline's Stage 2 output automatically.

Output:
  results/perspective_ablation_outputs.jsonl   — raw responses
  results/perspective_ablation_scores.csv      — per-record judge scores
  results/perspective_ablation_bootstrap.txt   — bootstrap results + LaTeX

Usage:
    python run_perspective_ablation.py --input moel_contexts_400.txt
    python run_perspective_ablation.py --input comparison_sample.json
    python run_perspective_ablation.py --input my_contexts.jsonl --n-samples 200
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
from typing import Optional

import anthropic
import numpy as np
import pandas as pd
from dotenv import load_dotenv

from config import ANTHROPIC_API_KEY, SONNET_MODEL, HAIKU_MODEL, MAX_TOKENS, TEMPERATURE, USE_PROMPT_CACHING
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
logger = logging.getLogger("perspective_ablation")

DIMS = ["emotional_acknowledgment", "perspective_taking",
        "tone_match", "avoids_harm", "meets_core_need"]

# Wave numbers — avoid collision with main pipeline (1-5) and other scripts (97-99)
WAVE_STAGE2    = 11   # Stage 2 (shared between conditions)
WAVE_FP_S3     = 12   # First-person Stage 3
WAVE_TP_S3     = 13   # Third-person Stage 3
WAVE_FP_S5     = 14   # First-person Stage 5 (response generation)
WAVE_TP_S5     = 15   # Third-person Stage 5
WAVE_FP_JUDGE  = 16   # Judge scores for first-person
WAVE_TP_JUDGE  = 17   # Judge scores for third-person


# ═══════════════════════════════════════════════════════════════════════════════
# PROMPTS
# ═══════════════════════════════════════════════════════════════════════════════

# ── Stage 2 system prompt (shared) ────────────────────────────────────────────
STAGE2_SYSTEM = """You are an expert emotion analyst. Given a dialogue utterance, produce a structured appraisal.

Return a JSON object with these fields:
{
  "primary_emotion": "<Plutchik primary: joy/trust/fear/surprise/sadness/disgust/anger/anticipation>",
  "intensity": <0.0-1.0>,
  "regulation_state": "<unregulated|reappraisal|suppression|seeking_support>",
  "bdi": {
    "belief": "<what the person believes about their situation>",
    "desire": "<what they want>",
    "intention": "<what they intend to do or want from this conversation>"
  },
  "context_type": "<brief situational description>",
  "is_ambiguous": <true|false>
}

Return ONLY the JSON object. No markdown, no explanation."""

# ── Stage 3 — FIRST-PERSON simulation (original) ──────────────────────────────
STAGE3_FP_SYSTEM = """You are performing a first-person empathy simulation.
Your output will be used to guide empathic response generation.
Return a JSON object with these fields:
{
  "perspective": "<first-person narrative: how you feel, what you need, what would land well>",
  "core_need": "<the single most important thing you need to hear right now>",
  "feared_responses": ["<response type that would feel dismissive or harmful>"],
  "helpful_responses": ["<response type that would feel genuinely supportive>"],
  "tone_recommendation": "<specific tone guidance for the responder>",
  "avoid_specifically": ["<specific phrases or approaches to avoid>"],
  "regulation_note": "<how your current regulation state should affect the response>",
  "token_count_check": "<write 'matched' here>"
}
Return ONLY the JSON object. No markdown."""

STAGE3_FP_USER = """Perform a first-person perspective simulation for this person.

Emotion: {emotion}
Intensity: {intensity}
Context: {context}
Appraisal (BDI + regulation state): {appraisal}

Adopt their first-person perspective completely. As if you ARE this person:
1. How am I feeling right now? What is the texture of this emotion?
2. What triggered this feeling? What appraisals am I making about my situation?
3. What do I most need to hear from someone right now?
4. What kind of response would feel genuinely understanding vs. dismissive or hollow?
5. If someone said something to me right now, what tone would land best? What words would feel authentic vs. performative?

Respond in the JSON format specified. The 'perspective' field should be a rich first-person account grounded in the actual situation — not generic. Be specific about this person's exact situation."""

# ── Stage 3 — THIRD-PERSON analytical reasoning (matched compute) ─────────────
# Carefully designed to match cognitive depth without first-person framing.
# Covers identical dimensions: emotion texture, BDI, regulation, tone, needs.
# NEVER uses "If I were...", "I would feel...", "imagining myself..."
STAGE3_TP_SYSTEM = """You are performing a third-person analytical empathy assessment.
Your output will be used to guide empathic response generation.
Return a JSON object with these fields:
{
  "analysis": "<third-person analytical account: the user's emotional state, what drives it, what they need>",
  "core_need": "<the single most important thing this person needs to hear right now>",
  "feared_responses": ["<response type that would feel dismissive or harmful to this person>"],
  "helpful_responses": ["<response type that would feel genuinely supportive to this person>"],
  "tone_recommendation": "<specific tone guidance for the responder>",
  "avoid_specifically": ["<specific phrases or approaches to avoid with this person>"],
  "regulation_note": "<how their current regulation state should affect the response>",
  "token_count_check": "<write 'matched' here>"
}
Return ONLY the JSON object. No markdown."""

STAGE3_TP_USER = """Perform a third-person analytical assessment of this person's emotional state and needs.

Emotion: {emotion}
Intensity: {intensity}
Context: {context}
Appraisal (BDI + regulation state): {appraisal}

Analyse their situation from an expert third-person perspective:
1. What is the emotional experience of this person? What is the specific texture and quality of what they are feeling?
2. What triggered this emotional state? What appraisals are they making about their situation?
3. What does this person most need to hear from a responder right now, given their specific situation?
4. What kinds of responses would feel genuinely supportive vs. dismissive or harmful to this specific person?
5. What tone, register, and approach would best serve this person's emotional needs? What should be specifically avoided?

Respond in the JSON format specified. The 'analysis' field should be a thorough analytical account grounded in this person's exact situation — not generic. Be specific about the psychological dynamics at play for this individual."""

# ── Stage 5 system prompt (shared) ────────────────────────────────────────────
STAGE5_SYSTEM = """You are an empathic conversational assistant.
Generate a single empathic response to the person's message.
Use the simulation/analysis output to calibrate your response.
Return a JSON object:
{"response": "<your empathic response>", "tone_used": "<brief description of tone>"}
Return ONLY the JSON object. No markdown."""

STAGE5_USER = """Generate an empathic response to this person.

Their message: {context}

Simulation/Analysis output: {stage3_output}

Generate a response that addresses their core need, matches the recommended tone,
and avoids the identified harmful patterns. Be specific to their situation."""

# ── Judge prompt (shared) ──────────────────────────────────────────────────────
JUDGE_SYSTEM = """You are an expert evaluator of empathic dialogue responses.
Return ONLY a JSON object — no markdown, no explanation:
{"emotional_acknowledgment": X, "perspective_taking": X, "tone_match": X, "avoids_harm": X, "meets_core_need": X}
where X is an integer 1-5.

1=Poor 2=Below average 3=Adequate 4=Good 5=Excellent

- emotional_acknowledgment: explicitly recognises the speaker's emotional state
- perspective_taking: demonstrates genuine first-person understanding of how the speaker feels
- tone_match: tone matches the speaker's emotional register
- avoids_harm: avoids minimising, victim-blaming, or dysregulating content
- meets_core_need: addresses what the speaker most needs right now"""

JUDGE_USER = """Rate this empathic response.

Dialogue context: {context}

Response: {response}"""


# ═══════════════════════════════════════════════════════════════════════════════
# INPUT LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def load_contexts(path: str, n_samples: Optional[int], seed: int) -> list[dict]:
    """
    Load contexts from various formats. Returns list of
    {'idx': int, 'context': str} dicts.
    Supports: .txt (one per line), .jsonl, .json (list or comparison_sample format)
    """
    p = Path(path)
    records = []

    if p.suffix == ".txt":
        with open(p, encoding="utf-8") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if line:
                    records.append({"idx": i, "context": line})

    elif p.suffix == ".jsonl":
        with open(p, encoding="utf-8") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                ctx = rec.get("context") or rec.get("utterance") or ""
                if ctx:
                    records.append({"idx": i, "context": ctx})

    elif p.suffix == ".json":
        data = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(data, list):
            for i, rec in enumerate(data):
                ctx = rec.get("context") or rec.get("utterance") or ""
                if ctx:
                    records.append({"idx": i, "context": ctx})
        elif isinstance(data, dict) and "results" in data:
            for i, rec in enumerate(data["results"]):
                if rec.get("success") and not rec.get("error"):
                    records.append({"idx": i, "context": rec["utterance"]})

    logger.info("Loaded %d contexts from %s", len(records), path)

    if n_samples and n_samples < len(records):
        rng = np.random.default_rng(seed)
        indices = rng.choice(len(records), size=n_samples, replace=False)
        indices.sort()
        records = [records[i] for i in indices]
        logger.info("Sampled %d contexts (seed=%d)", len(records), seed)

    return records


# ═══════════════════════════════════════════════════════════════════════════════
# BATCH REQUEST BUILDERS
# ═══════════════════════════════════════════════════════════════════════════════

def _sys(prompt: str) -> list[dict] | str:
    if USE_PROMPT_CACHING:
        return [{"type": "text", "text": prompt,
                 "cache_control": {"type": "ephemeral"}}]
    return prompt


def build_stage2_request(idx: int, context: str) -> dict:
    return {
        "custom_id": f"s2_{idx:05d}",
        "params": {
            "model": HAIKU_MODEL,
            "max_tokens": MAX_TOKENS.get("stage2", 512),
            "temperature": TEMPERATURE.get("stage2", 0.3),
            "system": _sys(STAGE2_SYSTEM),
            "messages": [{"role": "user", "content": f"Utterance:\n{context}"}],
        },
    }


def build_stage3_fp_request(idx: int, context: str, appraisal: dict) -> dict:
    emotion    = appraisal.get("primary_emotion", "unknown")
    intensity  = appraisal.get("intensity", 0.5)
    appr_str   = json.dumps(appraisal)
    return {
        "custom_id": f"fp_{idx:05d}",
        "params": {
            "model": SONNET_MODEL,
            "max_tokens": MAX_TOKENS.get("stage3", 1024),
            "temperature": TEMPERATURE.get("stage3", 0.7),
            "system": _sys(STAGE3_FP_SYSTEM),
            "messages": [{"role": "user", "content": STAGE3_FP_USER.format(
                emotion=emotion, intensity=intensity,
                context=context, appraisal=appr_str,
            )}],
        },
    }


def build_stage3_tp_request(idx: int, context: str, appraisal: dict) -> dict:
    emotion    = appraisal.get("primary_emotion", "unknown")
    intensity  = appraisal.get("intensity", 0.5)
    appr_str   = json.dumps(appraisal)
    return {
        "custom_id": f"tp_{idx:05d}",
        "params": {
            "model": SONNET_MODEL,
            "max_tokens": MAX_TOKENS.get("stage3", 1024),
            "temperature": TEMPERATURE.get("stage3", 0.7),
            "system": _sys(STAGE3_TP_SYSTEM),
            "messages": [{"role": "user", "content": STAGE3_TP_USER.format(
                emotion=emotion, intensity=intensity,
                context=context, appraisal=appr_str,
            )}],
        },
    }


def build_stage5_request(custom_id: str, context: str, stage3_output: dict) -> dict:
    return {
        "custom_id": custom_id,
        "params": {
            "model": SONNET_MODEL,
            "max_tokens": MAX_TOKENS.get("stage5", 512),
            "temperature": TEMPERATURE.get("stage5", 0.7),
            "system": _sys(STAGE5_SYSTEM),
            "messages": [{"role": "user", "content": STAGE5_USER.format(
                context=context,
                stage3_output=json.dumps(stage3_output),
            )}],
        },
    }


def build_judge_request(custom_id: str, context: str, response: str) -> dict:
    return {
        "custom_id": custom_id,
        "params": {
            "model": HAIKU_MODEL,
            "max_tokens": 256,
            "temperature": 0.0,
            "system": _sys(JUDGE_SYSTEM),
            "messages": [{"role": "user", "content": JUDGE_USER.format(
                context=context, response=response,
            )}],
        },
    }


# ═══════════════════════════════════════════════════════════════════════════════
# WAVE RUNNERS
# ═══════════════════════════════════════════════════════════════════════════════

def run_wave(
    requests: list[dict],
    client: anthropic.Anthropic,
    cp_dir: Path,
    wave_num: int,
    label: str,
) -> dict:
    cp     = _checkpoint_path(cp_dir, wave_num)
    cached = _load_checkpoint(cp)
    if cached:
        logger.info("Wave %d (%s): loaded from checkpoint (%d records)",
                    wave_num, label, len(cached))
        return cached
    logger.info("Wave %d (%s): submitting %d requests…",
                wave_num, label, len(requests))
    batch_id = submit_batch(requests, client)
    results  = poll_until_complete(batch_id, client)
    _save_checkpoint(cp, results)
    return results


def extract(results: dict, prefix: str) -> dict[int, dict]:
    """Extract {idx: data} from batch results, stripping the custom_id prefix."""
    out = {}
    for cid, result in results.items():
        idx = int(cid.replace(prefix, ""))
        if result.get("ok"):
            out[idx] = result.get("data", {})
        else:
            logger.warning("Record %d failed: %s", idx, result.get("error"))
    return out


def extract_text(results: dict, prefix: str) -> dict[int, str]:
    """Extract plain text responses (for stage5 / judge where we want the string)."""
    out = {}
    for cid, result in results.items():
        idx = int(cid.replace(prefix, ""))
        if result.get("ok"):
            data = result.get("data", {})
            raw  = result.get("raw", "")
            text = data.get("response") or raw or ""
            out[idx] = text.strip()
        else:
            logger.warning("Record %d failed: %s", idx, result.get("error"))
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# BOOTSTRAP
# ═══════════════════════════════════════════════════════════════════════════════

def paired_bootstrap(
    a: np.ndarray,
    b: np.ndarray,
    n: int = 10_000,
    seed: int = 42,
) -> tuple[float, float, float]:
    rng     = np.random.default_rng(seed)
    diff    = a - b
    d_obs   = diff.mean()
    centred = diff - d_obs
    boot    = np.array([
        rng.choice(centred, size=len(centred), replace=True).mean()
        for _ in range(n)
    ])
    p = (np.abs(boot) >= np.abs(d_obs)).mean()
    return float(d_obs), float(p), float(boot.std())


# ═══════════════════════════════════════════════════════════════════════════════
# REPORTING
# ═══════════════════════════════════════════════════════════════════════════════

def report(
    fp_df: pd.DataFrame,
    tp_df: pd.DataFrame,
    outputs: list[dict],
    n_resamples: int,
    output_path: str,
) -> None:
    # Align on shared indices
    shared = set(fp_df["idx"]) & set(tp_df["idx"])
    fp = fp_df[fp_df["idx"].isin(shared)].set_index("idx").sort_index()
    tp = tp_df[tp_df["idx"].isin(shared)].set_index("idx").sort_index()
    n  = len(fp)

    # Token count comparison
    fp_tokens = [r["fp_stage3_tokens"] for r in outputs
                 if r.get("fp_stage3_tokens") and r["idx"] in shared]
    tp_tokens = [r["tp_stage3_tokens"] for r in outputs
                 if r.get("tp_stage3_tokens") and r["idx"] in shared]

    lines = []
    lines.append("=" * 68)
    lines.append("  PERSPECTIVE ABLATION: First-Person vs. Third-Person Stage 3")
    lines.append("=" * 68)
    lines.append(f"  N aligned pairs      : {n}")
    lines.append(f"  N resamples          : {n_resamples:,}")
    if fp_tokens and tp_tokens:
        lines.append(f"  Stage 3 token count  : FP={np.mean(fp_tokens):.0f} "
                     f"TP={np.mean(tp_tokens):.0f} "
                     f"(Δ={np.mean(fp_tokens)-np.mean(tp_tokens):+.0f})")
    lines.append("")

    # Overall
    d_obs, p_val, _ = paired_bootstrap(
        fp["overall"].values, tp["overall"].values, n_resamples
    )
    lines.append(f"  {'Dimension':<28} {'FP':>8} {'TP':>8} {'Δ(FP-TP)':>10} {'p':>8}")
    lines.append(f"  {'-'*64}")

    all_dims = DIMS + ["overall"]
    for dim in all_dims:
        d, p, _ = paired_bootstrap(
            fp[dim].values, tp[dim].values, n_resamples
        )
        sig = " ***" if p < 0.001 else (" **" if p < 0.01 else
              (" *"  if p < 0.05 else ""))
        lines.append(
            f"  {dim:<28} {fp[dim].mean():>8.4f} {tp[dim].mean():>8.4f} "
            f"{d:>+10.4f} {p:>8.4f}{sig}"
        )

    lines.append("")
    lines.append("  (* p<0.05  ** p<0.01  *** p<0.001)")
    lines.append("")

    # Interpretation
    if p_val > 0.05:
        lines.append("  OVERALL: Not significant — first-person framing does not")
        lines.append("  significantly outperform matched third-person reasoning.")
        lines.append("  → The gain may come from compute/depth, not perspective frame.")
    else:
        winner = "First-person" if d_obs > 0 else "Third-person"
        lines.append(f"  OVERALL: Significant (p={p_val:.4f}) — {winner} wins.")
        if d_obs > 0:
            lines.append("  → First-person perspective framing is the active ingredient.")

    # Check Perspective-Taking specifically — the theoretically motivated dimension
    pt_d, pt_p, _ = paired_bootstrap(
        fp["perspective_taking"].values,
        tp["perspective_taking"].values,
        n_resamples,
    )
    lines.append("")
    lines.append(f"  KEY DIMENSION — Perspective-Taking:")
    lines.append(f"  FP={fp['perspective_taking'].mean():.4f} "
                 f"TP={tp['perspective_taking'].mean():.4f} "
                 f"Δ={pt_d:+.4f} p={pt_p:.4f}")
    if pt_p < 0.05 and pt_d > 0:
        lines.append("  ✓ FP significantly leads on Perspective-Taking —")
        lines.append("    strong evidence for the first-person mechanism.")
    elif pt_p > 0.05:
        lines.append("  ✗ No significant difference on Perspective-Taking —")
        lines.append("    first-person framing may not be the active ingredient.")

    # LaTeX
    lines.append("")
    lines.append("-" * 68)
    lines.append("  LaTeX rows for ablation table:")
    fp_vals = " & ".join(f"{fp[d].mean():.4f}" for d in all_dims)
    tp_vals = " & ".join(f"{tp[d].mean():.4f}" for d in all_dims)
    lines.append(f"  Stage 3 (first-person)  & {fp_vals} \\\\")
    lines.append(f"  Stage 3 (third-person)  & {tp_vals} \\\\")
    lines.append("")

    fp_mean = fp["overall"].mean()
    tp_mean = tp["overall"].mean()
    p_str   = "p < 0.001" if p_val < 0.001 else f"p = {p_val:.3f}"
    sig_str = "significantly" if p_val < 0.05 else "not significantly"
    lines.append("  LaTeX sentence:")
    lines.append(
        f"  Replacing \\stage{{3}}'s first-person simulation with a "
        f"matched-compute third-person analytical step yields an overall "
        f"empathy score of {tp_mean:.4f} vs.\\ {fp_mean:.4f}, a difference "
        f"that is {sig_str} significant "
        f"($\\Delta = {d_obs:+.4f}$, ${p_str}$, paired bootstrap, "
        f"$n = {n_resamples:,}$ resamples)."
    )
    lines.append("=" * 68)

    output = "\n".join(lines)
    print("\n" + output)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(output)
    logger.info("Bootstrap results saved → %s", output_path)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="First-person vs. third-person Stage 3 ablation"
    )
    parser.add_argument("--input",
                        default="moel_contexts_400.txt",
                        help="Input contexts (.txt, .jsonl, or .json)")
    parser.add_argument("--n-samples",  type=int, default=None,
                        help="Subsample N contexts (default: use all)")
    parser.add_argument("--seed",       type=int, default=42)
    parser.add_argument("--checkpoint-dir", default=".checkpoints/perspective_ablation")
    parser.add_argument("--outputs",    default="results/perspective_ablation_outputs.jsonl")
    parser.add_argument("--scores",     default="results/perspective_ablation_scores.csv")
    parser.add_argument("--bootstrap",  default="results/perspective_ablation_bootstrap.txt")
    parser.add_argument("--n-resamples", type=int, default=10_000)
    parser.add_argument("--skip-to-bootstrap", action="store_true",
                        help="Skip generation/scoring, run bootstrap on existing scores CSV")
    args = parser.parse_args()

    load_dotenv(".env")
    api_key = os.environ.get("ANTHROPIC_API_KEY") or ANTHROPIC_API_KEY
    client  = anthropic.Anthropic(api_key=api_key)
    cp_dir  = Path(args.checkpoint_dir)
    cp_dir.mkdir(parents=True, exist_ok=True)

    # ── Bootstrap only ────────────────────────────────────────────────────
    if args.skip_to_bootstrap:
        if not Path(args.scores).exists():
            logger.error("Scores CSV not found: %s", args.scores)
            return
        df     = pd.read_csv(args.scores)
        fp_df  = df[df["condition"] == "first_person"].copy()
        tp_df  = df[df["condition"] == "third_person"].copy()
        report(fp_df, tp_df, [], args.n_resamples, args.bootstrap)
        return

    # ── Load contexts ─────────────────────────────────────────────────────
    records = load_contexts(args.input, args.n_samples, args.seed)
    logger.info("Running ablation on %d contexts", len(records))

    # ── Wave 1: Stage 2 (shared appraisal for both conditions) ───────────
    logger.info("=== Wave: Stage 2 (appraisal) ===")
    s2_requests = [build_stage2_request(r["idx"], r["context"]) for r in records]
    s2_results  = run_wave(s2_requests, client, cp_dir, WAVE_STAGE2, "stage2")
    s2_map      = extract(s2_results, "s2_")
    logger.info("Stage 2 complete: %d / %d succeeded", len(s2_map), len(records))

    # ── Wave 2a: Stage 3 first-person ─────────────────────────────────────
    logger.info("=== Wave: Stage 3 — First-Person ===")
    fp_requests = [
        build_stage3_fp_request(r["idx"], r["context"], s2_map.get(r["idx"], {}))
        for r in records if r["idx"] in s2_map
    ]
    fp_s3_results = run_wave(fp_requests, client, cp_dir, WAVE_FP_S3, "fp_stage3")
    fp_s3_map     = extract(fp_s3_results, "fp_")

    # ── Wave 2b: Stage 3 third-person ─────────────────────────────────────
    logger.info("=== Wave: Stage 3 — Third-Person ===")
    tp_requests = [
        build_stage3_tp_request(r["idx"], r["context"], s2_map.get(r["idx"], {}))
        for r in records if r["idx"] in s2_map
    ]
    tp_s3_results = run_wave(tp_requests, client, cp_dir, WAVE_TP_S3, "tp_stage3")
    tp_s3_map     = extract(tp_s3_results, "tp_")

    # ── Wave 3a: Stage 5 first-person ─────────────────────────────────────
    logger.info("=== Wave: Stage 5 — First-Person responses ===")
    fp5_requests = [
        build_stage5_request(f"fp5_{r['idx']:05d}", r["context"],
                             fp_s3_map.get(r["idx"], {}))
        for r in records if r["idx"] in fp_s3_map
    ]
    fp5_results = run_wave(fp5_requests, client, cp_dir, WAVE_FP_S5, "fp_stage5")
    fp5_map     = extract_text(fp5_results, "fp5_")

    # ── Wave 3b: Stage 5 third-person ─────────────────────────────────────
    logger.info("=== Wave: Stage 5 — Third-Person responses ===")
    tp5_requests = [
        build_stage5_request(f"tp5_{r['idx']:05d}", r["context"],
                             tp_s3_map.get(r["idx"], {}))
        for r in records if r["idx"] in tp_s3_map
    ]
    tp5_results = run_wave(tp5_requests, client, cp_dir, WAVE_TP_S5, "tp_stage5")
    tp5_map     = extract_text(tp5_results, "tp5_")

    # ── Save raw outputs ───────────────────────────────────────────────────
    outputs = []
    for r in records:
        idx = r["idx"]
        fp_s3 = fp_s3_map.get(idx, {})
        tp_s3 = tp_s3_map.get(idx, {})
        fp_resp = fp5_map.get(idx, "")
        tp_resp = tp5_map.get(idx, "")

        # Rough token count proxy: word count * 1.3
        fp_tokens = int(len(json.dumps(fp_s3).split()) * 1.3) if fp_s3 else 0
        tp_tokens = int(len(json.dumps(tp_s3).split()) * 1.3) if tp_s3 else 0

        outputs.append({
            "idx":               idx,
            "context":           r["context"],
            "fp_stage3_output":  fp_s3,
            "tp_stage3_output":  tp_s3,
            "fp_response":       fp_resp,
            "tp_response":       tp_resp,
            "fp_stage3_tokens":  fp_tokens,
            "tp_stage3_tokens":  tp_tokens,
        })

    Path(args.outputs).parent.mkdir(parents=True, exist_ok=True)
    with open(args.outputs, "w", encoding="utf-8") as f:
        for o in outputs:
            f.write(json.dumps(o, ensure_ascii=False) + "\n")
    logger.info("Raw outputs saved → %s", args.outputs)

    # ── Wave 4a: Judge first-person responses ──────────────────────────────
    logger.info("=== Wave: Judge — First-Person ===")
    fpj_requests = [
        build_judge_request(f"fpj_{r['idx']:05d}", r["context"],
                            fp5_map.get(r["idx"], ""))
        for r in records if r["idx"] in fp5_map and fp5_map.get(r["idx"])
    ]
    fpj_results = run_wave(fpj_requests, client, cp_dir, WAVE_FP_JUDGE, "fp_judge")
    fpj_map     = extract(fpj_results, "fpj_")

    # ── Wave 4b: Judge third-person responses ─────────────────────────────
    logger.info("=== Wave: Judge — Third-Person ===")
    tpj_requests = [
        build_judge_request(f"tpj_{r['idx']:05d}", r["context"],
                            tp5_map.get(r["idx"], ""))
        for r in records if r["idx"] in tp5_map and tp5_map.get(r["idx"])
    ]
    tpj_results = run_wave(tpj_requests, client, cp_dir, WAVE_TP_JUDGE, "tp_judge")
    tpj_map     = extract(tpj_results, "tpj_")

    # ── Build scores DataFrame ─────────────────────────────────────────────
    rows = []
    for r in records:
        idx = r["idx"]
        for condition, scores_map in [("first_person", fpj_map),
                                      ("third_person", tpj_map)]:
            sc = scores_map.get(idx)
            if sc is None:
                continue
            if not all(k in sc for k in DIMS):
                continue
            sc["overall"] = round(sum(sc[k] for k in DIMS) / len(DIMS), 4)
            rows.append({
                "idx":       idx,
                "condition": condition,
                "context":   r["context"][:200],
                **{k: sc[k] for k in DIMS},
                "overall":   sc["overall"],
            })

    df = pd.DataFrame(rows)
    df.to_csv(args.scores, index=False)
    logger.info("Scores saved → %s", args.scores)

    # Print quick summary
    means = df.groupby("condition")[DIMS + ["overall"]].mean().round(4)
    print("\n" + "=" * 60)
    print("  QUICK SUMMARY")
    print("=" * 60)
    print(means.to_string())

    # ── Bootstrap ──────────────────────────────────────────────────────────
    fp_df = df[df["condition"] == "first_person"].copy()
    tp_df = df[df["condition"] == "third_person"].copy()
    report(fp_df, tp_df, outputs, args.n_resamples, args.bootstrap)


if __name__ == "__main__":
    main()
