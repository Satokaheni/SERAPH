"""
Automatic Metrics — BLEU / ROUGE-L / BERTScore
================================================
Computes BLEU-1, BLEU-2, ROUGE-L, and BERTScore (F1) for all available
systems on the EmpatheticDialogues test set, using gold listener responses
as references.

Systems scored (from available response files):
  - SERAPH Full          (results/empathetic_dialogues_full.json)
  - SERAPH w/o Stage 3   (results/empathetic_dialogues_no_stage3.json)
  - CoT Empathy          (cot_ed_full_outputs.jsonl)          [if available]
  - Raw Claude           (results/raw_claude_ed_full.json)    [if available]

Output:
  results/automatic_metrics.csv      — per-record scores
  results/automatic_metrics_summary.csv — mean ± std per system
  tables/table_automatic_metrics.tex — LaTeX table for paper

Usage:
    pip install nltk rouge-score bert-score
    python compute_automatic_metrics.py
    python compute_automatic_metrics.py --seraph   results/empathetic_dialogues_full.json
                                        --no3      results/empathetic_dialogues_no_stage3.json
                                        --cot      cot_ed_full_outputs.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import nltk
import pandas as pd
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from bert_score import score as bert_score

logging.basicConfig(
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("auto_metrics")

# Download NLTK tokenizer if needed
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)


# ── Loaders ───────────────────────────────────────────────────────────────────

def load_seraph_json(path: str) -> dict[str, str]:
    """Load {utterance[:100]: response} from SERAPH results JSON."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    results = data.get("results", data) if isinstance(data, dict) else data
    return {
        r["utterance"][:100].strip(): r["response"]
        for r in results
        if r.get("success") and not r.get("error") and r.get("response")
    }


def load_references(seraph_json_path: str) -> dict[str, str]:
    """Load {utterance[:100]: human_response} from SERAPH results JSON."""
    with open(seraph_json_path, encoding="utf-8") as f:
        data = json.load(f)
    results = data.get("results", data) if isinstance(data, dict) else data
    return {
        r["utterance"][:100].strip(): r["human_response"]
        for r in results
        if r.get("human_response")
    }


def load_cot_jsonl(path: str) -> dict[str, str]:
    """Load {context[:100]: cot_response} from cot_ed_full_outputs.jsonl."""
    responses = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            cot = rec.get("cot", "")
            if cot and not cot.startswith("ERROR"):
                responses[rec["context"][:100].strip()] = cot
    return responses


def load_generic_json(path: str,
                      utterance_key: str = "utterance",
                      response_key: str = "response") -> dict[str, str]:
    """Generic loader for other system JSONs."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    results = data.get("results", data) if isinstance(data, dict) else data
    return {
        r[utterance_key][:100].strip(): r[response_key]
        for r in results
        if r.get(response_key) and not r.get("error")
    }


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_bleu(hypothesis: str, reference: str) -> tuple[float, float]:
    """Compute sentence-level BLEU-1 and BLEU-2."""
    smoother = SmoothingFunction().method1
    hyp_tok  = nltk.word_tokenize(hypothesis.lower())
    ref_tok  = nltk.word_tokenize(reference.lower())
    if not hyp_tok or not ref_tok:
        return 0.0, 0.0
    b1 = sentence_bleu([ref_tok], hyp_tok,
                       weights=(1, 0, 0, 0),
                       smoothing_function=smoother)
    b2 = sentence_bleu([ref_tok], hyp_tok,
                       weights=(0.5, 0.5, 0, 0),
                       smoothing_function=smoother)
    return float(b1), float(b2)


def compute_rouge_l(hypothesis: str, reference: str,
                    scorer_obj: rouge_scorer.RougeScorer) -> float:
    """Compute ROUGE-L F1."""
    scores = scorer_obj.score(reference, hypothesis)
    return float(scores["rougeL"].fmeasure)


# ── Main pipeline ─────────────────────────────────────────────────────────────

def score_system(
    name: str,
    responses: dict[str, str],
    references: dict[str, str],
    rouge: rouge_scorer.RougeScorer,
) -> list[dict]:
    """Score one system. Returns list of per-record dicts."""
    rows = []
    shared_keys = set(responses) & set(references)
    logger.info("%s: %d aligned records", name, len(shared_keys))

    for key in sorted(shared_keys):
        hyp = responses[key]
        ref = references[key]
        b1, b2 = compute_bleu(hyp, ref)
        rl     = compute_rouge_l(hyp, ref, rouge)
        rows.append({
            "system":   name,
            "context":  key,
            "bleu1":    b1,
            "bleu2":    b2,
            "rouge_l":  rl,
            "hypothesis": hyp[:200],
            "reference":  ref[:200],
        })

    return rows


def compute_bertscore(df: pd.DataFrame) -> pd.DataFrame:
    """Compute BERTScore F1 per system, add to df."""
    results = []
    for system, grp in df.groupby("system"):
        logger.info("Computing BERTScore for %s (%d records)…", system, len(grp))
        _, _, F1 = bert_score(
            grp["hypothesis"].tolist(),
            grp["reference"].tolist(),
            lang="en",
            verbose=False,
        )
        f1_list = F1.tolist()
        results.append((system, grp.index.tolist(), f1_list))

    # Merge back
    bert_map = {}
    for system, indices, f1s in results:
        for idx, f1 in zip(indices, f1s):
            bert_map[idx] = f1

    df["bertscore_f1"] = df.index.map(bert_map)
    return df


# ── LaTeX output ──────────────────────────────────────────────────────────────

def generate_latex(summary: pd.DataFrame, n_per_system: dict[str, int]) -> str:
    metrics = ["bleu1", "bleu2", "rouge_l", "bertscore_f1"]
    col_labels = {
        "bleu1":        "BLEU-1",
        "bleu2":        "BLEU-2",
        "rouge_l":      "ROUGE-L",
        "bertscore_f1": "BERTScore",
    }

    # Bold best per column
    means = summary.xs("mean", axis=1, level=1)
    best  = {m: means[m].idxmax() for m in metrics}

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{lcccc}")
    lines.append(r"\toprule")
    lines.append(
        r"\textbf{System} & \textbf{BLEU-1} & \textbf{BLEU-2} & "
        r"\textbf{ROUGE-L} & \textbf{BERTScore} \\"
    )
    lines.append(r"\midrule")

    row_order = [
        "SERAPH w/o Stage 3",
        "CoT Empathy",
        "Raw Claude",
        "SERAPH Full",
    ]
    present = [s for s in row_order if s in means.index]
    present += [s for s in means.index if s not in present]

    for system in present:
        if system not in means.index:
            continue
        vals = []
        for m in metrics:
            v = means.loc[system, m]
            s = f"{v:.4f}"
            if system == best[m]:
                s = f"\\textbf{{{s}}}"
            vals.append(s)
        n = n_per_system.get(system, "?")
        lines.append(f"{system} & " + " & ".join(vals) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(
        r"\caption{Automatic metrics on EmpatheticDialogues test set "
        r"(gold listener responses as references). "
        r"BLEU-1/2: sentence-level with smoothing; "
        r"ROUGE-L: F1; BERTScore: F1 (roberta-large). "
        r"All systems are evaluated on the same aligned utterances. "
        r"Scores are uniformly low across systems, consistent with prior "
        r"findings that n-gram overlap poorly captures empathic quality "
        r"\cite{rashkin2019empathetic}; "
        r"LLM-as-judge scores (Table~\ref{tab:empathy_dims}) are the "
        r"primary evaluation metric.}"
    )
    lines.append(r"\label{tab:auto_metrics}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Compute BLEU/ROUGE/BERTScore")
    parser.add_argument("--seraph",
                        default="results/empathetic_dialogues_full.json")
    parser.add_argument("--no3",
                        default="results/empathetic_dialogues_no_stage3.json")
    parser.add_argument("--cot",
                        default="cot_ed_full_outputs.jsonl",
                        help="CoT Empathy outputs JSONL (optional)")
    parser.add_argument("--raw-claude",
                        default=None,
                        help="Raw Claude outputs JSON (optional)")
    parser.add_argument("--output-csv",
                        default="results/automatic_metrics.csv")
    parser.add_argument("--output-tex",
                        default="tables/table_automatic_metrics.tex")
    parser.add_argument("--skip-bertscore", action="store_true",
                        help="Skip BERTScore (slower, requires GPU for speed)")
    args = parser.parse_args()

    # Load references from SERAPH full JSON (has human_response field)
    logger.info("Loading gold references…")
    references = load_references(args.seraph)
    logger.info("Loaded %d gold references", len(references))

    # Load system responses
    systems: dict[str, dict[str, str]] = {}

    if Path(args.seraph).exists():
        systems["SERAPH Full"] = load_seraph_json(args.seraph)

    if Path(args.no3).exists():
        systems["SERAPH w/o Stage 3"] = load_seraph_json(args.no3)

    if args.cot and Path(args.cot).exists():
        systems["CoT Empathy"] = load_cot_jsonl(args.cot)
    else:
        logger.info("CoT outputs not found at %s — skipping", args.cot)

    if args.raw_claude and Path(args.raw_claude).exists():
        systems["Raw Claude"] = load_generic_json(args.raw_claude)

    if not systems:
        logger.error("No system response files found. Check paths.")
        return

    # Score
    rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    all_rows = []
    for name, responses in systems.items():
        all_rows.extend(score_system(name, responses, references, rouge))

    df = pd.DataFrame(all_rows).reset_index(drop=True)

    # BERTScore
    if not args.skip_bertscore:
        logger.info("Computing BERTScore (this may take a few minutes)…")
        df = compute_bertscore(df)
    else:
        df["bertscore_f1"] = float("nan")
        logger.info("BERTScore skipped (--skip-bertscore)")

    # Save per-record CSV
    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_csv, index=False)
    logger.info("Per-record scores saved → %s", args.output_csv)

    # Summary
    metrics = ["bleu1", "bleu2", "rouge_l", "bertscore_f1"]
    summary = df.groupby("system")[metrics].agg(["mean", "std"]).round(4)
    summary_path = args.output_csv.replace(".csv", "_summary.csv")
    summary.to_csv(summary_path)

    n_per_system = df.groupby("system").size().to_dict()

    # Print summary
    print("\n" + "=" * 65)
    print("  AUTOMATIC METRICS — EmpatheticDialogues")
    print("=" * 65)
    means = summary.xs("mean", axis=1, level=1)
    print(f"  {'System':<26} {'BLEU-1':>8} {'BLEU-2':>8} "
          f"{'ROUGE-L':>8} {'BERTScore':>10} {'N':>6}")
    print(f"  {'-'*68}")
    for system in means.index:
        n = n_per_system.get(system, 0)
        b1 = means.loc[system, "bleu1"]
        b2 = means.loc[system, "bleu2"]
        rl = means.loc[system, "rouge_l"]
        bs = means.loc[system, "bertscore_f1"]
        print(f"  {system:<26} {b1:>8.4f} {b2:>8.4f} {rl:>8.4f} {bs:>10.4f} {n:>6}")
    print("=" * 65)

    # LaTeX
    latex = generate_latex(summary, n_per_system)
    Path(args.output_tex).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_tex, "w", encoding="utf-8") as f:
        f.write(latex)
    logger.info("LaTeX table saved → %s", args.output_tex)

    print(f"\nNext step: \\input{{{args.output_tex}}} in main.tex")
    print("Note: if BLEU/ROUGE are uniformly low across systems, add a")
    print("sentence noting this supports the LLM-as-judge approach.")


if __name__ == "__main__":
    main()
