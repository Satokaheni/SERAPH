"""
score_empathy.py
================
Runs the SERAPH LLM-as-judge empathy scorer on generated outputs from
MoEL, MIME, CEM, Raw Claude, and SERAPH, producing a unified comparison
table suitable for inclusion in the EMNLP paper.

Usage
-----
    pip install anthropic tqdm pandas python-dotenv

    python score_empathy.py \
        --contexts   data/empathetic_dialogues_test_contexts.txt \
        --seraph     data/seraph_outputs.txt \
        --moel       data/moel_generation_result.txt \
        --mime       data/mime_output.txt \
        --cem        data/cem_results.txt \
        --raw_claude data/raw_claude_outputs.txt \
        --output     results/empathy_comparison.csv \
        --n_samples  300 \
        # API key loaded automatically from .env

Input file format (one entry per line)
---------------------------------------
contexts file  : one dialogue context per line (the input utterance(s))
system files   : one generated response per line, aligned with contexts

Output
------
results/empathy_comparison.csv  — per-sample scores for all systems
results/empathy_summary.csv     — mean ± std per system per dimension
Prints a LaTeX table to stdout ready to paste into the paper.
"""

import argparse
import json
import os
import time
import random
from pathlib import Path

import anthropic
from dotenv import load_dotenv
import pandas as pd

# Load API key from .env in project root
load_dotenv(".env")
from tqdm import tqdm


# ── Judge prompt ──────────────────────────────────────────────────────────────

JUDGE_SYSTEM = """You are an expert evaluator of empathic dialogue responses.
You will be given a dialogue context (what a person said) and a system response.
Rate the response on five dimensions using a 1-5 Likert scale.

Return ONLY a valid JSON object with these exact keys:
{
  "emotional_acknowledgment": <1-5>,
  "perspective_taking": <1-5>,
  "tone_match": <1-5>,
  "avoids_harm": <1-5>,
  "meets_core_need": <1-5>
}

Scoring guide:
1 = Poor  2 = Below average  3 = Adequate  4 = Good  5 = Excellent

Definitions:
- emotional_acknowledgment: Does the response explicitly recognise the speaker's emotional state?
- perspective_taking: Does it demonstrate genuine first-person understanding of how the speaker feels?
- tone_match: Does the tone match the speaker's emotional register (not too clinical, not over-the-top)?
- avoids_harm: Does it avoid minimising, victim-blaming, unsolicited advice, or dysregulating content?
- meets_core_need: Does it address what the speaker most needs right now?

Return ONLY the JSON object. No explanation, no markdown."""

JUDGE_USER_TEMPLATE = """Dialogue context:
{context}

System response:
{response}"""


# ── Scoring function ──────────────────────────────────────────────────────────

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


def score_response(client: anthropic.Anthropic,
                   context: str,
                   response: str,
                   retries: int = 3) -> dict | None:
    """Score a single response using claude-haiku as judge."""
    text = ""
    prompt = JUDGE_TEMPLATE.replace("CONTEXT_PLACEHOLDER", context.strip()).replace("RESPONSE_PLACEHOLDER", response.strip())
    for attempt in range(retries):
        try:
            result = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=256,
                messages=[{"role": "user", "content": prompt}]
            )
            text = result.content[0].text.strip()
            # Strip markdown fences if present
            import re as _re
            text = _re.sub(r'^```(?:json)?\s*', '', text, flags=_re.MULTILINE)
            text = _re.sub(r'\s*```$', '', text, flags=_re.MULTILINE)
            text = text.strip()
            scores = json.loads(text)
            dims = ["emotional_acknowledgment", "perspective_taking",
                    "tone_match", "avoids_harm", "meets_core_need"]
            assert all(k in scores for k in dims), f"Missing keys: {scores}"
            assert all(1 <= scores[k] <= 5 for k in dims), f"Out of range: {scores}"
            scores["overall"] = round(sum(scores[k] for k in dims) / len(dims), 4)
            return scores
        except Exception as e:
            if attempt < retries - 1:
                wait = 2 ** attempt
                print(f"  Retry {attempt+1}: {e}. Raw={repr(text)[:80]}")
                time.sleep(wait)
            else:
                print(f"  FAILED: {e}. Raw={repr(text)[:80]}")
                return None

# ── File loading ──────────────────────────────────────────────────────────────

def load_lines(path: str) -> list[str]:
    """Load non-empty lines from a text file."""
    with open(path, "r", encoding="utf-8") as f:
        lines = [l.rstrip("\n") for l in f if l.strip()]
    return lines


def load_contexts(path: str) -> list[str]:
    """
    Load contexts from either:
    - A .npy file (MoEL format: numpy array of token lists)
    - A plain .txt file (one context per line)
    """
    if path.endswith(".npy"):
        try:
            import numpy as np
            data = np.load(path, allow_pickle=True)
            contexts = []
            for item in data:
                if isinstance(item, (list, np.ndarray)):
                    # Each item is a list of sentences/tokens
                    if isinstance(item[0], (list, np.ndarray)):
                        # List of token lists — join each sentence, then join sentences
                        ctx = " | ".join(" ".join(str(t) for t in sent) for sent in item)
                    else:
                        ctx = " ".join(str(t) for t in item)
                else:
                    ctx = str(item)
                contexts.append(ctx.strip())
            print(f"Loaded {len(contexts)} contexts from {path} (numpy)")
            return contexts
        except Exception as e:
            print(f"Failed to load npy: {e}. Trying as text file.")
    return load_lines(path)


def load_moel(path: str) -> list[str]:
    """
    MoEL generation_result.txt format:
    Emotion: X
    Context: ...
    Beam: response text
    ---
    This parser extracts the Beam (generated) response.
    Falls back to treating each line as a response if format not detected.
    """
    responses = []
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    # Try structured format first
    blocks = content.strip().split("---")
    for block in blocks:
        lines = [l.strip() for l in block.strip().splitlines() if l.strip()]
        beam_lines = [l for l in lines if l.lower().startswith("beam:")]
        if beam_lines:
            responses.append(beam_lines[0][5:].strip())
        else:
            # Try "Pred:" prefix used in some versions
            pred_lines = [l for l in lines if l.lower().startswith("pred:")]
            if pred_lines:
                responses.append(pred_lines[0][5:].strip())

    # Fallback: one response per line
    if not responses:
        print("MoEL: structured format not detected, reading line-by-line")
        responses = load_lines(path)

    print(f"Loaded {len(responses)} MoEL responses")
    return responses


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="SERAPH empathy scorer")
    parser.add_argument("--jsonl",       default=None,
                        help="Pre-aligned JSONL file (recommended). Each line: {context, seraph, raw_claude, ...}")
    parser.add_argument("--contexts",   default=None,
                        help="Test contexts file (one per line)")
    parser.add_argument("--seraph",     default=None,
                        help="SERAPH Full outputs (one per line)")
    parser.add_argument("--moel",       default=None,
                        help="MoEL generation_result.txt")
    parser.add_argument("--mime",       default=None,
                        help="MIME output.txt")
    parser.add_argument("--cem",        default=None,
                        help="CEM results.txt")
    parser.add_argument("--raw_claude", default=None,
                        help="Raw Claude outputs (one per line)")
    parser.add_argument("--seraph_no3", default=None,
                        help="SERAPH w/o Stage 3 outputs (one per line)")
    parser.add_argument("--output",     default="results/empathy_comparison.csv",
                        help="Output CSV path")
    parser.add_argument("--n_samples",  type=int, default=300,
                        help="Number of samples to score (default 300)")
    parser.add_argument("--seed",       type=int, default=42,
                        help="Random seed for sample selection")
    args = parser.parse_args()

    # ── Setup ──────────────────────────────────────────────────────────────
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY not found. Make sure .env exists with ANTHROPIC_API_KEY=sk-...")
        raise ValueError("ANTHROPIC_API_KEY not found in .env")

    client = anthropic.Anthropic(api_key=api_key)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)

    # ── Load data ──────────────────────────────────────────────────────────
    if args.jsonl:
        # Load from pre-aligned JSONL file (recommended — avoids newline alignment issues)
        import json as _json
        records = []
        with open(args.jsonl, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(_json.loads(line))
        print(f"Loaded {len(records)} aligned records from {args.jsonl}")
        contexts = [r['context'] for r in records]
        systems = {}
        if 'seraph' in records[0]:
            systems['SERAPH Full'] = [r['seraph'] for r in records]
        if 'raw_claude' in records[0]:
            systems['Raw Claude'] = [r['raw_claude'] for r in records]
        if 'cot' in records[0]:
            systems['CoT Empathy'] = [r['cot'] for r in records]
        if 'seraph_no3' in records[0]:
            systems['SERAPH w/o Stage 3'] = [r['seraph_no3'] for r in records]
        if 'moel' in records[0]:
            systems['MoEL'] = [r['moel'] for r in records]
        if 'mime' in records[0]:
            systems['MIME'] = [r['mime'] for r in records]
        if 'cem' in records[0]:
            systems['CEM'] = [r['cem'] for r in records]
        n = len(records)
    else:
        contexts = load_contexts(args.contexts)
        print(f"Loaded {len(contexts)} contexts")
        systems = {"SERAPH Full": load_lines(args.seraph)}
        if args.seraph_no3:
            systems["SERAPH w/o Stage 3"] = load_lines(args.seraph_no3)
        if args.raw_claude:
            systems["Raw Claude"] = load_lines(args.raw_claude)
        if args.moel:
            systems["MoEL"] = load_moel(args.moel)
        if args.mime:
            systems["MIME"] = load_lines(args.mime)
        if args.cem:
            systems["CEM"] = load_lines(args.cem)
        n = len(contexts)
        for name, outputs in systems.items():
            if len(outputs) != n:
                print(f"WARNING: {name} has {len(outputs)} outputs vs "
                      f"{n} contexts — truncating to min")
            systems[name] = outputs[:n]

    # Sample indices
    indices = list(range(n))
    if args.n_samples < n:
        indices = random.sample(indices, args.n_samples)
        indices.sort()
        print(f"Sampling {args.n_samples} of {n} examples (seed={args.seed})")

    # ── Score ──────────────────────────────────────────────────────────────
    dims = ["emotional_acknowledgment", "perspective_taking",
            "tone_match", "avoids_harm", "meets_core_need", "overall"]

    all_rows = []
    total_calls = len(indices) * len(systems)
    call_count = 0

    for system_name, outputs in systems.items():
        print(f"\n{'='*60}")
        print(f"Scoring: {system_name}")
        print(f"{'='*60}")

        for idx in tqdm(indices, desc=system_name):
            context  = contexts[idx]
            response = outputs[idx]
            scores   = score_response(client, context, response)
            call_count += 1

            if scores is None:
                # Record NaN on failure
                scores = {d: float("nan") for d in dims}

            row = {
                "system": system_name,
                "sample_idx": idx,
                "context": context[:200],   # truncate for CSV readability
                "response": response[:200],
                **scores
            }
            all_rows.append(row)

            # Rate limit: ~40 req/min on Haiku tier
            time.sleep(0.5)

    # ── Save results ───────────────────────────────────────────────────────
    df = pd.DataFrame(all_rows)
    df.to_csv(args.output, index=False)
    print(f"\nSaved {len(df)} rows to {args.output}")

    # ── Summary table ──────────────────────────────────────────────────────
    summary = (df.groupby("system")[dims]
                 .agg(["mean", "std"])
                 .round(4))

    summary_path = args.output.replace(".csv", "_summary.csv")
    summary.to_csv(summary_path)
    print(f"Saved summary to {summary_path}")

    # ── Print mean scores ──────────────────────────────────────────────────
    means = df.groupby("system")[dims].mean().round(4)
    print("\n" + "="*60)
    print("MEAN EMPATHY SCORES BY SYSTEM")
    print("="*60)
    print(means.to_string())

    # ── Generate LaTeX table ───────────────────────────────────────────────
    print("\n" + "="*60)
    print("LATEX TABLE (paste into paper)")
    print("="*60)

    # Define preferred row order
    row_order = ["MoEL", "MIME", "CEM", "Raw Claude", "CoT Empathy",
                 "SERAPH w/o Stage 3", "SERAPH Full"]
    present = [s for s in row_order if s in means.index]
    # Add any unexpected systems
    present += [s for s in means.index if s not in present]

    col_labels = {
        "emotional_acknowledgment": "Ack.",
        "perspective_taking":       "Persp.",
        "tone_match":               "Tone",
        "avoids_harm":              "Safe",
        "meets_core_need":          "Need",
        "overall":                  "Overall"
    }

    latex = []
    latex.append(r"\begin{table*}[!ht]")
    latex.append(r"\centering")
    latex.append(r"\small")
    latex.append(r"\begin{tabular}{lcccccc}")
    latex.append(r"\toprule")
    latex.append(r"\textbf{System} & \textbf{Ack.} & \textbf{Persp.} & "
                 r"\textbf{Tone} & \textbf{Safe} & \textbf{Need} & "
                 r"\textbf{Overall} \\")
    latex.append(r"\midrule")
    latex.append(r"\multicolumn{7}{l}{\textit{Fine-tuned supervised systems}} \\")
    latex.append(r"\midrule")

    supervised = ["MoEL", "MIME", "CEM"]
    zeroshot   = ["Raw Claude", "CoT Empathy", "SERAPH w/o Stage 3", "SERAPH Full"]

    def fmt_row(name, row_means, bold=False):
        vals = [f"{row_means[d]:.4f}" for d in dims]
        if bold:
            vals = [f"\\textbf{{{v}}}" for v in vals]
        return f"{name} & " + " & ".join(vals) + r" \\"

    for s in [x for x in supervised if x in present]:
        latex.append(fmt_row(s, means.loc[s]))

    latex.append(r"\midrule")
    latex.append(r"\multicolumn{7}{l}{\textit{Zero-shot / training-free systems}} \\")
    latex.append(r"\midrule")

    for s in [x for x in zeroshot if x in present]:
        bold = (s == "SERAPH Full")
        name_tex = r"\seraph{} Full" if s == "SERAPH Full" else \
                   r"\seraph{} w/o Stage~3" if "Stage 3" in s else s
        latex.append(fmt_row(name_tex, means.loc[s], bold=bold))

    latex.append(r"\bottomrule")
    latex.append(r"\end{tabular}")
    latex.append(
        r"\caption{Unified empathy evaluation across all systems on "
        r"EmpatheticDialogues ($n=" + str(args.n_samples) + r"$ samples). "
        r"Scores are LLM-as-judge ratings (1--5 scale) applied uniformly "
        r"to all systems' generated outputs. "
        r"\textbf{Ack.}~= Emotional Acknowledgment; "
        r"\textbf{Persp.}~= Perspective-Taking; "
        r"\textbf{Tone}~= Tone Match; "
        r"\textbf{Safe}~= Avoids Harmful Patterns; "
        r"\textbf{Need}~= Meets Core Need. "
        r"Bold = best per column. "
        r"Supervised systems were fine-tuned on the training split; "
        r"\seraph{} is zero-shot.}"
    )
    latex.append(r"\label{tab:unified_empathy}")
    latex.append(r"\end{table*}")

    latex_str = "\n".join(latex)
    print(latex_str)

    # Save LaTeX
    latex_path = args.output.replace(".csv", "_table.tex")
    with open(latex_path, "w") as f:
        f.write(latex_str)
    print(f"\nSaved LaTeX to {latex_path}")

    # ── Cost estimate ──────────────────────────────────────────────────────
    approx_cost = call_count * 0.0002  # ~$0.0002 per Haiku call
    print(f"\nApproximate API cost: ${approx_cost:.2f} "
          f"({call_count} Haiku calls)")


if __name__ == "__main__":
    main()
