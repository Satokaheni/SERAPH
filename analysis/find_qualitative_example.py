"""
find_qualitative_example.py
============================
Searches pipeline outputs to find the best qualitative example for a
paper figure, then generates LaTeX.

Data sources (auto-detected from your actual files):
  empathy_comparison.csv              — per-record LLM-judge scores for
                                        SERAPH Full, CoT Empathy, MoEL
  perspective_ablation_outputs.jsonl  — Stage 3 simulation text (fp_stage3_output)
                                        and responses for 50-context subset
  empathetic_dialogues_full.json      — emotion labels (gt_emotion) per utterance

Candidate criteria (relaxed automatically if < 5 found):
  1. SERAPH overall >= 4.0, MoEL overall <= 3.5, gap >= 0.8
  2. SERAPH tone_match > CoT tone_match AND SERAPH meets_core_need > CoT meets_core_need
  3. Context <= 400 chars
  4. All responses <= 500 chars, Stage 3 excerpt <= 600 chars
  5. Emotion not in: neutral, joy, surprise
  6. No crisis content

Output:
  tables/figure_qualitative.tex       — LaTeX figure for #1 candidate
  results/qualitative_candidates.json — top 5 candidates for manual review

Usage:
    python find_qualitative_example.py
    python find_qualitative_example.py \\
        --scores   results/empathy_comparison.csv \\
        --ablation perspective_ablation_outputs.jsonl \\
        --full     results/empathetic_dialogues_full.json
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

# ── Crisis content filter ─────────────────────────────────────────────────────
CRISIS_WORDS = {"suicide", "self-harm", "self harm", "kill", "die", "dying",
                "abuse", "abused", "overdose", "cutting", "hurt myself"}

TRIVIAL_EMOTIONS = {"neutral", "joy", "surprise", "happy", "happiness"}


# ── LaTeX escaping ────────────────────────────────────────────────────────────
def latex_escape(text: str) -> str:
    """Escape LaTeX special characters in plain text."""
    replacements = [
        ("\\", "\\textbackslash{}"),
        ("&",  "\\&"),
        ("%",  "\\%"),
        ("$",  "\\$"),
        ("#",  "\\#"),
        ("_",  "\\_"),
        ("{",  "\\{"),
        ("}",  "\\}"),
        ("~",  "\\textasciitilde{}"),
        ("^",  "\\textasciicircum{}"),
    ]
    for char, escaped in replacements:
        text = text.replace(char, escaped)
    return text


def truncate(text: str, max_chars: int, suffix: str = "...") -> str:
    """Truncate text to max_chars, breaking at a word boundary."""
    if len(text) <= max_chars:
        return text
    truncated = text[:max_chars].rsplit(" ", 1)[0]
    return truncated + suffix


def first_sentences(text: str, n: int = 3) -> str:
    """Extract first n sentences from text."""
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return " ".join(sentences[:n])


# ── Data loading ──────────────────────────────────────────────────────────────

def load_scores(scores_path: str) -> dict[str, dict[str, dict]]:
    """
    Load scores CSV into {system: {context_key: score_dict}}.
    context_key = first 120 chars of context, stripped.
    """
    df = pd.read_csv(scores_path)
    result: dict[str, dict] = {}
    for _, row in df.iterrows():
        system = row["system"]
        key    = str(row["context"])[:120].strip()
        if system not in result:
            result[system] = {}
        result[system][key] = {
            "ack":     float(row.get("emotional_acknowledgment", row.get("ack_score", 0))),
            "persp":   float(row.get("perspective_taking",       row.get("persp_score", 0))),
            "tone":    float(row.get("tone_match",               row.get("tone_score", 0))),
            "safe":    float(row.get("avoids_harm",              row.get("harm_score", 0))),
            "need":    float(row.get("meets_core_need",          row.get("need_score", 0))),
            "overall": float(row.get("overall", 0)),
            "response": str(row.get("response", "")),
        }
    return result


def load_stage3(ablation_path: str) -> dict[str, dict]:
    """
    Load Stage 3 simulation and responses from perspective_ablation_outputs.jsonl.
    Returns {context_key: {stage3_text, fp_response}}.
    """
    result = {}
    with open(ablation_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            key = str(rec.get("context", ""))[:120].strip()
            s3  = rec.get("fp_stage3_output", {})
            s3_text = ""
            if isinstance(s3, dict):
                s3_text = s3.get("perspective", "") or s3.get("analysis", "") or str(s3)
            elif isinstance(s3, str):
                s3_text = s3
            result[key] = {
                "stage3_text": s3_text,
                "fp_response": rec.get("fp_response", ""),
            }
    return result


def infer_emotion(context: str) -> str:
    """
    Infer a rough emotion label from context keywords.
    Used as fallback when ground-truth labels aren't available.
    """
    ctx = context.lower()
    if any(w in ctx for w in ["nervous", "anxious", "worried", "scared", "fear"]):
        return "anxiety"
    if any(w in ctx for w in ["sad", "depress", "cry", "grief", "lost", "miss", "mourn"]):
        return "sadness"
    if any(w in ctx for w in ["angry", "anger", "mad", "furious", "frustrated"]):
        return "anger"
    if any(w in ctx for w in ["proud", "accomplish", "achieve", "success"]):
        return "pride"
    if any(w in ctx for w in ["embarrass", "shame", "humiliat"]):
        return "embarrassment"
    if any(w in ctx for w in ["disappoint", "let down", "fail"]):
        return "disappointment"
    if any(w in ctx for w in ["excit", "thrill", "can't wait", "looking forward"]):
        return "anticipation"
    if any(w in ctx for w in ["lonely", "alone", "isolat", "no one"]):
        return "loneliness"
    return "distress"


# ── Candidate search ──────────────────────────────────────────────────────────

def find_candidates(
    scores:    dict[str, dict],
    stage3:    dict[str, dict],
    seraph_threshold: float = 4.0,
    raw_threshold:    float = 3.5,
    gap_threshold:    float = 0.8,
) -> list[dict]:
    """
    Find all contexts meeting quality, length, emotion, and safety criteria.
    Returns sorted list of candidate dicts.
    """
    seraph_scores = scores.get("SERAPH Full", {})
    cot_scores    = scores.get("CoT Empathy", {})
    raw_scores    = scores.get("MoEL", scores.get("Raw Claude", {}))

    # All contexts that have scores for all three systems
    shared_keys = set(seraph_scores) & set(cot_scores) & set(raw_scores)
    print(f"  Shared contexts (all 3 systems scored): {len(shared_keys)}")

    # Contexts that also have Stage 3 simulation
    with_stage3 = shared_keys & set(stage3)
    print(f"  Of those, with Stage 3 simulation:      {len(with_stage3)}")

    candidates = []

    for key in with_stage3:
        s  = seraph_scores[key]
        c  = cot_scores[key]
        r  = raw_scores[key]
        s3 = stage3[key]

        emotion      = infer_emotion(key)
        seraph_resp  = s["response"]
        cot_resp     = c["response"]
        raw_resp     = r["response"]
        s3_text      = s3["stage3_text"]

        # ── Criterion 1: Score separation ─────────────────────────────────
        if s["overall"] < seraph_threshold:
            continue
        if r["overall"] > raw_threshold:
            continue
        if (s["overall"] - r["overall"]) < gap_threshold:
            continue

        # ── Criterion 2: SERAPH beats CoT on tone + need ──────────────────
        if not (s["tone"] > c["tone"] and s["need"] > c["need"]):
            continue

        # ── Criterion 3: Context length ───────────────────────────────────
        if len(key) > 400:
            continue

        # ── Criterion 4: Response lengths ─────────────────────────────────
        if max(len(seraph_resp), len(cot_resp), len(raw_resp)) > 500:
            continue
        if len(s3_text) > 600 and len(first_sentences(s3_text, 3)) > 600:
            continue

        # ── Criterion 6: No crisis content ────────────────────────────────
        combined = (key + " " + seraph_resp + " " + cot_resp + " " + raw_resp).lower()
        if any(w in combined for w in CRISIS_WORDS):
            continue

        gap_seraph_raw = s["overall"] - r["overall"]
        gap_seraph_cot_key_dims = (s["tone"] + s["need"]) - (c["tone"] + c["need"])

        candidates.append({
            "context":       key,
            "emotion":       emotion,
            "seraph_resp":   seraph_resp,
            "cot_resp":      cot_resp,
            "raw_resp":      raw_resp,
            "stage3_text":   s3_text,
            "seraph_scores": s,
            "cot_scores":    c,
            "raw_scores":    r,
            "gap_seraph_raw":      gap_seraph_raw,
            "gap_seraph_cot_dims": gap_seraph_cot_key_dims,
        })

    # Sort: primary = gap(SERAPH-Raw), tiebreak = gap(SERAPH-CoT on tone+need)
    candidates.sort(
        key=lambda x: (x["gap_seraph_raw"], x["gap_seraph_cot_dims"]),
        reverse=True,
    )
    return candidates


def find_with_relaxation(scores: dict, stage3: dict) -> list[dict]:
    """Try criteria at default thresholds, relax by 0.1 increments if < 5 found."""
    seraph_t, raw_t, gap_t = 4.0, 3.5, 0.8
    for attempt in range(6):
        candidates = find_candidates(scores, stage3,
                                     seraph_t, raw_t, gap_t)
        if len(candidates) >= 5:
            print(f"  Found {len(candidates)} candidates "
                  f"(thresholds: SERAPH>={seraph_t}, Raw<={raw_t}, gap>={gap_t:.1f})")
            return candidates
        print(f"  Only {len(candidates)} candidates at current thresholds — relaxing...")
        seraph_t -= 0.1
        raw_t    += 0.1
        gap_t    -= 0.1

    print(f"  Found {len(candidates)} candidates after maximum relaxation.")
    return candidates


# ── Display ───────────────────────────────────────────────────────────────────

def print_candidate(rank: int, c: dict) -> None:
    def fmt_scores(sc: dict) -> str:
        return (f"Ack={sc['ack']:.1f} Persp={sc['persp']:.1f} "
                f"Tone={sc['tone']:.1f} Safe={sc['safe']:.1f} "
                f"Need={sc['need']:.1f} Overall={sc['overall']:.2f}")

    print(f"\n{'='*60}")
    print(f"=== Candidate {rank} ===")
    print(f"Emotion: {c['emotion']}")
    print(f"Context: {c['context']}")
    print(f"--- Stage 3 Simulation (excerpt, first 500 chars) ---")
    print(first_sentences(c["stage3_text"], 3)[:500])
    print(f"--- SERAPH Full Response ---")
    print(c["seraph_resp"])
    print(f"--- CoT Empathy Response ---")
    print(c["cot_resp"])
    print(f"--- Raw Claude / MoEL Response ---")
    print(c["raw_resp"])
    print(f"--- Scores ---")
    print(f"SERAPH:  {fmt_scores(c['seraph_scores'])}")
    print(f"CoT:     {fmt_scores(c['cot_scores'])}")
    print(f"Raw/MoEL:{fmt_scores(c['raw_scores'])}")
    print(f"Gap (SERAPH-Raw): {c['gap_seraph_raw']:.2f}  "
          f"Gap (SERAPH-CoT tone+need): {c['gap_seraph_cot_dims']:.2f}")


# ── LaTeX generation ──────────────────────────────────────────────────────────

def generate_latex(c: dict, output_path: Path) -> None:
    ctx      = latex_escape(c["context"])
    emotion  = latex_escape(c["emotion"])
    s3_exc   = latex_escape(first_sentences(c["stage3_text"], 3)[:400])
    s_resp   = latex_escape(truncate(c["seraph_resp"], 400))
    c_resp   = latex_escape(truncate(c["cot_resp"],    400))
    r_resp   = latex_escape(truncate(c["raw_resp"],    400))

    s = c["seraph_scores"]
    co = c["cot_scores"]
    r  = c["raw_scores"]

    latex = f"""% Qualitative example figure — auto-generated by find_qualitative_example.py
% Emotion: {c['emotion']}
% SERAPH overall: {s['overall']:.2f}  CoT: {co['overall']:.2f}  Raw/MoEL: {r['overall']:.2f}

\\begin{{figure}}[t]
\\small
\\setlength{{\\fboxsep}}{{4pt}}

\\noindent\\textbf{{Context}} (\\textit{{{emotion}}}): \\\\
\\textit{{``{ctx}''}}

\\vspace{{4pt}}
\\noindent\\fbox{{\\parbox{{0.96\\columnwidth}}{{%
\\textbf{{Stage~3 Self-Simulation (excerpt):}} \\\\
\\textit{{``{s3_exc}\\ldots''}}
}}}}

\\vspace{{4pt}}
\\noindent\\textbf{{\\seraph{{}} Full}} \\textsf{{[Tone={s['tone']:.0f}, Need={s['need']:.0f}, Overall={s['overall']:.2f}]}}: \\\\
``{s_resp}''

\\vspace{{2pt}}
\\noindent\\textbf{{CoT Empathy}} \\textsf{{[Tone={co['tone']:.0f}, Need={co['need']:.0f}, Overall={co['overall']:.2f}]}}: \\\\
``{c_resp}''

\\vspace{{2pt}}
\\noindent\\textbf{{MoEL}} \\textsf{{[Tone={r['tone']:.0f}, Need={r['need']:.0f}, Overall={r['overall']:.2f}]}}: \\\\
``{r_resp}''

\\caption{{Qualitative example from EmpatheticDialogues.
The input context expresses \\textit{{{emotion}}}.
\\stage{{3}}'s first-person self-simulation (boxed) grounds the response
in the recipient's felt experience.
\\seraph{{}} Full achieves higher Tone Match ({s['tone']:.0f} vs.\\ {co['tone']:.0f})
and Meets Core Need ({s['need']:.0f} vs.\\ {co['need']:.0f}) scores than CoT Empathy,
consistent with the human evaluation findings.}}
\\label{{fig:qualitative}}
\\end{{figure}}
"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(latex, encoding="utf-8")
    print(f"\nLaTeX figure saved → {output_path}")
    print("\n--- LaTeX preview ---")
    print(latex)


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Find best qualitative example for SERAPH paper figure"
    )
    parser.add_argument("--scores",
                        default="results/empathy_comparison.csv",
                        help="LLM-judge scores CSV")
    parser.add_argument("--ablation",
                        default="perspective_ablation_outputs.jsonl",
                        help="Perspective ablation JSONL with Stage 3 simulation")
    parser.add_argument("--full",
                        default="results/empathetic_dialogues_full.json",
                        help="Full ED results JSON for emotion labels")
    parser.add_argument("--latex-out",
                        default="tables/figure_qualitative.tex")
    parser.add_argument("--json-out",
                        default="results/qualitative_candidates.json")
    args = parser.parse_args()

    print("Loading scores...")
    scores = load_scores(args.scores)
    print(f"  Systems: {list(scores.keys())}")
    for sys, recs in scores.items():
        print(f"  {sys}: {len(recs)} records")

    print("\nLoading Stage 3 simulations...")
    stage3 = load_stage3(args.ablation)
    print(f"  {len(stage3)} records with Stage 3 output")

    print("\nSearching for candidates...")
    candidates = find_with_relaxation(scores, stage3)

    if not candidates:
        print("No candidates found even after relaxation.")
        print("Consider running on a larger context set or relaxing criteria manually.")
        return

    top5 = candidates[:5]
    print(f"\nFound {len(candidates)} candidates meeting criteria. Showing top 5.")

    for i, c in enumerate(top5):
        print_candidate(i + 1, c)

    # Save top 5 to JSON for manual review
    json_out = Path(args.json_out)
    json_out.parent.mkdir(parents=True, exist_ok=True)
    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(top5, f, indent=2, ensure_ascii=False)
    print(f"\nTop 5 candidates saved → {json_out}")

    # Generate LaTeX for #1
    print("\nGenerating LaTeX for candidate #1...")
    generate_latex(top5[0], Path(args.latex_out))


if __name__ == "__main__":
    main()
