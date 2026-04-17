"""
prepare_data.py
===============
Prepares aligned input files for score_empathy.py by extracting
contexts from the EmpatheticDialogues test set and parsing each
system's output file into a consistent one-response-per-line format.

Usage
-----
    python prepare_data.py \
        --ed_test   path/to/EmpatheticDialogues/test.csv \
        --moel_out  path/to/MoEL/generation_result.txt \
        --mime_out  path/to/MIME/save/test/output.txt \
        --cem_out   path/to/CEM/results/results.txt \
        --seraph_out path/to/seraph_outputs.jsonl \
        --out_dir   data/

After running, pass the output files to score_empathy.py.

EmpatheticDialogues test.csv columns:
    conv_id, utterance_idx, context, prompt, speaker_idx,
    utterance, emotion, selfeval, tags

Each conversation's final utterance is the "context" (what the
human said that the system should respond to empathically).
"""

import argparse
import csv
import json
import re
from pathlib import Path
from collections import defaultdict


# ── EmpatheticDialogues context extraction ────────────────────────────────────

def extract_ed_contexts(test_csv: str) -> list[str]:
    """
    Extract the listener-turn contexts from EmpatheticDialogues test.csv.
    Each conversation has alternating speaker/listener turns.
    We extract the last speaker utterance as the context to respond to.
    """
    convs = defaultdict(list)
    with open(test_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            conv_id = row["conv_id"]
            convs[conv_id].append(row)

    contexts = []
    for conv_id, turns in sorted(convs.items()):
        # Sort by utterance index
        turns = sorted(turns, key=lambda r: int(r["utterance_idx"]))
        # The context is the last utterance from the speaker (odd indices)
        # Concatenate full history up to last speaker turn
        history = []
        last_speaker_utt = None
        for t in turns:
            utt = t["utterance"].strip()
            idx = int(t["utterance_idx"])
            if idx % 2 == 0:  # speaker turns (0-indexed even = speaker)
                last_speaker_utt = utt
                history.append(f"Speaker: {utt}")
            else:
                history.append(f"Listener: {utt}")
        if last_speaker_utt:
            # Use last 3 turns as context for conciseness
            context = " | ".join(history[-3:]) if len(history) >= 3 \
                      else " | ".join(history)
            contexts.append(context)

    print(f"Extracted {len(contexts)} contexts from EmpatheticDialogues test set")
    return contexts


# ── MoEL parser ───────────────────────────────────────────────────────────────

def parse_moel(path: str) -> list[str]:
    """
    Parse MoEL's generation_result.txt.
    Format (repeated blocks):
        Emotion: X
        Context: ...
        Beam: response
        Ref: reference
        ---
    Falls back to one-response-per-line if format not found.
    """
    responses = []
    with open(path, encoding="utf-8") as f:
        content = f.read()

    blocks = re.split(r"-{3,}", content)
    for block in blocks:
        block = block.strip()
        if not block:
            continue
        beam_match = re.search(r"^Beam:\s*(.+)$", block, re.MULTILINE | re.IGNORECASE)
        if beam_match:
            responses.append(beam_match.group(1).strip())
        else:
            # Try "Generated:" prefix used in some versions
            gen_match = re.search(r"^Generated:\s*(.+)$", block,
                                   re.MULTILINE | re.IGNORECASE)
            if gen_match:
                responses.append(gen_match.group(1).strip())

    if not responses:
        print("MoEL: structured format not found, reading line by line")
        responses = [l.strip() for l in open(path, encoding="utf-8")
                     if l.strip()]

    print(f"Parsed {len(responses)} MoEL responses")
    return responses


# ── MIME parser ───────────────────────────────────────────────────────────────

def parse_mime(path: str) -> list[str]:
    """
    MIME output.txt — one response per line, plain text.
    Some versions prefix with 'Pred: '.
    """
    responses = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.lower().startswith("pred:"):
                line = line[5:].strip()
            responses.append(line)
    print(f"Parsed {len(responses)} MIME responses")
    return responses


# ── CEM parser ────────────────────────────────────────────────────────────────

def parse_cem(path: str) -> list[str]:
    """
    CEM results.txt — structured blocks or one response per line.
    Common format:
        Pred: response text
        Ref:  reference text
    """
    responses = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.lower().startswith("pred:"):
                responses.append(line[5:].strip())

    if not responses:
        print("CEM: 'Pred:' prefix not found, reading line by line")
        responses = [l.strip() for l in open(path, encoding="utf-8")
                     if l.strip()]

    print(f"Parsed {len(responses)} CEM responses")
    return responses


# ── SERAPH JSONL parser ───────────────────────────────────────────────────────

def parse_seraph_jsonl(path: str) -> list[str]:
    """
    Parse SERAPH outputs from a JSONL file.
    Each line: {"context": "...", "response": "...", ...}
    Also accepts plain text (one response per line).
    """
    responses = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                responses.append(obj.get("response", obj.get("output", "")))
            except json.JSONDecodeError:
                responses.append(line)  # plain text fallback
    print(f"Parsed {len(responses)} SERAPH responses")
    return responses


# ── Alignment helper ──────────────────────────────────────────────────────────

def align_and_save(contexts: list[str],
                   system_outputs: dict[str, list[str]],
                   out_dir: str):
    """
    Align all system outputs to the context list length,
    then save one file per system.
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    n = len(contexts)

    # Save contexts
    ctx_path = Path(out_dir) / "contexts.txt"
    with open(ctx_path, "w", encoding="utf-8") as f:
        f.write("\n".join(contexts))
    print(f"Saved contexts to {ctx_path}")

    for name, outputs in system_outputs.items():
        if len(outputs) != n:
            print(f"WARNING: {name}: {len(outputs)} outputs vs {n} contexts. "
                  f"Truncating to {min(len(outputs), n)}.")
            outputs = outputs[:n]
            # Pad with empty string if too short
            while len(outputs) < n:
                outputs.append("[NO OUTPUT]")

        safe_name = name.lower().replace(" ", "_").replace("/", "_")
        out_path = Path(out_dir) / f"{safe_name}_outputs.txt"
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("\n".join(outputs))
        print(f"Saved {name} outputs ({len(outputs)} lines) to {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Prepare aligned data for SERAPH empathy scoring")
    parser.add_argument("--ed_test",    required=True,
                        help="EmpatheticDialogues test.csv")
    parser.add_argument("--moel_out",   default=None,
                        help="MoEL generation_result.txt")
    parser.add_argument("--mime_out",   default=None,
                        help="MIME output.txt")
    parser.add_argument("--cem_out",    default=None,
                        help="CEM results.txt")
    parser.add_argument("--seraph_out", default=None,
                        help="SERAPH outputs (.jsonl or .txt)")
    parser.add_argument("--seraph_no3", default=None,
                        help="SERAPH w/o Stage 3 outputs (.jsonl or .txt)")
    parser.add_argument("--raw_claude", default=None,
                        help="Raw Claude outputs (.txt)")
    parser.add_argument("--out_dir",    default="data/prepared",
                        help="Output directory for aligned files")
    args = parser.parse_args()

    # Extract contexts
    contexts = extract_ed_contexts(args.ed_test)

    # Parse each system
    system_outputs = {}

    if args.moel_out:
        system_outputs["MoEL"] = parse_moel(args.moel_out)

    if args.mime_out:
        system_outputs["MIME"] = parse_mime(args.mime_out)

    if args.cem_out:
        system_outputs["CEM"] = parse_cem(args.cem_out)

    if args.seraph_out:
        system_outputs["SERAPH Full"] = parse_seraph_jsonl(args.seraph_out)

    if args.seraph_no3:
        system_outputs["SERAPH w/o Stage 3"] = parse_seraph_jsonl(args.seraph_no3)

    if args.raw_claude:
        system_outputs["Raw Claude"] = [
            l.strip() for l in open(args.raw_claude, encoding="utf-8")
            if l.strip()
        ]

    # Align and save
    align_and_save(contexts, system_outputs, args.out_dir)

    # Print next step
    print("\n" + "="*60)
    print("Next step — run the scorer:")
    print("="*60)
    cmd_parts = [
        "python score_empathy.py",
        f"  --contexts {args.out_dir}/contexts.txt",
    ]
    if args.seraph_out:
        cmd_parts.append(f"  --seraph   {args.out_dir}/seraph_full_outputs.txt")
    if args.moel_out:
        cmd_parts.append(f"  --moel     {args.out_dir}/moel_outputs.txt")
    if args.mime_out:
        cmd_parts.append(f"  --mime     {args.out_dir}/mime_outputs.txt")
    if args.cem_out:
        cmd_parts.append(f"  --cem      {args.out_dir}/cem_outputs.txt")
    if args.raw_claude:
        cmd_parts.append(f"  --raw_claude {args.out_dir}/raw_claude_outputs.txt")
    if args.seraph_no3:
        cmd_parts.append(
            f"  --seraph_no3 {args.out_dir}/seraph_w/o_stage_3_outputs.txt")
    cmd_parts += [
        "  --output   results/empathy_comparison.csv",
        "  --n_samples 200",
        "  --api_key  $ANTHROPIC_API_KEY",
    ]
    print(" \\\n".join(cmd_parts))


if __name__ == "__main__":
    main()
