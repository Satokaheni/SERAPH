"""
SERAPH Results Table Generator
================================
Reads benchmark result JSON files and generates publication-ready
LaTeX table environments for the paper.

Produces four tables:
  Table 1 — Main results: SERAPH vs baselines across all datasets
  Table 2 — Ablation study: all 6 variants on EmpatheticDialogues
  Table 3 — Per-class F1 breakdown (SERAPH full vs no_stage3)
  Table 4 — Empathy dimension breakdown (5-dimension scores)

Usage:
    python analysis/results_table_generator.py
    python analysis/results_table_generator.py --results-dir results --output paper/tables
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Optional

from paths import PATHS
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_metrics(path: Path) -> Optional[dict]:
    """Load the metrics dict from a benchmark result file.
    Automatically checks ablations/ subdirectory if primary path not found.
    """
    # Try primary path
    if path.exists():
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
        return data.get("metrics", data)
    # Try ablations/ subdirectory
    ablation_path = path.parent / "ablations" / path.name
    if ablation_path.exists():
        with ablation_path.open(encoding="utf-8") as f:
            data = json.load(f)
        return data.get("metrics", data)
    logger.warning("Result file not found: %s", path)
    return None


def _fmt(value: Optional[float], decimals: int = 4, bold_threshold: Optional[float] = None) -> str:
    """Format a float for LaTeX. Bold if it meets the threshold."""
    if value is None:
        return r"\textemdash"
    formatted = f"{value:.{decimals}f}"
    if bold_threshold is not None and value >= bold_threshold:
        return r"\textbf{" + formatted + "}"
    return formatted


def _bold_max(values: list[Optional[float]], decimals: int = 4) -> list[str]:
    """Format a row of values, bolding the maximum."""
    clean = [v for v in values if v is not None]
    if not clean:
        return [_fmt(v, decimals) for v in values]
    max_val = max(clean)
    return [
        r"\textbf{" + f"{v:.{decimals}f}" + "}" if (v is not None and abs(v - max_val) < 1e-9)
        else _fmt(v, decimals)
        for v in values
    ]


# ---------------------------------------------------------------------------
# Table 1 — Main Results
# ---------------------------------------------------------------------------

def generate_main_results_table(results_dir: Path, output_dir: Path) -> str:
    """
    Main results table: SERAPH vs baselines on all 3 datasets.

    Rows: systems (SERAPH Full, No Stage 3, Raw Claude, DialogueRNN)
    Cols: dataset × metric (Weighted F1, Empathy Score)
    """
    systems = [
        ("SERAPH (Full)",     "full"),
        ("SERAPH w/o Stage 3","no_stage3"),
        ("SERAPH w/o Stage 4","no_stage4"),
        ("Raw Claude",        "raw_claude"),
        ("DialogueRNN",       "dialogue_rnn"),
    ]
    all_datasets = [
        ("MELD",                "meld"),
        ("EmpathDial",          "empathetic_dialogues"),
    ]
    # Only include datasets that have at least one result file present
    datasets = [
        (ds_name, ds_key) for ds_name, ds_key in all_datasets
        if any(
            (results_dir / f"{ds_key}_{sys_key}.json").exists() or
            (results_dir / "ablations" / f"{ds_key}_{sys_key}.json").exists()
            for _, sys_key in systems
        )
    ]

    # Load all metrics
    data: dict[str, dict[str, dict]] = {}  # {dataset_key: {system_key: metrics}}
    for _, ds_key in datasets:
        data[ds_key] = {}
        for _, sys_key in systems:
            fname = results_dir / f"{ds_key}_{sys_key}.json"
            m = _load_metrics(fname)
            data[ds_key][sys_key] = m or {}

    lines = []
    lines.append(r"% ── Table 1: Main Results ──────────────────────────────────────")
    lines.append(r"\begin{table*}[t]")
    lines.append(r"  \centering")
    lines.append(r"  \caption{Main results across three benchmark datasets. "
                 r"\textbf{Bold} = best in column. "
                 r"Weighted F1 measures emotion classification accuracy; "
                 r"Empathy Score is the LLM-as-judge mean (1--5 scale, normalised to 0--1).}")
    lines.append(r"  \label{tab:main_results}")
    lines.append(r"  \begin{tabular}{l" + "cc" * len(datasets) + r"}")
    lines.append(r"    \toprule")

    # Header row 1 — dataset names spanning 2 cols each
    ds_headers = " & ".join(
        r"\multicolumn{2}{c}{" + ds_name + "}" for ds_name, _ in datasets
    )
    lines.append(r"    \textbf{System} & " + ds_headers + r" \\")

    # Header row 2 — metric names and cmidrules (dynamic based on dataset count)
    metric_headers = " & ".join([r"W-F1 & Emp."] * len(datasets))
    cmidrules = "".join(
        f"\cmidrule(lr){{{2 + i*2}-{3 + i*2}}}" for i in range(len(datasets))
    )
    lines.append(f"    {cmidrules}")
    lines.append(r"     & " + metric_headers + r" \\")
    lines.append(r"    \midrule")

    # Collect column values for bolding
    col_wf1:  dict[str, list] = {ds: [] for _, ds in datasets}
    col_emp:  dict[str, list] = {ds: [] for _, ds in datasets}
    for _, sys_key in systems:
        for _, ds_key in datasets:
            m = data[ds_key].get(sys_key, {})
            col_wf1[ds_key].append(m.get("weighted_f1"))
            col_emp[ds_key].append(m.get("mean_empathy_score"))

    # Data rows
    for sys_idx, (sys_name, sys_key) in enumerate(systems):
        cells = []
        for _, ds_key in datasets:
            m = data[ds_key].get(sys_key, {})
            wf1 = m.get("weighted_f1")
            emp = m.get("mean_empathy_score")
            # Bold if max in column
            wf1_fmt = _bold_max(col_wf1[ds_key])[sys_idx]
            emp_fmt = _bold_max(col_emp[ds_key])[sys_idx]
            cells.extend([wf1_fmt, emp_fmt])

        if sys_name == "Raw Claude":
            lines.append(r"    \midrule")  # separator before non-pipeline baselines

        lines.append(f"    {sys_name} & " + " & ".join(cells) + r" \\")

    lines.append(r"    \bottomrule")
    lines.append(r"  \end{tabular}")
    lines.append(r"\end{table*}")

    latex = "\n".join(lines)
    out_path = output_dir / "table1_main_results.tex"
    out_path.write_text(latex, encoding="utf-8")
    logger.info("Table 1 written to %s", out_path)
    return latex


# ---------------------------------------------------------------------------
# Table 2 — Ablation Study
# ---------------------------------------------------------------------------

def generate_ablation_table(results_dir: Path, output_dir: Path) -> str:
    """
    Ablation table: all 6 variants on EmpatheticDialogues (most informative dataset).
    Also includes MELD numbers for the two key variants (full vs no_stage3).
    """
    ablation_dir = results_dir / "ablations"

    variants = [
        ("Full Pipeline",           "full"),
        (r"w/o Stage 3 (no sim.)",  "no_stage3"),
        (r"w/o Stage 4 (no gate)",  "no_stage4"),
        ("Merged Stages 1+2",       "merged_1_2"),
        ("Stage 3 Only",            "stage3_only"),
        ("Random Emotion Label",    "random_emotion"),
    ]

    lines = []
    lines.append(r"% ── Table 2: Ablation Study ────────────────────────────────────")
    lines.append(r"\begin{table}[t]")
    lines.append(r"  \centering")
    lines.append(r"  \caption{Ablation study on EmpatheticDialogues. "
                 r"Each row removes or modifies one component. "
                 r"\textbf{Bold} = best. "
                 r"$\Delta$ Emp. = change in empathy score relative to full pipeline.}")
    lines.append(r"  \label{tab:ablation}")
    lines.append(r"  \begin{tabular}{lcccc}")
    lines.append(r"    \toprule")
    lines.append(r"    \textbf{Variant} & \textbf{W-F1} & \textbf{Emp.} "
                 r"& \textbf{Tone} & \textbf{$\Delta$ Emp.} \\")
    lines.append(r"    \midrule")

    # Load metrics
    metrics_list = []
    for _, var_key in variants:
        fname = ablation_dir / f"empathetic_dialogues_{var_key}.json"
        m = _load_metrics(fname) or {}
        metrics_list.append(m)

    # Get full-pipeline scores as reference
    full_emp = metrics_list[0].get("mean_empathy_score", 0.0) or 0.0

    wf1_vals  = [m.get("weighted_f1")        for m in metrics_list]
    emp_vals  = [m.get("mean_empathy_score")  for m in metrics_list]
    tone_vals = [m.get("mean_tone_match")     for m in metrics_list]

    bold_wf1  = _bold_max(wf1_vals)
    bold_emp  = _bold_max(emp_vals)
    bold_tone = _bold_max(tone_vals)

    for i, (var_name, _) in enumerate(variants):
        wf1_fmt  = bold_wf1[i]
        emp_fmt  = bold_emp[i]
        tone_fmt = bold_tone[i]

        emp_val = emp_vals[i]
        if emp_val is not None and full_emp:
            delta = emp_val - full_emp
            delta_fmt = f"{delta:+.4f}"
            if delta < -0.05:
                delta_fmt = r"\textcolor{red}{" + delta_fmt + "}"
        else:
            delta_fmt = r"\textemdash"

        row = f"    {var_name} & {wf1_fmt} & {emp_fmt} & {tone_fmt} & {delta_fmt} \\\\"
        lines.append(row)

    lines.append(r"    \bottomrule")
    lines.append(r"  \end{tabular}")
    lines.append(r"\end{table}")

    latex = "\n".join(lines)
    out_path = output_dir / "table2_ablation.tex"
    out_path.write_text(latex, encoding="utf-8")
    logger.info("Table 2 written to %s", out_path)
    return latex


# ---------------------------------------------------------------------------
# Table 3 — Per-Class F1 Breakdown
# ---------------------------------------------------------------------------

def generate_per_class_table(results_dir: Path, output_dir: Path) -> str:
    """
    Per-class F1 for SERAPH Full vs no_stage3 on MELD.
    Shows which emotion classes benefit most from self-simulation.
    """
    from pipeline.stage2_classifier import PLUTCHIK_PRIMARIES

    full_path     = results_dir / "meld_full.json"
    no_sim_path   = results_dir / "meld_no_stage3.json"

    full_metrics  = _load_metrics(full_path) or {}
    nosim_metrics = _load_metrics(no_sim_path) or {}

    full_pcf  = full_metrics.get("per_class_f1",  {})
    nosim_pcf = nosim_metrics.get("per_class_f1", {})

    # Filter to only emotions with non-zero F1 in either system
    # This removes hallucinated/irrelevant labels and keeps only the
    # dataset-relevant emotion classes (e.g. the 7 MELD classes)
    all_emotions = sorted(set(full_pcf.keys()) | set(nosim_pcf.keys()))
    emotions = [
        e for e in all_emotions
        if (full_pcf.get(e, 0) or 0) > 0 or (nosim_pcf.get(e, 0) or 0) > 0
    ]
    if not emotions:
        emotions = sorted(full_pcf.keys() or PLUTCHIK_PRIMARIES)

    lines = []
    lines.append(r"% ── Table 3: Per-Class F1 ──────────────────────────────────────")
    lines.append(r"\begin{table}[t]")
    lines.append(r"  \centering")
    lines.append(r"  \caption{Per-class F1 on MELD: SERAPH Full vs without self-simulation (Stage 3). "
                 r"\textbf{Bold} = better per row.}")
    lines.append(r"  \label{tab:per_class}")
    lines.append(r"  \begin{tabular}{lcc}")
    lines.append(r"    \toprule")
    lines.append(r"    \textbf{Emotion} & \textbf{Full} & \textbf{w/o Stage 3} \\")
    lines.append(r"    \midrule")

    for emotion in emotions:
        full_val  = full_pcf.get(emotion)
        nosim_val = nosim_pcf.get(emotion)
        row_vals  = _bold_max([full_val, nosim_val])
        lines.append(f"    {emotion.capitalize()} & {row_vals[0]} & {row_vals[1]} \\\\")

    # Summary row
    lines.append(r"    \midrule")
    lines.append(
        r"    \textit{Weighted Avg.} & "
        + _fmt(full_metrics.get("weighted_f1")) + " & "
        + _fmt(nosim_metrics.get("weighted_f1")) + r" \\"
    )
    lines.append(r"    \bottomrule")
    lines.append(r"  \end{tabular}")
    lines.append(r"\end{table}")

    latex = "\n".join(lines)
    out_path = output_dir / "table3_per_class_f1.tex"
    out_path.write_text(latex, encoding="utf-8")
    logger.info("Table 3 written to %s", out_path)
    return latex


# ---------------------------------------------------------------------------
# Table 4 — Empathy Dimension Breakdown
# ---------------------------------------------------------------------------

def generate_empathy_dimensions_table(results_dir: Path, output_dir: Path) -> str:
    """
    Empathy dimension scores for SERAPH Full, no_stage3, and Raw Claude
    on EmpatheticDialogues. 5 dimensions × 3 systems.
    """
    systems = [
        ("SERAPH Full",           "empathetic_dialogues_full.json"),
        ("w/o Stage 3",           "ablations/empathetic_dialogues_no_stage3.json"),
        ("Raw Claude",            "empathetic_dialogues_raw_claude.json"),
    ]

    dimensions = [
        ("Emotional Ack.",    "mean_emotional_acknowledgment"),
        ("Perspective-Taking","mean_perspective_taking"),
        ("Tone Match",        "mean_tone_match"),
        ("Avoids Harm",       "mean_avoids_harmful"),
        ("Meets Need",        "mean_meets_core_need"),
        (r"\textit{Overall}", "mean_empathy_score"),
    ]

    all_metrics = []
    for _, fname in systems:
        m = _load_metrics(results_dir / fname) or {}
        all_metrics.append(m)

    lines = []
    lines.append(r"% ── Table 4: Empathy Dimensions ────────────────────────────────")
    lines.append(r"\begin{table}[t]")
    lines.append(r"  \centering")
    lines.append(r"  \caption{Empathy alignment scores on EmpatheticDialogues, "
                 r"broken down by LLM-as-judge dimension (1--5 scale). "
                 r"\textbf{Bold} = best per row.}")
    lines.append(r"  \label{tab:empathy_dims}")
    col_spec = "l" + "c" * len(systems)
    lines.append(r"  \begin{tabular}{" + col_spec + r"}")
    lines.append(r"    \toprule")

    sys_headers = " & ".join(r"\textbf{" + s + "}" for s, _ in systems)
    lines.append(r"    \textbf{Dimension} & " + sys_headers + r" \\")
    lines.append(r"    \midrule")

    for dim_label, dim_key in dimensions:
        if dim_label.startswith(r"\textit"):
            lines.append(r"    \midrule")
        vals = [m.get(dim_key) for m in all_metrics]
        bold = _bold_max(vals)
        lines.append(f"    {dim_label} & " + " & ".join(bold) + r" \\")

    lines.append(r"    \bottomrule")
    lines.append(r"  \end{tabular}")
    lines.append(r"\end{table}")

    latex = "\n".join(lines)
    out_path = output_dir / "table4_empathy_dimensions.tex"
    out_path.write_text(latex, encoding="utf-8")
    logger.info("Table 4 written to %s", out_path)
    return latex


# ---------------------------------------------------------------------------
# Master runner
# ---------------------------------------------------------------------------

def generate_all_tables(
    results_dir: str = "results",
    output_dir: str = "paper/tables",
) -> None:
    rdir = Path(results_dir)
    odir = Path(output_dir)
    odir.mkdir(parents=True, exist_ok=True)

    print("\nGenerating LaTeX tables …\n")

    try:
        generate_main_results_table(rdir, odir)
        print("  ✓ Table 1 — Main Results")
    except Exception as e:
        print(f"  ✗ Table 1 failed: {e}")

    try:
        generate_ablation_table(rdir, odir)
        print("  ✓ Table 2 — Ablation Study")
    except Exception as e:
        print(f"  ✗ Table 2 failed: {e}")

    try:
        generate_per_class_table(rdir, odir)
        print("  ✓ Table 3 — Per-Class F1")
    except Exception as e:
        print(f"  ✗ Table 3 failed: {e}")

    try:
        generate_empathy_dimensions_table(rdir, odir)
        print("  ✓ Table 4 — Empathy Dimensions")
    except Exception as e:
        print(f"  ✗ Table 4 failed: {e}")

    print(f"\nAll tables saved to {odir}/\n")
    print("Include in LaTeX with:")
    for i in range(1, 5):
        stub = ["main_results", "ablation", "per_class_f1", "empathy_dimensions"][i - 1]
        print(f"  \\input{{tables/table{i}_{stub}}}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate LaTeX tables from SERAPH results")
    parser.add_argument("--results-dir", default="results",  help="Results JSON directory")
    parser.add_argument("--output",      default="paper/tables", help="Output directory for .tex files")
    args = parser.parse_args()
    generate_all_tables(results_dir=args.results_dir, output_dir=args.output)


if __name__ == "__main__":
    logging.basicConfig(format="%(levelname)-8s | %(message)s", level=logging.INFO)
    main()
