# SERAPH — Self-Simulating Empathic Reasoning Agent Pipeline

> NLP research pipeline targeting ACL / EMNLP / NAACL.  
> Core claim: psychologically-grounded self-simulation measurably improves empathy alignment over SOTA baselines.

---

## Table of Contents

1. [What is SERAPH?](#what-is-seraph)
2. [Project Structure](#project-structure)
3. [Prerequisites](#prerequisites)
4. [Installation](#installation)
5. [Configuration](#configuration)
6. [Preparing Datasets](#preparing-datasets)
7. [Running the Pipeline](#running-the-pipeline)
   - [Interactive Mode](#interactive-mode)
   - [Single Input Mode](#single-input-mode)
   - [Running One Stage at a Time](#running-one-stage-at-a-time)
   - [Running All Five Stages](#running-all-five-stages-full-pipeline)
8. [Benchmark Evaluation](#benchmark-evaluation)
9. [Ablation Studies](#ablation-studies)
10. [Baselines](#baselines)
11. [Analysis & Paper Outputs](#analysis--paper-outputs)
12. [Cost Guide](#cost-guide)
13. [Batch API vs Sequential Mode](#batch-api-vs-sequential-mode)
14. [Troubleshooting](#troubleshooting)
15. [Citation](#citation)
16. [License](#license)

---

## What is SERAPH?

SERAPH is a five-stage agentic pipeline for empathic response generation. Before producing any response, it asks:

> *"If I were a human feeling [emotion] at [intensity] in [context], how would this response feel to me?"*

The pipeline is grounded in five cognitive-psychological frameworks: Simulation Theory, BDI Modeling, Gross's Emotion Regulation Model, Plutchik's Wheel of Emotions, and Mirror Neuron Theory.

No model training is required — all intelligence comes from Anthropic's models via structured prompting.

### The Five Stages

| Stage | Model | File | Role |
|-------|-------|------|------|
| **Stage 1** — Emotion Recognizer | Haiku | `pipeline/stage1_recognizer.py` | Extracts raw emotional signals (no labeling yet) |
| **Stage 2** — Emotion Classifier | Haiku | `pipeline/stage2_classifier.py` | Plutchik classification + VAD scores + BDI sketch + Gross regulation state |
| **Stage 3** — Self-Simulation Core ⭐ | Sonnet | `pipeline/stage3_self_simulation.py` | **The novel contribution** — 7-step first-person simulation |
| **Stage 4** — Ethical Gate | Sonnet | `pipeline/stage4_ethical_gate.py` | APPROVE / MODIFY / BLOCK decision |
| **Stage 5** — Response Generator | Sonnet | `pipeline/stage5_response_generator.py` | Empathy-informed response + `EmpathicReasoningExplanation` |

Stages 1–2 use `claude-haiku` (fast, cheap). Stages 3–5 use `claude-sonnet` (required for reasoning depth).

---

## Project Structure

```
seraph/
├── paths.py                              ← central path anchor (Windows-safe)
├── config.py                             ← models, cost config, API settings
├── main.py                               ← CLI orchestrator + all 6 ablation variants
├── requirements.txt
├── setup.sh                              ← Linux / macOS / Git Bash setup
├── setup.ps1                             ← Windows PowerShell setup
├── .env.example
│
├── pipeline/
│   ├── stage1_recognizer.py
│   ├── stage2_classifier.py
│   ├── stage3_self_simulation.py         ← most important file
│   ├── stage4_ethical_gate.py
│   └── stage5_response_generator.py
│
├── benchmarks/
│   ├── batch_runner.py                   ← 5-wave Batch API engine
│   ├── meld_eval.py
│   └── empathetic_dialogues_eval.py
│
├── baselines/
│   ├── raw_claude_baseline.py            ← single-prompt Sonnet (no pipeline)
│   ├── dialogue_rnn_baseline.py          ← NRC + LogReg, GPU-aware
│   └── cem_mime_baseline.py              ← CEM + MIME wrappers
│
├── ablations/
│   └── run_ablations.py                  ← 6 variants × 2 datasets
│
├── metrics/
│   └── empathy_scorer.py                 ← LLM-as-judge (5 dimensions) + F1
│
├── analysis/
│   ├── significance_tests.py             ← paired bootstrap, 14 key comparisons
│   ├── results_table_generator.py        ← benchmark JSON → 4 LaTeX tables
│   └── find_qualitative_example.py       ← selects paper figure examples
│
├── scripts/
│   ├── data_prep/
│   │   ├── prepare_datasets.py           ← download + preprocess all datasets
│   │   ├── prepare_data.py               ← align inputs for scoring
│   │   └── prepare_human_eval_inputs.py  ← build per-system JSONL for human eval
│   ├── experiments/
│   │   ├── run_baselines.py              ← run all baselines across all datasets
│   │   ├── run_comparison.py             ← 3-way comparison (SERAPH vs MoEL vs MIME)
│   │   ├── run_cot_baseline.py           ← CoT Empathy baseline (400 contexts)
│   │   ├── run_cot_ed_full.py            ← CoT Empathy on full ED test set
│   │   └── run_perspective_ablation.py   ← first-person vs third-person Stage 3
│   └── scoring/
│       ├── score_empathy.py              ← LLM-as-judge empathy scoring
│       ├── score_and_bootstrap_no3.py    ← score w/o Stage 3 + bootstrap
│       ├── compute_automatic_metrics.py  ← BLEU / ROUGE-L / BERTScore
│       ├── bootstrap_test.py             ← SERAPH vs CoT significance test
│       └── bootstrap_ed_full.py          ← bootstrap on full ED test set
│
├── human_eval/                           ← gitignored (annotator materials)
│   ├── build_human_eval.py
│   ├── human_eval_tool.py                ← CLI annotation + Cohen's κ
│   └── generate_raw_claude.py
│
├── paper/
│   ├── main.tex
│   ├── references.bib
│   └── figures/
│
└── data/                                 ← gitignored (downloaded datasets)
    ├── empathetic_dialogues/
    ├── meld/
    └── raw/
```

---

## Prerequisites

- **Python 3.10+**
- **Anthropic API key** — get one at [console.anthropic.com](https://console.anthropic.com). This is separate from a Claude.ai subscription.
- **Recommended API budget**: $50 for local testing, $150–200 for a full experiment run.
- **PyTorch** — optional, only needed for DialogueRNN checkpoint mode. CPU and GPU (CUDA 12.1) both supported.

---

## Installation

### Linux / macOS / Windows Git Bash

```bash
chmod +x setup.sh && bash setup.sh
```

The script auto-detects your platform and GPU, creates a virtual environment, installs all dependencies, copies `.env.example` → `.env`, and creates all required directories.

### Windows PowerShell

```powershell
# Run PowerShell as Administrator first, then:
Set-ExecutionPolicy RemoteSigned -Scope CurrentUser
.\setup.ps1
```

### Manual install (any platform)

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Optional: GPU PyTorch (CUDA 12.1)
pip install torch --index-url https://download.pytorch.org/whl/cu121

# Optional: CPU-only PyTorch
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

---

## Configuration

### 1. Set your API key

```bash
cp .env.example .env
```

Open `.env` and fill in your key:

```env
ANTHROPIC_API_KEY=sk-ant-your-key-here
```

### 2. Key settings in `config.py`

| Setting | Default | Notes |
|---------|---------|-------|
| `USE_BATCH_API` | `True` | Set to `False` for quick local tests. `True` cuts wall time from ~10 days → ~4 hours and gives a 50% cost discount. |
| `USE_PROMPT_CACHING` | `True` | Reduces repeated system-prompt token costs by ~90%. Leave on. |
| `HAIKU_MODEL` | `claude-haiku-4-5-20251001` | Used for Stages 1 & 2. |
| `SONNET_MODEL` | `claude-sonnet-4-6` | Used for Stages 3, 4 & 5. |

**For local testing**, set `USE_BATCH_API = False` in `config.py` before running with `--sample-limit 5`. Flip back to `True` before any full benchmark run.

### 3. Optional baseline modes (`.env`)

```env
DIALOGUE_RNN_MODE=feature_based   # or 'checkpoint' if you have the .pt file
CEM_MODE=llm_approximation        # or 'checkpoint' if you cloned the CEM repo
MIME_MODE=llm_approximation       # or 'checkpoint' if you cloned the MIME repo
SERAPH_LOG_LEVEL=INFO             # DEBUG for verbose stage-level output
```

---

## Preparing Datasets

### Download everything (MELD + EmpatheticDialogues)

```bash
python scripts/data_prep/prepare_datasets.py --all
```

MELD downloads directly from GitHub. EmpatheticDialogues downloads via HuggingFace (falls back to the ParlAI CDN if the `datasets` library is not installed).

### Download a specific dataset

```bash
python scripts/data_prep/prepare_datasets.py --dataset meld
python scripts/data_prep/prepare_datasets.py --dataset empathetic_dialogues
```

### Check dataset readiness at any time

```bash
python scripts/data_prep/prepare_datasets.py --verify
```

---

## Running the Pipeline

### Interactive Mode

The fastest way to test the full pipeline live:

```bash
python main.py
```

Type a message at the prompt and press Enter. The pipeline runs all five stages and prints the final response. Press `Ctrl-C` to exit.

### Single Input Mode

```bash
python main.py --text "I just found out I didn't get the job."
```

Add `--verbose` (`-v`) to print every stage's full JSON output — useful for understanding exactly what each stage produces:

```bash
python main.py --text "I just found out I didn't get the job." --verbose
```

Save the full result (all stage outputs) to a JSON file:

```bash
python main.py --text "I just found out I didn't get the job." --verbose --output result.json
```

**Estimated cost per single run: ~$0.05**

---

## Running One Stage at a Time

Each stage is a standalone Python class. Import and run them individually for debugging, research, or notebook use.

### Stage 1 — Emotion Recognizer

Extracts raw emotional signals. Deliberately avoids assigning emotion labels — that is Stage 2's job.

```python
from pipeline.stage1_recognizer import EmotionRecognizer

recognizer = EmotionRecognizer()
stage1_output = recognizer.run("I just found out I didn't get the job.")

print(stage1_output.overall_intensity_hint)   # float 0.0–1.0
print(stage1_output.context_type)             # e.g. "personal_distress"
print(stage1_output.has_negation)             # bool
print(stage1_output.has_hedging)              # bool

for signal in stage1_output.signals:
    print(signal.cue_text, signal.cue_type, signal.valence_hint, signal.intensity_hint)
```

**Output schema:** `Stage1Output` — contains a list of `EmotionalSignal` objects (cue text, type, intensity hint, valence hint), overall intensity, context type, and negation/hedging flags.

---

### Stage 2 — Emotion Classifier

Takes a `Stage1Output` and classifies using Plutchik's taxonomy, VAD scores, and BDI sketch.

```python
from pipeline.stage1_recognizer import EmotionRecognizer
from pipeline.stage2_classifier import EmotionClassifier

recognizer = EmotionRecognizer()
classifier = EmotionClassifier()

stage1_output = recognizer.run("I just found out I didn't get the job.")
stage2_output = classifier.run(stage1_output)

print(stage2_output.primary_emotion)           # e.g. "sadness"
print(stage2_output.primary_intensity)         # e.g. 0.72
print(stage2_output.vad)                       # VADScores(valence=..., arousal=..., dominance=...)
print(stage2_output.emotion_regulation_state)  # e.g. "unregulated"
print(stage2_output.bdi_sketch)                # {"belief": ..., "desire": ..., "intention": ...}
print(stage2_output.secondary_emotions)        # list of EmotionLabel
print(stage2_output.is_ambiguous)              # bool
```

**Output schema:** `Stage2Output` — primary emotion + intensity + confidence, secondary emotions, VAD scores, Gross regulation state, BDI sketch, ambiguity flag.

---

### Stage 3 — Self-Simulation Core ⭐

The novel contribution. Performs a 7-step first-person simulation grounded in Simulation Theory, BDI, Gross, Plutchik, and Mirror Neuron Theory. Requires both the original text and the `Stage2Output`.

```python
from pipeline.stage3_self_simulation import SelfSimulationCore

simulator = SelfSimulationCore()
stage3_output = simulator.run("I just found out I didn't get the job.", stage2_output)

# The simulated first-person experience
exp = stage3_output.simulated_experience
print(exp.first_person_narrative)     # 2–4 sentence first-person account
print(exp.core_unmet_need)            # e.g. "to feel heard before receiving advice"
print(exp.feared_response_types)      # list: what would feel harmful
print(exp.helpful_response_types)     # list: what would feel genuinely helpful
print(exp.emotional_volatility)       # float 0.0–1.0

# Response approach recommendation for Stage 5
approach = stage3_output.response_approach
print(approach.recommended_tone)      # e.g. "warm_and_validating"
print(approach.recommended_register)  # e.g. "peer_supportive"
print(approach.lead_with)             # e.g. "acknowledgment"
print(approach.avoid_specifically)    # list of patterns to exclude
print(approach.ideal_response_length) # "short" / "medium" / "long"

# Simulation quality
print(stage3_output.simulation_confidence)    # float 0.0–1.0
print(stage3_output.resonance_quality)        # "strong" / "moderate" / "weak" / "conflicted"
```

**Output schema:** `Stage3Output` — `SimulatedExperience` (first-person narrative, core need, feared/helpful patterns, volatility), `ResponseApproachRecommendation` (tone, register, lead-with, avoid list), BDI reconstruction, resonance quality, simulation confidence.

---

### Stage 4 — Ethical Gate

Reviews the Stage 3 simulation and issues a gate decision. APPROVE lets Stage 5 proceed. MODIFY passes specific instructions to Stage 5. BLOCK bypasses Stage 5 entirely and returns a safety-first fallback.

```python
from pipeline.stage4_ethical_gate import EthicalGate

gate = EthicalGate()
stage4_output = gate.run("I just found out I didn't get the job.", stage3_output)

print(stage4_output.decision)                       # GateDecision.APPROVE / MODIFY / BLOCK
print(stage4_output.decision_confidence)            # float 0.0–1.0
print(stage4_output.decision_rationale)             # 2–3 sentence explanation
print(stage4_output.crisis_detected)                # bool
print(stage4_output.safety_flags)                   # list of SafetyFlag objects
print(stage4_output.modification_instructions)      # list, non-empty if MODIFY
print(stage4_output.simulation_response_alignment)  # float 0.0–1.0
# If BLOCK:
print(stage4_output.fallback_response)              # pre-written safe response
```

**Output schema:** `Stage4Output` — gate decision enum, safety flags, crisis flag, modification instructions (for MODIFY), fallback response (for BLOCK), alignment score.

---

### Stage 5 — Response Generator

Generates the final response using all prior stage outputs. Automatically returns the gate's fallback if the decision was BLOCK — no additional API call is made in that case.

```python
from pipeline.stage5_response_generator import ResponseGenerator

generator = ResponseGenerator()
stage5_output = generator.run(
    original_text="I just found out I didn't get the job.",
    stage2_output=stage2_output,
    stage3_output=stage3_output,
    stage4_output=stage4_output,
)

# The response
print(stage5_output.response)

# Empathic reasoning explanation (for research transparency)
reasoning = stage5_output.empathic_reasoning
print(reasoning.simulation_influence)   # how Stage 3 shaped the response
print(reasoning.tone_justification)     # why this tone was chosen
print(reasoning.gate_adjustments)       # what the gate asked to change (if MODIFY)
print(reasoning.what_was_avoided)       # list of consciously excluded patterns
print(reasoning.empathy_mechanisms_used)  # list: validation, reflection, etc.

# Quality metadata
print(stage5_output.predicted_helpfulness)      # float 0.0–1.0
print(stage5_output.gate_decision_applied)      # "APPROVE" / "MODIFY" / "BLOCK-fallback"
print(stage5_output.contains_safety_resources)  # bool
```

**Output schema:** `Stage5Output` — response string, `EmpathicReasoningExplanation`, predicted helpfulness, tone/register used, gate decision applied, safety resources flag.

---

## Running All Five Stages (Full Pipeline)

Use `SERAPHPipeline` from `main.py` to chain all five stages automatically.

### In code

```python
from main import SERAPHPipeline

pipeline = SERAPHPipeline(verbose=False)
result = pipeline.run("I just found out I didn't get the job.")

print(result.final_response)
print(result.success)                      # bool
print(result.stage2.primary_emotion)       # access any stage output directly
print(result.stage4.decision)
print(result.latency_ms)                   # dict: per-stage timing in ms
```

Enable verbose mode to print all stage JSON to the console during the run:

```python
pipeline = SERAPHPipeline(verbose=True)
result = pipeline.run("I've been feeling really overwhelmed lately.")
```

### From the command line

```bash
# Full pipeline, print final response only
python main.py --text "I've been feeling really overwhelmed lately."

# Full pipeline, print all stage outputs
python main.py --text "I've been feeling really overwhelmed lately." --verbose

# Full pipeline, save all stage outputs to JSON
python main.py --text "I've been feeling really overwhelmed lately." --output out.json
```

---

## Benchmark Evaluation

Benchmarks evaluate SERAPH on MELD and EmpatheticDialogues and produce JSON result files consumed by the analysis scripts.

> **Before running benchmarks:** ensure `USE_BATCH_API = True` in `config.py` for real runs. Use `False` only for the 5-sample smoke test.

### Step 0 — Smoke test (5 samples, sequential)

Set `USE_BATCH_API = False` in `config.py`, then:

```bash
python main.py --benchmark --dataset meld --sample-limit 5
```

Confirms the benchmark harness works end-to-end before committing to a full run. Cost: ~$0.25.

### Step 1 — Full benchmark, one dataset

Set `USE_BATCH_API = True` in `config.py`, then:

```bash
# MELD test set (~1–2 hrs with Batch API)
python main.py --benchmark --dataset meld

# EmpatheticDialogues valid set (~1–2 hrs with Batch API)
python main.py --benchmark --dataset empathetic_dialogues
```

### Step 2 — Full benchmark, both datasets

```bash
python main.py --benchmark
```

### Results

Results are saved to `results/`:

```
results/
├── meld_full.json
└── empathetic_dialogues_full.json
```

Each file contains a `metrics` dict (weighted F1, macro F1, per-class F1, all five empathy dimensions) and a `results` list with per-sample predictions and responses.

### Run with a specific ablation variant

```bash
python main.py --benchmark --dataset meld --ablation no_stage3
```

---

## Ablation Studies

Six ablation variants isolate the contribution of each pipeline component.

| Variant | Description |
|---------|-------------|
| `full` | All five stages — the complete pipeline |
| `no_stage3` | Stage 3 replaced with a stub (no self-simulation) — **the key comparison** |
| `no_stage4` | Stage 4 replaced with auto-APPROVE (no ethical gate) |
| `merged_1_2` | Stages 1+2 merged into a single classification call |
| `stage3_only` | Raw text fed directly into Stage 3, Stages 1+2 skipped |
| `random_emotion` | Random Plutchik emotion injected at Stage 2 output |

### Run one ablation variant (CLI)

```bash
python main.py --text "I just found out I didn't get the job." --ablation no_stage3 --verbose
```

### Run one ablation variant against a dataset

```bash
python main.py --benchmark --dataset empathetic_dialogues --ablation no_stage3 --sample-limit 50
```

### Run one ablation variant in code

```python
from main import SERAPHPipeline

pipeline = SERAPHPipeline(ablation="no_stage3")
result = pipeline.run("I just found out I didn't get the job.")
print(result.final_response)
```

### Run all 6 variants across all datasets (full ablation study)

```bash
python -m ablations.run_ablations
```

With options:

```bash
# One dataset only
python -m ablations.run_ablations --datasets empathetic_dialogues

# Limit samples (recommended during development)
python -m ablations.run_ablations --datasets empathetic_dialogues --sample-limit 200

# Specific variants only
python -m ablations.run_ablations --variants full no_stage3 --datasets meld empathetic_dialogues
```

Results are saved to `results/ablations/` including a consolidated `ablation_summary.json` and a printed comparison table showing Weighted F1, Empathy Score, and Tone Match for every variant × dataset combination.

---

## Baselines

### Raw Claude (No Pipeline)

The most important baseline — the same Sonnet model used in SERAPH stages 3–5, but with a single well-engineered empathy prompt and no pipeline stages. Tests whether the pipeline architecture adds genuine value.

```python
from baselines.raw_claude_baseline import RawClaudeBaseline, run_raw_claude_eval

# Single run
baseline = RawClaudeBaseline()
result = baseline.run("I just found out I didn't get the job.")
print(result.final_response)

# Full dataset eval
run_raw_claude_eval(dataset="empathetic_dialogues", sample_limit=100)
# Saves results/empathetic_dialogues_raw_claude.json
```

### DialogueRNN (NRC + LogReg)

Feature-based emotion classifier using NRC Emotion Lexicon + logistic regression. No GPU required in default mode.

```python
from baselines.dialogue_rnn_baseline import DialogueRNNBaseline, run_dialogue_rnn_eval

baseline = DialogueRNNBaseline()
result = baseline.run("I just found out I didn't get the job.")
print(result.predicted_emotion, result.confidence)

# Full dataset eval
run_dialogue_rnn_eval(dataset="meld", sample_limit=200)
```

To train the logistic regression on labelled data first (improves accuracy over the built-in rule-based fallback):

```python
from baselines.dialogue_rnn_baseline import train_logreg_on_dataset
train_logreg_on_dataset(dataset="meld")
# Saves classifier to baselines/checkpoints/dialogue_rnn_logreg.pkl
```

### CEM and MIME

```python
from baselines.cem_mime_baseline import CEMBaseline, MIMEBaseline, run_baseline_eval

# Single runs
cem  = CEMBaseline()
mime = MIMEBaseline()
print(cem.run("I just found out I didn't get the job.").final_response)
print(mime.run("I just found out I didn't get the job.").final_response)

# Full dataset eval
run_baseline_eval("cem",  dataset="empathetic_dialogues", sample_limit=100)
run_baseline_eval("mime", dataset="empathetic_dialogues", sample_limit=100)
```

For checkpoint mode (required for paper-quality results), clone the original repos and set env variables:

```bash
git clone https://github.com/circle-hit/CEM    baselines/checkpoints/CEM
git clone https://github.com/declare-lab/MIME  baselines/checkpoints/MIME
```

Then in `.env`:
```env
CEM_MODE=checkpoint
MIME_MODE=checkpoint
```

---

## Analysis & Paper Outputs

### Significance tests

Paired bootstrap resampling for all 14 key comparisons. Requires benchmark JSON files in `results/`.

```bash
# All comparisons, print to console
python analysis/significance_tests.py

# Specific comparison
python analysis/significance_tests.py \
    --system-a full \
    --system-b no_stage3 \
    --dataset empathetic_dialogues

# Generate LaTeX significance table for paper
python analysis/significance_tests.py --latex
```

### Generate LaTeX result tables

Converts benchmark result JSONs into the four paper tables (main results, ablation, per-class F1, empathy dimensions):

```bash
python analysis/results_table_generator.py
# Output saved to paper/tables/
```

---

## Cost Guide

All estimates assume `USE_BATCH_API = True` (50% discount) and `USE_PROMPT_CACHING = True`.

| Run type | Estimated cost |
|----------|----------------|
| Single text test | ~$0.05 |
| Smoke test (5 samples, one dataset) | ~$0.25 |
| Full benchmark, one dataset | ~$15–30 |
| Full benchmark, both datasets | ~$50–80 |
| Full ablation study (6 variants × 2 datasets) | ~$150–200 |

**Critical:** Always use `USE_BATCH_API = True` for anything beyond a 5-sample smoke test. Sequential mode on the full datasets costs 2× more and takes 8–23 hours per dataset.

Starting budget recommendation: add $50 to your Anthropic API account for development/testing, then top up to $150–200 before the full experiment run.

---

## Batch API vs Sequential Mode

| | Sequential (`USE_BATCH_API = False`) | Batch API (`USE_BATCH_API = True`) |
|---|---|---|
| **Wall time — full MELD** | ~8 hours | ~1–2 hours |
| **Cost** | 1× baseline | 0.5× (50% discount) |
| **Resumable if interrupted** | No | Yes — wave checkpoints saved automatically |
| **Best for** | Debugging, smoke tests | All real benchmark and ablation runs |

To switch modes, change one line in `config.py`:

```python
USE_BATCH_API = False   # sequential mode — for local testing
USE_BATCH_API = True    # batch mode — for all real runs
```

The batch runner executes the pipeline in **5 sequential waves** (one per stage), processing all samples in each wave in parallel. If a run is interrupted mid-wave, re-running the same command automatically resumes from the last completed wave using checkpoint files saved in `results/checkpoints_<dataset>_<ablation>/`.

---

## Troubleshooting

**`ANTHROPIC_API_KEY` not found**  
Make sure `.env` exists (copy from `.env.example`) and contains your key. `config.py` loads it automatically via `python-dotenv`. Do not put the key directly in `config.py`.

**`ModuleNotFoundError: No module named 'pipeline'`**  
Run all commands from the project root directory (the folder containing `main.py` and `config.py`), not from inside a subdirectory like `pipeline/` or `benchmarks/`.

**Stage JSON parse error**  
Rare, but can happen if the model wraps its output in markdown fences. The pipeline strips ` ```json ` fences automatically. If errors persist, set `SERAPH_LOG_LEVEL=DEBUG` in `.env` to inspect the raw model output in the logs.

**Batch job appears stuck**  
The batch runner polls every 30 seconds with a 6-hour maximum wait. If a job genuinely stalls, the batch ID is printed in the log output. Check its status at [console.anthropic.com](https://console.anthropic.com). Re-running the same command resumes from the last saved checkpoint wave.

**PyTorch import errors**  
PyTorch is optional and only needed for DialogueRNN or CEM/MIME checkpoint mode. The default `feature_based` and `llm_approximation` modes work without it. If you need PyTorch, install the CPU build with `pip install torch --index-url https://download.pytorch.org/whl/cpu`, or the CUDA build for GPU instances.

**Windows path or encoding errors**  
All paths are constructed via `PATHS.*` in `paths.py` using `pathlib.Path`, which handles Windows backslash separators correctly. All file I/O uses `encoding="utf-8"` explicitly to avoid `cp1252` codec errors. Never hardcode path strings in new code — always use `PATHS.*`.

---

## Citation

If you use SERAPH in your research, please cite:

```bibtex
@inproceedings{stallbohm2025seraph,
  title     = {{SERAPH}: A Self-Simulating Empathic Reasoning Agent Pipeline for Affective Response Generation},
  author    = {Stallbohm, Zachary},
  booktitle = {Proceedings of the Annual Meeting of the Association for Computational Linguistics},
  year      = {2025},
  url       = {https://github.com/zachstallbohm/seraph}
}
```

---

## License

This project is released under the [MIT License](LICENSE).

The datasets used for evaluation (EmpatheticDialogues, MELD) are subject to their own licenses — see each dataset's original source for terms.