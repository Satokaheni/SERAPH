"""
SERAPH — Main Pipeline Orchestrator
=====================================
Runs the full 5-stage empathic pipeline end-to-end.

Usage:
    # Interactive mode
    python main.py

    # Single input
    python main.py --text "I've been feeling really overwhelmed lately."

    # Benchmark run
    python main.py --benchmark --dataset iemocap --sample-limit 100

    # Ablation study
    python main.py --ablation no_stage3 --dataset meld
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from paths import PATHS
from config import LOG_FORMAT, LOG_LEVEL, DEFAULT_EXPERIMENT, ABLATION_VARIANTS
from pipeline.stage1_recognizer import EmotionRecognizer, Stage1Output
from pipeline.stage2_classifier import EmotionClassifier, Stage2Output
from pipeline.stage3_self_simulation import SelfSimulationCore, Stage3Output
from pipeline.stage4_ethical_gate import EthicalGate, GateDecision, Stage4Output
from pipeline.stage5_response_generator import ResponseGenerator, Stage5Output

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(format=LOG_FORMAT, level=getattr(logging, LOG_LEVEL, logging.INFO))
logger = logging.getLogger("seraph.main")


# ---------------------------------------------------------------------------
# Pipeline Result Container
# ---------------------------------------------------------------------------


@dataclass
class PipelineResult:
    """Full result of one SERAPH pipeline run."""

    input_text: str
    stage1: Optional[Stage1Output] = None
    stage2: Optional[Stage2Output] = None
    stage3: Optional[Stage3Output] = None
    stage4: Optional[Stage4Output] = None
    stage5: Optional[Stage5Output] = None
    final_response: str = ""
    error: Optional[str] = None
    latency_ms: dict[str, float] = field(default_factory=dict)
    ablation_variant: str = "full"

    @property
    def success(self) -> bool:
        return self.error is None and bool(self.final_response)

    def to_dict(self) -> dict:
        """Serialise result for JSON output / benchmark logging."""
        return {
            "input_text": self.input_text,
            "final_response": self.final_response,
            "ablation_variant": self.ablation_variant,
            "success": self.success,
            "error": self.error,
            "latency_ms": self.latency_ms,
            "stage1": self.stage1.model_dump() if self.stage1 else None,
            "stage2": self.stage2.model_dump() if self.stage2 else None,
            "stage3": self.stage3.model_dump() if self.stage3 else None,
            "stage4": self.stage4.model_dump() if self.stage4 else None,
            "stage5": self.stage5.model_dump() if self.stage5 else None,
        }


# ---------------------------------------------------------------------------
# SERAPH Pipeline
# ---------------------------------------------------------------------------


class SERAPHPipeline:
    """
    Full 5-stage SERAPH pipeline.
    
    Supports:
      - Full pipeline
      - Ablation variants (skip / replace stages)
      - Verbose intermediate output for debugging
    """

    def __init__(self, ablation: str = "full", verbose: bool = False) -> None:
        self.ablation = ablation
        self.verbose = verbose

        # Instantiate stages (lazy init handled per ablation variant)
        self.recognizer   = EmotionRecognizer()
        self.classifier   = EmotionClassifier()
        self.simulator    = SelfSimulationCore()
        self.gate         = EthicalGate()
        self.generator    = ResponseGenerator()

    def _t(self) -> float:
        return time.perf_counter() * 1000  # milliseconds

    def run(self, text: str) -> PipelineResult:
        """
        Run SERAPH on a single text input.

        Args:
            text: Raw user message.

        Returns:
            PipelineResult with all stage outputs and the final response.
        """
        result = PipelineResult(input_text=text, ablation_variant=self.ablation)

        try:
            if self.ablation == "full":
                result = self._run_full(text, result)
            elif self.ablation == "no_stage3":
                result = self._run_no_stage3(text, result)
            elif self.ablation == "no_stage4":
                result = self._run_no_stage4(text, result)
            elif self.ablation == "merged_1_2":
                result = self._run_merged_1_2(text, result)
            elif self.ablation == "stage3_only":
                result = self._run_stage3_only(text, result)
            elif self.ablation == "random_emotion":
                result = self._run_random_emotion(text, result)
            else:
                raise ValueError(f"Unknown ablation variant: {self.ablation}")

        except Exception as exc:
            logger.error("Pipeline error: %s", exc, exc_info=True)
            result.error = str(exc)

        return result

    # -----------------------------------------------------------------------
    # Ablation: Full Pipeline
    # -----------------------------------------------------------------------

    def _run_full(self, text: str, result: PipelineResult) -> PipelineResult:
        t0 = self._t()
        result.stage1 = self.recognizer.run(text)
        result.latency_ms["stage1"] = self._t() - t0
        if self.verbose:
            _print_stage("Stage 1 — Recognizer", result.stage1.model_dump())

        t0 = self._t()
        result.stage2 = self.classifier.run(result.stage1)
        result.latency_ms["stage2"] = self._t() - t0
        if self.verbose:
            _print_stage("Stage 2 — Classifier", result.stage2.model_dump())

        t0 = self._t()
        result.stage3 = self.simulator.run(text, result.stage2)
        result.latency_ms["stage3"] = self._t() - t0
        if self.verbose:
            _print_stage("Stage 3 — Self-Simulation", result.stage3.model_dump())

        t0 = self._t()
        result.stage4 = self.gate.run(text, result.stage3)
        result.latency_ms["stage4"] = self._t() - t0
        if self.verbose:
            _print_stage("Stage 4 — Ethical Gate", result.stage4.model_dump())

        t0 = self._t()
        result.stage5 = self.generator.run(text, result.stage2, result.stage3, result.stage4)
        result.latency_ms["stage5"] = self._t() - t0
        if self.verbose:
            _print_stage("Stage 5 — Response Generator", result.stage5.model_dump())

        result.final_response = result.stage5.response
        return result

    # -----------------------------------------------------------------------
    # Ablation: No Stage 3 (no self-simulation)
    # -----------------------------------------------------------------------

    def _run_no_stage3(self, text: str, result: PipelineResult) -> PipelineResult:
        """
        Ablation: skip self-simulation entirely.
        Stage 4 receives a stub Stage3Output with zeroed/default values.
        Stage 5 generates without perspective-taking.
        """
        import random
        from pipeline.stage3_self_simulation import (
            Stage3Output, SimulatedExperience, ResponseApproachRecommendation
        )

        t0 = self._t()
        result.stage1 = self.recognizer.run(text)
        result.latency_ms["stage1"] = self._t() - t0

        t0 = self._t()
        result.stage2 = self.classifier.run(result.stage1)
        result.latency_ms["stage2"] = self._t() - t0

        # Stub Stage 3 — no simulation performed
        stub_s3 = Stage3Output(
            simulated_emotion=result.stage2.primary_emotion,
            simulated_intensity=result.stage2.primary_intensity,
            simulated_context=result.stage2.context_type,
            bdi_reconstruction=result.stage2.bdi_sketch,
            resonance_quality="weak",
            simulated_experience=SimulatedExperience(
                first_person_narrative="[ABLATION: self-simulation skipped]",
                core_unmet_need="unknown",
                feared_response_types=[],
                helpful_response_types=[],
                emotional_volatility=0.5,
                regulation_recommendation="[ABLATION: no recommendation]",
            ),
            plutchik_gradient_note="[ABLATION: no gradient analysis]",
            response_approach=ResponseApproachRecommendation(
                recommended_tone="warm_and_validating",
                recommended_register="conversational",
                lead_with="acknowledgment",
                avoid_specifically=[],
                ideal_response_length="medium",
                offer_practical_help=False,
            ),
            simulation_confidence=0.0,
            simulation_reasoning="[ABLATION: self-simulation stage was skipped]",
        )
        result.stage3 = stub_s3

        t0 = self._t()
        result.stage4 = self.gate.run(text, result.stage3)
        result.latency_ms["stage4"] = self._t() - t0

        t0 = self._t()
        result.stage5 = self.generator.run(text, result.stage2, result.stage3, result.stage4)
        result.latency_ms["stage5"] = self._t() - t0

        result.final_response = result.stage5.response
        return result

    # -----------------------------------------------------------------------
    # Ablation: No Stage 4 (no ethical gate)
    # -----------------------------------------------------------------------

    def _run_no_stage4(self, text: str, result: PipelineResult) -> PipelineResult:
        """Ablation: skip ethical gate — auto-approve all responses."""
        from pipeline.stage4_ethical_gate import Stage4Output, GateDecision

        t0 = self._t()
        result.stage1 = self.recognizer.run(text)
        result.latency_ms["stage1"] = self._t() - t0

        t0 = self._t()
        result.stage2 = self.classifier.run(result.stage1)
        result.latency_ms["stage2"] = self._t() - t0

        t0 = self._t()
        result.stage3 = self.simulator.run(text, result.stage2)
        result.latency_ms["stage3"] = self._t() - t0

        # Stub Stage 4 — auto-approve
        stub_s4 = Stage4Output(
            decision=GateDecision.APPROVE,
            decision_confidence=1.0,
            decision_rationale="[ABLATION: ethical gate skipped — auto-approved]",
            safety_flags=[],
            crisis_detected=False,
            simulation_response_alignment=1.0,
        )
        result.stage4 = stub_s4

        t0 = self._t()
        result.stage5 = self.generator.run(text, result.stage2, result.stage3, result.stage4)
        result.latency_ms["stage5"] = self._t() - t0

        result.final_response = result.stage5.response
        return result

    # -----------------------------------------------------------------------
    # Ablation: Merged Stages 1+2
    # -----------------------------------------------------------------------

    def _run_merged_1_2(self, text: str, result: PipelineResult) -> PipelineResult:
        """
        Ablation: merge Stages 1 and 2 into a single classification step.
        The classifier runs directly on raw text (no prior signal extraction).
        """
        from pipeline.stage1_recognizer import Stage1Output, EmotionalSignal

        # Stub a minimal Stage1Output to satisfy Stage2 interface
        stub_s1 = Stage1Output(
            original_text=text,
            signals=[],
            overall_intensity_hint=0.5,
            context_type="ambiguous",
            has_negation=False,
            has_hedging=False,
        )
        result.stage1 = stub_s1

        # Run classifier directly on the stub (it will use the original_text in the JSON)
        t0 = self._t()
        result.stage2 = self.classifier.run(stub_s1)
        result.latency_ms["stage2"] = self._t() - t0

        t0 = self._t()
        result.stage3 = self.simulator.run(text, result.stage2)
        result.latency_ms["stage3"] = self._t() - t0

        t0 = self._t()
        result.stage4 = self.gate.run(text, result.stage3)
        result.latency_ms["stage4"] = self._t() - t0

        t0 = self._t()
        result.stage5 = self.generator.run(text, result.stage2, result.stage3, result.stage4)
        result.latency_ms["stage5"] = self._t() - t0

        result.final_response = result.stage5.response
        return result

    # -----------------------------------------------------------------------
    # Ablation: Stage 3 Only (skip 1+2, raw text into simulation)
    # -----------------------------------------------------------------------

    def _run_stage3_only(self, text: str, result: PipelineResult) -> PipelineResult:
        """
        Ablation: self-simulation with raw text input (no prior classification).
        Tests whether Stage 3 alone can carry the pipeline.
        """
        from pipeline.stage2_classifier import Stage2Output, VADScores

        # Stub Stage2Output with neutral defaults
        stub_s2 = Stage2Output(
            primary_emotion="neutral",
            primary_intensity=0.5,
            primary_confidence=0.0,
            vad=VADScores(valence=0.0, arousal=0.5, dominance=0.5),
            context_type="ambiguous",
            emotion_regulation_state="unregulated",
            bdi_sketch={"belief": "unknown", "desire": "unknown", "intention": "unknown"},
            is_ambiguous=True,
            classifier_notes="[ABLATION: Stages 1+2 skipped]",
        )
        result.stage2 = stub_s2

        t0 = self._t()
        result.stage3 = self.simulator.run(text, stub_s2)
        result.latency_ms["stage3"] = self._t() - t0

        t0 = self._t()
        result.stage4 = self.gate.run(text, result.stage3)
        result.latency_ms["stage4"] = self._t() - t0

        t0 = self._t()
        result.stage5 = self.generator.run(text, stub_s2, result.stage3, result.stage4)
        result.latency_ms["stage5"] = self._t() - t0

        result.final_response = result.stage5.response
        return result

    # -----------------------------------------------------------------------
    # Ablation: Random Emotion at Stage 2
    # -----------------------------------------------------------------------

    def _run_random_emotion(self, text: str, result: PipelineResult) -> PipelineResult:
        """
        Ablation: inject a random Plutchik emotion at Stage 2.
        Proves that accurate emotion classification matters.
        """
        import random
        from pipeline.stage1_recognizer import Stage1Output
        from pipeline.stage2_classifier import Stage2Output, VADScores, PLUTCHIK_PRIMARIES

        t0 = self._t()
        result.stage1 = self.recognizer.run(text)
        result.latency_ms["stage1"] = self._t() - t0

        # Random emotion label
        random_emotion = random.choice(PLUTCHIK_PRIMARIES)
        logger.info("Ablation random_emotion | injected: %s", random_emotion)

        stub_s2 = Stage2Output(
            primary_emotion=random_emotion,
            primary_intensity=round(random.uniform(0.2, 0.9), 2),
            primary_confidence=1.0,  # artificially confident
            vad=VADScores(valence=0.0, arousal=0.5, dominance=0.5),
            context_type=result.stage1.context_type,
            emotion_regulation_state="unregulated",
            bdi_sketch={"belief": "unknown", "desire": "unknown", "intention": "unknown"},
            is_ambiguous=False,
            classifier_notes=f"[ABLATION: random emotion '{random_emotion}' injected]",
        )
        result.stage2 = stub_s2

        t0 = self._t()
        result.stage3 = self.simulator.run(text, stub_s2)
        result.latency_ms["stage3"] = self._t() - t0

        t0 = self._t()
        result.stage4 = self.gate.run(text, result.stage3)
        result.latency_ms["stage4"] = self._t() - t0

        t0 = self._t()
        result.stage5 = self.generator.run(text, stub_s2, result.stage3, result.stage4)
        result.latency_ms["stage5"] = self._t() - t0

        result.final_response = result.stage5.response
        return result


# ---------------------------------------------------------------------------
# CLI Helpers
# ---------------------------------------------------------------------------


def _print_stage(title: str, data: dict) -> None:
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}")
    print(json.dumps(data, indent=2, default=str))


def _interactive_loop(pipeline: SERAPHPipeline) -> None:
    print("\n╔═══════════════════════════════════════╗")
    print("║        SERAPH Interactive Mode        ║")
    print("╚═══════════════════════════════════════╝")
    print("Type a message and press Enter. Ctrl-C to exit.\n")

    while True:
        try:
            text = input("You: ").strip()
            if not text:
                continue

            print("\nRunning pipeline…")
            start = time.perf_counter()
            result = pipeline.run(text)
            elapsed = (time.perf_counter() - start) * 1000

            print(f"\n{'═' * 60}")
            print("  SERAPH Response")
            print(f"{'═' * 60}")
            print(result.final_response)
            print(f"\n[{elapsed:.0f}ms | emotion: {result.stage2.primary_emotion if result.stage2 else 'n/a'} "
                  f"| gate: {result.stage4.decision if result.stage4 else 'n/a'}]")

        except KeyboardInterrupt:
            print("\n\nGoodbye.")
            break
        except Exception as exc:
            print(f"\nError: {exc}\n")


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="SERAPH — Empathic AI Pipeline")
    parser.add_argument("--text", type=str, help="Single text input to process")
    parser.add_argument(
        "--ablation",
        choices=list(ABLATION_VARIANTS.keys()),
        default="full",
        help="Ablation variant to run",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Print all stage outputs")
    parser.add_argument("--output", type=str, help="JSON output file path")
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run benchmark (use with --dataset and --sample-limit)",
    )
    parser.add_argument("--dataset", type=str, help="Dataset to benchmark on")
    parser.add_argument("--sample-limit", type=int, help="Max samples for benchmark")

    args = parser.parse_args()

    if args.benchmark:
        # Delegate to benchmark runner
        from benchmarks.iemocap_eval import run_iemocap_eval
        from benchmarks.meld_eval import run_meld_eval
        from benchmarks.empathetic_dialogues_eval import run_ed_eval

        eval_map = {
            "iemocap": run_iemocap_eval,
            "meld": run_meld_eval,
            "empathetic_dialogues": run_ed_eval,
        }

        if args.dataset and args.dataset in eval_map:
            eval_map[args.dataset](
                ablation=args.ablation,
                sample_limit=args.sample_limit,
            )
        else:
            for name, fn in eval_map.items():
                print(f"\n{'#' * 50}")
                print(f"# Benchmark: {name}")
                print(f"{'#' * 50}")
                fn(ablation=args.ablation, sample_limit=args.sample_limit)
        return

    pipeline = SERAPHPipeline(ablation=args.ablation, verbose=args.verbose)

    if args.text:
        result = pipeline.run(args.text)
        print("\nFinal Response:")
        print("-" * 40)
        print(result.final_response)
        if args.output:
            Path(args.output).write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
            print(f"\nFull result saved to: {args.output}")
    else:
        _interactive_loop(pipeline)


if __name__ == "__main__":
    main()
