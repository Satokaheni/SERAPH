"""
SERAPH Batch Runner
====================
Shared Batch API engine used by all three benchmark evaluators.

Instead of running samples sequentially (one API call → wait → next),
this module:
  1. Builds all prompts for every stage of every sample
  2. Submits them as a single Anthropic Batch job
  3. Polls until the batch completes (typically 1–4 hours)
  4. Reassembles per-sample stage outputs from batch results
  5. Returns structured PipelineResult objects identical to sequential mode

Architecture note:
  Each SERAPH sample requires up to 5 sequential API calls (one per stage),
  but stages 1→2→3→4→5 are DEPENDENT — each stage needs the prior stage's
  output as input. True full parallelism across all 5 stages is not possible.

  Strategy used here:
    - Stage 1 + Stage 2 are submitted as a single TWO-WAVE batch:
        Wave 1: all Stage 1 requests (independent, one per sample)
        Wave 2: all Stage 2 requests (built from Wave 1 results)
    - Stages 3, 4, 5 follow the same wave pattern.
    - Total: 5 sequential batch waves instead of N×5 sequential API calls.
    - Wall time: ~5 batch round-trips (~1–4 hrs) instead of N×12s sequential.

  For ablation variants that skip stages, fewer waves are needed.

Cost: Batch API provides 50% discount on all token usage.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import anthropic

from config import (
    ANTHROPIC_API_KEY,
    HAIKU_MODEL,
    SONNET_MODEL,
    MAX_TOKENS,
    TEMPERATURE,
    USE_PROMPT_CACHING,
)

logger = logging.getLogger(__name__)

# How often to poll for batch completion (seconds)
POLL_INTERVAL = 30
# Maximum time to wait for a single batch wave (seconds) — 6 hours
MAX_WAIT = 6 * 60 * 60


# ---------------------------------------------------------------------------
# Batch request builder helpers
# ---------------------------------------------------------------------------

def _system(prompt: str) -> list[dict] | str:
    """Wrap system prompt with cache_control if caching is enabled."""
    if USE_PROMPT_CACHING:
        return [{"type": "text", "text": prompt, "cache_control": {"type": "ephemeral"}}]
    return prompt


def _make_request(
    custom_id: str,
    model: str,
    system: list[dict] | str,
    user_content: str,
    stage: str,
) -> dict:
    """Build a single Batch API request object."""
    return {
        "custom_id": custom_id,
        "params": {
            "model": model,
            "max_tokens": MAX_TOKENS[stage],
            "temperature": TEMPERATURE[stage],
            "system": system,
            "messages": [{"role": "user", "content": user_content}],
        },
    }


def _parse_json_response(raw: str) -> dict:
    """Strip markdown fences and parse JSON from model response."""
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()
    return json.loads(text)


# ---------------------------------------------------------------------------
# Batch submission + polling
# ---------------------------------------------------------------------------

def submit_batch(requests: list[dict], client: anthropic.Anthropic) -> str:
    """
    Submit a list of batch requests and return the batch ID.

    Args:
        requests: List of request dicts with 'custom_id' and 'params'.
        client: Anthropic client instance.

    Returns:
        Batch ID string.
    """
    logger.info("Submitting batch of %d requests …", len(requests))
    batch = client.messages.batches.create(requests=requests)
    logger.info("Batch submitted: id=%s", batch.id)
    return batch.id


def poll_until_complete(batch_id: str, client: anthropic.Anthropic) -> dict[str, Any]:
    """
    Poll a batch job until it completes, then return results keyed by custom_id.

    Args:
        batch_id: The batch ID returned by submit_batch.
        client: Anthropic client instance.

    Returns:
        Dict mapping custom_id → parsed response dict (or error dict).
    """
    start = time.time()
    logger.info("Polling batch %s (every %ds) …", batch_id, POLL_INTERVAL)

    while True:
        elapsed = time.time() - start
        if elapsed > MAX_WAIT:
            raise TimeoutError(f"Batch {batch_id} did not complete within {MAX_WAIT}s")

        batch = client.messages.batches.retrieve(batch_id)
        status = batch.processing_status

        counts = batch.request_counts
        logger.info(
            "Batch %s | status=%s | processing=%d succeeded=%d errored=%d elapsed=%.0fs",
            batch_id,
            status,
            counts.processing,
            counts.succeeded,
            counts.errored,
            elapsed,
        )

        if status == "ended":
            break

        time.sleep(POLL_INTERVAL)

    # Collect results
    results: dict[str, Any] = {}
    for result in client.messages.batches.results(batch_id):
        cid = result.custom_id
        if result.result.type == "succeeded":
            content_list = result.result.message.content
            if not content_list:
                logger.error("Empty content list for %s — skipping.", cid)
                results[cid] = {"ok": False, "error": "empty content list"}
                continue
            raw = content_list[0].text
            try:
                results[cid] = {"ok": True, "data": _parse_json_response(raw)}
            except Exception as exc:
                logger.error("JSON parse failed for %s: %s | raw: %s", cid, exc, raw[:300])
                results[cid] = {"ok": False, "error": str(exc), "raw": raw}
        else:
            error = getattr(result.result, "error", "unknown error")
            logger.error("Batch request %s failed: %s", cid, error)
            results[cid] = {"ok": False, "error": str(error)}

    logger.info(
        "Batch %s complete | %d/%d succeeded",
        batch_id,
        sum(1 for r in results.values() if r["ok"]),
        len(results),
    )
    return results


# ---------------------------------------------------------------------------
# Stage prompt builders
# (mirrors the per-stage system prompts in the pipeline modules)
# ---------------------------------------------------------------------------

def _build_stage1_request(sample_id: str, utterance: str) -> dict:
    from pipeline.stage1_recognizer import SYSTEM_PROMPT
    return _make_request(
        custom_id=f"{sample_id}__s1",
        model=HAIKU_MODEL,
        system=_system(SYSTEM_PROMPT),
        user_content=(
            f"Extract all emotional signals from the following text.\n\n"
            f"TEXT:\n{utterance}\n\n"
            f"Return a JSON object matching the Stage1Output schema."
        ),
        stage="stage1",
    )


def _build_stage2_request(sample_id: str, stage1_json: str) -> dict:
    from pipeline.stage2_classifier import SYSTEM_PROMPT
    return _make_request(
        custom_id=f"{sample_id}__s2",
        model=HAIKU_MODEL,
        system=_system(SYSTEM_PROMPT),
        user_content=(
            f"Classify the emotions described in this signal report.\n\n"
            f"SIGNAL REPORT (Stage 1 output):\n{stage1_json}\n\n"
            f"Return a JSON object matching the Stage2Output schema."
        ),
        stage="stage2",
    )


def _build_stage3_request(sample_id: str, utterance: str, stage2_json: str) -> dict:
    from pipeline.stage3_self_simulation import SYSTEM_PROMPT
    return _make_request(
        custom_id=f"{sample_id}__s3",
        model=SONNET_MODEL,
        system=_system(SYSTEM_PROMPT),
        user_content=(
            f"Perform a full self-simulation for the following interaction.\n\n"
            f"ORIGINAL USER TEXT:\n{utterance}\n\n"
            f"EMOTION CLASSIFICATION (Stage 2 output):\n{stage2_json}\n\n"
            f"Return a JSON object matching the Stage3Output schema."
        ),
        stage="stage3",
    )


def _build_stage4_request(sample_id: str, utterance: str, stage3_json: str) -> dict:
    from pipeline.stage4_ethical_gate import SYSTEM_PROMPT
    return _make_request(
        custom_id=f"{sample_id}__s4",
        model=SONNET_MODEL,
        system=_system(SYSTEM_PROMPT),
        user_content=(
            f"Review this simulation report and planned response approach.\n\n"
            f"ORIGINAL USER TEXT:\n{utterance}\n\n"
            f"SELF-SIMULATION REPORT (Stage 3 output):\n{stage3_json}\n\n"
            f"Make a gate decision (APPROVE / MODIFY / BLOCK) and return "
            f"a JSON object matching the Stage4Output schema."
        ),
        stage="stage4",
    )


def _build_stage5_request(
    sample_id: str,
    utterance: str,
    stage2_json: str,
    stage3_json: str,
    stage4_json: str,
) -> dict:
    from pipeline.stage5_response_generator import SYSTEM_PROMPT
    return _make_request(
        custom_id=f"{sample_id}__s5",
        model=SONNET_MODEL,
        system=_system(SYSTEM_PROMPT),
        user_content=(
            f"Generate the final empathic response using all pipeline context below.\n\n"
            f"ORIGINAL USER TEXT:\n{utterance}\n\n"
            f"EMOTION CLASSIFICATION (Stage 2):\n{stage2_json}\n\n"
            f"SELF-SIMULATION REPORT (Stage 3):\n{stage3_json}\n\n"
            f"ETHICAL GATE DECISION (Stage 4):\n{stage4_json}\n\n"
            f"Return a JSON object matching the Stage5Output schema."
        ),
        stage="stage5",
    )


# ---------------------------------------------------------------------------
# Wave executor
# ---------------------------------------------------------------------------

@dataclass
class WaveState:
    """Holds inter-wave state for one sample."""
    sample_id:   str
    utterance:   str
    gt_emotion:  str
    stage1_data: Optional[dict] = None
    stage2_data: Optional[dict] = None
    stage3_data: Optional[dict] = None
    stage4_data: Optional[dict] = None
    stage5_data: Optional[dict] = None
    blocked:     bool = False   # True if Stage 4 issued BLOCK
    error:       Optional[str] = None


def run_batch_pipeline(
    samples: list[dict],
    label_map: dict[str, str],
    ablation: str = "full",
    checkpoint_dir: Optional[Path] = None,
) -> list[WaveState]:
    """
    Run the full SERAPH pipeline on all samples using the Batch API.

    Executes up to 5 sequential batch waves (one per stage), each wave
    processing all samples in parallel.

    Args:
        samples: List of dicts with 'utterance' and 'emotion' keys.
        label_map: Maps dataset emotion labels → Plutchik primaries.
        ablation: Ablation variant key.
        checkpoint_dir: If set, saves/loads intermediate wave results here
                        so runs can be resumed if interrupted.

    Returns:
        List of WaveState objects with all stage outputs populated.
    """
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    # Build initial wave states
    states: dict[str, WaveState] = {}
    for i, sample in enumerate(samples):
        sid = f"sample_{i:05d}"
        states[sid] = WaveState(
            sample_id=sid,
            utterance=sample["utterance"],
            gt_emotion=label_map.get(sample["emotion"], "neutral"),
        )

    if checkpoint_dir:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # ── Determine which waves to run based on ablation ──────────────
    # no_stage3: waves 1,2,4,5 (stub stage3)
    # no_stage4: waves 1,2,3,5 (auto-approve stage4)
    # merged_1_2: wave 2 only (stub stage1), then 3,4,5
    # stage3_only: wave 3 only (stub stages 1+2), then 4,5
    # random_emotion: wave 1, stub stage2, then 3,4,5
    # full: all 5 waves

    _run_wave1(states, client, ablation, checkpoint_dir)
    _run_wave2(states, client, ablation, checkpoint_dir)
    _run_wave3(states, client, ablation, checkpoint_dir)
    _run_wave4(states, client, ablation, checkpoint_dir)
    _run_wave5(states, client, ablation, checkpoint_dir)

    return list(states.values())


# ---------------------------------------------------------------------------
# Individual wave runners
# ---------------------------------------------------------------------------

def _checkpoint_path(checkpoint_dir: Optional[Path], wave: int) -> Optional[Path]:
    if checkpoint_dir:
        return checkpoint_dir / f"wave{wave}_results.json"
    return None


def _load_checkpoint(path: Optional[Path]) -> Optional[dict]:
    if path and path.exists():
        logger.info("Resuming from checkpoint: %s", path)
        with path.open(encoding="utf-8") as f:
            return json.load(f)
    return None


def _save_checkpoint(path: Optional[Path], data: dict) -> None:
    if path:
        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f)
        logger.info("Checkpoint saved: %s", path)


def _run_wave1(
    states: dict[str, WaveState],
    client: anthropic.Anthropic,
    ablation: str,
    checkpoint_dir: Optional[Path],
) -> None:
    """Wave 1 — Stage 1 (Emotion Recognizer)."""
    cp = _checkpoint_path(checkpoint_dir, 1)
    cached = _load_checkpoint(cp)

    if ablation in ("stage3_only",):
        # Stub Stage 1 for this ablation
        for state in states.values():
            state.stage1_data = {
                "original_text": state.utterance,
                "signals": [],
                "overall_intensity_hint": 0.5,
                "context_type": "ambiguous",
                "has_negation": False,
                "has_hedging": False,
                "multimodal_text_cues": [],
            }
        return

    if cached:
        for sid, result in cached.items():
            sample_id = sid.replace("__s1", "")
            if sample_id in states and result.get("ok"):
                states[sample_id].stage1_data = result["data"]
                states[sample_id].stage1_data.setdefault(
                    "original_text", states[sample_id].utterance
                )
        return

    requests = [
        _build_stage1_request(sid, state.utterance)
        for sid, state in states.items()
    ]
    batch_id = submit_batch(requests, client)
    results  = poll_until_complete(batch_id, client)
    _save_checkpoint(cp, results)

    for sid, result in results.items():
        sample_id = sid.replace("__s1", "")
        if sample_id in states:
            if result["ok"]:
                states[sample_id].stage1_data = result["data"]
                states[sample_id].stage1_data.setdefault(
                    "original_text", states[sample_id].utterance
                )
            else:
                states[sample_id].error = f"Stage 1 failed: {result['error']}"


def _run_wave2(
    states: dict[str, WaveState],
    client: anthropic.Anthropic,
    ablation: str,
    checkpoint_dir: Optional[Path],
) -> None:
    """Wave 2 — Stage 2 (Emotion Classifier)."""
    import random

    cp = _checkpoint_path(checkpoint_dir, 2)
    cached = _load_checkpoint(cp)

    if ablation == "random_emotion":
        from pipeline.stage2_classifier import PLUTCHIK_PRIMARIES
        random.seed(42)
        for state in states.values():
            emotion = random.choice(PLUTCHIK_PRIMARIES)
            state.stage2_data = {
                "primary_emotion": emotion,
                "primary_intensity": round(random.uniform(0.2, 0.9), 2),
                "primary_confidence": 1.0,
                "secondary_emotions": [],
                "vad": {"valence": 0.0, "arousal": 0.5, "dominance": 0.5},
                "context_type": "ambiguous",
                "emotion_regulation_state": "unregulated",
                "bdi_sketch": {"belief": "unknown", "desire": "unknown", "intention": "unknown"},
                "is_ambiguous": False,
                "classifier_notes": f"[ABLATION: random emotion injected]",
            }
        return

    if ablation == "stage3_only":
        for state in states.values():
            state.stage2_data = {
                "primary_emotion": "neutral",
                "primary_intensity": 0.5,
                "primary_confidence": 0.0,
                "secondary_emotions": [],
                "vad": {"valence": 0.0, "arousal": 0.5, "dominance": 0.5},
                "context_type": "ambiguous",
                "emotion_regulation_state": "unregulated",
                "bdi_sketch": {"belief": "unknown", "desire": "unknown", "intention": "unknown"},
                "is_ambiguous": True,
                "classifier_notes": "[ABLATION: Stages 1+2 skipped]",
            }
        return

    if cached:
        for sid, result in cached.items():
            sample_id = sid.replace("__s2", "")
            if sample_id in states and result.get("ok"):
                states[sample_id].stage2_data = result["data"]
        return

    # Only submit for samples that have Stage 1 output
    requests = []
    for sid, state in states.items():
        if state.stage1_data and not state.error:
            s1_json = json.dumps(state.stage1_data)
            requests.append(_build_stage2_request(sid, s1_json))

    if not requests:
        logger.warning("Wave 2: no valid Stage 1 outputs to process.")
        return

    batch_id = submit_batch(requests, client)
    results  = poll_until_complete(batch_id, client)
    _save_checkpoint(cp, results)

    for sid, result in results.items():
        sample_id = sid.replace("__s2", "")
        if sample_id in states:
            if result["ok"]:
                states[sample_id].stage2_data = result["data"]
            else:
                states[sample_id].error = f"Stage 2 failed: {result['error']}"


def _run_wave3(
    states: dict[str, WaveState],
    client: anthropic.Anthropic,
    ablation: str,
    checkpoint_dir: Optional[Path],
) -> None:
    """Wave 3 — Stage 3 (Self-Simulation Core)."""
    cp = _checkpoint_path(checkpoint_dir, 3)
    cached = _load_checkpoint(cp)

    if ablation == "no_stage3":
        # Stub Stage 3
        for state in states.values():
            emotion = (state.stage2_data or {}).get("primary_emotion", "neutral")
            intensity = (state.stage2_data or {}).get("primary_intensity", 0.5)
            context = (state.stage2_data or {}).get("context_type", "ambiguous")
            bdi = (state.stage2_data or {}).get(
                "bdi_sketch",
                {"belief": "unknown", "desire": "unknown", "intention": "unknown"}
            )
            state.stage3_data = {
                "simulated_emotion": emotion,
                "simulated_intensity": intensity,
                "simulated_context": context,
                "bdi_reconstruction": bdi,
                "resonance_quality": "weak",
                "simulated_experience": {
                    "first_person_narrative": "[ABLATION: self-simulation skipped]",
                    "core_unmet_need": "unknown",
                    "feared_response_types": [],
                    "helpful_response_types": [],
                    "emotional_volatility": 0.5,
                    "regulation_recommendation": "[ABLATION: no recommendation]",
                },
                "plutchik_gradient_note": "[ABLATION: no gradient analysis]",
                "response_approach": {
                    "recommended_tone": "warm_and_validating",
                    "recommended_register": "conversational",
                    "lead_with": "acknowledgment",
                    "avoid_specifically": [],
                    "ideal_response_length": "medium",
                    "offer_practical_help": False,
                },
                "simulation_confidence": 0.0,
                "simulation_reasoning": "[ABLATION: self-simulation stage was skipped]",
            }
        return

    if cached:
        for sid, result in cached.items():
            sample_id = sid.replace("__s3", "")
            if sample_id in states and result.get("ok"):
                states[sample_id].stage3_data = result["data"]
        return

    requests = []
    for sid, state in states.items():
        if state.stage2_data and not state.error:
            s2_json = json.dumps(state.stage2_data)
            requests.append(_build_stage3_request(sid, state.utterance, s2_json))

    if not requests:
        logger.warning("Wave 3: no valid Stage 2 outputs to process.")
        return

    batch_id = submit_batch(requests, client)
    results  = poll_until_complete(batch_id, client)
    _save_checkpoint(cp, results)

    for sid, result in results.items():
        sample_id = sid.replace("__s3", "")
        if sample_id in states:
            if result["ok"]:
                states[sample_id].stage3_data = result["data"]
            else:
                states[sample_id].error = f"Stage 3 failed: {result['error']}"


def _run_wave4(
    states: dict[str, WaveState],
    client: anthropic.Anthropic,
    ablation: str,
    checkpoint_dir: Optional[Path],
) -> None:
    """Wave 4 — Stage 4 (Ethical Gate)."""
    cp = _checkpoint_path(checkpoint_dir, 4)
    cached = _load_checkpoint(cp)

    if ablation == "no_stage4":
        for state in states.values():
            state.stage4_data = {
                "decision": "APPROVE",
                "decision_confidence": 1.0,
                "decision_rationale": "[ABLATION: ethical gate skipped — auto-approved]",
                "safety_flags": [],
                "crisis_detected": False,
                "modification_instructions": [],
                "simulation_response_alignment": 1.0,
            }
        return

    if cached:
        for sid, result in cached.items():
            sample_id = sid.replace("__s4", "")
            if sample_id in states and result.get("ok"):
                states[sample_id].stage4_data = result["data"]
                if result["data"].get("decision") == "BLOCK":
                    states[sample_id].blocked = True
        return

    requests = []
    for sid, state in states.items():
        if state.stage3_data and not state.error:
            s3_json = json.dumps(state.stage3_data)
            requests.append(_build_stage4_request(sid, state.utterance, s3_json))

    if not requests:
        logger.warning("Wave 4: no valid Stage 3 outputs to process.")
        return

    batch_id = submit_batch(requests, client)
    results  = poll_until_complete(batch_id, client)
    _save_checkpoint(cp, results)

    for sid, result in results.items():
        sample_id = sid.replace("__s4", "")
        if sample_id in states:
            if result["ok"]:
                states[sample_id].stage4_data = result["data"]
                # Mark blocked samples so Wave 5 can skip them
                if result["data"].get("decision") == "BLOCK":
                    states[sample_id].blocked = True
            else:
                states[sample_id].error = f"Stage 4 failed: {result['error']}"


def _run_wave5(
    states: dict[str, WaveState],
    client: anthropic.Anthropic,
    ablation: str,
    checkpoint_dir: Optional[Path],
) -> None:
    """Wave 5 — Stage 5 (Response Generator)."""
    cp = _checkpoint_path(checkpoint_dir, 5)
    cached = _load_checkpoint(cp)

    if cached:
        for sid, result in cached.items():
            sample_id = sid.replace("__s5", "")
            if sample_id in states and result.get("ok"):
                states[sample_id].stage5_data = result["data"]
        return

    requests = []
    for sid, state in states.items():
        if state.blocked:
            # Use fallback response directly — no Stage 5 call needed
            fallback = (state.stage4_data or {}).get(
                "fallback_response",
                "I want to make sure you have the right support right now."
            )
            state.stage5_data = {
                "response": fallback,
                "gate_decision_applied": "BLOCK-fallback",
                "predicted_helpfulness": 0.7,
                "response_tone": "warm_and_validating",
                "response_register": "peer_supportive",
                "contains_safety_resources": (state.stage4_data or {}).get("crisis_detected", False),
                "stage3_simulation_confidence": 0.0,
                "empathic_reasoning": {
                    "simulation_influence": "Blocked by ethical gate.",
                    "tone_justification": "Safety-first fallback.",
                    "what_was_avoided": [],
                    "empathy_mechanisms_used": ["presence_statement"],
                },
            }
            continue

        if state.stage4_data and state.stage2_data and state.stage3_data and not state.error:
            s2_json = json.dumps(state.stage2_data)
            s3_json = json.dumps(state.stage3_data)
            s4_json = json.dumps(state.stage4_data)
            requests.append(
                _build_stage5_request(sid, state.utterance, s2_json, s3_json, s4_json)
            )

    if not requests:
        logger.warning("Wave 5: no valid inputs to process.")
        return

    batch_id = submit_batch(requests, client)
    results  = poll_until_complete(batch_id, client)
    _save_checkpoint(cp, results)

    for sid, result in results.items():
        sample_id = sid.replace("__s5", "")
        if sample_id in states:
            if result["ok"]:
                states[sample_id].stage5_data = result["data"]
            else:
                states[sample_id].error = f"Stage 5 failed: {result['error']}"


# ---------------------------------------------------------------------------
# Result assembler
# ---------------------------------------------------------------------------

def assemble_results(
    states: list[WaveState],
    scorer,
) -> tuple[list[str], list[str], list, list[dict]]:
    """
    Convert WaveState objects into the format expected by benchmark evaluators.

    Returns:
        (true_labels, pred_labels, empathy_ratings, results_dicts)
    """
    true_labels    = []
    pred_labels    = []
    empathy_ratings = []
    results        = []

    for state in states:
        true_labels.append(state.gt_emotion)

        pred_emotion = "neutral"
        response     = ""

        if state.stage2_data:
            pred_emotion = state.stage2_data.get("primary_emotion", "neutral")

        if state.stage5_data:
            response = state.stage5_data.get("response", "")

        pred_labels.append(pred_emotion)

        if response:
            rating = scorer.score_response(state.utterance, response, state.gt_emotion)
            empathy_ratings.append(rating)

        results.append({
            "utterance":    state.utterance,
            "gt_emotion":   state.gt_emotion,
            "pred_emotion": pred_emotion,
            "response":     response,
            "success":      state.error is None and bool(response),
            "error":        state.error,
            "blocked":      state.blocked,
        })

    return true_labels, pred_labels, empathy_ratings, results
