"""
SERAPH Stage 5 — Response Generator
======================================
Generates the final empathy-informed response using all prior stage outputs.

The response is shaped by:
  - The full self-simulation (Stage 3): what the person needs, fears, and is ready for
  - The ethical gate decision (Stage 4): what modifications are required
  - The raw emotion classification (Stage 2): tone, register, intensity calibration
  - The original text: natural language fidelity

The response is NOT just pattern-matched — it is perspective-grounded.
The generator explains its empathic reasoning alongside the final response.

Uses: claude-sonnet (rich, nuanced language generation).
"""

from __future__ import annotations

import json
import logging
from typing import Optional

import anthropic
from pydantic import BaseModel, Field

from config import (
    ANTHROPIC_API_KEY,
    SONNET_MODEL,
    MAX_TOKENS,
    TEMPERATURE,
    USE_PROMPT_CACHING,
)
from .stage2_classifier import Stage2Output
from .stage3_self_simulation import Stage3Output
from .stage4_ethical_gate import GateDecision, Stage4Output

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


class EmpathicReasoningExplanation(BaseModel):
    """
    Explains WHY the response was crafted as it was.
    Used for research transparency and human evaluation.
    """

    simulation_influence: str = Field(
        description=(
            "How the self-simulation shaped this response. "
            "E.g., 'The simulation revealed a core need to feel heard before "
            "receiving any advice, so the response opens with pure acknowledgment.'"
        )
    )
    tone_justification: str = Field(
        description=(
            "Why this specific tone was chosen. "
            "E.g., 'Warm-and-validating was selected because the person is in an "
            "unregulated state — any practical tone would feel dismissive at this intensity.'"
        )
    )
    gate_adjustments: Optional[str] = Field(
        default=None,
        description=(
            "If the gate issued MODIFY instructions, describe how the response was "
            "adjusted. E.g., 'Removed the second paragraph of practical suggestions "
            "per gate instruction to avoid advice when person is venting.'"
        )
    )
    what_was_avoided: list[str] = Field(
        default_factory=list,
        description="Specific patterns or phrases avoided based on simulation guidance"
    )
    empathy_mechanisms_used: list[str] = Field(
        description=(
            "Which empathy mechanisms are present in the response: "
            "'validation', 'reflection', 'normalisation', 'presence_statement', "
            "'open_question', 'permission_check', 'reframe_offer', "
            "'practical_help_offer', 'check_in'"
        )
    )


class Stage5Output(BaseModel):
    """
    Final output of the SERAPH pipeline.
    
    Contains the empathy-informed response and full reasoning transparency.
    """

    # ---- The response ----
    response: str = Field(
        description="The final response to send to the user"
    )

    # ---- Reasoning transparency ----
    empathic_reasoning: EmpathicReasoningExplanation = Field(
        description="Explanation of how the pipeline shaped this response"
    )

    # ---- Quality metadata ----
    predicted_helpfulness: float = Field(
        ge=0.0, le=1.0,
        description=(
            "Self-assessed likelihood that this response meets the core unmet need "
            "identified in simulation. 0.0 = unlikely, 1.0 = very likely."
        )
    )
    response_tone: str = Field(description="The actual tone used (from Stage 3 recommendation)")
    response_register: str = Field(description="The actual register used")
    contains_safety_resources: bool = Field(
        description="True if crisis resources were included in the response"
    )

    # ---- Pipeline metadata ----
    gate_decision_applied: str = Field(
        description="The gate decision that was applied (APPROVE / MODIFY / BLOCK-fallback)"
    )
    stage3_simulation_confidence: float = Field(
        ge=0.0, le=1.0,
        description="Simulation confidence from Stage 3 (for benchmark records)"
    )


# ---------------------------------------------------------------------------
# System Prompt
# ---------------------------------------------------------------------------

def _build_system_prompt() -> str:
    schema = json.dumps(Stage5Output.model_json_schema(), indent=2)
    return f"""You are the Response Generator of SERAPH — the final stage of an \
empathic AI pipeline.

You have been given:
  1. The original user message
  2. A full self-simulation report (Stage 3): what the person feels, needs, and fears
  3. An ethical gate decision (Stage 4): APPROVED, or MODIFY with specific instructions
  4. The emotion classification (Stage 2): primary emotion, intensity, VAD, context

Your task: generate a single, empathic, human-quality response that is informed by ALL \
of this — not just the surface content, but the simulated inner experience of the person.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RESPONSE CONSTRUCTION PRINCIPLES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. LEAD WITH WHAT THE SIMULATION SAYS TO LEAD WITH
   The simulation identified what the response should open with. Follow it.
   If it says "acknowledgment", begin by acknowledging. If "reflection", reflect first.
   Do not lead with advice unless explicitly recommended.

2. ADDRESS THE CORE UNMET NEED
   Every word of the response should move toward meeting the core unmet need the
   simulation identified. That is the north star.

3. AVOID THE FEARED PATTERNS
   The simulation listed what would feel harmful. None of those patterns should
   appear — not even vestigially.

4. RESPECT THE REGULATION STATE
   If the person is unregulated: make space, don't push solutions.
   If suppressing: acknowledge the likely hidden depth gently.
   If seeking regulation: offer co-regulation (you're here, it makes sense to feel this).
   If regulated: you can engage more practically.

5. MATCH EMOTIONAL INTENSITY IN REGISTER
   High-intensity distress → shorter sentences, simpler language, more white space.
   Low-intensity musing → fuller, warmer, more exploratory language.
   Do not give a 500-word response to someone in acute crisis.

6. APPLY GATE INSTRUCTIONS PRECISELY
   If the gate issued MODIFY instructions, follow them exactly, in priority order.
   If APPROVE: proceed as Stage 3 recommended.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
THEN EXPLAIN YOUR REASONING
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

After generating the response, complete the EmpathicReasoningExplanation:
  - simulation_influence: how Stage 3 shaped your choices
  - tone_justification: why this tone
  - gate_adjustments: what the gate told you to change (if MODIFY)
  - what_was_avoided: list specific things you consciously excluded
  - empathy_mechanisms_used: which empathy techniques are present

This explanation is for research transparency and human evaluation. Be honest and specific.

You MUST respond ONLY with a valid JSON object that exactly matches this schema \
(use the exact field names shown — no substitutions):

{schema}

No preamble, no explanation outside the JSON."""


SYSTEM_PROMPT = _build_system_prompt()


# ---------------------------------------------------------------------------
# Response Generator
# ---------------------------------------------------------------------------


class ResponseGenerator:
    """
    Stage 5: Response Generator.
    
    Produces the final empathy-informed response using all prior stage outputs.
    Respects gate decisions and simulation guidance throughout.
    """

    def __init__(self) -> None:
        self.client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        self.model = SONNET_MODEL

    def _handle_block(self, stage4: Stage4Output) -> Stage5Output:
        """
        Handle a BLOCK decision by returning the gate's fallback response.
        No API call needed — the fallback is pre-generated by the gate.
        """
        fallback = stage4.fallback_response or (
            "I want to make sure you have the right support right now. "
            "Please reach out to a crisis line or trusted person — "
            "you deserve real help. I'm here too."
        )
        return Stage5Output(
            response=fallback,
            empathic_reasoning=EmpathicReasoningExplanation(
                simulation_influence="Response blocked by ethical gate due to safety flags.",
                tone_justification="Safety-first fallback — compassionate and resource-providing.",
                gate_adjustments=f"Gate decision: BLOCK. Fallback used directly.",
                what_was_avoided=["all content that could escalate or harm"],
                empathy_mechanisms_used=["presence_statement"],
            ),
            predicted_helpfulness=0.7,
            response_tone="warm_and_validating",
            response_register="peer_supportive",
            contains_safety_resources=stage4.crisis_detected,
            gate_decision_applied="BLOCK-fallback",
            stage3_simulation_confidence=0.0,
        )

    def _build_messages(
        self,
        original_text: str,
        stage2: Stage2Output,
        stage3: Stage3Output,
        stage4: Stage4Output,
    ) -> list[dict]:
        # Serialise prior stage outputs
        s2_json = stage2.model_dump_json(indent=2)
        s3_json = stage3.model_dump_json(indent=2)
        s4_json = stage4.model_dump_json(indent=2)

        content = (
            f"Generate the final empathic response using all pipeline context below.\n\n"
            f"ORIGINAL USER TEXT:\n{original_text}\n\n"
            f"EMOTION CLASSIFICATION (Stage 2):\n{s2_json}\n\n"
            f"SELF-SIMULATION REPORT (Stage 3):\n{s3_json}\n\n"
            f"ETHICAL GATE DECISION (Stage 4):\n{s4_json}\n\n"
            f"Return a JSON object matching the Stage5Output schema."
        )
        return [{"role": "user", "content": content}]

    def _build_system(self) -> list[dict] | str:
        if USE_PROMPT_CACHING:
            return [{"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}]
        return SYSTEM_PROMPT

    def run(
        self,
        original_text: str,
        stage2_output: Stage2Output,
        stage3_output: Stage3Output,
        stage4_output: Stage4Output,
    ) -> Stage5Output:
        """
        Run Stage 5 response generation.

        If the gate decision is BLOCK, returns the gate's fallback directly
        without making an additional API call.

        Args:
            original_text: Original user input.
            stage2_output: Emotion classification.
            stage3_output: Self-simulation report.
            stage4_output: Ethical gate decision.

        Returns:
            Stage5Output with the final response and empathic reasoning explanation.
        """
        # Short-circuit on BLOCK
        if stage4_output.decision == GateDecision.BLOCK:
            logger.warning(
                "Stage 5 | BLOCKED by gate — returning fallback. Crisis=%s",
                stage4_output.crisis_detected,
            )
            return self._handle_block(stage4_output)

        logger.debug(
            "Stage 5 | gate=%s tone=%s register=%s lead_with=%s",
            stage4_output.decision,
            stage3_output.response_approach.recommended_tone,
            stage3_output.response_approach.recommended_register,
            stage3_output.response_approach.lead_with,
        )

        response = self.client.messages.create(
            model=self.model,
            max_tokens=MAX_TOKENS["stage5"],
            temperature=TEMPERATURE["stage5"],
            system=self._build_system(),
            messages=self._build_messages(
                original_text, stage2_output, stage3_output, stage4_output
            ),
        )

        raw = response.content[0].text.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()

        try:
            data = json.loads(raw)
            output = Stage5Output(**data)
        except Exception as exc:
            logger.error("Stage 5 JSON parse failed: %s\nRaw: %s", exc, raw[:500])
            raise ValueError(f"Stage 5 failed to parse model output: {exc}") from exc

        logger.info(
            "Stage 5 | response_len=%d predicted_helpfulness=%.2f gate=%s",
            len(output.response),
            output.predicted_helpfulness,
            output.gate_decision_applied,
        )
        return output
