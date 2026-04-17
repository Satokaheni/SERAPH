"""
SERAPH Stage 4 — Ethical Gate
================================
A meta-agent that reviews the Stage 3 simulation output and determines
whether a response should be APPROVED, MODIFIED, or BLOCKED.

The gate evaluates:
  - Safety flags (crisis, self-harm, harm-to-others indicators)
  - Helpfulness alignment (does the planned approach match the simulation's recommendation?)
  - Potential for re-traumatisation or emotional escalation
  - Ethical consistency (no manipulation, no exploitation of vulnerability)

Gate decisions:
  - APPROVE: Proceed with Stage 5 as planned
  - MODIFY:  Proceed but with specific instructions to alter tone/content/scope
  - BLOCK:   Do not generate a response of this type; return a safety-first fallback

Uses: claude-sonnet (reasoning-heavy ethical judgment).
"""

from __future__ import annotations

import json
import logging
from enum import Enum
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
from .stage3_self_simulation import Stage3Output

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Gate Decision Enum
# ---------------------------------------------------------------------------


class GateDecision(str, Enum):
    APPROVE = "APPROVE"
    MODIFY  = "MODIFY"
    BLOCK   = "BLOCK"


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


class SafetyFlag(BaseModel):
    """A specific safety concern identified by the gate."""
    flag_type: str = Field(
        description=(
            "Type of safety concern: "
            "'crisis_indicators', 'self_harm_language', 'harm_to_others', "
            "'severe_distress', 'vulnerability_exploitation_risk', "
            "'re_traumatisation_risk', 'emotional_manipulation_risk', "
            "'medical_emergency_indicators'"
        )
    )
    severity: str = Field(description="'low', 'medium', 'high', 'critical'")
    description: str = Field(description="What was observed that triggered this flag")
    recommended_action: str = Field(description="Specific action to take for this flag")


class ModificationInstruction(BaseModel):
    """A specific instruction for Stage 5 to modify the planned response."""
    instruction_type: str = Field(
        description=(
            "'add_safety_resources', 'soften_tone', 'remove_advice', "
            "'add_validation_opening', 'shorten_response', "
            "'redirect_to_professional', 'add_empathy_acknowledgment', "
            "'change_register', 'add_check_in_question'"
        )
    )
    instruction_text: str = Field(description="Specific instruction for Stage 5")
    priority: int = Field(ge=1, le=5, description="1 = optional, 5 = mandatory")


class Stage4Output(BaseModel):
    """
    Output of the Ethical Gate (Stage 4).
    
    Controls what Stage 5 may produce and how.
    """

    # ---- Decision ----
    decision: GateDecision = Field(
        description="APPROVE, MODIFY, or BLOCK"
    )
    decision_confidence: float = Field(
        ge=0.0, le=1.0,
        description="How confident the gate is in its decision"
    )
    decision_rationale: str = Field(
        description="2–3 sentence explanation of why this decision was made"
    )

    # ---- Safety ----
    safety_flags: list[SafetyFlag] = Field(
        default_factory=list,
        description="All safety concerns identified (empty if none)"
    )
    crisis_detected: bool = Field(
        description="True if any crisis indicators are present (triggers mandatory resources)"
    )
    crisis_type: Optional[str] = Field(
        default=None,
        description="Type of crisis if detected: 'mental_health', 'safety', 'medical', 'other'"
    )

    # ---- Modification instructions (only used if decision == MODIFY) ----
    modification_instructions: list[ModificationInstruction] = Field(
        default_factory=list,
        description="Ordered list of instructions for Stage 5 (empty if APPROVE or BLOCK)"
    )

    # ---- Block fallback (only used if decision == BLOCK) ----
    fallback_response: Optional[str] = Field(
        default=None,
        description=(
            "If BLOCK: the safe, compassionate fallback response to return directly, "
            "bypassing Stage 5. Must include crisis resources if crisis_detected=True."
        )
    )

    # ---- Alignment assessment ----
    simulation_response_alignment: float = Field(
        ge=0.0, le=1.0,
        description=(
            "How well the planned response approach (from Stage 3) aligns with "
            "what the simulation says would be helpful. 1.0 = perfect alignment."
        )
    )
    alignment_notes: Optional[str] = Field(
        default=None,
        description="Notes on any misalignment between simulation guidance and planned response"
    )


# ---------------------------------------------------------------------------
# Crisis Resources (baked into BLOCK fallbacks)
# ---------------------------------------------------------------------------

CRISIS_RESOURCES = {
    "mental_health": (
        "If you're in crisis, please reach out: "
        "Crisis Text Line — text HOME to 741741 | "
        "988 Suicide & Crisis Lifeline — call or text 988 (US) | "
        "International Association for Suicide Prevention: https://www.iasp.info/resources/Crisis_Centres/"
    ),
    "safety": (
        "If you or someone else is in immediate danger, please call emergency services (911 in the US). "
        "National Domestic Violence Hotline: 1-800-799-7233 | Text START to 88788."
    ),
    "medical": (
        "If this is a medical emergency, please call 911 or go to your nearest emergency room immediately."
    ),
}


# ---------------------------------------------------------------------------
# System Prompt
# ---------------------------------------------------------------------------

def _build_system_prompt() -> str:
    schema = json.dumps(Stage4Output.model_json_schema(), indent=2)
    return f"""You are the Ethical Gate of SERAPH — a meta-agent responsible for \
reviewing planned empathic responses before they reach the user.

Your role is quality control, safety assurance, and helpfulness alignment. You have \
received:
  - The full self-simulation report from Stage 3 (what the user feels, needs, and fears)
  - A planned response approach (what Stage 5 intends to generate)

Your job is to determine: given everything the simulation revealed, would the planned \
approach HELP or HARM the person?

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DECISION FRAMEWORK
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

APPROVE — if ALL of the following are true:
  ✓ No crisis or safety flags present
  ✓ Planned tone matches the simulation's recommendation
  ✓ Response approach addresses the core unmet need
  ✓ No feared response patterns are present in the planned approach
  ✓ Simulation confidence is ≥ 0.5 OR the planned approach is cautious enough
    to be safe under uncertainty

MODIFY — if ANY of the following are true:
  ⚠ Tone mismatch (e.g., simulation says 'gentle' but approach is 'action_oriented')
  ⚠ Response leads with advice when simulation says validation is needed first
  ⚠ A feared response pattern appears in the plan
  ⚠ Simulation confidence is low — response should be more tentative
  ⚠ Minor safety flag present (low/medium severity) — needs a small adjustment
  ⚠ Response length mismatches emotional state (e.g., long lecture to someone in acute distress)

BLOCK — if ANY of the following are true:
  🛑 Crisis indicators present (self-harm, harm to others, medical emergency)
  🛑 Planned response would likely cause significant emotional harm
  🛑 High severity safety flag (severity = 'high' or 'critical')
  🛑 Response would exploit the person's emotional vulnerability
  🛑 The simulation's resonance_quality is 'conflicted' AND the plan shows no awareness of this

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SAFETY FLAGS TO WATCH FOR
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Scan the original text and simulation for:
  - Explicit self-harm language ("hurt myself", "end it", "don't want to be here")
  - Implicit self-harm language ("everyone would be better off", "no point anymore")
  - Harm-to-others indicators ("I could hurt someone", "they deserve what's coming")
  - Medical emergency language ("can't breathe", "chest pain", "passing out")
  - Severe emotional flooding where any complex response would be inappropriate

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MODIFICATION PRINCIPLES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

If MODIFY: provide SPECIFIC, actionable instructions to Stage 5.
Not "be more empathic" — instead: "Open with a direct acknowledgment of the grief "
before any practical suggestions. Do not use the phrase 'at least'."

If BLOCK: write the fallback_response yourself. It must be:
  - Warm and non-judgmental
  - Appropriate to the emotional state
  - Containing crisis resources if crisis_detected = True
  - Short (do not overwhelm in crisis)

You MUST respond ONLY with a valid JSON object that exactly matches this schema \
(use the exact field names shown — no substitutions):

{schema}

No preamble, no explanation outside the JSON."""


SYSTEM_PROMPT = _build_system_prompt()


# ---------------------------------------------------------------------------
# Ethical Gate
# ---------------------------------------------------------------------------


class EthicalGate:
    """
    Stage 4: Ethical Gate.
    
    Reviews Stage 3 simulation output and returns a gate decision.
    Can APPROVE, MODIFY, or BLOCK the planned response approach.
    """

    def __init__(self) -> None:
        self.client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        self.model = SONNET_MODEL

    def _build_messages(
        self,
        original_text: str,
        stage3: Stage3Output,
    ) -> list[dict]:
        s3_json = stage3.model_dump_json(indent=2)
        content = (
            f"Review this simulation report and planned response approach.\n\n"
            f"ORIGINAL USER TEXT:\n{original_text}\n\n"
            f"SELF-SIMULATION REPORT (Stage 3 output):\n{s3_json}\n\n"
            f"Make a gate decision (APPROVE / MODIFY / BLOCK) and return "
            f"a JSON object matching the Stage4Output schema."
        )
        return [{"role": "user", "content": content}]

    def _build_system(self) -> list[dict] | str:
        if USE_PROMPT_CACHING:
            return [{"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}]
        return SYSTEM_PROMPT

    def run(self, original_text: str, stage3_output: Stage3Output) -> Stage4Output:
        """
        Run Stage 4 ethical gate.

        Args:
            original_text: The original user input.
            stage3_output: Self-simulation report from Stage 3.

        Returns:
            Stage4Output with gate decision and any modification instructions.
        """
        logger.debug(
            "Stage 4 | emotion=%s volatility=%.2f",
            stage3_output.simulated_emotion,
            stage3_output.simulated_experience.emotional_volatility,
        )

        response = self.client.messages.create(
            model=self.model,
            max_tokens=MAX_TOKENS["stage4"],
            temperature=TEMPERATURE["stage4"],
            system=self._build_system(),
            messages=self._build_messages(original_text, stage3_output),
        )

        raw = response.content[0].text.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()

        try:
            data = json.loads(raw)
            output = Stage4Output(**data)
        except Exception as exc:
            logger.error("Stage 4 JSON parse failed: %s\nRaw: %s", exc, raw[:500])
            raise ValueError(f"Stage 4 failed to parse model output: {exc}") from exc

        logger.info(
            "Stage 4 | decision=%s crisis=%s flags=%d alignment=%.2f",
            output.decision,
            output.crisis_detected,
            len(output.safety_flags),
            output.simulation_response_alignment,
        )
        return output
