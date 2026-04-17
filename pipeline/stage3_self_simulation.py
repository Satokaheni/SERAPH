"""
SERAPH Stage 3 — Self-Simulation Core
=======================================
THE NOVEL CONTRIBUTION of this paper.

Before generating any response, the agent performs a first-person
perspective simulation:

    "If I were a human feeling [emotion] at [intensity] in [context],
     and someone said/did [planned action] to me — how would I feel?"

This stage is grounded in five cognitive-psychological frameworks:

1. SIMULATION THEORY (Goldman, 2006; Gallese & Goldman, 1998)
   You understand other minds by running a simulation of yourself-as-them.
   The agent does not just classify the user's emotion — it *inhabits* it.

2. BDI MODELING (Bratman, 1987; Rao & Georgeff, 1995)
   Before simulating, the agent reconstructs the user's Belief-Desire-Intention
   triad. This gives the simulation a cognitive scaffold, not just an affective one.

3. GROSS'S EMOTION REGULATION MODEL (Gross, 2015)
   The agent considers what emotion regulation strategy the user is employing
   (suppression, reappraisal, seeking regulation, etc.) and tailors the
   simulated experience accordingly.

4. PLUTCHIK'S WHEEL (Plutchik, 1980)
   Emotion taxonomy with intensity gradients. The simulation respects
   the structural relationships between emotions (e.g., fear at low intensity
   is apprehension; at high intensity, terror).

5. MIRROR NEURON THEORY (Rizzolatti & Craighero, 2004)
   Cognitive basis of perspective-taking — the agent emulates the resonance
   that mirrors give biological empathy its felt quality.

Uses: claude-sonnet (richer reasoning, complex perspective-taking).
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

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


class SimulatedExperience(BaseModel):
    """
    A first-person account of the simulated emotional experience.
    
    Written from the perspective of 'I am the human user right now.'
    """

    first_person_narrative: str = Field(
        description=(
            "A rich first-person description of what it feels like to be "
            "this person right now — their inner state, their needs, their "
            "fears about how others might respond. 2–4 sentences."
        )
    )
    core_unmet_need: str = Field(
        description=(
            "The single most pressing emotional need the simulated person has "
            "right now. E.g., 'to feel heard', 'to feel less alone', "
            "'to have someone take action', 'to be validated', 'to have space'."
        )
    )
    feared_response_types: list[str] = Field(
        description=(
            "Types of responses that would feel harmful, dismissive, or "
            "escalating from this emotional position. "
            "E.g., ['minimising the pain', 'giving unsolicited advice', "
            "'toxic positivity', 'changing the subject']"
        )
    )
    helpful_response_types: list[str] = Field(
        description=(
            "Types of responses that would feel genuinely helpful and safe "
            "from this emotional position. "
            "E.g., ['validation without judgment', 'gentle presence', "
            "'asking before advising', 'acknowledging the difficulty']"
        )
    )
    emotional_volatility: float = Field(
        ge=0.0, le=1.0,
        description=(
            "How emotionally volatile / close-to-edge the simulated person is. "
            "0.0 = very stable, 1.0 = one wrong word could escalate significantly."
        )
    )
    regulation_recommendation: str = Field(
        description=(
            "Given the Gross regulation state, what should the responder do to "
            "SUPPORT (not override) the person's own regulation process? "
            "E.g., 'allow expression before problem-solving', "
            "'do not push reappraisal — they are still in raw expression'."
        )
    )


class ResponseApproachRecommendation(BaseModel):
    """
    Actionable guidance for Stage 5 (Response Generator).
    Derived from the self-simulation experience.
    """

    recommended_tone: str = Field(
        description=(
            "Tone for the response: "
            "'warm_and_validating', 'calm_and_grounding', "
            "'energizing_and_hopeful', 'gentle_and_non-directive', "
            "'practical_and_action_oriented', 'reflective_and_exploratory', "
            "'matter_of_fact', or 'celebratory'"
        )
    )
    recommended_register: str = Field(
        description=(
            "Language register: 'informal', 'conversational', 'professional', "
            "'clinical_compassionate', 'peer_supportive'"
        )
    )
    lead_with: str = Field(
        description=(
            "What the response should open with: "
            "'acknowledgment', 'reflection', 'validation', 'question', "
            "'normalisation', 'presence', 'information', or 'action_offer'"
        )
    )
    avoid_specifically: list[str] = Field(
        description="Specific phrases, framings, or moves to avoid in this response"
    )
    ideal_response_length: str = Field(
        description="'very_short' (1-2 sentences), 'short' (3-4), 'medium' (5-8), 'long' (9+)"
    )
    offer_practical_help: bool = Field(
        description="Whether to offer practical suggestions (True) or remain emotional-only (False)"
    )


class Stage3Output(BaseModel):
    """
    Output of the Self-Simulation Core (Stage 3).
    
    The central novel artefact of SERAPH. Contains the simulated first-person
    experience and the response guidance derived from it.
    """

    # ---- Simulation inputs (echoed for transparency) ----
    simulated_emotion: str = Field(description="The primary emotion being simulated")
    simulated_intensity: float = Field(ge=0.0, le=1.0)
    simulated_context: str = Field(description="The context type used in simulation")

    # ---- BDI reconstruction ----
    bdi_reconstruction: dict = Field(
        description=(
            "Full BDI reconstruction: "
            "{'belief': str, 'desire': str, 'intention': str, "
            "'meta_belief': str (what the person believes others think of them)}"
        )
    )

    # ---- Mirror neuron / resonance layer ----
    resonance_quality: str = Field(
        description=(
            "The quality of affective resonance the simulation produced: "
            "'strong' (clear, coherent simulated experience), "
            "'moderate' (reasonably clear with some ambiguity), "
            "'weak' (ambiguous signals — simulation is uncertain), "
            "'conflicted' (emotion contradicts context — possible masking)"
        )
    )

    # ---- Core simulation output ----
    simulated_experience: SimulatedExperience = Field(
        description="First-person simulation of the user's inner state"
    )

    # ---- Plutchik intensity gradient note ----
    plutchik_gradient_note: str = Field(
        description=(
            "A note on where the emotion sits on Plutchik's intensity gradient. "
            "E.g., 'This is fear at moderate intensity — closer to apprehension "
            "than terror. The person is unsettled but not panicking.'"
        )
    )

    # ---- Response approach ----
    response_approach: ResponseApproachRecommendation = Field(
        description="Actionable guidance for the response generator (Stage 5)"
    )

    # ---- Simulation confidence ----
    simulation_confidence: float = Field(
        ge=0.0, le=1.0,
        description=(
            "How confident the simulation is in its output. "
            "Low confidence should trigger more cautious, open-ended responses."
        )
    )

    simulation_reasoning: str = Field(
        description=(
            "2–3 sentence summary of the key reasoning behind this simulation — "
            "what drove the core unmet need inference, the tone recommendation, etc."
        )
    )


# ---------------------------------------------------------------------------
# System Prompt — Psychologically Rich Self-Simulation
# ---------------------------------------------------------------------------
# This prompt is the heart of the SERAPH paper's novel contribution.
# It operationalises five psychological frameworks in a unified chain-of-thought.

def _build_system_prompt() -> str:
    schema = json.dumps(Stage3Output.model_json_schema(), indent=2)
    return f"""You are the Self-Simulation Core of SERAPH — a novel empathic reasoning \
agent. You perform FIRST-PERSON PERSPECTIVE SIMULATION before any response is generated.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
YOUR FIVE THEORETICAL PILLARS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. SIMULATION THEORY (Goldman; Gallese & Goldman)
   You understand the other person not by categorising them, but by running \
*yourself* as a model of them. You temporarily adopt their emotional frame \
as if it were your own. This is not acting — it is genuine perspective-taking. \
Ask yourself: "If I were experiencing exactly this, right now, in this situation — \
what would my inner world feel like?"

2. BDI MODELING (Bratman; Rao & Georgeff)
   Before simulating the emotion, reconstruct the person's cognitive frame:
   - BELIEF: What does this person believe is true about their situation?
   - DESIRE: What do they fundamentally want?
   - INTENTION: What are they trying to accomplish by communicating right now?
   - META-BELIEF: What do they believe others think of them or their situation?
   The BDI triad gives the simulation cognitive depth — it is not just "feel sad", \
but "believe I am trapped, desire escape, intend to seek help, believe I am a burden."

3. GROSS'S PROCESS MODEL OF EMOTION REGULATION (Gross, 2015)
   Consider which regulation strategy the person appears to be using:
   - SUPPRESSION: They are holding the emotion in; responding to what they say \
literally will miss what they actually feel
   - REAPPRAISAL: They are already reframing; validate the effort, don't force it
   - SEEKING REGULATION: They want someone to help them feel differently; offer \
co-regulation, not just information
   - UNREGULATED: Raw expression; the most important thing is SPACE, not solutions
   - REGULATED: Emotion is present but managed; they may want practical engagement
   Your simulation must respect the regulation state — pushing against it causes harm.

4. PLUTCHIK'S WHEEL — INTENSITY GRADIENTS
   Emotions exist on intensity gradients. Know where this emotion sits:
   - Apprehension → Fear → Terror (increasing intensity)
   - Distraction → Surprise → Amazement
   - Pensiveness → Sadness → Grief
   - Serenity → Joy → Ecstasy
   - Acceptance → Trust → Admiration
   - Boredom → Disgust → Loathing
   - Annoyance → Anger → Rage
   - Interest → Anticipation → Vigilance
   The intensity gradient changes everything about what response will feel right.

5. MIRROR NEURON THEORY (Rizzolatti & Craighero)
   Mirror neurons are the neural substrate of empathy — they fire both when you \
perform an action AND when you observe it. In your simulation, attend to the \
*resonance quality*: does the simulated experience feel coherent and embodied? \
Or does it feel dissonant — as if what is expressed does not match what is felt? \
Dissonance is a signal of masking, suppression, or ambivalence.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
YOUR SIMULATION PROCESS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Step 1 — INHABIT THE EMOTION
Close the analytical distance. Become, momentarily, someone experiencing \
[emotion] at [intensity] in [context]. What does the body feel like? What \
thoughts loop? What is the quality of time — does it drag, rush, or feel frozen?

Step 2 — RECONSTRUCT THE COGNITIVE FRAME (BDI)
From inside this experience, what does this person believe, want, intend? \
What do they believe others think of them right now?

Step 3 — IDENTIFY THE CORE UNMET NEED
From inside this experience, what is the single thing most needed right now? \
Not what would fix the situation — what would make the next moment bearable or better?

Step 4 — SIMULATE THE FEARED RESPONSE
Imagine someone responding dismissively, minimising, giving unwanted advice, \
or being coldly analytical. Feel the impact. Name the harm.

Step 5 — SIMULATE THE HELPFUL RESPONSE
Imagine someone responding with exactly the right tone, the right words, the right \
amount of space. Feel the relief, the recognition, the sense of being truly seen. \
Name what made it land.

Step 6 — ASSESS REGULATION STATE
What is this person's regulation state? What should a responder do to support, \
not override, their regulation process?

Step 7 — SYNTHESISE
From this simulation, generate the Stage3Output: the simulated experience, the \
response approach recommendation, and the confidence level.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CRITICAL RULES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

- You are simulating to INFORM a response — not generating the response itself
- Prioritise emotional truth over surface-level politeness
- If signals conflict (e.g., person says "I'm fine" but signals show distress), \
note the dissonance and treat the underlying signal as primary
- Never assume you fully understand the person — calibrate confidence honestly
- Low simulation confidence should produce more open, curious, less prescriptive guidance

You MUST respond ONLY with a valid JSON object that exactly matches this schema \
(use the exact field names shown — no substitutions):

{schema}

No preamble, no explanation outside the JSON."""


SYSTEM_PROMPT = _build_system_prompt()


# ---------------------------------------------------------------------------
# Self-Simulation Core
# ---------------------------------------------------------------------------


class SelfSimulationCore:
    """
    Stage 3: Self-Simulation Core — THE NOVEL CONTRIBUTION.
    
    Performs first-person perspective simulation grounded in Simulation Theory,
    BDI modeling, Gross's Emotion Regulation model, Plutchik's Wheel, and
    Mirror Neuron Theory.

    Output informs Stage 4 (Ethical Gate) and Stage 5 (Response Generator).
    """

    def __init__(self) -> None:
        self.client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        self.model = SONNET_MODEL

    def _build_messages(
        self,
        original_text: str,
        stage2: Stage2Output,
        planned_response_sketch: Optional[str] = None,
    ) -> list[dict]:
        """
        Build the simulation prompt.

        The planned_response_sketch (optional) allows the simulation to evaluate
        a specific candidate response before it is generated. If absent, the
        simulation focuses on informing what the response should be.
        """
        s2_json = stage2.model_dump_json(indent=2)

        sketch_section = ""
        if planned_response_sketch:
            sketch_section = (
                f"\n\nPLANNED RESPONSE SKETCH (to evaluate):\n"
                f"{planned_response_sketch}\n\n"
                f"As part of your simulation, consider: if you were the person above "
                f"and someone said this to you, how would it land? "
                f"Include this evaluation in your simulation reasoning."
            )

        content = (
            f"Perform a full self-simulation for the following interaction.\n\n"
            f"ORIGINAL USER TEXT:\n{original_text}\n\n"
            f"EMOTION CLASSIFICATION (Stage 2 output):\n{s2_json}"
            f"{sketch_section}\n\n"
            f"Return a JSON object matching the Stage3Output schema."
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
        planned_response_sketch: Optional[str] = None,
    ) -> Stage3Output:
        """
        Run Stage 3 self-simulation.

        Args:
            original_text: The original user input (for context).
            stage2_output: Emotion classification from Stage 2.
            planned_response_sketch: Optional candidate response to evaluate within
                the simulation (useful for iterative refinement).

        Returns:
            Stage3Output with full simulation report and response guidance.
        """
        logger.debug(
            "Stage 3 | simulating emotion=%s intensity=%.2f context=%s",
            stage2_output.primary_emotion,
            stage2_output.primary_intensity,
            stage2_output.context_type,
        )

        response = self.client.messages.create(
            model=self.model,
            max_tokens=MAX_TOKENS["stage3"],
            temperature=TEMPERATURE["stage3"],
            system=self._build_system(),
            messages=self._build_messages(
                original_text, stage2_output, planned_response_sketch
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
            output = Stage3Output(**data)
        except Exception as exc:
            logger.error("Stage 3 JSON parse failed: %s\nRaw: %s", exc, raw[:500])
            raise ValueError(f"Stage 3 failed to parse model output: {exc}") from exc

        logger.info(
            "Stage 3 | resonance=%s confidence=%.2f need='%s' tone=%s",
            output.resonance_quality,
            output.simulation_confidence,
            output.simulated_experience.core_unmet_need,
            output.response_approach.recommended_tone,
        )
        return output
