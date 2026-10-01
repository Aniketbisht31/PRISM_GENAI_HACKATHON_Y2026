"""
Retrieval Controller — decides when to trigger retrieval during streaming transcript input.
Two variants: LLM-based (RetrievalController) and heuristic-based (RuleBasedController).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from src.llm.adapter import get_adapter, BaseLLMAdapter
from src.telemetry.logger import get_logger, TelemetryLogger


@dataclass
class ControllerDecision:
    """Result of evaluating a transcript chunk."""
    decision: str  # WAIT, RETRIEVE, NO_RETRIEVAL
    reason: str
    confidence: float
    timestamp_s: float


class RetrievalController:
    """LLM-based controller that classifies intent stability via the adapter."""

    def __init__(self):
        self.adapter: BaseLLMAdapter = get_adapter()
        self.logger: TelemetryLogger = get_logger()
        self._prompt_template = self.adapter.load_prompt_template("prompts/intent_stability.txt")

    def evaluate_chunk(
        self, cumulative_text: str, timestamp_s: float, session_id: str
    ) -> ControllerDecision:
        """Classify the cumulative transcript as WAIT / RETRIEVE / NO_RETRIEVAL."""
        prompt = self._prompt_template.replace("{transcript}", cumulative_text)


        try:
            response = self.adapter.complete(
                prompt=prompt,
                system_prompt="Classify the user's streaming transcript for retrieval readiness.",
                json_mode=True,
                temperature=0.1,
                max_tokens=256,
            )

            # Log the LLM call
            self.logger.log_llm_call(
                session_id=session_id,
                purpose="controller_intent_stability",
                model=response.model,
                usage=response.usage,
                latency_ms=response.latency_ms,
            )

            # Defensive JSON parsing
            parsed = self._parse_json(response.content)
            decision = parsed.get("decision", "WAIT").upper()
            reason = parsed.get("reason", "Parsed from LLM")
            confidence = float(parsed.get("confidence", 0.5))

            # Validate decision value
            if decision not in ("WAIT", "RETRIEVE", "NO_RETRIEVAL"):
                decision = "WAIT"
                reason = f"Invalid decision from LLM, defaulting to WAIT"

        except Exception as e:
            decision = "WAIT"
            reason = "The intent controller is temporarily unavailable; waiting for more context."
            confidence = 0.0
            self.logger.log_llm_failure(session_id, "controller_intent_stability", e)

        result = ControllerDecision(
            decision=decision,
            reason=reason,
            confidence=confidence,
            timestamp_s=timestamp_s,
        )

        # Log the decision via telemetry
        self.logger.log_controller_decision(
            session_id=session_id,
            decision=result.decision,
            reason=result.reason,
            confidence=result.confidence,
            transcript=cumulative_text,
            timestamp_s=timestamp_s,
            controller_type="llm",
        )

        return result

    @staticmethod
    def _parse_json(content: str) -> dict:
        """Defensively parse JSON from LLM output, stripping code fences."""
        content = content.strip()
        # Strip code fences
        content = re.sub(r"^```(?:json)?\s*", "", content)
        content = re.sub(r"\s*```$", "", content)
        content = content.strip()

        try:
            return json.loads(content)
        except json.JSONDecodeError:
            # Try extracting JSON object from the text
            match = re.search(r"\{[^{}]*\}", content, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group())
                except json.JSONDecodeError:
                    pass
            return {"decision": "WAIT", "reason": "Failed to parse LLM JSON", "confidence": 0.0}


class RuleBasedController:
    """
    Heuristic-based controller — no LLM call.
    Used for the ablation study comparing against the LLM controller.
    """

    # Presentation-only keywords
    PRESENTATION_PATTERNS = [
        r"\brepeat\b", r"\bshorter\b", r"\bsimpler\b", r"\bformal\b", r"\btranslate\b",
        r"\bbullet\s*points?\b", r"\bsummarize\s+that\b", r"\bmake\s+(it|that)\s+shorter\b",
        r"\bsimplify\b", r"\breformat\b", r"\brephrase\b", r"\bsay\s+that\s+again\b",
    ]

    # Conversational fillers / closings
    FILLER_PATTERNS = [
        r"^thanks?[\s,\.]*", r"^thank\s+you[\s,\.]*", r"^got\s+it[\s,\.]*",
        r"^okay[\s,\.]*", r"^ok[\s,\.]*", r"^yes[\s,\.]*$", r"^no[\s,\.]*$",
        r"^yeah[\s,\.]*", r"^right[\s,\.]*$", r"^sure[\s,\.]*", r"^great[\s,\.]*",
        r".*that'?s?\s+helpful.*", r"^understood[\s,\.]*", r".*thank\s*you.*",
    ]


    # Named entities that indicate enough context even in short utterances
    ENTITY_PATTERNS = [
        r"\bvenue\s+[ab]\b", r"\bdoc[_\s]*\d+\b", r"\binr\s*\d+",
        r"\bpune\b", r"\bhinjewadi\b", r"\bkharadi\b",
    ]

    def __init__(self):
        self.logger: TelemetryLogger = get_logger()

    def evaluate_chunk(
        self, cumulative_text: str, timestamp_s: float, session_id: str
    ) -> ControllerDecision:
        """Classify using word count, punctuation, named-entity regex, and keyword detection."""
        text = cumulative_text.strip()
        text_lower = text.lower()
        words = text_lower.split()
        word_count = len(words)

        decision = "WAIT"
        reason = "Default: waiting for more context"
        confidence = 0.3

        # === Check NO_RETRIEVAL first ===
        # Presentation-only
        for pattern in self.PRESENTATION_PATTERNS:
            if re.search(pattern, text_lower):
                decision = "NO_RETRIEVAL"
                reason = f"Presentation-only keyword detected: {pattern}"
                confidence = 0.9
                break

        # Conversational filler
        if decision != "NO_RETRIEVAL":
            for pattern in self.FILLER_PATTERNS:
                if re.match(pattern, text_lower):
                    decision = "NO_RETRIEVAL"
                    reason = "Conversational filler/closing"
                    confidence = 0.85
                    break

        # === Check RETRIEVE ===
        if decision == "WAIT":
            has_entity = any(re.search(p, text_lower) for p in self.ENTITY_PATTERNS)
            has_question_word = bool(re.search(r"\b(what|where|when|who|why|how|which|does|do|can|is|are)\b", text_lower))
            has_sentence_end = bool(re.search(r"[.?!]$", text))
            ends_with_conjunction = bool(re.search(r"\b(and|or|but|also|with|for|the|a|in|to)\s*\.{0,3}$", text_lower))

            if word_count >= 4 and has_sentence_end and not ends_with_conjunction:
                decision = "RETRIEVE"
                reason = "Sufficient words with sentence-ending punctuation"
                confidence = 0.8
            elif word_count >= 4 and has_question_word and not ends_with_conjunction:
                decision = "RETRIEVE"
                reason = "Question word detected with adequate context"
                confidence = 0.7
            elif has_entity and word_count >= 3:
                decision = "RETRIEVE"
                reason = "Named entity detected with minimal context"
                confidence = 0.65
            elif word_count >= 7 and not ends_with_conjunction:
                decision = "RETRIEVE"
                reason = "Long utterance, likely complete"
                confidence = 0.6
            elif word_count < 4 and not has_entity:
                decision = "WAIT"
                reason = f"Fragment too short ({word_count} words), no named entity"
                confidence = 0.5

        result = ControllerDecision(
            decision=decision,
            reason=reason,
            confidence=confidence,
            timestamp_s=timestamp_s,
        )

        # Log the decision via telemetry
        self.logger.log_controller_decision(
            session_id=session_id,
            decision=result.decision,
            reason=result.reason,
            confidence=result.confidence,
            transcript=cumulative_text,
            timestamp_s=timestamp_s,
            controller_type="rule_based",
        )

        return result
