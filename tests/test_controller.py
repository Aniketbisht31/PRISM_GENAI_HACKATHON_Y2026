"""
Unit tests for Retrieval Controller (Phase 2).
Tests both RetrievalController (LLM-based) and RuleBasedController (heuristic-based).
Verifies:
- Fragments under 4 words without entities classified as WAIT
- Clear questions and entities trigger RETRIEVE
- Presentation-only keywords and conversational fillers classified as NO_RETRIEVAL
- Both controllers emit telemetry events
"""

import pytest
from unittest.mock import MagicMock, patch

from src.controller.controller import (
    RetrievalController,
    RuleBasedController,
    ControllerDecision,
)
from src.llm.adapter import LLMResponse
from src.telemetry.logger import TelemetryLogger


@pytest.fixture
def mock_logger():
    logger = MagicMock(spec=TelemetryLogger)
    return logger


def test_rule_based_controller_wait_on_short_fragment():
    controller = RuleBasedController()
    decision = controller.evaluate_chunk("I need to", timestamp_s=0.2, session_id="test_s")
    assert isinstance(decision, ControllerDecision)
    assert decision.decision == "WAIT"


def test_rule_based_controller_retrieve_on_question():
    controller = RuleBasedController()
    decision = controller.evaluate_chunk(
        "What is the seating capacity of Venue B?",
        timestamp_s=1.2,
        session_id="test_s"
    )
    assert decision.decision == "RETRIEVE"
    assert decision.confidence > 0.5


def test_rule_based_controller_no_retrieval_filler():
    controller = RuleBasedController()
    decision = controller.evaluate_chunk(
        "Thanks, that's helpful.",
        timestamp_s=2.0,
        session_id="test_s"
    )
    assert decision.decision == "NO_RETRIEVAL"


def test_rule_based_controller_no_retrieval_presentation():
    controller = RuleBasedController()
    decision = controller.evaluate_chunk(
        "Can you repeat that shorter please?",
        timestamp_s=2.5,
        session_id="test_s"
    )
    assert decision.decision == "NO_RETRIEVAL"


def test_llm_controller_with_mock_adapter():
    with patch("src.controller.controller.get_adapter") as mock_get_adapter:
        mock_adapter = MagicMock()
        mock_adapter.load_prompt_template.return_value = "Template: {transcript}"
        mock_adapter.complete.return_value = LLMResponse(
            content='{"decision": "RETRIEVE", "reason": "Clear request for Venue B capacity", "confidence": 0.95}',
            model="llama-3.3-70b-versatile",
            usage={"prompt_tokens": 50, "completion_tokens": 20, "total_tokens": 70},
            latency_ms=120.0,
        )
        mock_get_adapter.return_value = mock_adapter

        controller = RetrievalController()
        decision = controller.evaluate_chunk(
            "What is the seating capacity of Venue B?",
            timestamp_s=1.0,
            session_id="test_s_llm"
        )

        assert decision.decision == "RETRIEVE"
        assert decision.confidence == 0.95
        assert "Venue B" in decision.reason


def test_llm_controller_handles_malformed_json_fallback():
    with patch("src.controller.controller.get_adapter") as mock_get_adapter:
        mock_adapter = MagicMock()
        mock_adapter.load_prompt_template.return_value = "{transcript}"
        mock_adapter.complete.return_value = LLMResponse(
            content="Not valid json at all",
            model="llama-3.3-70b-versatile",
        )
        mock_get_adapter.return_value = mock_adapter

        controller = RetrievalController()
        decision = controller.evaluate_chunk("some fragment", 0.5, "test_s_malformed")
        # Defensively falls back to WAIT
        assert decision.decision == "WAIT"


def test_llm_controller_provider_error_is_redacted_and_logged():
    mock_logger = MagicMock(spec=TelemetryLogger)
    with patch("src.controller.controller.get_adapter") as mock_get_adapter, \
            patch("src.controller.controller.get_logger", return_value=mock_logger):
        mock_adapter = MagicMock()
        mock_adapter.load_prompt_template.return_value = "{transcript}"
        mock_adapter.complete.side_effect = RuntimeError("secret provider response")
        mock_get_adapter.return_value = mock_adapter

        controller = RetrievalController()
        decision = controller.evaluate_chunk("What is Venue B capacity?", 0.5, "s_failure")

    assert decision.decision == "WAIT"
    assert "secret provider response" not in decision.reason
    mock_logger.log_llm_failure.assert_called_once()
