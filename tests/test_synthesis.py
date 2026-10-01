"""
Unit tests for Session-Aware Synthesizer (Phase 5).
Tests:
- In-memory ephemeral session state (isolated per session_id)
- Initial synthesis with citation validation
- Strict prevention of hallucinated doc IDs
- Refinement with late constraints (Version N+1 without restarting)
- Presentation-only handling (0 new retrievals, reformatting)
- Explicit uncertainty emission when corpus lacks coverage
"""

import json
import pytest
from unittest.mock import MagicMock, patch

from src.synthesis.synthesizer import (
    SessionSynthesizer,
    SynthesisResult,
    SessionState,
    AnswerVersion,
)
from src.llm.adapter import LLMResponse


@pytest.fixture
def sample_chunks():
    return [
        {
            "doc_id": "Doc_02",
            "section": "\u00a71",
            "title": "Venue B - Overview",
            "text": "Venue B is located in Kharadi, Pune. Maximum capacity 100 people theatre-style, 50 people workshop-style. Base rental cost INR 35,000 per day.",
        },
        {
            "doc_id": "Doc_04",
            "section": "\u00a71",
            "title": "Venue B - Cancellation Policy",
            "text": "Full refund if cancelled 10 or more days before the event, 50% between 5 and 10 days.",
        }
    ]


def test_initial_synthesis_with_citations(sample_chunks):
    with patch("src.synthesis.synthesizer.get_adapter") as mock_get_adapter:
        mock_adapter = MagicMock()
        mock_adapter.load_prompt_template.return_value = "{chunks}\n{sub_queries}\n{prior_context}"
        mock_adapter.complete.return_value = LLMResponse(
            content=json.dumps({
                "answer": "\"Maximum capacity 100 people theatre-style, 50 people workshop-style.\" [Doc_02 \u00a71]. \"Full refund if cancelled 10 or more days before the event, 50% between 5 and 10 days.\" [Doc_04 \u00a71].",
                "citations": [
                    {"doc_id": "Doc_02", "section": "\u00a71", "claim": "Maximum capacity 100 people theatre-style, 50 people workshop-style."},
                    {"doc_id": "Doc_04", "section": "\u00a71", "claim": "Full refund if cancelled 10 or more days before the event, 50% between 5 and 10 days."}
                ],
                "uncertainty": None
            }),
            model="llama-3.3-70b-versatile",
            usage={"prompt_tokens": 100, "completion_tokens": 40, "total_tokens": 140},
        )
        mock_get_adapter.return_value = mock_adapter

        synth = SessionSynthesizer()
        session_id = synth.create_session()

        result = synth.synthesize(
            session_id=session_id,
            sub_queries=["Venue B workshop capacity", "Venue B cancellation policy"],
            retrieved_chunks=sample_chunks,
            retrieval_events=[{"query": "Venue B", "num_chunks": 2}],
        )

        assert isinstance(result, SynthesisResult)
        assert result.answer_version == 1
        assert len(result.citations) == 2
        assert result.uncertainty is None
        assert "50 people" in result.answer


def test_hallucinated_doc_ids_filtered(sample_chunks):
    with patch("src.synthesis.synthesizer.get_adapter") as mock_get_adapter:
        mock_adapter = MagicMock()
        mock_adapter.load_prompt_template.return_value = "{chunks}\n{sub_queries}\n{prior_context}"
        # LLM returns a hallucinated doc ID (Doc_99) alongside valid Doc_02
        mock_adapter.complete.return_value = LLMResponse(
            content=json.dumps({
                "answer": "\"Maximum capacity 100 people theatre-style, 50 people workshop-style.\" [Doc_02 \u00a71]. \"Music license included\" [Doc_99 \u00a71].",
                "citations": [
                    {"doc_id": "Doc_02", "section": "\u00a71", "claim": "Maximum capacity 100 people theatre-style, 50 people workshop-style."},
                    {"doc_id": "Doc_99", "section": "\u00a71", "claim": "fake claim"}
                ],
                "uncertainty": None
            }),
            model="llama-3.3-70b-versatile",
        )
        mock_get_adapter.return_value = mock_adapter

        synth = SessionSynthesizer()
        session_id = synth.create_session()

        result = synth.synthesize(
            session_id=session_id,
            sub_queries=["Venue B"],
            retrieved_chunks=sample_chunks,
            retrieval_events=[],
        )

        # Doc_99 must be filtered out because it is NOT in sample_chunks
        doc_ids = [c["doc_id"] for c in result.citations]
        assert "Doc_02" in doc_ids
        assert "Doc_99" not in doc_ids
        assert "Doc_99" not in result.answer
        assert "Music license included" not in result.answer
        assert result.uncertainty is not None


def test_unsupported_generation_falls_back_to_exact_retrieved_excerpt(sample_chunks):
    with patch("src.synthesis.synthesizer.get_adapter") as mock_get_adapter:
        mock_adapter = MagicMock()
        mock_adapter.load_prompt_template.return_value = "{chunks} {sub_queries} {prior_context}"
        mock_adapter.complete.return_value = LLMResponse(
            content=json.dumps({
                "answer": "Venue B can seat 50 workshop attendees. [Doc_02 \u00a71]",
                "citations": [{
                    "doc_id": "Doc_02", "section": "\u00a71",
                    "claim": "Venue B can seat 50 workshop attendees.",
                }],
                "uncertainty": None,
            }),
            model="test-model",
        )
        mock_get_adapter.return_value = mock_adapter
        result = SessionSynthesizer().synthesize(
            session_id="extractive-fallback",
            sub_queries=["Venue B workshop capacity"],
            retrieved_chunks=[sample_chunks[0]],
            retrieval_events=[],
        )

    assert result.citations == [{
        "doc_id": "Doc_02", "section": "\u00a71",
        "claim": "Maximum capacity 100 people theatre-style, 50 people workshop-style.",
    }]
    assert "Maximum capacity 100 people theatre-style, 50 people workshop-style." in result.answer
    assert "could not be tied" in result.uncertainty


def test_answer_about_wrong_named_venue_is_rejected(sample_chunks):
    venue_a = {
        "doc_id": "Doc_01", "section": "\u00a71", "title": "Venue A - Overview",
        "text": "Venue A is a corporate event space with a maximum seating capacity of 80 people.",
    }
    with patch("src.synthesis.synthesizer.get_adapter") as mock_get_adapter:
        mock_adapter = MagicMock()
        mock_adapter.load_prompt_template.return_value = "{chunks} {sub_queries} {prior_context}"
        mock_adapter.complete.return_value = LLMResponse(
            content=json.dumps({
                "answer": "Venue A is a corporate event space with a maximum seating capacity of 80 people. [Doc_01 \u00a71]",
                "citations": [{
                    "doc_id": "Doc_01", "section": "\u00a71",
                    "claim": "Venue A is a corporate event space with a maximum seating capacity of 80 people.",
                }],
                "uncertainty": None,
            }),
            model="test-model",
        )
        mock_get_adapter.return_value = mock_adapter
        result = SessionSynthesizer().synthesize(
            session_id="wrong-entity",
            sub_queries=["What is the seating capacity of Venue B?"],
            retrieved_chunks=[venue_a, sample_chunks[0]],
            retrieval_events=[],
        )

    assert all(citation["doc_id"] != "Doc_01" for citation in result.citations)
    assert "Venue A" not in result.answer
    assert "Venue B" in result.answer


def test_provider_synthesis_failure_uses_exact_excerpt_fallback(sample_chunks):
    with patch("src.synthesis.synthesizer.get_adapter") as mock_get_adapter:
        mock_adapter = MagicMock()
        mock_adapter.load_prompt_template.return_value = "{chunks} {sub_queries} {prior_context}"
        mock_adapter.complete.side_effect = RuntimeError("provider unavailable")
        mock_get_adapter.return_value = mock_adapter
        result = SessionSynthesizer().synthesize(
            session_id="provider-fallback",
            sub_queries=["Venue B workshop capacity"],
            retrieved_chunks=[sample_chunks[0]],
            retrieval_events=[],
        )

    assert result.citations
    assert "Maximum capacity 100 people theatre-style" in result.answer
    assert "model could not complete synthesis" in result.uncertainty


def test_refine_with_late_constraint(sample_chunks):
    with patch("src.synthesis.synthesizer.get_adapter") as mock_get_adapter:
        mock_adapter = MagicMock()
        mock_adapter.load_prompt_template.return_value = "{chunks}\n{sub_queries}\n{prior_context}"
        
        # Call 1: initial answer
        mock_adapter.complete.side_effect = [
            LLMResponse(
                content=json.dumps({
                    "answer": "\"Base rental cost INR 35,000 per day.\" [Doc_02 \u00a71].",
                    "citations": [
                    {"doc_id": "Doc_02", "section": "\u00a71", "claim": "Base rental cost INR 35,000 per day."},
                    ],
                    "uncertainty": None
                }),
                model="llama-3.3-70b-versatile",
            ),
            # Call 2: refinement with monsoon season constraint
            LLMResponse(
                content=json.dumps({
                    "answer": "\"Base rental cost INR 35,000 per day.\" [Doc_02 \u00a71]. \"During monsoon season, Pune experiences heavy rainfall from June to September, with an indoor contingency plan recommended.\" [Doc_12 \u00a71].",
                    "citations": [
                        {"doc_id": "Doc_02", "section": "\u00a71", "claim": "Base rental cost INR 35,000 per day."},
                        {"doc_id": "Doc_12", "section": "\u00a71", "claim": "During monsoon season, Pune experiences heavy rainfall from June to September, with an indoor contingency plan recommended."}
                    ],
                    "uncertainty": None
                }),
                model="llama-3.3-70b-versatile",
            )
        ]
        mock_get_adapter.return_value = mock_adapter

        synth = SessionSynthesizer()
        session_id = synth.create_session()

        # Version 1
        res1 = synth.synthesize(
            session_id=session_id,
            sub_queries=["Venue B rental cost"],
            retrieved_chunks=[sample_chunks[0]],
            retrieval_events=[],
        )
        assert res1.answer_version == 1

        # Version 2 (Refinement with late constraint)
        monsoon_chunk = {
            "doc_id": "Doc_12",
            "section": "\u00a71",
            "title": "Weather and Seasonal Considerations - Pune",
            "text": "During monsoon season, Pune experiences heavy rainfall from June to September, with an indoor contingency plan recommended.",
        }
        res2 = synth.refine_with_late_constraint(
            session_id=session_id,
            new_sub_queries=["monsoon season considerations"],
            new_retrieved_chunks=[monsoon_chunk],
            new_retrieval_events=[{"query": "monsoon", "num_chunks": 1}],
        )

        assert res2.answer_version == 2
        assert len(res2.citations) == 2
        assert "Doc_02" in [c["doc_id"] for c in res2.citations]
        assert "Doc_12" in [c["doc_id"] for c in res2.citations]


def test_handle_presentation_only(sample_chunks):
    with patch("src.synthesis.synthesizer.get_adapter") as mock_get_adapter:
        mock_adapter = MagicMock()
        def mock_load_template(path):
            if "reformat" in path:
                return "{previous_answer}\n{request}\n{citations}"
            return "{chunks}\n{sub_queries}\n{prior_context}"
        mock_adapter.load_prompt_template.side_effect = mock_load_template


        
        mock_adapter.complete.side_effect = [
            # Call 1: initial answer
            LLMResponse(
                content=json.dumps({
                    "answer": "\"Maximum capacity 100 people theatre-style, 50 people workshop-style.\" [Doc_02 \u00a71].",
                    "citations": [
                    {"doc_id": "Doc_02", "section": "\u00a71", "claim": "Maximum capacity 100 people theatre-style, 50 people workshop-style."},
                    ],
                    "uncertainty": None
                }),
                model="llama-3.3-70b-versatile",
            ),
            # Call 2: presentation reformatting
            LLMResponse(
                content=json.dumps({
                    "answer": "\u2022 \"Maximum capacity 100 people theatre-style, 50 people workshop-style.\" [Doc_02 \u00a71]",
                    "citations": [
                    {"doc_id": "Doc_02", "section": "\u00a71", "claim": "Maximum capacity 100 people theatre-style, 50 people workshop-style."},
                    ],
                    "uncertainty": None
                }),
                model="llama-3.3-70b-versatile",
            )
        ]
        mock_get_adapter.return_value = mock_adapter

        synth = SessionSynthesizer()
        session_id = synth.create_session()

        res1 = synth.synthesize(
            session_id=session_id,
            sub_queries=["Venue B capacity"],
            retrieved_chunks=[sample_chunks[0]],
            retrieval_events=[],
        )

        # Presentation turn: Make it bullet points
        res2 = synth.handle_presentation_only(session_id, "Make that a bullet point")

        assert res2.answer_version == 2
        assert len(res2.retrieval_events) == 0  # ZERO new retrievals
        assert len(res2.citations) == 1
        assert "•" in res2.answer


def test_uncertainty_handling():
    with patch("src.synthesis.synthesizer.get_adapter") as mock_get_adapter:
        mock_adapter = MagicMock()
        mock_adapter.load_prompt_template.return_value = "{chunks}\n{sub_queries}\n{prior_context}"
        mock_adapter.complete.return_value = LLMResponse(
            content=json.dumps({
                "answer": "There is no information in the available documents regarding live music licensing.",
                "citations": [],
                "uncertainty": "Live music performance licensing is not covered in the workshop planning corpus."
            }),
            model="llama-3.3-70b-versatile",
        )
        mock_get_adapter.return_value = mock_adapter

        synth = SessionSynthesizer()
        session_id = synth.create_session()

        result = synth.synthesize(
            session_id=session_id,
            sub_queries=["Does Venue A offer live music licensing?"],
            retrieved_chunks=[],
            retrieval_events=[],
        )

        assert result.uncertainty is not None
        assert "live music" in result.uncertainty.lower()
        assert len(result.citations) == 0
        mock_adapter.complete.assert_not_called()


def test_uncited_sentences_are_removed_from_grounded_answer():
    answer, removed = SessionSynthesizer._keep_cited_sentences(
        "Supported capacity is 50 [Doc_02 Â§1]. A fabricated detail is also true.",
        {("Doc_02", "Â§1")},
    )

    assert "Supported capacity" in answer
    assert "fabricated detail" not in answer
    assert removed is True


def test_trailing_citation_does_not_cover_preceding_facts():
    answer, removed = SessionSynthesizer._keep_cited_sentences(
        "Capacity is 100. Workshop layout capacity is 50. [Doc_02 \u00c2\u00a71]",
        {("Doc_02", "\u00c2\u00a71")},
    )
    assert "Capacity is 100" not in answer
    assert "Workshop layout capacity is 50" not in answer
    assert removed is True


def test_sections_repair_and_exact_source_claims_only():
    from src.provenance import normalize_section, evidence_supports_claim

    assert normalize_section("\ufffd1") == "\u00a71"
    source = "Venue B has capacity for 50 people in workshop-style layout."
    assert evidence_supports_claim("capacity for 50 people in workshop-style layout.", source)
    assert not evidence_supports_claim("Venue B seats 100 people.", source)


def test_uncited_sentence_does_not_inherit_following_citation():
    answer, removed = SessionSynthesizer._keep_cited_sentences(
        "Unsupported extra fact. Supported capacity is 50 [Doc_02 \u00c2\u00a71].",
        {("Doc_02", "\u00c2\u00a71")},
    )
    assert "Unsupported extra fact" not in answer
    assert removed is True


def test_sessions_have_independent_ephemeral_answer_state():
    synth = SessionSynthesizer.__new__(SessionSynthesizer)
    synth._sessions = {}
    first = synth._get_or_create_session("session-one")
    first.answer_versions.append(
        AnswerVersion(
            version=1, answer="private prior answer", citations=[], uncertainty=None,
            sub_queries=[], retrieved_chunks=[],
        )
    )
    second = synth._get_or_create_session("session-two")
    assert first.latest_answer.answer == "private prior answer"
    assert second.latest_answer is None
    synth.clear_session("session-one")
    assert synth.get_session("session-one") is None
