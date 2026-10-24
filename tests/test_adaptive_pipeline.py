from unittest.mock import MagicMock

import pytest

from src.pipeline.adaptive import AdaptiveRetrievalPipeline, AdaptiveRoutingResult


@pytest.fixture
def mock_pipeline():
    """Construct pipeline with mocked subcomponents for deterministic behavior."""
    mock_router = MagicMock()
    mock_retriever = MagicMock()
    mock_generator = MagicMock()
    mock_extractor = MagicMock()

    pipeline = AdaptiveRetrievalPipeline(
        config_path="config.yaml",
        router=mock_router,
        retriever=mock_retriever,
        generator=mock_generator,
        extractor=mock_extractor,
    )
    return pipeline, mock_router, mock_retriever, mock_generator


def test_next_higher_k_progression(mock_pipeline):
    pipeline, _, _, _ = mock_pipeline
    assert pipeline._get_next_higher_k(0) == 3
    assert pipeline._get_next_higher_k(3) == 8
    assert pipeline._get_next_higher_k(8) == 15
    assert pipeline._get_next_higher_k(15) == 15


def test_uncertainty_detection(mock_pipeline):
    pipeline, _, _, _ = mock_pipeline
    assert pipeline._is_uncertain_answer("") is True
    assert pipeline._is_uncertain_answer("   ") is True
    assert pipeline._is_uncertain_answer("I do not know the answer.") is True
    assert pipeline._is_uncertain_answer("There is not enough information to tell.") is True
    assert pipeline._is_uncertain_answer("It is unclear from the passage.") is True

    assert pipeline._is_uncertain_answer("Christopher Nolan directed Inception.") is False
    assert pipeline._is_uncertain_answer("1844") is False


def test_confidence_fallback_trigger(mock_pipeline):
    pipeline, mock_router, mock_retriever, mock_generator = mock_pipeline

    # Router predicts k=3 with low confidence (0.45 < 0.65)
    mock_router.predict_single.return_value = (3, 0.45, {0: 0.1, 3: 0.45, 8: 0.25, 15: 0.2})
    mock_retriever.retrieve.return_value = [{"id": "p1", "text": "pass 1", "title": "t1", "score": 0.9}]
    mock_generator.generate.return_value = {
        "answer": "Christopher Nolan",
        "tokens_in": 120,
        "tokens_out": 5,
        "latency_ms": 15.0,
        "metadata": {"k": 8},
        "cached": False,
    }

    result: AdaptiveRoutingResult = pipeline.run("Who directed Inception?", use_cache=False)

    assert result.predicted_k == 3
    assert result.fallback_triggered is True
    assert result.executed_k == 8  # stepped up from 3 to 8
    assert result.escalation_triggered is False
    assert result.answer == "Christopher Nolan"
    mock_retriever.retrieve.assert_called_with("Who directed Inception?", k=8)


def test_sufficiency_escalation_trigger(mock_pipeline):
    pipeline, mock_router, mock_retriever, mock_generator = mock_pipeline

    # High confidence for k=3, but answer is insufficient
    mock_router.predict_single.return_value = (3, 0.90, {0: 0.05, 3: 0.90, 8: 0.03, 15: 0.02})

    # First retrieval returns top-3, second returns top-15
    mock_retriever.retrieve.side_effect = [
        [{"id": "p1", "text": "vague text", "title": "t1", "score": 0.7}],
        [{"id": "p1", "text": "vague text", "title": "t1", "score": 0.7}, {"id": "p2", "text": "exact answer", "title": "t2", "score": 0.95}],
    ]

    # First generation says uncertain, escalated generation succeeds
    mock_generator.generate.side_effect = [
        {
            "answer": "I do not know based on the provided text.",
            "tokens_in": 80,
            "tokens_out": 10,
            "latency_ms": 10.0,
            "metadata": {"k": 3},
            "cached": False,
        },
        {
            "answer": "Christopher Nolan",
            "tokens_in": 350,
            "tokens_out": 5,
            "latency_ms": 25.0,
            "metadata": {"k": 15},
            "cached": False,
        },
    ]

    result = pipeline.run("Who directed Inception?", use_cache=False)

    assert result.predicted_k == 3
    assert result.fallback_triggered is False
    assert result.escalation_triggered is True
    assert result.executed_k == 15
    assert result.answer == "Christopher Nolan"
    assert result.prompt_tokens == 80 + 350
    assert result.completion_tokens == 10 + 5
    assert result.total_tokens == 445


def test_real_pipeline_execution():
    """Verify integration against the actual Chroma DB and saved router model."""
    pipeline = AdaptiveRetrievalPipeline()
    res = pipeline.run("When was Arthur's Magazine first published?", use_cache=True)

    assert isinstance(res, AdaptiveRoutingResult)
    assert res.predicted_k in [0, 3, 8, 15]
    assert res.executed_k in [0, 3, 8, 15]
    assert 0.0 <= res.confidence <= 1.0
    assert res.total_tokens >= 0
    assert res.total_latency_ms > 0
    assert res.router_latency_ms >= 0
    assert res.retrieval_latency_ms >= 0
    assert res.generation_latency_ms >= 0
