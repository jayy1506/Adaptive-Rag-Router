import json
import time
from pathlib import Path

import pytest

from src.router.features import QueryFeatureExtractor
from src.router.models import RouterInferenceWrapper


@pytest.fixture(scope="module")
def extractor():
    return QueryFeatureExtractor()


def test_feature_extractor_shapes(extractor):
    query = "When was Arthur's Magazine first published?"
    ling = extractor.extract_linguistic_features(query)
    emb = extractor.extract_embedding(query)
    hybrid = extractor.extract_batch([query], feature_type="hybrid")

    assert ling.shape == (14,)
    assert emb.shape == (384,)
    assert hybrid.shape == (1, 14 + 384)


def test_router_inference_and_latency(extractor):
    model_path = Path("results/router_model.joblib")
    assert model_path.exists(), "router_model.joblib must exist after training"

    router = RouterInferenceWrapper.load(str(model_path))

    query = "Who directed Inception?"
    # Warm-up inference to avoid cold-start PyTorch JIT overhead
    _ = router.predict_single(query, extractor)

    t0 = time.perf_counter()
    pred_k, confidence, prob_dict = router.predict_single(query, extractor)
    latency_ms = (time.perf_counter() - t0) * 1000.0

    assert pred_k in router.candidate_k
    assert 0.0 <= confidence <= 1.0
    assert pytest.approx(sum(prob_dict.values()), rel=1e-2) == 1.0
    # Steady-state classifier inference should be fast (under 50ms)
    assert latency_ms < 50.0, f"Router inference too slow: {latency_ms:.2f}ms"


def test_router_ablation_results():
    ablation_path = Path("results/router_ablation.json")
    assert ablation_path.exists(), "router_ablation.json must exist"

    with open(ablation_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert len(data) >= 6
    exp_ids = {d["experiment_id"] for d in data}
    assert "logreg_linguistic" in exp_ids
    assert "logreg_embedding" in exp_ids
    assert "xgboost_hybrid" in exp_ids
    assert "small_encoder_mlp_hybrid" in exp_ids

    for d in data:
        assert 0.0 <= d["accuracy"] <= 1.0
        assert 0.0 <= d["macro_f1"] <= 1.0
        assert "latency_ms_per_query" in d
