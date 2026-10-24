import json
from pathlib import Path

from src.router.labeler import assign_cheapest_k


def test_assign_cheapest_k_prefers_cheapest():
    candidate_k = [0, 3, 8, 15]

    # k=0 achieves threshold -> should return 0 (no retrieval!)
    scores_k0 = {
        0: {"em": 1.0, "f1": 1.0},
        3: {"em": 1.0, "f1": 1.0},
        8: {"em": 1.0, "f1": 1.0},
        15: {"em": 1.0, "f1": 1.0},
    }
    assert assign_cheapest_k(scores_k0, candidate_k, f1_threshold=0.5) == 0

    # k=0 fails (f1=0.2), k=3 passes (f1=0.6) -> should return 3
    scores_k3 = {
        0: {"em": 0.0, "f1": 0.2},
        3: {"em": 0.0, "f1": 0.6},
        8: {"em": 1.0, "f1": 1.0},
        15: {"em": 1.0, "f1": 1.0},
    }
    assert assign_cheapest_k(scores_k3, candidate_k, f1_threshold=0.5) == 3

    # k=0, 3 fail, k=8 passes -> should return 8
    scores_k8 = {
        0: {"em": 0.0, "f1": 0.0},
        3: {"em": 0.0, "f1": 0.3},
        8: {"em": 0.0, "f1": 0.75},
        15: {"em": 0.0, "f1": 0.8},
    }
    assert assign_cheapest_k(scores_k8, candidate_k, f1_threshold=0.5) == 8

    # None crosses 0.5 threshold -> choose k with highest f1
    scores_low = {
        0: {"em": 0.0, "f1": 0.1},
        3: {"em": 0.0, "f1": 0.2},
        8: {"em": 0.0, "f1": 0.35},
        15: {"em": 0.0, "f1": 0.45},
    }
    assert assign_cheapest_k(scores_low, candidate_k, f1_threshold=0.5) == 15


def test_labeled_dataset_structure():
    summary_path = Path("results/label_distribution.json")
    assert summary_path.exists(), "results/label_distribution.json must exist"

    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)

    assert "val" in summary
    assert "train" in summary
    assert summary["val"]["total"] > 0
    assert summary["train"]["total"] > 0

    val_labeled_path = Path("data/processed/val_labeled.jsonl")
    assert val_labeled_path.exists()
    with open(val_labeled_path, "r", encoding="utf-8") as f:
        first_line = json.loads(f.readline())
        assert "optimal_k" in first_line
        assert first_line["optimal_k"] in [0, 3, 8, 15]
        assert "k_evaluations" in first_line
        opt_k_str = str(first_line["optimal_k"])
        assert opt_k_str in first_line["k_evaluations"] or first_line["optimal_k"] in first_line["k_evaluations"]
        eval_entry = first_line["k_evaluations"].get(opt_k_str) or first_line["k_evaluations"].get(first_line["optimal_k"])
        assert "em" in eval_entry
        assert "f1" in eval_entry
        assert "tokens" in eval_entry
