import csv
import json
import logging
import time
from pathlib import Path
from typing import Any

import yaml
from sklearn.metrics import accuracy_score, f1_score

from src.router.features import QueryFeatureExtractor
from src.router.models import (
    LogisticRegressionRouter,
    RouterInferenceWrapper,
    SmallEncoderMLPRouter,
    XGBoostRouter,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_config(config_path: str = "config.yaml") -> dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_labeled_data(path: str) -> tuple[list[str], list[int]]:
    queries: list[str] = []
    labels: list[int] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                queries.append(item["query"])
                labels.append(int(item["optimal_k"]))
    return queries, labels


def run_training_ablation(
    train_path: str = "data/processed/train_labeled.jsonl",
    val_path: str = "data/processed/val_labeled.jsonl",
    results_dir: str = "results",
) -> dict[str, Any]:
    cfg = load_config()
    candidate_k = cfg.get("router", {}).get("candidate_k_values", [0, 3, 8, 15])
    model_save_path = cfg.get("router", {}).get("model_save_path", "results/router_model.joblib")

    out_dir = Path(results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading training data from {train_path}...")
    train_queries, y_train = load_labeled_data(train_path)
    logger.info(f"Loading validation data from {val_path}...")
    val_queries, y_val = load_labeled_data(val_path)

    logger.info(f"Loaded {len(train_queries)} train records and {len(val_queries)} val records.")

    # Feature Extractor
    extractor = QueryFeatureExtractor()

    logger.info("Extracting linguistic, embedding, and hybrid feature matrices...")
    feature_cache = {
        "train": {
            "linguistic": extractor.extract_batch(train_queries, feature_type="linguistic"),
            "embedding": extractor.extract_batch(train_queries, feature_type="embedding"),
            "hybrid": extractor.extract_batch(train_queries, feature_type="hybrid"),
        },
        "val": {
            "linguistic": extractor.extract_batch(val_queries, feature_type="linguistic"),
            "embedding": extractor.extract_batch(val_queries, feature_type="embedding"),
            "hybrid": extractor.extract_batch(val_queries, feature_type="hybrid"),
        },
    }

    ablation_experiments = [
        {
            "id": "logreg_linguistic",
            "model_cls": LogisticRegressionRouter,
            "feature_type": "linguistic",
            "kwargs": {"c": 1.0},
        },
        {
            "id": "logreg_embedding",
            "model_cls": LogisticRegressionRouter,
            "feature_type": "embedding",
            "kwargs": {"c": 1.0},
        },
        {
            "id": "xgboost_linguistic",
            "model_cls": XGBoostRouter,
            "feature_type": "linguistic",
            "kwargs": {"n_estimators": 60, "max_depth": 3},
        },
        {
            "id": "xgboost_hybrid",
            "model_cls": XGBoostRouter,
            "feature_type": "hybrid",
            "kwargs": {"n_estimators": 80, "max_depth": 4},
        },
        {
            "id": "small_encoder_mlp_embedding",
            "model_cls": SmallEncoderMLPRouter,
            "feature_type": "embedding",
            "kwargs": {"hidden_dim": 64},
        },
        {
            "id": "small_encoder_mlp_hybrid",
            "model_cls": SmallEncoderMLPRouter,
            "feature_type": "hybrid",
            "kwargs": {"hidden_dim": 64},
        },
    ]

    ablation_results: list[dict[str, Any]] = []
    best_model_obj = None
    best_f1 = -1.0
    best_config = None

    for exp in ablation_experiments:
        exp_id = exp["id"]
        feat_type = exp["feature_type"]
        logger.info(f"--- Training Experiment: {exp_id} ({feat_type}) ---")

        X_tr = feature_cache["train"][feat_type]
        X_v = feature_cache["val"][feat_type]

        model = exp["model_cls"](candidate_k=candidate_k, **exp["kwargs"])

        t0_train = time.perf_counter()
        model.fit(X_tr, y_train)
        train_time_ms = (time.perf_counter() - t0_train) * 1000.0

        # Inference timing and evaluation
        t0_inf = time.perf_counter()
        preds = model.predict(X_v)
        total_inf_ms = (time.perf_counter() - t0_inf) * 1000.0
        avg_inf_latency_ms = total_inf_ms / len(val_queries)

        acc = float(accuracy_score(y_val, preds))
        macro_f1 = float(f1_score(y_val, preds, average="macro", zero_division=0))
        weighted_f1 = float(f1_score(y_val, preds, average="weighted", zero_division=0))

        row = {
            "experiment_id": exp_id,
            "model_name": model.name,
            "feature_type": feat_type,
            "accuracy": round(acc, 4),
            "macro_f1": round(macro_f1, 4),
            "weighted_f1": round(weighted_f1, 4),
            "latency_ms_per_query": round(avg_inf_latency_ms, 3),
            "train_time_ms": round(train_time_ms, 1),
            "feature_dim": X_tr.shape[1],
        }
        ablation_results.append(row)
        logger.info(
            f"Result: Acc={acc:.4f} | Macro F1={macro_f1:.4f} | Weighted F1={weighted_f1:.4f} | Latency={avg_inf_latency_ms:.3f}ms"
        )

        if weighted_f1 > best_f1:
            best_f1 = weighted_f1
            best_model_obj = model
            best_config = exp

    logger.info(f"Top model selected: {best_config['id']} with Weighted F1={best_f1:.4f}")

    # Wrap and save best router model
    inference_wrapper = RouterInferenceWrapper(
        model=best_model_obj,
        feature_type=best_config["feature_type"],
        candidate_k=candidate_k,
    )
    inference_wrapper.save(model_save_path)

    # Save ablation outputs
    json_path = out_dir / "router_ablation.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(ablation_results, f, indent=2)

    csv_path = out_dir / "router_ablation.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(ablation_results[0].keys()))
        writer.writeheader()
        writer.writerows(ablation_results)

    logger.info(f"Saved ablation results to {json_path} and {csv_path}")
    return {"best_experiment": best_config["id"], "ablation": ablation_results}


if __name__ == "__main__":
    run_training_ablation()
