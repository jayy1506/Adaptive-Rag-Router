import json
import logging
from pathlib import Path
from typing import Any

import yaml

from src.data.schema import UnifiedQueryRecord
from src.evaluation.metrics import evaluate_prediction
from src.generation.generator import RAGGenerator
from src.retrieval.indexer import ChromaRetriever

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_config(config_path: str = "config.yaml") -> dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def assign_cheapest_k(
    k_scores: dict[int, dict[str, float]],
    candidate_k_values: list[int],
    f1_threshold: float = 0.5,
) -> int:
    """Assigns the minimal (cheapest) k that achieves correctness (F1 >= threshold or EM == 1.0).

    If none satisfies the threshold, chooses the k with the maximum F1 score (ties broken by higher k).
    """
    # ponytail: monotonic cost check - earliest k meeting threshold is optimal
    for k in candidate_k_values:
        scores = k_scores.get(k, {"em": 0.0, "f1": 0.0})
        if scores["em"] >= 1.0 or scores["f1"] >= f1_threshold:
            return k

    # If none crossed threshold, pick k with highest F1
    best_k = candidate_k_values[-1]
    best_f1 = -1.0
    for k in candidate_k_values:
        f1 = k_scores.get(k, {}).get("f1", 0.0)
        if f1 >= best_f1:
            best_f1 = f1
            best_k = k
    return best_k


def label_dataset(
    input_path: str,
    output_path: str,
    candidate_k: list[int],
    retriever: ChromaRetriever,
    generator: RAGGenerator,
    f1_threshold: float = 0.5,
    max_records: int | None = None,
) -> dict[str, Any]:
    in_file = Path(input_path)
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    # Check for existing labeled records to resume
    existing_records: dict[str, dict[str, Any]] = {}
    if out_file.exists():
        with open(out_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    existing_records[item["query_id"]] = item
        logger.info(f"Resuming {output_path}: found {len(existing_records)} already labeled records.")

    queries: list[UnifiedQueryRecord] = []
    with open(in_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                queries.append(UnifiedQueryRecord.model_validate_json(line))

    if max_records:
        queries = queries[:max_records]

    total_queries = len(queries)
    logger.info(f"Labeling {total_queries} queries from {input_path} across candidate k={candidate_k}...")

    labeled_items: list[dict[str, Any]] = list(existing_records.values())
    label_counts: dict[int, int] = {k: 0 for k in candidate_k}

    with open(out_file, "a" if existing_records else "w", encoding="utf-8") as out_f:
        for idx, q in enumerate(queries):
            if q.query_id in existing_records:
                opt_k = existing_records[q.query_id]["optimal_k"]
                label_counts[opt_k] = label_counts.get(opt_k, 0) + 1
                continue

            k_evaluations: dict[int, dict[str, Any]] = {}
            k_scores_summary: dict[int, dict[str, float]] = {}

            # Retrieve max candidate context once to optimize vector DB queries
            max_k = max(candidate_k)
            all_retrieved = retriever.retrieve(q.query, k=max_k) if max_k > 0 else []

            # Evaluate candidate k in ascending order; break as soon as threshold is met (cheapest k)
            optimal_k = None
            for k in candidate_k:
                passages_k = all_retrieved[:k] if k > 0 else []
                gen_res = generator.generate(q.query, k=k, passages=passages_k)
                scores = evaluate_prediction(gen_res["answer"], q.answers)

                k_evaluations[k] = {
                    "answer": gen_res["answer"],
                    "em": scores["em"],
                    "f1": scores["f1"],
                    "tokens": gen_res["tokens_in"] + gen_res["tokens_out"],
                    "latency_ms": gen_res["latency_ms"],
                }
                k_scores_summary[k] = scores

                # ponytail: early stop once cheapest sufficient k achieves target F1
                if scores["em"] >= 1.0 or scores["f1"] >= f1_threshold:
                    optimal_k = k
                    break

            if optimal_k is None:
                optimal_k = assign_cheapest_k(k_scores_summary, candidate_k, f1_threshold=f1_threshold)
            label_counts[optimal_k] = label_counts.get(optimal_k, 0) + 1

            record = {
                "query_id": q.query_id,
                "query": q.query,
                "answers": q.answers,
                "dataset": q.dataset,
                "optimal_k": optimal_k,
                "k_evaluations": k_evaluations,
            }
            out_f.write(json.dumps(record) + "\n")
            out_f.flush()
            labeled_items.append(record)

            if (idx + 1) % 50 == 0 or (idx + 1) == total_queries:
                logger.info(
                    f"Processed {idx + 1}/{total_queries} queries. "
                    f"Current label distribution: {label_counts}"
                )

    return {
        "total": len(labeled_items),
        "distribution": label_counts,
        "output_path": str(out_file),
    }


def run_labeling():
    cfg = load_config()
    candidate_k = cfg.get("router", {}).get("candidate_k_values", [0, 3, 8, 15])
    retriever = ChromaRetriever()
    generator = RAGGenerator()

    # Label validation split
    logger.info("--- Labeling Validation Split ---")
    val_stats = label_dataset(
        input_path="data/processed/val.jsonl",
        output_path="data/processed/val_labeled.jsonl",
        candidate_k=candidate_k,
        retriever=retriever,
        generator=generator,
    )

    # Label training split
    logger.info("--- Labeling Training Split ---")
    train_stats = label_dataset(
        input_path="data/processed/train.jsonl",
        output_path="data/processed/train_labeled.jsonl",
        candidate_k=candidate_k,
        retriever=retriever,
        generator=generator,
    )

    summary = {
        "candidate_k": candidate_k,
        "val": val_stats,
        "train": train_stats,
    }

    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    summary_path = results_dir / "label_distribution.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Labeling finished! Summary saved to {summary_path}")
    return summary


if __name__ == "__main__":
    run_labeling()
