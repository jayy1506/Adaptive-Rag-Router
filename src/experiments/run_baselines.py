import csv
import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
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


def load_queries(path: str = "data/processed/val.jsonl") -> list[UnifiedQueryRecord]:
    records: list[UnifiedQueryRecord] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(UnifiedQueryRecord.model_validate_json(line))
    return records


def run_baselines(
    val_path: str = "data/processed/val.jsonl",
    baseline_k_values: list[int] | None = None,
    results_dir: str = "results",
) -> dict[str, Any]:
    cfg = load_config()
    if baseline_k_values is None:
        baseline_k_values = cfg.get("router", {}).get("baseline_k_values", [0, 3, 5, 10])

    out_dir = Path(results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    queries = load_queries(val_path)
    logger.info(f"Loaded {len(queries)} evaluation queries from {val_path}")

    retriever = ChromaRetriever()
    generator = RAGGenerator()

    results: dict[str, Any] = {}
    summary_rows: list[dict[str, Any]] = []

    for k in baseline_k_values:
        logger.info(f"--- Running Baseline k={k} on {len(queries)} queries ---")
        em_list: list[float] = []
        f1_list: list[float] = []
        token_costs: list[int] = []
        retrieval_latencies: list[float] = []
        gen_latencies: list[float] = []
        total_latencies: list[float] = []
        detailed_records: list[dict[str, Any]] = []

        for q in queries:
            t0 = time.perf_counter()
            # Retrieval
            if k > 0:
                passages = retriever.retrieve(q.query, k=k)
            else:
                passages = []
            t_retrieval = (time.perf_counter() - t0) * 1000.0

            # Generation
            gen_res = generator.generate(q.query, k=k, passages=passages)
            t_total = (time.perf_counter() - t0) * 1000.0

            scores = evaluate_prediction(gen_res["answer"], q.answers)
            em_list.append(scores["em"])
            f1_list.append(scores["f1"])
            tokens = gen_res["tokens_in"] + gen_res["tokens_out"]
            token_costs.append(tokens)

            retrieval_latencies.append(t_retrieval)
            gen_latencies.append(gen_res["latency_ms"])
            total_latencies.append(t_total)

            detailed_records.append(
                {
                    "query_id": q.query_id,
                    "dataset": q.dataset,
                    "k": k,
                    "em": scores["em"],
                    "f1": scores["f1"],
                    "tokens": tokens,
                    "latency_retrieval_ms": round(t_retrieval, 2),
                    "latency_gen_ms": gen_res["latency_ms"],
                    "latency_total_ms": round(t_total, 2),
                }
            )

        # Aggregate metrics
        mean_em = float(np.mean(em_list))
        mean_f1 = float(np.mean(f1_list))
        mean_tokens = float(np.mean(token_costs))
        p50_latency = float(np.percentile(total_latencies, 50))
        p95_latency = float(np.percentile(total_latencies, 95))
        p50_retrieval = float(np.percentile(retrieval_latencies, 50))
        p95_retrieval = float(np.percentile(retrieval_latencies, 95))

        k_summary = {
            "k": k,
            "queries_count": len(queries),
            "em": round(mean_em, 4),
            "f1": round(mean_f1, 4),
            "avg_tokens": round(mean_tokens, 1),
            "p50_latency_ms": round(p50_latency, 2),
            "p95_latency_ms": round(p95_latency, 2),
            "p50_retrieval_ms": round(p50_retrieval, 2),
            "p95_retrieval_ms": round(p95_retrieval, 2),
        }
        results[f"k_{k}"] = {
            "summary": k_summary,
            "details": detailed_records,
        }
        summary_rows.append(k_summary)

        logger.info(
            f"k={k} -> EM: {mean_em:.4f} | F1: {mean_f1:.4f} | Avg Tokens: {mean_tokens:.1f} | p50: {p50_latency:.1f}ms | p95: {p95_latency:.1f}ms"
        )

    # Save outputs
    json_path = out_dir / "baselines.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    csv_path = out_dir / "baselines.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    logger.info(f"Saved baseline results to {json_path} and {csv_path}")
    return results


if __name__ == "__main__":
    run_baselines()
