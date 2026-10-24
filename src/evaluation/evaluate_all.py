import csv
import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np

from src.evaluation.metrics import evaluate_prediction
from src.generation.generator import RAGGenerator
from src.pipeline.adaptive import AdaptiveRetrievalPipeline
from src.retrieval.indexer import ChromaRetriever

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def run_benchmark(
    test_path: str = "data/processed/test.jsonl",
    output_json: str = "results/final_benchmark.json",
    output_csv: str = "results/final_benchmark.csv",
    sample_limit: int | None = None,
) -> dict[str, Any]:
    test_file = Path(test_path)
    if not test_file.exists():
        raise FileNotFoundError(f"Test split not found at {test_path}")

    records: list[dict[str, Any]] = []
    with open(test_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    if sample_limit:
        records = records[:sample_limit]

    logger.info(f"Running comprehensive benchmark across {len(records)} test queries...")

    retriever = ChromaRetriever()
    generator = RAGGenerator()
    pipeline = AdaptiveRetrievalPipeline(retriever=retriever, generator=generator)

    systems = ["baseline_k0", "baseline_k3", "baseline_k5", "baseline_k10", "adaptive_router"]
    # We also evaluate k=15 as an oracle benchmark for under-retrieval failure detection
    candidate_systems = systems + ["oracle_k15"]

    per_system_results: dict[str, list[dict[str, Any]]] = {s: [] for s in candidate_systems}

    for idx, rec in enumerate(records):
        query = rec["query"]
        answers = rec["answers"]

        if (idx + 1) % 50 == 0 or idx == 0:
            logger.info(f"Evaluating query {idx + 1}/{len(records)}: '{query[:50]}...'")

        # 1. Evaluate fixed baselines and oracle k=15
        for k_val, sys_name in [(0, "baseline_k0"), (3, "baseline_k3"), (5, "baseline_k5"), (10, "baseline_k10"), (15, "oracle_k15")]:
            t0_ret = time.perf_counter()
            passages = retriever.retrieve(query, k=k_val) if k_val > 0 else []
            ret_latency_ms = (time.perf_counter() - t0_ret) * 1000.0

            t0_gen = time.perf_counter()
            gen_res = generator.generate(query=query, passages=passages, k=k_val, use_cache=True)
            gen_latency_ms = (time.perf_counter() - t0_gen) * 1000.0

            ans = gen_res["answer"]
            m_res = evaluate_prediction(ans, answers)
            em, f1 = m_res["em"], m_res["f1"]

            per_system_results[sys_name].append({
                "query_id": rec["query_id"],
                "em": em,
                "f1": f1,
                "tokens_in": gen_res["tokens_in"],
                "tokens_out": gen_res["tokens_out"],
                "total_tokens": gen_res["tokens_in"] + gen_res["tokens_out"],
                "router_latency_ms": 0.0,
                "retrieval_latency_ms": ret_latency_ms,
                "generation_latency_ms": gen_latency_ms,
                "total_latency_ms": ret_latency_ms + gen_latency_ms,
                "executed_k": k_val,
            })

        # 2. Evaluate Adaptive Router
        adapt_res = pipeline.run(query, use_cache=True)
        adapt_m = evaluate_prediction(adapt_res.answer, answers)
        adapt_em, adapt_f1 = adapt_m["em"], adapt_m["f1"]

        per_system_results["adaptive_router"].append({
            "query_id": rec["query_id"],
            "em": adapt_em,
            "f1": adapt_f1,
            "tokens_in": adapt_res.prompt_tokens,
            "tokens_out": adapt_res.completion_tokens,
            "total_tokens": adapt_res.total_tokens,
            "router_latency_ms": adapt_res.router_latency_ms,
            "retrieval_latency_ms": adapt_res.retrieval_latency_ms + adapt_res.escalation_retrieval_latency_ms,
            "generation_latency_ms": adapt_res.generation_latency_ms + adapt_res.escalation_generation_latency_ms,
            "total_latency_ms": adapt_res.total_latency_ms,
            "executed_k": adapt_res.executed_k,
            "fallback_triggered": adapt_res.fallback_triggered,
            "escalation_triggered": adapt_res.escalation_triggered,
        })

    # 3. Aggregate benchmark metrics
    summary_rows: list[dict[str, Any]] = []
    oracle_f1s = [r["f1"] for r in per_system_results["oracle_k15"]]

    for sys_name in systems:
        items = per_system_results[sys_name]
        n = len(items)
        em_avg = float(np.mean([r["em"] for r in items]))
        f1_avg = float(np.mean([r["f1"] for r in items]))
        avg_tokens_in = float(np.mean([r["tokens_in"] for r in items]))
        avg_tokens_out = float(np.mean([r["tokens_out"] for r in items]))
        avg_total_tokens = float(np.mean([r["total_tokens"] for r in items]))

        total_lats = [r["total_latency_ms"] for r in items]
        rout_lats = [r["router_latency_ms"] for r in items]
        ret_lats = [r["retrieval_latency_ms"] for r in items]
        gen_lats = [r["generation_latency_ms"] for r in items]

        # Under-retrieval failure rate: cases where sys f1 < 0.5, but oracle k=15 f1 >= 0.5
        under_retrieval_failures = sum(
            1 for i in range(n) if items[i]["f1"] < 0.5 and oracle_f1s[i] >= 0.5
        )
        under_retrieval_rate = under_retrieval_failures / n if n > 0 else 0.0

        avg_executed_k = float(np.mean([r["executed_k"] for r in items]))

        summary_rows.append({
            "system": sys_name,
            "exact_match": round(em_avg, 4),
            "token_f1": round(f1_avg, 4),
            "avg_tokens_in": round(avg_tokens_in, 1),
            "avg_tokens_out": round(avg_tokens_out, 1),
            "avg_total_tokens": round(avg_total_tokens, 1),
            "avg_executed_k": round(avg_executed_k, 2),
            "p50_total_latency_ms": round(float(np.percentile(total_lats, 50)), 2),
            "p95_total_latency_ms": round(float(np.percentile(total_lats, 95)), 2),
            "p50_router_latency_ms": round(float(np.percentile(rout_lats, 50)), 2),
            "p95_router_latency_ms": round(float(np.percentile(rout_lats, 95)), 2),
            "p50_retrieval_latency_ms": round(float(np.percentile(ret_lats, 50)), 2),
            "p95_retrieval_latency_ms": round(float(np.percentile(ret_lats, 95)), 2),
            "p50_generation_latency_ms": round(float(np.percentile(gen_lats, 50)), 2),
            "p95_generation_latency_ms": round(float(np.percentile(gen_lats, 95)), 2),
            "under_retrieval_rate": round(under_retrieval_rate, 4),
        })

    # Save results to JSON and CSV
    Path(output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(summary_rows, f, indent=2)
    logger.info(f"Saved benchmark summary to {output_json}")

    with open(output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    logger.info(f"Saved benchmark summary to {output_csv}")

    return {"summary": summary_rows, "num_queries": len(records)}


if __name__ == "__main__":
    run_benchmark()
