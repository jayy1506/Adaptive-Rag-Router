import csv
import json
from pathlib import Path


def test_final_benchmark_files_exist():
    json_path = Path("results/final_benchmark.json")
    csv_path = Path("results/final_benchmark.csv")
    assert json_path.exists(), "results/final_benchmark.json must exist"
    assert csv_path.exists(), "results/final_benchmark.csv must exist"


def test_benchmark_metrics_validity():
    json_path = Path("results/final_benchmark.json")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    expected_systems = {"baseline_k0", "baseline_k3", "baseline_k5", "baseline_k10", "adaptive_router"}
    found_systems = {d["system"] for d in data}
    assert expected_systems.issubset(found_systems)

    for row in data:
        assert 0.0 <= row["exact_match"] <= 1.0
        assert 0.0 <= row["token_f1"] <= 1.0
        assert row["avg_total_tokens"] >= 0.0
        assert row["p50_total_latency_ms"] >= 0.0
        assert row["p95_total_latency_ms"] >= row["p50_total_latency_ms"]
        assert 0.0 <= row["under_retrieval_rate"] <= 1.0

        # Latency breakdown checks
        assert row["p50_router_latency_ms"] >= 0.0
        assert row["p50_retrieval_latency_ms"] >= 0.0
        assert row["p50_generation_latency_ms"] >= 0.0


def test_pareto_plots_exist():
    cost_plot = Path("results/pareto_accuracy_vs_cost.png")
    lat_plot = Path("results/pareto_accuracy_vs_latency.png")

    assert cost_plot.exists(), "Cost Pareto plot missing"
    assert lat_plot.exists(), "Latency Pareto plot missing"
    assert cost_plot.stat().st_size > 1000, "Cost plot is empty or corrupt"
    assert lat_plot.stat().st_size > 1000, "Latency plot is empty or corrupt"


def test_csv_matches_json():
    json_path = Path("results/final_benchmark.json")
    csv_path = Path("results/final_benchmark.csv")

    with open(json_path, "r", encoding="utf-8") as f:
        json_data = json.load(f)

    with open(csv_path, "r", encoding="utf-8") as f:
        csv_rows = list(csv.DictReader(f))

    assert len(json_data) == len(csv_rows)
    for j_row, c_row in zip(json_data, csv_rows):
        assert j_row["system"] == c_row["system"]
        assert round(j_row["token_f1"], 3) == round(float(c_row["token_f1"]), 3)
