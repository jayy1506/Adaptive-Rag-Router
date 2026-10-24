import json
import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def generate_pareto_plots(
    benchmark_json_path: str = "results/final_benchmark.json",
    cost_plot_path: str = "results/pareto_accuracy_vs_cost.png",
    latency_plot_path: str = "results/pareto_accuracy_vs_latency.png",
) -> dict[str, str]:
    with open(benchmark_json_path, "r", encoding="utf-8") as f:
        data: list[dict[str, Any]] = json.load(f)

    labels = {
        "baseline_k0": "Fixed k=0 (No Retrieval)",
        "baseline_k3": "Fixed k=3",
        "baseline_k5": "Fixed k=5",
        "baseline_k10": "Fixed k=10",
        "adaptive_router": "Adaptive Router (Ours)",
    }

    colors = {
        "baseline_k0": "#64748b",      # Slate grey
        "baseline_k3": "#0284c7",      # Sky blue
        "baseline_k5": "#0d9488",      # Teal
        "baseline_k10": "#d97706",     # Amber
        "adaptive_router": "#dc2626",  # Vibrant Red/Crimson
    }

    # Style configuration
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.size"] = 10

    # 1. Plot Accuracy vs Cost (Avg Tokens)
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)

    # Sort baselines by tokens for curve
    baselines = [d for d in data if d["system"] != "adaptive_router"]
    baselines.sort(key=lambda x: x["avg_total_tokens"])

    b_tokens = [d["avg_total_tokens"] for d in baselines]
    b_f1 = [d["token_f1"] for d in baselines]
    ax.plot(b_tokens, b_f1, linestyle="--", color="#94a3b8", alpha=0.8, label="Fixed Retrieval Frontier")

    for d in data:
        sys = d["system"]
        x = d["avg_total_tokens"]
        y = d["token_f1"]
        is_router = (sys == "adaptive_router")
        size = 180 if is_router else 110
        marker = "*" if is_router else "o"
        ax.scatter(x, y, s=size, color=colors.get(sys, "#334155"), marker=marker, zorder=5, label=labels.get(sys, sys))

        offset_y = 0.015 if is_router else -0.02
        offset_x = 0
        ax.annotate(
            f"{labels.get(sys, sys)}\n({x:.0f} tokens, F1={y:.3f})",
            (x, y),
            textcoords="offset points",
            xytext=(offset_x, offset_y * 100),
            ha="center",
            fontsize=8.5,
            fontweight="bold" if is_router else "normal",
            bbox={"boxstyle": "round,pad=0.25", "fc": "#ffffff", "ec": colors.get(sys, "#cbd5e1"), "alpha": 0.9},
        )

    ax.set_title("Pareto Frontier: Accuracy (F1) vs. Inference Token Cost", fontsize=13, fontweight="bold", pad=14)
    ax.set_xlabel("Average Total Tokens per Query (Prompt + Generation Cost)", fontsize=11, labelpad=8)
    ax.set_ylabel("Token F1 Score", fontsize=11, labelpad=8)
    ax.legend(loc="lower right", framealpha=0.9)
    ax.grid(True, linestyle=":", alpha=0.6)

    Path(cost_plot_path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(cost_plot_path)
    plt.close(fig)
    logger.info(f"Saved Pareto Accuracy vs Cost plot to {cost_plot_path}")

    # 2. Plot Accuracy vs p50 Latency
    fig2, ax2 = plt.subplots(figsize=(8, 6), dpi=300)

    baselines_lat = sorted(baselines, key=lambda x: x["p50_total_latency_ms"])
    b_lat = [d["p50_total_latency_ms"] for d in baselines_lat]
    b_lat_f1 = [d["token_f1"] for d in baselines_lat]
    ax2.plot(b_lat, b_lat_f1, linestyle="--", color="#94a3b8", alpha=0.8, label="Fixed Baseline Frontier")

    for d in data:
        sys = d["system"]
        x = d["p50_total_latency_ms"]
        y = d["token_f1"]
        is_router = (sys == "adaptive_router")
        size = 180 if is_router else 110
        marker = "*" if is_router else "o"
        ax2.scatter(x, y, s=size, color=colors.get(sys, "#334155"), marker=marker, zorder=5, label=labels.get(sys, sys))

        offset_y = 0.015 if is_router else -0.02
        ax2.annotate(
            f"{labels.get(sys, sys)}\n({x:.1f}ms, F1={y:.3f})",
            (x, y),
            textcoords="offset points",
            xytext=(0, offset_y * 100),
            ha="center",
            fontsize=8.5,
            fontweight="bold" if is_router else "normal",
            bbox={"boxstyle": "round,pad=0.25", "fc": "#ffffff", "ec": colors.get(sys, "#cbd5e1"), "alpha": 0.9},
        )

    ax2.set_title("Pareto Frontier: Accuracy (F1) vs. Total Latency (p50)", fontsize=13, fontweight="bold", pad=14)
    ax2.set_xlabel("p50 Total Query Latency (ms)", fontsize=11, labelpad=8)
    ax2.set_ylabel("Token F1 Score", fontsize=11, labelpad=8)
    ax2.legend(loc="lower right", framealpha=0.9)
    ax2.grid(True, linestyle=":", alpha=0.6)

    fig2.tight_layout()
    fig2.savefig(latency_plot_path)
    plt.close(fig2)
    logger.info(f"Saved Pareto Accuracy vs Latency plot to {latency_plot_path}")

    return {"cost_plot": cost_plot_path, "latency_plot": latency_plot_path}


if __name__ == "__main__":
    generate_pareto_plots()
