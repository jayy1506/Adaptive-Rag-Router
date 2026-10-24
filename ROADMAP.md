# Detailed Execution Roadmap: Adaptive-Retrieval RAG Router

## Phase 1: Data Pipeline, Unified Schema & Leakage Validation
- **Deliverables**:
  - `src/data/download.py`: Hugging Face dataset sampler (1k HotpotQA distractor, 1k Natural Questions).
  - `src/data/schema.py`: Pydantic / dataclass schema unification.
  - `src/data/split.py`: Deterministic 70/15/15 split generation.
  - `tests/test_data_leakage.py`: Tests for train/val/test query isolation, passage overlap, and format validity.
- **Verification**: `pytest tests/test_data_leakage.py` (Ralph loop with `<promise>PHASE_1_DONE</promise>`).
- **CodeRabbit Review**: Verify diff, address findings, clean commit.

## Phase 2: Indexing & Baseline Evaluation (k=0, 3, 5, 10)
- **Deliverables**:
  - `src/retrieval/indexer.py`: ChromaDB indexing with `sentence-transformers` (`all-MiniLM-L6-v2`).
  - `src/generation/generator.py`: LLM generation interface with prompt templates.
  - `src/evaluation/metrics.py`: Normalized EM & Token F1 implementation.
  - `src/experiments/run_baselines.py`: Execute fixed retrieval baselines ($k \in \{0, 3, 5, 10\}$) on val split.
  - `tests/test_baselines.py`: Functional checks for retriever and metrics.
- **Verification**: Baseline results logged in `results/baselines.json`.
- **CodeRabbit Review**: Clean diff review.

## Phase 3: Router Ground Truth Label Generation
- **Deliverables**:
  - `src/router/cache.py`: Deterministic query+k generation cache (avoiding duplicate LLM calls).
  - `src/router/labeler.py`: Run queries at $k \in \{0, 3, 8, 15\}$, compute EM/F1, label with minimum $k$ achieving correct response (or max if none).
  - `data/labeled/`: Saved ground truth labeled dataset with distribution summaries.
  - `tests/test_labeler.py`: Tests for monotonic cost labeling and caching logic.
- **Verification**: Validate label distributions and cache efficiency.
- **CodeRabbit Review**: Clean diff review.

## Phase 4: Router Training & Ablation
- **Deliverables**:
  - `src/router/features.py`: Surface & statistical feature extractor (length, question type, ambiguity, lexical density).
  - `src/router/models.py`:
    - Model A: Tabular Classifier (Logistic Regression / XGBoost) on handcrafted features.
    - Model B: Embedding-based Classifier (Logistic Regression / MLP on sentence embeddings).
    - Model C: Hybrid (Features + Embeddings).
  - `src/router/train.py`: Training pipeline with cross-validation on train set and evaluation on val set.
  - `results/router_ablation.json`: Ablation comparing Accuracy, Macro F1, and Routing Latency.
  - `tests/test_router.py`: Unit tests for router feature extraction and inference latency.
- **Verification**: Train/Val performance metrics and model artifacts saved.
- **CodeRabbit Review**: Clean diff review.

## Phase 5: Adaptive Pipeline with Confidence Fallback & Escalation
- **Deliverables**:
  - `src/pipeline/adaptive.py`: End-to-end adaptive orchestrator:
    - Step 1: Predict optimal $k$ and prediction confidence probability.
    - Step 2: Confidence fallback: If $\max(P) < \tau$, step up to higher $k$.
    - Step 3: Retrieval & Generation.
    - Step 4: Sufficiency heuristic / Escalation: If generated answer contains uncertainty indicators or low token confidence, trigger escalation to $k_{\text{escalate}}$.
  - `tests/test_adaptive_pipeline.py`: Comprehensive test coverage of fallback and escalation paths.
- **Verification**: Pass Ralph loop `<promise>PHASE_5_DONE</promise>`.
- **CodeRabbit Review**: Clean diff review.

## Phase 6: Comprehensive Benchmark & Evaluation
- **Deliverables**:
  - `src/evaluation/evaluate_all.py`: Benchmark All Baselines vs Adaptive Router on isolated Test split (15%).
  - Metrics collected:
    - Exact Match (EM) & Token F1.
    - Average tokens per query (prompt + completion cost).
    - p50 and p95 latency broken down into: Router Latency, Retrieval Latency, Generation Latency.
    - Under-retrieval failure rate (cases where routing $k$ failed, but higher $k$ would succeed).
  - `src/evaluation/plots.py`: Generate publication-grade Pareto plots (`results/pareto_accuracy_vs_cost.png`, `results/pareto_accuracy_vs_latency.png`).
  - `results/final_benchmark.json` and `results/final_benchmark.csv`.
- **Verification**: Verified benchmark results and generated charts.
- **CodeRabbit Review**: Clean diff review.

## Phase 7: FastAPI Endpoint, Live Dashboard & Final Documentation
- **Deliverables**:
  - `src/api/server.py`: FastAPI server exposing `/predict`, `/route`, and `/query` endpoints with detailed timing headers.
  - `src/dashboard/`: Interactive modern UI displaying real-time routing decisions, confidence gauge, retrieved context chunks, and latency waterfall.
  - `README.md`: Architectural documentation, replication commands, benchmark summary tables, and embedded Pareto charts.
  - `tests/test_api.py`: Endpoint regression tests.
- **Verification**: End-to-end API test and dashboard validation.
- **CodeRabbit Review**: Final review on entire repo.
