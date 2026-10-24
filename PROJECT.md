# Adaptive-Retrieval RAG Router

## 1. Project Overview & Problem Statement
Standard Retrieval-Augmented Generation (RAG) pipelines retrieve a fixed number of passages ($k$) for every user query regardless of query complexity. This leads to:
1. **Unnecessary Latency & Token Cost**: Simple/factual queries that can be answered from parametric memory or with $k=0$ still incur retrieval overhead and bloated context windows.
2. **Context Dilution / Distraction**: Irrelevant chunks can confuse the generator.
3. **Under-Retrieval**: Complex multi-hop queries receive too few chunks (e.g., $k=3$), failing to answer or producing incomplete answers.

The **Adaptive-Retrieval RAG Router** implements a lightweight classifier that dynamically predicts the optimal retrieval decision per query:
- Skip retrieval ($k=0$) for parametric/simple queries.
- Fetch low context ($k=3$), medium context ($k=8$), or high context ($k=15$) for more complex queries.
- Incorporate confidence-based fallback and escalation triggers if the generated answer is incomplete.
- Benchmark cost, latency, accuracy, and under-retrieval failure rates against fixed baselines ($k=0, 3, 5, 10$).

---

## 2. Technical Stack & Standards
- **Language & Runtime**: Python 3.11+
- **Vector Database**: ChromaDB (`chromadb`)
- **Embeddings**: `sentence-transformers` (`all-MiniLM-L6-v2` or `BAAI/bge-small-en-v1.5`)
- **Classifier Models**: `scikit-learn` (Logistic Regression, Random Forest), `xgboost`, and small encoder / MLP
- **Datasets**: Hugging Face `datasets` (HotpotQA distractor + Natural Questions)
- **API & Serving**: FastAPI, Uvicorn, Pydantic
- **Evaluation & Testing**: `pytest`, custom EM/F1 metrics, LLM-as-a-judge harness
- **Configuration**: Unified `config.yaml` with deterministic random seeds
- **Artifacts**: All benchmark metrics, ablation tables, and Pareto plots saved to `/results` (CSV, JSON, PNG)

---

## 3. Tool Protocols & Operational Rules
- **GSD (Get-Shit-Done)**: Every phase follows `discuss` -> `plan` -> `execute` -> `verify`. Keep plans small and modular.
- **Ponytail**: Keep active across the entire project.
  - Before writing code, verify stdlib or existing dependency suitability.
  - One-liners where appropriate.
  - Mark deliberate shortcuts with `# ponytail: <rationale>`.
  - Continuous review to prune bloat without compromising tests, input validation, or evaluation harness.
- **Ralph Loop**: Clear pass/fail checkpoints executed with iteration cap (20-25) and completion promise `<promise>PHASE_N_DONE</promise>` emitted strictly upon green tests.
- **CodeRabbit**: Clean review on uncommitted diffs before moving between phases. Critical & major findings must be resolved.

---

## 4. Dataset & Partitioning Specification
- **Sources**:
  - HotpotQA (distractor split, multi-hop reasoning)
  - Natural Questions (Google web queries, single-hop / direct factual)
- **Sampling**: ~1,000 questions sampled from each dataset (Total: ~2,000 queries)
- **Unified Schema**:
  ```json
  {
    "query_id": "str",
    "query": "str",
    "answers": ["str"],
    "dataset": "hotpotqa | natural_questions",
    "passages": [
      {
        "id": "str",
        "title": "str",
        "text": "str",
        "is_supporting": true | false
      }
    ]
  }
  ```
- **Splits**: 70% Train, 15% Validation, 15% Test (Deterministic seed). Test split is strictly isolated for final evaluation.
- **Leakage Prevention**: Passage-level and query-level disjoint checks verified in automated unit tests.

---

## 5. Phased Roadmap
1. **Phase 1: Data Pipeline & Hygiene**
   - Ingestion, sampling (~1k HotpotQA, ~1k NQ), unified schema normalization.
   - Deterministic 70/15/15 train/val/test splits.
   - Rigorous data leakage test suite.
2. **Phase 2: Indexing & Fixed Baselines**
   - Chroma vector indexing of passage corpus with sentence-transformers.
   - Fixed baseline runners: $k=0$ (parametric), $k=3$, $k=5$, $k=10$.
   - Output generation and baseline metrics harness.
3. **Phase 3: Router Ground Truth Labeling**
   - Run training & val queries across $k \in \{0, 3, 8, 15\}$.
   - Evaluate exact match (EM) and token F1 against reference answers.
   - Ground truth label assignment: minimal (cheapest) $k$ satisfying correctness threshold.
   - Persistent LLM call caching (SQLite/diskcache/JSONL) to prevent duplicate spend.
4. **Phase 4: Router Training & Ablation**
   - Feature engineering: query length, question words, semantic ambiguity, perplexity/entropy proxy, embedding vectors.
   - Model exploration: Logistic Regression, XGBoost, and Small Encoder / MLP.
   - Ablation study comparing query-feature models vs. embedding-based classifiers.
5. **Phase 5: Adaptive Retrieval Pipeline**
   - Integration of classifier router with vector retriever.
   - Confidence thresholding & fallback to higher $k$ when router margin is low.
   - Answer sufficiency check & reactive escalation loop.
6. **Phase 6: Comprehensive Benchmark & Evaluation**
   - Evaluation on isolated 15% Test split.
   - Metrics: EM, F1, LLM-as-a-judge score, average tokens/cost per query, p50/p95 latency breakdown (Router vs Retrieval vs Generation), under-retrieval failure rate.
   - Pareto frontier visualization (Accuracy vs Cost, Accuracy vs Latency).
7. **Phase 7: Serving, Dashboard & Documentation**
   - Production FastAPI router endpoint (`/predict`, `/query`).
   - Interactive diagnostic dashboard visualizing real-time routing decisions, confidence scores, and retrieved chunks.
   - Final `README.md` containing full benchmarks, system architecture, and Pareto curves.
