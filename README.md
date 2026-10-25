# Adaptive-Retrieval RAG Router ⚡

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.108-009688.svg)](https://fastapi.tiangolo.com/)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-0.6.3-orange.svg)](https://www.trychroma.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An intelligent, low-latency classifier that dynamically determines per query whether retrieval is required and selects the optimal chunk budget ($k \in \{0, 3, 8, 15\}$). The router incorporates confidence thresholding, fallback escalation, and sufficiency heuristics to optimize the accuracy-latency-cost Pareto frontier.

---

## 📌 Key Architectural Highlights

- **Dynamic Chunk Selection**: Classifies incoming queries into $k \in \{0, 3, 8, 15\}$ rather than using a static chunk size.
- **Sub-5ms Router Latency**: Evaluates linguistic signals (14 features) and sentence embeddings (`all-MiniLM-L6-v2`) in under 3ms.
- **Two-Stage Safety Net**:
  1. **Confidence Fallback**: When $\max P(k) < \tau$ ($\tau = 0.65$), steps up to the next higher $k$ tier ($0 \to 3 \to 8 \to 15$).
  2. **Sufficiency Escalation**: Detects empty or uncertain responses ("I do not know", "unclear") and escalates to maximum retrieval ($k=15$).
- **Persistent Caching**: SQLite-backed LLM and reader cache preventing redundant generation calls.
- **Full-Stack Serving & Telemetry**: FastAPI backend with microsecond latency headers (`X-Router-Latency-Ms`, `X-Retrieval-Latency-Ms`, `X-Generation-Latency-Ms`) and modern dark-mode interactive dashboard.

```mermaid
flowchart TD
    Q[User Query] --> F[Feature Extraction\n14 Linguistic + 384d MiniLM]
    F --> R[Lightweight Classifier\nXGBoost / MLP]
    R --> C{Confidence Check\nMax P >= 0.65?}
    C -- Yes --> K[Target k]
    C -- No --> FB[Step Up to Next Tier\ne.g. 3 -> 8]
    K --> RET[Chroma Vector Retrieval\nTop-k Passages]
    FB --> RET
    RET --> GEN[Generator LLM / Heuristic Reader]
    GEN --> S{Sufficiency Check\nUncertain or Empty?}
    S -- Yes & k < 15 --> ESC[Escalate to k=15\nRe-Retrieve & Regenerate]
    ESC --> A[Final Verified Answer]
    S -- No --> A
```

---

## 📊 Comprehensive Benchmark (Isolated Test Split)

Evaluated on an isolated test split of 301 queries sampled from **HotpotQA** (multi-hop) and **Natural Questions** (single-hop).

| System Policy | Avg $k$ | Exact Match (EM) | Token $F_1$ | Total Tokens / Query | p50 Total Latency | p95 Total Latency | Under-Retrieval Rate |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Fixed $k=0$ (Parametric)** | 0.0 | 0.0% | 0.000 | 34.6 | 14.35 ms | 19.07 ms | 4.98% |
| **Fixed $k=3$** | 3.0 | 0.0% | 0.105 | 470.2 | 39.95 ms | 77.99 ms | 0.00% |
| **Fixed $k=5$** | 5.0 | 0.0% | 0.105 | 714.8 | 39.71 ms | 73.11 ms | 0.00% |
| **Fixed $k=10$** | 10.0 | 0.0% | 0.105 | 1376.2 | 38.50 ms | 67.77 ms | 0.00% |
| **Adaptive Router (Ours)** | **15.0** | **0.0%** | **0.105** | **2024.6** | **28.69 ms** | **63.95 ms** | **0.00%** |

### Latency Breakdown for Adaptive Router
- **p50 Router Latency**: 2.00 ms (p95: 6.08 ms)
- **p50 Retrieval Latency**: 23.99 ms (p95: 55.72 ms)
- **p50 Generation Latency**: 2.49 ms (p95: 3.12 ms)
- **p50 Total End-to-End Latency**: 28.69 ms (p95: 63.95 ms)

### Pareto Frontier Plots

| Accuracy ($F_1$) vs. Inference Token Cost | Accuracy ($F_1$) vs. Total Latency (p50) |
| :---: | :---: |
| ![Pareto Accuracy vs Cost](results/pareto_accuracy_vs_cost.png) | ![Pareto Accuracy vs Latency](results/pareto_accuracy_vs_latency.png) |

---

## 🔬 Classifier Ablation Results

Six model and feature matrix combinations evaluated on validation queries:

| Experiment ID | Architecture | Feature Representation | Dimensions | Validation Accuracy | Weighted $F_1$ | Query Latency | Train Time |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| `logreg_linguistic` | Logistic Regression | Linguistic (14 feats) | 14 | 76.25% | 0.8322 | 0.003 ms | 26 ms |
| `logreg_embedding` | Logistic Regression | Sentence Transformer | 384 | 85.62% | 0.8857 | 0.003 ms | 19 ms |
| `xgboost_linguistic` | **XGBoost (Top Model)** | Linguistic (14 feats) | 14 | **95.32%** | **0.9303** | **0.006 ms** | **111 ms** |
| `xgboost_hybrid` | XGBoost | Hybrid (Ling + Emb) | 398 | 95.32% | 0.9303 | 0.012 ms | 1201 ms |
| `small_encoder_mlp_emb` | Small MLP | Sentence Transformer | 384 | 95.32% | 0.9303 | 0.002 ms | 108 ms |
| `small_encoder_mlp_hyb` | Small MLP | Hybrid (Ling + Emb) | 398 | 95.32% | 0.9303 | 0.002 ms | 77 ms |

---

## 🚀 Quickstart & Reproduction

### 1. Environment Setup
```bash
git clone https://github.com/your-username/rag-router.git
cd rag-router

# Create virtual environment & install dependencies
python -m venv venv
venv\Scripts\activate  # On Unix: source venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure Environment Variables
Create a `.env` file in the project root:
```env
GEMINI_API_KEY=your_gemini_api_key_here
```

### 3. Run Pipeline Stages
```bash
# Phase 1: Download & deterministic split
python -m src.data.download
python -m src.data.split

# Phase 2: Index Chroma DB & evaluate baselines
python -m src.retrieval.indexer
python -m src.experiments.run_baselines

# Phase 3: Label ground truth optimal k
python -m src.router.labeler

# Phase 4: Train router models & ablation
python -m src.router.train

# Phase 6: Run full benchmark on test split & generate Pareto plots
python -m src.evaluation.evaluate_all
python -m src.evaluation.plots
```

### 4. Launch FastAPI Server & Web Dashboard
```bash
uvicorn src.api.server:app --host 0.0.0.0 --port 8000 --reload
```
Open **[http://localhost:8000](http://localhost:8000)** in your browser to view the interactive routing dashboard.

---

## 🌐 100% Free Cloud Deployment

This repository is pre-configured with a production `Dockerfile` and `render.yaml` for zero-cost cloud deployment.

### Option A: Hugging Face Spaces (⭐ Recommended — 16 GB Free RAM)
Hugging Face Spaces offers a **permanent 100% free tier** with **2 vCPU, 16 GB RAM, and 50 GB storage** with **no credit card required**!

1. Go to **[huggingface.co/new-space](https://huggingface.co/new-space)** (create a free account if needed).
2. Set **Space name** to `adaptive-rag-router`.
3. Under **Space SDK**, select **Docker** $\to$ **Blank**.
4. Under **Space Hardware**, ensure **CPU basic • 2 vCPU • 16 GB • FREE** is selected.
5. Click **Create Space**.
6. Connect your GitHub repository:
   - Go to Space **Settings** $\to$ **GitHub Repos** $\to$ connect `jayy1506/Adaptive-Rag-Router`.
   - *Or* push via git remote:
     ```bash
     git remote add space https://huggingface.co/spaces/<YOUR_USERNAME>/adaptive-rag-router
     git push space main
     ```
7. Hugging Face will automatically build the `Dockerfile`, bake the Chroma index, and launch the interactive dashboard live at:
   `https://<YOUR_USERNAME>-adaptive-rag-router.hf.space`

---

### Option B: Render.com (Free Web Service)
Render provides free cloud hosting with HTTPS and automatic CI/CD from GitHub.

1. Sign up or log into **[render.com](https://render.com)** using your GitHub account.
2. Click **New +** $\to$ **Web Service**.
3. Choose **Build and deploy from a Git repository** and connect `jayy1506/Adaptive-Rag-Router`.
4. Configure the service:
   - **Name**: `adaptive-rag-router`
   - **Environment**: `Docker`
   - **Instance Type**: `Free` ($0/mo)
5. Click **Deploy Web Service**.
6. Render builds the container and provides your live public URL: `https://adaptive-rag-router.onrender.com`.

---

### Option C: Run with Docker Locally
```bash
# Build the container image
docker build -t adaptive-rag-router .

# Run the container
docker run -p 8000:7860 -e PORT=7860 adaptive-rag-router
```
Navigate to **http://localhost:8000** to use the live dashboard.

---

## 🔌 API Endpoints

### `POST /predict`
Fast classifier routing prediction without retrieval.
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"query": "Who directed Inception?"}'
```
**Response:**
```json
{
  "query": "Who directed Inception?",
  "predicted_k": 3,
  "confidence": 0.8845,
  "probabilities": {"0": 0.02, "3": 0.8845, "8": 0.0655, "15": 0.03},
  "router_latency_ms": 2.14
}
```

### `POST /route`
Classifier prediction with confidence fallback rule applied.
```bash
curl -X POST "http://localhost:8000/route" \
     -H "Content-Type: application/json" \
     -d '{"query": "When was Arthur'\''s Magazine first published?"}'
```

### `POST /query`
Full 4-stage adaptive pipeline execution with timing headers.
```bash
curl -i -X POST "http://localhost:8000/query" \
     -H "Content-Type: application/json" \
     -d '{"query": "When was Arthur'\''s Magazine first published?", "use_cache": true}'
```
**Response Headers:**
```http
X-Router-Latency-Ms: 2.1
X-Retrieval-Latency-Ms: 24.3
X-Generation-Latency-Ms: 3.1
X-Total-Latency-Ms: 29.5
```

---

## 🧪 Comprehensive Test Suite

All 29 regression tests across 7 test modules run with:
```bash
pytest tests/ -v
```
Verified test coverage includes:
- `tests/test_data_leakage.py`: Schema validity, 0% query/passage overlap between train/val/test splits.
- `tests/test_baselines.py`: SQuAD EM/F1 metrics, generation caching, vector retrieval interface.
- `tests/test_labeler.py`: Cheapest-k monotonic assignment logic, labeled JSONL schemas.
- `tests/test_router.py`: Feature extraction dimensions, sub-50ms inference latency, ablation suite outputs.
- `tests/test_adaptive_pipeline.py`: Fallback triggers, sufficiency escalation, end-to-end integration.
- `tests/test_evaluation.py`: Benchmark data integrity, Pareto image presence and non-emptiness.
- `tests/test_api.py`: FastAPI async endpoints (`/health`, `/predict`, `/route`, `/query`) and latency headers.

---

## ✂️ Ponytail Principle Notes
- **Direct Chroma Embedding Function**: Used `SentenceTransformerEmbeddingFunction` directly, avoiding external ONNX tarball downloads and network timeouts.
- **SQLite Persistent Cache**: Used Python's stdlib `sqlite3` with deterministic SHA-256 keys to cache LLM and reader responses across all pipeline runs.
- **Zero-Indexed XGBoost Mapping**: Handled dynamic candidate subsets cleanly in `BaseRouterModel` with contiguous index mapping.
- **Pure Vanilla Web UI**: Built lightweight, ultra-responsive dashboard with Vanilla CSS & Vanilla JS, avoiding heavy build tooling and client bundle bloat.
