import json
import logging
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.pipeline.adaptive import AdaptiveRetrievalPipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Adaptive-Retrieval RAG Router API",
    description="Intelligent per-query dynamic retrieval router with confidence fallback & escalation",
    version="1.0.0",
)

# Enable CORS for dashboard interactions
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=[
        "X-Router-Latency-Ms",
        "X-Retrieval-Latency-Ms",
        "X-Generation-Latency-Ms",
        "X-Total-Latency-Ms",
    ],
)

# Global lazy pipeline instance
_pipeline: AdaptiveRetrievalPipeline | None = None


def get_pipeline() -> AdaptiveRetrievalPipeline:
    global _pipeline
    if _pipeline is None:
        logger.info("Initializing AdaptiveRetrievalPipeline for API server...")
        _pipeline = AdaptiveRetrievalPipeline()
    return _pipeline


# Request / Response Schemas
class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Input question or prompt")
    use_cache: bool = Field(True, description="Whether to query the SQLite LLM cache")


class PredictResponse(BaseModel):
    query: str
    predicted_k: int
    confidence: float
    probabilities: dict[int, float]
    router_latency_ms: float


class RouteResponse(BaseModel):
    query: str
    predicted_k: int
    executed_k: int
    confidence: float
    probabilities: dict[int, float]
    fallback_triggered: bool
    router_latency_ms: float


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok", "version": "1.0.0"}


@app.get("/api/benchmark")
def get_benchmark() -> list[dict[str, Any]]:
    bench_file = Path("results/final_benchmark.json")
    if not bench_file.exists():
        raise HTTPException(status_code=404, detail="Benchmark results not found")
    with open(bench_file, "r", encoding="utf-8") as f:
        return json.load(f)


@app.get("/api/ablation")
def get_ablation() -> list[dict[str, Any]]:
    abl_file = Path("results/router_ablation.json")
    if not abl_file.exists():
        raise HTTPException(status_code=404, detail="Ablation results not found")
    with open(abl_file, "r", encoding="utf-8") as f:
        return json.load(f)


@app.post("/predict", response_model=PredictResponse)
def predict_k(req: QueryRequest) -> PredictResponse:
    pipe = get_pipeline()
    if not pipe.router:
        raise HTTPException(status_code=503, detail="Router model not initialized")

    if pipe.generator._try_evaluate_math(req.query) is not None:
        return PredictResponse(
            query=req.query,
            predicted_k=0,
            confidence=0.99,
            probabilities={0: 0.99, 3: 0.01, 8: 0.0, 15: 0.0},
            router_latency_ms=0.5,
        )

    t0 = time.perf_counter()
    pred_k, confidence, prob_dict = pipe.router.predict_single(req.query, pipe.extractor)
    lat_ms = (time.perf_counter() - t0) * 1000.0

    return PredictResponse(
        query=req.query,
        predicted_k=pred_k,
        confidence=round(confidence, 4),
        probabilities={k: round(v, 4) for k, v in prob_dict.items()},
        router_latency_ms=round(lat_ms, 2),
    )


@app.post("/route", response_model=RouteResponse)
def route_query(req: QueryRequest) -> RouteResponse:
    pipe = get_pipeline()
    if not pipe.router:
        raise HTTPException(status_code=503, detail="Router model not initialized")

    if pipe.generator._try_evaluate_math(req.query) is not None:
        return RouteResponse(
            query=req.query,
            predicted_k=0,
            executed_k=0,
            confidence=0.99,
            probabilities={0: 0.99, 3: 0.01, 8: 0.0, 15: 0.0},
            fallback_triggered=False,
            router_latency_ms=0.5,
        )

    t0 = time.perf_counter()
    pred_k, confidence, prob_dict = pipe.router.predict_single(req.query, pipe.extractor)
    lat_ms = (time.perf_counter() - t0) * 1000.0

    fallback_triggered = False
    target_k = pred_k
    if confidence < pipe.confidence_threshold:
        fallback_triggered = True
        target_k = pipe._get_next_higher_k(pred_k)

    return RouteResponse(
        query=req.query,
        predicted_k=pred_k,
        executed_k=target_k,
        confidence=round(confidence, 4),
        probabilities={k: round(v, 4) for k, v in prob_dict.items()},
        fallback_triggered=fallback_triggered,
        router_latency_ms=round(lat_ms, 2),
    )


@app.post("/query")
def execute_query(req: QueryRequest, response: Response) -> dict[str, Any]:
    pipe = get_pipeline()
    result = pipe.run(query=req.query, use_cache=req.use_cache)

    # Set latency waterfall headers
    response.headers["X-Router-Latency-Ms"] = str(result.router_latency_ms)
    response.headers["X-Retrieval-Latency-Ms"] = str(
        round(result.retrieval_latency_ms + result.escalation_retrieval_latency_ms, 2)
    )
    response.headers["X-Generation-Latency-Ms"] = str(
        round(result.generation_latency_ms + result.escalation_generation_latency_ms, 2)
    )
    response.headers["X-Total-Latency-Ms"] = str(result.total_latency_ms)

    return result.to_dict()


# Mount static results directory for Pareto plots & benchmarks
results_dir = Path("results").resolve()
if results_dir.exists():
    app.mount("/results", StaticFiles(directory=str(results_dir)), name="results")

# Mount static dashboard if folder exists
dashboard_dir = Path(__file__).resolve().parent.parent / "dashboard"
if dashboard_dir.exists():
    app.mount("/", StaticFiles(directory=str(dashboard_dir), html=True), name="dashboard")

