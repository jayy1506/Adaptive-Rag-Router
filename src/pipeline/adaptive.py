import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from src.generation.generator import RAGGenerator
from src.retrieval.indexer import ChromaRetriever
from src.router.features import QueryFeatureExtractor
from src.router.models import RouterInferenceWrapper

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

UNCERTAINTY_PHRASES = (
    "i do not know",
    "i don't know",
    "not enough information",
    "cannot determine",
    "unclear",
    "no information provided",
    "cannot be answered",
    "unknown",
    "not mentioned",
)


def load_config(config_path: str = "config.yaml") -> dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@dataclass
class AdaptiveRoutingResult:
    """Detailed telemetry and results from the adaptive retrieval pipeline."""

    query: str
    predicted_k: int
    executed_k: int
    confidence: float
    probabilities: dict[int, float] = field(default_factory=dict)
    fallback_triggered: bool = False
    escalation_triggered: bool = False
    passages: list[dict[str, Any]] = field(default_factory=list)
    answer: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    router_latency_ms: float = 0.0
    retrieval_latency_ms: float = 0.0
    generation_latency_ms: float = 0.0
    escalation_retrieval_latency_ms: float = 0.0
    escalation_generation_latency_ms: float = 0.0
    total_latency_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class AdaptiveRetrievalPipeline:
    """End-to-end adaptive orchestrator combining router prediction, confidence fallback,

    vector retrieval, answer generation, and sufficiency escalation.
    """

    def __init__(
        self,
        config_path: str = "config.yaml",
        router: RouterInferenceWrapper | None = None,
        retriever: ChromaRetriever | None = None,
        generator: RAGGenerator | None = None,
        extractor: QueryFeatureExtractor | None = None,
    ):
        self.config = load_config(config_path)
        self.candidate_k = sorted(self.config["router"]["candidate_k_values"])
        self.confidence_threshold = float(self.config["router"].get("confidence_threshold", 0.65))
        self.escalation_k = int(self.config["router"].get("escalation_k", 15))

        model_path = self.config["router"].get("model_save_path", "results/router_model.joblib")
        self.extractor = extractor or QueryFeatureExtractor()

        if router:
            self.router = router
        elif Path(model_path).exists():
            self.router = RouterInferenceWrapper.load(model_path)
        else:
            logger.warning(f"Router artifact {model_path} not found. Fallback mode active.")
            self.router = None

        self.retriever = retriever or ChromaRetriever(config_path=config_path)
        self.generator = generator or RAGGenerator(config_path=config_path)

    def _get_next_higher_k(self, current_k: int) -> int:
        """Ponytail: simple index lookup returns next higher k tier."""
        for k in self.candidate_k:
            if k > current_k:
                return k
        return current_k

    def _is_uncertain_answer(self, answer: str) -> bool:
        """Ponytail: regex-free string matching checks answer sufficiency."""
        clean = answer.strip().lower()
        if not clean:
            return True
        return any(phrase in clean for phrase in UNCERTAINTY_PHRASES)

    def run(self, query: str, use_cache: bool = True) -> AdaptiveRoutingResult:
        """Execute the 4-stage adaptive pipeline for a single query."""
        t0_total = time.perf_counter()

        # Stage 1: Feature Extraction & Routing Prediction
        t0_router = time.perf_counter()
        is_math = self.generator._try_evaluate_math(query) is not None

        if is_math:
            pred_k, confidence, prob_dict = 0, 0.99, {0: 0.99, 3: 0.01, 8: 0.0, 15: 0.0}
        elif self.router:
            pred_k, confidence, prob_dict = self.router.predict_single(query, self.extractor)
        else:
            # Default to middle tier if uninitialized
            pred_k, confidence, prob_dict = 3, 0.5, {0: 0.1, 3: 0.5, 8: 0.2, 15: 0.2}
        router_latency_ms = (time.perf_counter() - t0_router) * 1000.0

        # Stage 2: Confidence Fallback Check
        fallback_triggered = False
        target_k = pred_k
        if not is_math and confidence < self.confidence_threshold:
            fallback_triggered = True
            target_k = self._get_next_higher_k(pred_k)

        # Stage 3: Initial Retrieval & Generation
        t0_retrieval = time.perf_counter()
        passages = self.retriever.retrieve(query, k=target_k) if target_k > 0 else []
        retrieval_latency_ms = (time.perf_counter() - t0_retrieval) * 1000.0

        t0_gen = time.perf_counter()
        gen_res = self.generator.generate(query=query, passages=passages, k=target_k, use_cache=use_cache)
        generation_latency_ms = (time.perf_counter() - t0_gen) * 1000.0

        answer = gen_res["answer"]
        prompt_tokens = gen_res["tokens_in"]
        completion_tokens = gen_res["tokens_out"]
        executed_k = target_k

        # Stage 4: Sufficiency Check & Escalation Loop
        escalation_triggered = False
        esc_ret_latency_ms = 0.0
        esc_gen_latency_ms = 0.0

        if target_k < self.escalation_k and self._is_uncertain_answer(answer):
            escalation_triggered = True
            executed_k = self.escalation_k

            t0_esc_ret = time.perf_counter()
            escalated_passages = self.retriever.retrieve(query, k=self.escalation_k)
            esc_ret_latency_ms = (time.perf_counter() - t0_esc_ret) * 1000.0

            t0_esc_gen = time.perf_counter()
            esc_gen_res = self.generator.generate(
                query=query, passages=escalated_passages, k=self.escalation_k, use_cache=use_cache
            )
            esc_gen_latency_ms = (time.perf_counter() - t0_esc_gen) * 1000.0

            passages = escalated_passages
            answer = esc_gen_res["answer"]
            prompt_tokens += esc_gen_res["tokens_in"]
            completion_tokens += esc_gen_res["tokens_out"]

        total_latency_ms = (time.perf_counter() - t0_total) * 1000.0

        return AdaptiveRoutingResult(
            query=query,
            predicted_k=pred_k,
            executed_k=executed_k,
            confidence=round(confidence, 4),
            probabilities={k: round(v, 4) for k, v in prob_dict.items()},
            fallback_triggered=fallback_triggered,
            escalation_triggered=escalation_triggered,
            passages=passages,
            answer=answer,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            router_latency_ms=round(router_latency_ms, 2),
            retrieval_latency_ms=round(retrieval_latency_ms, 2),
            generation_latency_ms=round(generation_latency_ms, 2),
            escalation_retrieval_latency_ms=round(esc_ret_latency_ms, 2),
            escalation_generation_latency_ms=round(esc_gen_latency_ms, 2),
            total_latency_ms=round(total_latency_ms, 2),
        )

    def run_batch(self, queries: list[str], use_cache: bool = True) -> list[AdaptiveRoutingResult]:
        """Process a list of queries sequentially with the adaptive pipeline."""
        return [self.run(q, use_cache=use_cache) for q in queries]
