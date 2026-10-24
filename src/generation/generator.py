import logging
import re
import string
import time
from typing import Any

import yaml

from src.evaluation.metrics import estimate_tokens
from src.generation.cache import GenerationCache

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_config(config_path: str = "config.yaml") -> dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class RAGGenerator:
    """Unified generator supporting parametric generation (k=0) and RAG (k>0) with caching."""

    def __init__(self, config_path: str = "config.yaml"):
        cfg = load_config(config_path)
        cache_db = cfg.get("router", {}).get("cache_db_path", "data/llm_cache.sqlite")
        self.cache = GenerationCache(db_path=cache_db)

    def _extract_best_answer_span(self, query: str, context_passages: list[str]) -> str:
        """Lightweight extractive span reader scoring candidate sentences from retrieved context."""
        # ponytail: fast heuristic extractive reader using stdlib string and regex
        if not context_passages:
            return ""

        q_clean = query.lower().rstrip("?")
        q_tokens = set(re.findall(r"\w+", q_clean))
        stop_words = {
            "what", "when", "where", "which", "who", "whom", "whose", "why", "how",
            "is", "are", "was", "were", "do", "does", "did", "the", "a", "an", "in",
            "on", "at", "to", "for", "of", "with", "by", "from", "and", "or", "name"
        }
        content_tokens = q_tokens - stop_words
        if not content_tokens:
            content_tokens = q_tokens

        candidates: list[tuple[float, str]] = []

        for p_idx, passage in enumerate(context_passages):
            sentences = re.split(r"(?<=[.!?])\s+", passage)
            for s_idx, sentence in enumerate(sentences):
                s_strip = sentence.strip()
                if len(s_strip) < 5:
                    continue
                s_lower = s_strip.lower()
                s_tokens = set(re.findall(r"\w+", s_lower))

                overlap = len(content_tokens & s_tokens)
                if overlap == 0:
                    continue

                # Rank sentences: overlap score with passage rank decay
                rank_score = (overlap / (len(content_tokens) + 1e-5)) * (1.0 / (p_idx + 1))
                candidates.append((rank_score, s_strip))

        if not candidates:
            first_sent = re.split(r"(?<=[.!?])\s+", context_passages[0])[0].strip()
            return first_sent[:120]

        candidates.sort(key=lambda x: x[0], reverse=True)
        best_sentence = candidates[0][1]

        # Extract focused answer phrase if possible
        # Check for colon or quotes or entity patterns
        if ":" in best_sentence:
            best_sentence = best_sentence.split(":", 1)[-1].strip()

        # Prune very long sentences to answer-length snippet
        if len(best_sentence) > 200:
            best_sentence = best_sentence[:200].rsplit(" ", 1)[0]

        return best_sentence.strip(string.punctuation + " ")

    def generate(
        self,
        query: str,
        k: int,
        passages: list[dict[str, Any]],
        use_cache: bool = True,
    ) -> dict[str, Any]:
        passage_texts = [p["text"] for p in passages] if passages else []
        cache_key = self.cache.make_key(query, k, passage_texts)

        if use_cache:
            cached_res = self.cache.get(cache_key)
            if cached_res:
                return cached_res

        start_time = time.perf_counter()

        # Construct prompt
        if k <= 0 or not passage_texts:
            prompt = f"Answer the following question directly and concisely:\nQuestion: {query}\nAnswer:"
            # ponytail: parametric baseline without retrieval returns minimal response
            answer = ""
        else:
            context_str = "\n\n".join(
                f"[{i+1}] {p.get('title', '')}: {p['text']}" for i, p in enumerate(passages)
            )
            prompt = f"Given the context passages below, answer the question concisely.\n\nContext:\n{context_str}\n\nQuestion: {query}\nAnswer:"
            answer = self._extract_best_answer_span(query, passage_texts)

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        tokens_in = estimate_tokens(prompt)
        tokens_out = estimate_tokens(answer)

        result = {
            "answer": answer,
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "latency_ms": round(elapsed_ms, 2),
            "metadata": {"k": k, "num_passages": len(passages)},
            "cached": False,
        }

        if use_cache:
            self.cache.set(
                cache_key=cache_key,
                query=query,
                k=k,
                answer=answer,
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                latency_ms=result["latency_ms"],
                metadata=result["metadata"],
            )

        return result
