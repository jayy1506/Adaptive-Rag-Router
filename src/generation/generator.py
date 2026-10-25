import logging
import os
import re
import string
import time
from typing import Any

import yaml
from dotenv import load_dotenv

from src.evaluation.metrics import estimate_tokens
from src.generation.cache import GenerationCache

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_config(config_path: str = "config.yaml") -> dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class RAGGenerator:
    """Unified generator supporting parametric generation (k=0) and RAG (k>0) with Gemini LLM + local fallback & caching."""

    def __init__(self, config_path: str = "config.yaml"):
        cfg = load_config(config_path)
        cache_db = cfg.get("router", {}).get("cache_db_path", "data/llm_cache.sqlite")
        self.cache = GenerationCache(db_path=cache_db)

        # Initialize Gemini client if available
        self.gemini_client = None
        self.model_name = "gemini-2.5-flash"
        self._rate_limit_until = 0.0
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY_FALLBACK")
        if api_key:
            try:
                from google import genai
                self.gemini_client = genai.Client(api_key=api_key)
                logger.info(f"Initialized Gemini client with model '{self.model_name}'")
            except Exception as e:  # noqa: BLE001
                logger.warning(f"Could not initialize Gemini client: {e}. Falling back to local reader.")

    def _call_gemini(self, prompt: str) -> str | None:
        if not self.gemini_client:
            return None
        # ponytail: circuit breaker prevents waiting on 429 quota exhaustion
        if time.time() < self._rate_limit_until:
            return None
        try:
            resp = self.gemini_client.models.generate_content(
                model=self.model_name,
                contents=prompt,
            )
            return resp.text.strip() if resp and resp.text else None
        except Exception as e:  # noqa: BLE001
            err_str = str(e).lower()
            if "429" in err_str or "quota" in err_str or "resource_exhausted" in err_str:
                # trip circuit breaker for 20 seconds
                self._rate_limit_until = time.time() + 20.0
                logger.debug("Gemini rate limit encountered. Tripping circuit breaker for 20s.")
            else:
                logger.debug(f"Gemini API call failed: {e}")
            return None

    def _extract_best_answer_span(self, query: str, context_passages: list[str]) -> str:
        """Lightweight extractive fallback reader scoring candidate sentences from retrieved context."""
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
        content_tokens = q_tokens - stop_words or q_tokens
        candidates: list[tuple[float, str]] = []

        for p_idx, passage in enumerate(context_passages):
            sentences = re.split(r"(?<=[.!?])\s+", passage)
            for sentence in sentences:
                s_strip = sentence.strip()
                if len(s_strip) < 5:
                    continue
                s_lower = s_strip.lower()
                s_tokens = set(re.findall(r"\w+", s_lower))

                overlap = len(content_tokens & s_tokens)
                if overlap == 0:
                    continue

                rank_score = (overlap / (len(content_tokens) + 1e-5)) * (1.0 / (p_idx + 1))
                candidates.append((rank_score, s_strip))

        if not candidates:
            first_sent = re.split(r"(?<=[.!?])\s+", context_passages[0])[0].strip()
            return first_sent[:120]

        candidates.sort(key=lambda x: x[0], reverse=True)
        best_sentence = candidates[0][1]

        if ":" in best_sentence:
            best_sentence = best_sentence.split(":", 1)[-1].strip()

        if len(best_sentence) > 200:
            best_sentence = best_sentence[:200].rsplit(" ", 1)[0]

        return best_sentence.strip(string.punctuation + " ")

    @staticmethod
    def _try_evaluate_math(query: str) -> str | None:
        """Ponytail: safely evaluate basic arithmetic without external dependencies."""
        m = re.search(r"(\b\d+\s*[\+\-\*\/]\s*\d+(?:\s*[\+\-\*\/]\s*\d+)*\b)", query)
        if m:
            expr = m.group(1)
            # Strict safety whitelist: only digits, spaces, and basic arithmetic operators
            if re.match(r"^[\d\s\+\-\*\/\(\)\.]+$", expr):
                try:
                    res = eval(expr, {"__builtins__": None}, {})
                    if isinstance(res, float) and res.is_integer():
                        res = int(res)
                    return str(res)
                except (ValueError, SyntaxError, ZeroDivisionError, TypeError):
                    return None
        return None

    def generate(
        self,
        query: str,
        k: int,
        passages: list[dict[str, Any]] | None = None,
        use_cache: bool = True,
    ) -> dict[str, Any]:
        # Handle direct arithmetic immediately without noise from passages
        math_ans = self._try_evaluate_math(query)
        if math_ans is not None:
            prompt = f"Question: {query}\nAnswer: {math_ans}"
            return {
                "answer": math_ans,
                "tokens_in": estimate_tokens(prompt),
                "tokens_out": estimate_tokens(math_ans),
                "latency_ms": 0.5,
                "metadata": {"k": k, "type": "arithmetic"},
                "cached": False,
            }

        passage_texts = [p["text"] for p in passages] if passages else []
        cache_key = self.cache.make_key(query, k, passage_texts)

        if use_cache:
            cached_res = self.cache.get(cache_key)
            if cached_res:
                return cached_res

        start_time = time.perf_counter()

        # Construct prompt
        if k <= 0 or not passage_texts:
            prompt = (
                f"Answer the following question directly and concisely in a few words.\n"
                f"Question: {query}\n"
                f"Answer:"
            )
            llm_ans = self._call_gemini(prompt)
            answer = llm_ans if llm_ans else ""
        else:
            context_str = "\n\n".join(
                f"[{i+1}] {p.get('title', '')}: {p['text']}" for i, p in enumerate(passages)
            )
            prompt = (
                f"Context passages:\n{context_str}\n\n"
                f"Based ONLY on the context passages above, answer the question directly and concisely.\n"
                f"Question: {query}\n"
                f"Answer:"
            )
            llm_ans = self._call_gemini(prompt)
            answer = llm_ans if llm_ans else self._extract_best_answer_span(query, passage_texts)

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        tokens_in = estimate_tokens(prompt)
        tokens_out = estimate_tokens(answer)

        result = {
            "answer": answer,
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "latency_ms": round(elapsed_ms, 2),
            "metadata": {"k": k, "num_passages": len(passage_texts)},
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
