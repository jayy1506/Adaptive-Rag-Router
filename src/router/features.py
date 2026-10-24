import re
import string

import numpy as np
from sentence_transformers import SentenceTransformer


class QueryFeatureExtractor:
    """Extracts linguistic query features and dense sentence embeddings for routing decisions."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        # ponytail: reuse cached all-MiniLM-L6-v2 model
        self.embedding_model = SentenceTransformer(model_name)
        self.wh_words = [
            "what", "who", "when", "where", "why", "how", "which", "whose", "whom"
        ]
        self.comparison_cues = ["both", "and", "or", "between", "differ", "same", "first"]

    def extract_linguistic_features(self, query: str) -> np.ndarray:
        """Extracts 14 normalized linguistic and surface features from query text."""
        # ponytail: lightweight regex feature extraction without large NLP parsers
        q_clean = query.strip()
        words = re.findall(r"\w+", q_clean.lower())
        num_words = max(1, len(words))
        char_len = len(q_clean)

        # 1. Length features
        word_count_norm = min(1.0, num_words / 30.0)
        char_len_norm = min(1.0, char_len / 180.0)
        avg_word_len = sum(len(w) for w in words) / num_words / 10.0

        # 2. WH-Word indicators
        wh_feature = [1.0 if w in words[:3] else 0.0 for w in ["what", "who", "when", "where", "why", "how", "which"]]

        # 3. Multi-hop / Bridge / Comparison indicators
        has_comparison = 1.0 if any(cue in words for cue in self.comparison_cues) else 0.0

        # 4. Syntactic complexity cues
        has_question_mark = 1.0 if "?" in q_clean else 0.0
        punc_count = sum(1 for ch in q_clean if ch in string.punctuation)
        punc_density = min(1.0, punc_count / (num_words + 1e-5))

        # 5. Named Entity proxy (capitalized words excluding sentence start)
        original_words = q_clean.split()
        capitalized = sum(1 for w in original_words[1:] if w and w[0].isupper())
        cap_ratio = min(1.0, capitalized / max(1, len(original_words) - 1))

        features = [
            word_count_norm,
            char_len_norm,
            avg_word_len,
            has_comparison,
            has_question_mark,
            punc_density,
            cap_ratio,
            *wh_feature,
        ]
        return np.array(features, dtype=np.float32)

    def extract_embedding(self, query: str) -> np.ndarray:
        """Computes 384-dimensional dense semantic embedding."""
        emb = self.embedding_model.encode(query, show_progress_bar=False)
        return np.array(emb, dtype=np.float32)

    def extract_batch(
        self, queries: list[str], feature_type: str = "hybrid"
    ) -> np.ndarray:
        """Extracts features for a batch of queries.

        feature_type: 'linguistic', 'embedding', or 'hybrid'
        """
        if feature_type == "linguistic":
            return np.vstack([self.extract_linguistic_features(q) for q in queries])

        embeddings = self.embedding_model.encode(queries, show_progress_bar=False, batch_size=64)
        embeddings = np.array(embeddings, dtype=np.float32)

        if feature_type == "embedding":
            return embeddings

        # 'hybrid': concatenate linguistic features + dense embedding
        ling_feats = np.vstack([self.extract_linguistic_features(q) for q in queries])
        return np.hstack([ling_feats, embeddings])
