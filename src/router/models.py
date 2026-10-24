import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class BaseRouterModel:
    """Base class for routing classifiers."""

    def __init__(self, name: str, candidate_k: list[int]):
        self.name = name
        self.candidate_k = candidate_k
        self.k_to_train_idx: dict[int, int] = {}
        self.train_idx_to_k: dict[int, int] = {}
        self.classes_in_train_: list[int] = []
        self.model = None

    def _fit_label_mapping(self, y: list[int]):
        # ponytail: dynamic continuous indexing 0..N-1 ensures XGBoost and MLP compliance
        self.classes_in_train_ = sorted(set(y))
        self.k_to_train_idx = {k: i for i, k in enumerate(self.classes_in_train_)}
        self.train_idx_to_k = {i: k for i, k in enumerate(self.classes_in_train_)}

    def _encode_labels(self, y: list[int]) -> np.ndarray:
        return np.array([self.k_to_train_idx.get(k, 0) for k in y], dtype=np.int64)

    def _decode_labels(self, y_indices: np.ndarray) -> list[int]:
        return [self.train_idx_to_k.get(int(idx), self.candidate_k[-1]) for idx in y_indices]

    def fit(self, X: np.ndarray, y: list[int]):
        self._fit_label_mapping(y)
        y_enc = self._encode_labels(y)
        self.model.fit(X, y_enc)
        return self

    def predict(self, X: np.ndarray) -> list[int]:
        y_pred_idx = self.model.predict(X)
        return self._decode_labels(y_pred_idx)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        # ponytail: predict_proba maps observed class probabilities back to candidate_k
        raw_probs = self.model.predict_proba(X)
        aligned = np.zeros((len(X), len(self.candidate_k)), dtype=np.float32)
        classes = self.classes_in_train_ if self.classes_in_train_ else self.candidate_k
        for local_idx, k in enumerate(classes):
            if k in self.candidate_k:
                full_idx = self.candidate_k.index(k)
                aligned[:, full_idx] = raw_probs[:, local_idx]
        return aligned


class LogisticRegressionRouter(BaseRouterModel):
    def __init__(self, candidate_k: list[int], max_iter: int = 500, c: float = 1.0):
        super().__init__("LogisticRegression", candidate_k)
        # ponytail: class_weight='balanced' addresses class skew
        self.model = LogisticRegression(
            C=c,
            max_iter=max_iter,
            class_weight="balanced",
            random_state=42,
            solver="lbfgs",
        )


class XGBoostRouter(BaseRouterModel):
    def __init__(self, candidate_k: list[int], n_estimators: int = 80, max_depth: int = 4):
        super().__init__("XGBoost", candidate_k)
        self.model = xgb.XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=0.08,
            random_state=42,
            eval_metric="mlogloss",
            tree_method="hist",
        )


class SmallEncoderMLPRouter(BaseRouterModel):
    """Small neural classifier (2-layer MLP) acting as a lightweight encoder router."""

    def __init__(self, candidate_k: list[int], hidden_dim: int = 64):
        super().__init__("SmallEncoderMLP", candidate_k)
        # ponytail: MLPClassifier provides low-latency neural feedforward with early stopping
        self.model = MLPClassifier(
            hidden_layer_sizes=(hidden_dim, 32),
            activation="relu",
            max_iter=300,
            early_stopping=True,
            random_state=42,
            learning_rate_init=0.003,
        )


class RouterInferenceWrapper:
    """End-to-end wrapper pairing a trained model with feature extraction for real-time inference."""

    def __init__(
        self,
        model: BaseRouterModel,
        feature_type: str = "hybrid",
        candidate_k: list[int] | None = None,
    ):
        self.model = model
        self.feature_type = feature_type
        self.candidate_k = candidate_k or model.candidate_k

    def predict_single(self, query: str, extractor: Any) -> tuple[int, float, dict[int, float]]:
        """Returns: (predicted_k, confidence_probability, per_class_probabilities)."""
        feats = extractor.extract_batch([query], feature_type=self.feature_type)
        probs = self.model.predict_proba(feats)[0]
        pred_k = self.model.predict(feats)[0]
        max_prob = float(np.max(probs))

        prob_dict = {k: float(probs[i]) for i, k in enumerate(self.candidate_k)}
        return pred_k, max_prob, prob_dict

    def save(self, file_path: str):
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "model": self.model,
                "feature_type": self.feature_type,
                "candidate_k": self.candidate_k,
            },
            path,
        )
        logger.info(f"Saved router model artifact to {path}")

    @classmethod
    def load(cls, file_path: str) -> "RouterInferenceWrapper":
        data = joblib.load(file_path)
        return cls(
            model=data["model"],
            feature_type=data["feature_type"],
            candidate_k=data["candidate_k"],
        )
