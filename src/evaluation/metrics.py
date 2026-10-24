import re
import string
from collections import Counter


def normalize_answer(s: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""
    # ponytail: regex-based SQuAD normalization in four fast operations
    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text: str) -> str:
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def exact_match_score(prediction: str, ground_truth: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def f1_score(prediction: str, ground_truth: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gt_tokens = normalize_answer(ground_truth).split()

    if not pred_tokens or not gt_tokens:
        return float(pred_tokens == gt_tokens)

    # ponytail: collections.Counter for fast multiset intersection
    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0.0

    precision = 1.0 * num_same / len(pred_tokens)
    recall = 1.0 * num_same / len(gt_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def evaluate_prediction(prediction: str, ground_truths: list[str]) -> dict[str, float]:
    """Computes max EM and F1 across multiple valid reference answers."""
    if not ground_truths:
        return {"em": 0.0, "f1": 0.0}

    # ponytail: max over all ground truth answers
    em = max(exact_match_score(prediction, gt) for gt in ground_truths)
    f1 = max(f1_score(prediction, gt) for gt in ground_truths)
    return {"em": em, "f1": f1}


def estimate_tokens(text: str) -> int:
    """Approximate token count (1 word ≈ 1.3 tokens)."""
    # ponytail: lightweight token approximation without loading full BPE tokenizer
    words = text.split()
    return max(1, int(len(words) * 1.3))
