import re
from pathlib import Path

import pytest

from src.data.schema import UnifiedQueryRecord


def normalize_text(text: str) -> str:
    # ponytail: minimal normalization using stdlib re
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", "", text.lower())).strip()


@pytest.fixture(scope="module")
def split_data():
    base_dir = Path("data/processed")
    train_path = base_dir / "train.jsonl"
    val_path = base_dir / "val.jsonl"
    test_path = base_dir / "test.jsonl"

    assert train_path.exists(), f"Missing {train_path}"
    assert val_path.exists(), f"Missing {val_path}"
    assert test_path.exists(), f"Missing {test_path}"

    def load_jsonl(path: Path):
        records = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records.append(UnifiedQueryRecord.model_validate_json(line))
        return records

    return {
        "train": load_jsonl(train_path),
        "val": load_jsonl(val_path),
        "test": load_jsonl(test_path),
    }


def test_schema_validity(split_data):
    """Ensure every record complies with UnifiedQueryRecord and has valid fields."""
    for split_name, records in split_data.items():
        assert len(records) > 0, f"Split {split_name} is empty"
        for rec in records:
            assert isinstance(rec, UnifiedQueryRecord)
            assert rec.query_id.strip() != "", f"Empty query_id in {split_name}"
            assert rec.query.strip() != "", f"Empty query in {split_name}"
            assert len(rec.answers) > 0, f"No answers for query {rec.query_id} in {split_name}"
            assert rec.dataset in ("hotpotqa", "natural_questions")
            assert len(rec.passages) > 0, f"No passages for query {rec.query_id} in {split_name}"
            for p in rec.passages:
                assert p.id.strip() != ""
                assert p.text.strip() != ""


def test_no_query_id_leakage(split_data):
    """Ensure zero overlap of query IDs across train, val, and test splits."""
    train_ids = {r.query_id for r in split_data["train"]}
    val_ids = {r.query_id for r in split_data["val"]}
    test_ids = {r.query_id for r in split_data["test"]}

    train_val_overlap = train_ids.intersection(val_ids)
    train_test_overlap = train_ids.intersection(test_ids)
    val_test_overlap = val_ids.intersection(test_ids)

    assert not train_val_overlap, f"Query ID leakage between train and val: {train_val_overlap}"
    assert not train_test_overlap, f"Query ID leakage between train and test: {train_test_overlap}"
    assert not val_test_overlap, f"Query ID leakage between val and test: {val_test_overlap}"


def test_no_textual_query_leakage(split_data):
    """Ensure zero exact normalized question text overlap across splits."""
    train_texts = {normalize_text(r.query) for r in split_data["train"]}
    val_texts = {normalize_text(r.query) for r in split_data["val"]}
    test_texts = {normalize_text(r.query) for r in split_data["test"]}

    train_val_overlap = train_texts.intersection(val_texts)
    train_test_overlap = train_texts.intersection(test_texts)
    val_test_overlap = val_texts.intersection(test_texts)

    assert not train_val_overlap, f"Text leakage between train and val: {train_val_overlap}"
    assert not train_test_overlap, f"Text leakage between train and test: {train_test_overlap}"
    assert not val_test_overlap, f"Text leakage between val and test: {val_test_overlap}"


def test_split_proportions_and_representation(split_data):
    """Ensure ~70/15/15 ratio and fair representation of both datasets."""
    n_train = len(split_data["train"])
    n_val = len(split_data["val"])
    n_test = len(split_data["test"])
    total = n_train + n_val + n_test

    assert total >= 1800, f"Expected ~2000 total queries, found {total}"
    assert 0.65 <= (n_train / total) <= 0.75, f"Train proportion out of bounds: {n_train/total:.2f}"
    assert 0.12 <= (n_val / total) <= 0.18, f"Val proportion out of bounds: {n_val/total:.2f}"
    assert 0.12 <= (n_test / total) <= 0.18, f"Test proportion out of bounds: {n_test/total:.2f}"

    for split_name, records in split_data.items():
        hotpot_count = sum(1 for r in records if r.dataset == "hotpotqa")
        nq_count = sum(1 for r in records if r.dataset == "natural_questions")
        assert hotpot_count > 0, f"No HotpotQA in {split_name}"
        assert nq_count > 0, f"No Natural Questions in {split_name}"
