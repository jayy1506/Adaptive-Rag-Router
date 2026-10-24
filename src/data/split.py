import logging
import random
import re
from pathlib import Path

from src.data.download import load_config, run_sampling
from src.data.schema import Passage, UnifiedQueryRecord

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def _normalize_query(q: str) -> str:
    # ponytail: minimal regex normalization for query deduplication
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", "", q.lower())).strip()


def partition_records(
    records: list[UnifiedQueryRecord],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> dict[str, list[UnifiedQueryRecord]]:
    # ponytail: deduplicate on normalized query text using dict
    seen_texts: set[str] = set()
    unique_records: list[UnifiedQueryRecord] = []
    for r in records:
        norm_q = _normalize_query(r.query)
        if norm_q and norm_q not in seen_texts:
            seen_texts.add(norm_q)
            unique_records.append(r)

    rng = random.Random(seed)
    rng.shuffle(unique_records)

    n_total = len(unique_records)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    train = unique_records[:n_train]
    val = unique_records[n_train : n_train + n_val]
    test = unique_records[n_train + n_val :]

    return {"train": train, "val": val, "test": test}


def create_splits() -> dict[str, list[UnifiedQueryRecord]]:
    cfg = load_config()
    seed = cfg["project"]["seed"]
    splits_cfg = cfg["data"]["splits"]
    train_ratio = splits_cfg["train"]
    val_ratio = splits_cfg["val"]

    raw_dir = Path(cfg["data"]["raw_cache_dir"])
    proc_dir = Path(cfg["data"]["processed_dir"])
    proc_dir.mkdir(parents=True, exist_ok=True)

    hp_raw_file = raw_dir / "hotpotqa_raw.jsonl"
    nq_raw_file = raw_dir / "natural_questions_raw.jsonl"

    if not hp_raw_file.exists() or not nq_raw_file.exists():
        logger.info("Raw files missing, running sampling first...")
        run_sampling()

    def read_records(path: Path) -> list[UnifiedQueryRecord]:
        items = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    items.append(UnifiedQueryRecord.model_validate_json(line))
        return items

    hp_records = read_records(hp_raw_file)
    nq_records = read_records(nq_raw_file)

    logger.info(f"Loaded {len(hp_records)} HotpotQA and {len(nq_records)} NQ raw records.")

    # Partition each dataset independently with deterministic seed
    hp_splits = partition_records(hp_records, train_ratio, val_ratio, seed=seed)
    nq_splits = partition_records(nq_records, train_ratio, val_ratio, seed=seed + 1)

    combined_splits: dict[str, list[UnifiedQueryRecord]] = {
        "train": hp_splits["train"] + nq_splits["train"],
        "val": hp_splits["val"] + nq_splits["val"],
        "test": hp_splits["test"] + nq_splits["test"],
    }

    # Deterministic shuffle for each combined split
    for split_name, recs in combined_splits.items():
        rng = random.Random(seed + hash(split_name))
        rng.shuffle(recs)

    # Save to jsonl
    all_passages: dict[str, Passage] = {}
    for split_name, recs in combined_splits.items():
        out_path = proc_dir / f"{split_name}.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for r in recs:
                f.write(r.model_dump_json() + "\n")
                for p in r.passages:
                    if p.id not in all_passages:
                        all_passages[p.id] = p
        logger.info(f"Saved {len(recs)} records to {out_path}")

    # Save passages corpus for vector DB indexing
    passages_path = proc_dir / "passages.jsonl"
    with open(passages_path, "w", encoding="utf-8") as f:
        for p in all_passages.values():
            f.write(p.model_dump_json() + "\n")
    logger.info(f"Saved {len(all_passages)} unique passages to {passages_path}")

    return combined_splits


if __name__ == "__main__":
    create_splits()
