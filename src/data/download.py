import hashlib
import logging
from pathlib import Path

import yaml
from datasets import load_dataset

from src.data.schema import Passage, UnifiedQueryRecord

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _hash_str(val: str) -> str:
    # ponytail: stdlib hashlib for fast deterministic short hashes
    return hashlib.md5(val.encode("utf-8")).hexdigest()[:10]


def sample_hotpotqa(target_count: int = 1000) -> list[UnifiedQueryRecord]:
    logger.info(f"Streaming and sampling {target_count} records from HotpotQA (distractor)...")
    ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split="train", streaming=True)
    records: list[UnifiedQueryRecord] = []

    for item in ds:
        q_id = str(item.get("id", ""))
        question = str(item.get("question", "")).strip()
        answer = str(item.get("answer", "")).strip()
        if not question or not answer:
            continue

        context = item.get("context", {})
        titles = context.get("title", [])
        sentences_list = context.get("sentences", [])
        supp_titles = set(item.get("supporting_facts", {}).get("title", []))

        passages: list[Passage] = []
        for idx, (title, sents) in enumerate(zip(titles, sentences_list)):
            text = " ".join(sents).strip()
            if not text:
                continue
            p_id = f"hp_{q_id}_{idx}_{_hash_str(title)}"
            passages.append(
                Passage(
                    id=p_id,
                    title=title,
                    text=text,
                    is_supporting=(title in supp_titles),
                )
            )

        if not passages:
            continue

        records.append(
            UnifiedQueryRecord(
                query_id=f"hotpot_{q_id}",
                query=question,
                answers=[answer],
                dataset="hotpotqa",
                passages=passages,
            )
        )

        if len(records) >= target_count:
            break

    logger.info(f"Sampled {len(records)} HotpotQA records.")
    return records


def sample_natural_questions(target_count: int = 1000) -> list[UnifiedQueryRecord]:
    logger.info(f"Streaming and sampling {target_count} records from Natural Questions...")
    ds = load_dataset("cjlovering/natural-questions-short", split="train", streaming=True)
    records: list[UnifiedQueryRecord] = []

    for idx, item in enumerate(ds):
        raw_id = item.get("id") or str(idx)
        questions_raw = item.get("questions", [])
        if not questions_raw:
            continue
        question = str(questions_raw[0].get("input_text", "")).strip()

        answers_raw = item.get("answers", [])
        answers = [
            str(a.get("span_text", "")).strip()
            for a in answers_raw
            if a.get("span_text") and a.get("span_text").strip()
        ]
        # ponytail: fallback to candidate text if span text empty
        if not answers:
            answers = [
                str(a.get("input_text", "")).strip()
                for a in answers_raw
                if a.get("input_text") and a.get("input_text").strip()
            ]

        context_text = str(item.get("contexts", "")).strip()
        title = str(item.get("name", "")).strip()

        if not question or not answers or not context_text:
            continue

        p_id = f"nq_{raw_id}_{_hash_str(question)}"
        passage = Passage(
            id=p_id,
            title=title,
            text=context_text,
            is_supporting=bool(item.get("has_correct_context", True)),
        )

        records.append(
            UnifiedQueryRecord(
                query_id=f"nq_{raw_id}",
                query=question,
                answers=answers,
                dataset="natural_questions",
                passages=[passage],
            )
        )

        if len(records) >= target_count:
            break

    logger.info(f"Sampled {len(records)} Natural Questions records.")
    return records


def run_sampling():
    cfg = load_config()
    hp_count = cfg["data"]["hotpotqa_sample_size"]
    nq_count = cfg["data"]["nq_sample_size"]
    raw_dir = Path(cfg["data"]["raw_cache_dir"])
    raw_dir.mkdir(parents=True, exist_ok=True)

    hp_records = sample_hotpotqa(hp_count)
    nq_records = sample_natural_questions(nq_count)

    hp_out = raw_dir / "hotpotqa_raw.jsonl"
    nq_out = raw_dir / "natural_questions_raw.jsonl"

    with open(hp_out, "w", encoding="utf-8") as f:
        f.writelines(r.model_dump_json() + "\n" for r in hp_records)

    with open(nq_out, "w", encoding="utf-8") as f:
        f.writelines(r.model_dump_json() + "\n" for r in nq_records)

    logger.info(f"Saved raw sampled datasets to {hp_out} and {nq_out}")
    return hp_records, nq_records


if __name__ == "__main__":
    run_sampling()
