import logging
from pathlib import Path
from typing import Any

import chromadb
import yaml
from chromadb.utils import embedding_functions

from src.data.schema import Passage

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_config(config_path: str = "config.yaml") -> dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class ChromaRetriever:
    def __init__(self, config_path: str = "config.yaml"):
        cfg = load_config(config_path)
        self.db_dir = Path(cfg["retrieval"]["chroma_db_dir"])
        self.collection_name = cfg["retrieval"]["collection_name"]
        self.model_name = cfg["retrieval"]["embedding_model"]
        self.batch_size = cfg["retrieval"].get("batch_size", 128)

        self.db_dir.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(self.db_dir))
        # ponytail: reuse cached sentence-transformers embedding function
        self.ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=self.model_name
        )
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self.ef,
            metadata={"hnsw:space": "cosine"},
        )
        if self.collection.count() == 0 and Path("data/processed/passages.jsonl").exists():
            logger.info("Chroma index empty. Initializing index from passages.jsonl...")
            self.build_index()

    def count(self) -> int:
        return self.collection.count()

    def build_index(
        self, passages_path: str = "data/processed/passages.jsonl", force_reindex: bool = False
    ) -> int:
        current_count = self.count()
        if current_count > 0 and not force_reindex:
            logger.info(f"Collection '{self.collection_name}' already indexed with {current_count} passages.")
            return current_count

        if force_reindex and current_count > 0:
            logger.info("Force reindex requested. Resetting collection...")
            self.client.delete_collection(self.collection_name)
            self.collection = self.client.create_collection(
                name=self.collection_name,
                embedding_function=self.ef,
                metadata={"hnsw:space": "cosine"},
            )

        p_file = Path(passages_path)
        if not p_file.exists():
            raise FileNotFoundError(f"Passages file {passages_path} does not exist.")

        logger.info(f"Reading passages from {passages_path}...")
        passages: list[Passage] = []
        with open(p_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    passages.append(Passage.model_validate_json(line))

        total = len(passages)
        logger.info(f"Indexing {total} passages in batches of {self.batch_size}...")

        for i in range(0, total, self.batch_size):
            batch = passages[i : i + self.batch_size]
            ids = [p.id for p in batch]
            documents = [p.text for p in batch]
            metadatas = [{"title": p.title, "is_supporting": p.is_supporting} for p in batch]

            self.collection.upsert(
                ids=ids,
                documents=documents,
                metadatas=metadatas,
            )
            if (i + len(batch)) % 1024 == 0 or (i + len(batch)) == total:
                logger.info(f"Indexed {i + len(batch)} / {total} passages.")

        logger.info(f"Successfully completed indexing. Total in collection: {self.count()}")
        return self.count()

    def retrieve(self, query: str, k: int = 3) -> list[dict[str, Any]]:
        # ponytail: k=0 skips vector db call entirely
        if k <= 0:
            return []

        results = self.collection.query(
            query_texts=[query],
            n_results=k,
        )

        formatted: list[dict[str, Any]] = []
        if not results or not results.get("ids") or not results["ids"][0]:
            return formatted

        ids = results["ids"][0]
        docs = results["documents"][0] if results.get("documents") else [""] * len(ids)
        metas = results["metadatas"][0] if results.get("metadatas") else [{}] * len(ids)
        dists = results["distances"][0] if results.get("distances") else [0.0] * len(ids)

        for p_id, text, meta, dist in zip(ids, docs, metas, dists):
            formatted.append(
                {
                    "id": p_id,
                    "title": meta.get("title", ""),
                    "text": text,
                    "is_supporting": meta.get("is_supporting", False),
                    "score": 1.0 - dist if dist is not None else 1.0,
                }
            )
        return formatted


if __name__ == "__main__":
    retriever = ChromaRetriever()
    retriever.build_index()
    sample_res = retriever.retrieve("When was Arthur's Magazine started?", k=3)
    print(f"Sample retrieval test: {len(sample_res)} results returned.")
