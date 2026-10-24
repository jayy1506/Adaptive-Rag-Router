import hashlib
import json
import sqlite3
from pathlib import Path


class GenerationCache:
    """Thread-safe persistent cache for LLM and reader responses using SQLite."""

    def __init__(self, db_path: str = "data/llm_cache.sqlite"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        # ponytail: sqlite3 stdlib for persistent key-value cache
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        return sqlite3.connect(str(self.db_path))

    def _init_db(self):
        with self._get_connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS response_cache (
                    cache_key TEXT PRIMARY KEY,
                    query TEXT,
                    k INTEGER,
                    answer TEXT,
                    tokens_in INTEGER,
                    tokens_out INTEGER,
                    latency_ms REAL,
                    metadata_json TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.commit()

    @staticmethod
    def make_key(query: str, k: int, context_passages: list[str]) -> str:
        # ponytail: deterministic hash on query + k + sorted passage texts
        raw = f"{query.strip().lower()}::k={k}::" + "||".join(context_passages)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def get(self, cache_key: str) -> dict | None:
        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT answer, tokens_in, tokens_out, latency_ms, metadata_json
                FROM response_cache WHERE cache_key = ?
                """,
                (cache_key,),
            )
            row = cur.fetchone()
            if row:
                return {
                    "answer": row[0],
                    "tokens_in": row[1],
                    "tokens_out": row[2],
                    "latency_ms": row[3],
                    "metadata": json.loads(row[4]) if row[4] else {},
                    "cached": True,
                }
        return None

    def set(
        self,
        cache_key: str,
        query: str,
        k: int,
        answer: str,
        tokens_in: int,
        tokens_out: int,
        latency_ms: float,
        metadata: dict | None = None,
    ):
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO response_cache
                (cache_key, query, k, answer, tokens_in, tokens_out, latency_ms, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    cache_key,
                    query,
                    k,
                    answer,
                    tokens_in,
                    tokens_out,
                    latency_ms,
                    json.dumps(metadata or {}),
                ),
            )
            conn.commit()

    def count(self) -> int:
        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT COUNT(*) FROM response_cache")
            return cur.fetchone()[0]
