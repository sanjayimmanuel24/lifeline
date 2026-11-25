"""Long-term memory: episodic (vector), semantic SOP (vector), procedural templates and the self-model.

Episodic memory embeds the situation summary text and appends the CVAE latent mean,
so "similar situations" match on both what was said and the shape of demand.
"""
from __future__ import annotations

import json
import re
import sqlite3
import threading
from pathlib import Path

import numpy as np

from lifeline.config import ARTIFACTS

KNOWLEDGE = Path(__file__).resolve().parents[1] / "knowledge"
LATENT_WEIGHT = 0.15

_lock = threading.Lock()


class _Embedder:
    """Chroma's bundled all-MiniLM-L6-v2 (ONNX). Falls back to hashed bag-of-words offline."""

    def __init__(self):
        try:
            from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
            self.fn = DefaultEmbeddingFunction()
            self.fn(["warmup"])
            self.kind = "all-MiniLM-L6-v2"
        except Exception:                      # no network for the model download
            self.fn, self.kind = None, "hashed-bow"

    def __call__(self, texts: list[str]) -> np.ndarray:
        if self.fn is not None:
            return np.array(self.fn(texts), dtype=np.float32)
        out = np.zeros((len(texts), 384), np.float32)
        for i, t in enumerate(texts):
            for w in re.findall(r"[a-z0-9]+", t.lower()):
                out[i, hash(w) % 384] += 1
        return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-6)


class Memory:
    def __init__(self, path: Path | None = None):
        import chromadb
        path = path or ARTIFACTS / "memory"
        path.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(path / "chroma"))
        self.embed = _Embedder()
        self.episodes = self.client.get_or_create_collection("episodes", metadata={"hnsw:space": "cosine"})
        self.sop = self.client.get_or_create_collection("sop", metadata={"hnsw:space": "cosine"})
        if self.sop.count() == 0:
            self._index_sop()
        self.db = sqlite3.connect(path / "memory.sqlite", check_same_thread=False)
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS templates(name TEXT PRIMARY KEY, bucket TEXT, posts TEXT, score REAL, uses INT);
            CREATE TABLE IF NOT EXISTS lessons(id INTEGER PRIMARY KEY, t TEXT, bucket TEXT, text TEXT, source TEXT);
        """)

    # ---- semantic: SOP RAG --------------------------------------------------------
    def _index_sop(self) -> None:
        text = (KNOWLEDGE / "sop.md").read_text(encoding="utf-8")
        chunks = [c.strip() for c in re.split(r"\n(?=## )", text) if c.startswith("## ")]
        ids = [re.match(r"## (SOP-\d+)", c).group(1) for c in chunks]
        self.sop.add(ids=ids, documents=chunks, embeddings=self.embed(chunks).tolist())

    def search_sop(self, query: str, k: int = 2) -> list[dict]:
        r = self.sop.query(query_embeddings=self.embed([query]).tolist(), n_results=k)
        return [{"id": i, "text": d, "distance": float(s)}
                for i, d, s in zip(r["ids"][0], r["documents"][0], r["distances"][0])]

    # ---- episodic -------------------------------------------------------------------
    def _vec(self, text: str, latent: np.ndarray | None) -> list[float]:
        v = self.embed([text])[0]
        lat = np.zeros(8, np.float32) if latent is None else np.asarray(latent, np.float32)[:8]
        lat = np.pad(lat, (0, 8 - len(lat)))
        return np.concatenate([v, LATENT_WEIGHT * lat / max(np.linalg.norm(lat), 1.0)]).tolist()

    def remember(self, episode_id: str, summary: str, latent, meta: dict) -> None:
        with _lock:
            self.episodes.upsert(ids=[episode_id], documents=[summary], embeddings=[self._vec(summary, latent)],
                                 metadatas=[{k: (json.dumps(v) if isinstance(v, (dict, list)) else v)
                                             for k, v in meta.items() if v is not None}])

    def recall(self, summary: str, latent, k: int = 3) -> list[dict]:
        if self.episodes.count() == 0:
            return []
        r = self.episodes.query(query_embeddings=[self._vec(summary, latent)], n_results=min(k, self.episodes.count()))
        return [{"id": i, "summary": d, "similarity": round(1 - float(s), 3), **m}
                for i, d, s, m in zip(r["ids"][0], r["documents"][0], r["distances"][0], r["metadatas"][0])]

    # ---- procedural + lessons -----------------------------------------------------------
    def save_template(self, name: str, bucket: str, posts: list[int], score: float) -> None:
        with _lock:
            self.db.execute("INSERT INTO templates VALUES(?,?,?,?,1) ON CONFLICT(name) DO UPDATE SET "
                            "posts=excluded.posts, score=excluded.score, uses=uses+1",
                            (name, bucket, json.dumps(posts), score))
            self.db.commit()

    def templates(self, bucket: str) -> list[dict]:
        rows = self.db.execute("SELECT name, posts, score, uses FROM templates WHERE bucket=? ORDER BY score DESC LIMIT 3",
                               (bucket,)).fetchall()
        return [{"name": n, "posts": json.loads(p), "score": s, "uses": u} for n, p, s, u in rows]

    def add_lesson(self, t: str, bucket: str, text: str, source: str) -> None:
        with _lock:
            self.db.execute("INSERT INTO lessons(t,bucket,text,source) VALUES(?,?,?,?)", (t, bucket, text, source))
            self.db.commit()

    def lessons(self, bucket: str | None = None, k: int = 3) -> list[dict]:
        q = "SELECT t, bucket, text, source FROM lessons" + (" WHERE bucket=?" if bucket else "") + " ORDER BY id DESC LIMIT ?"
        rows = self.db.execute(q, ((bucket, k) if bucket else (k,))).fetchall()
        return [{"t": t, "bucket": b, "text": x, "source": s} for t, b, x, s in rows]

    def reset_episodes(self) -> None:
        self.client.delete_collection("episodes")
        self.episodes = self.client.get_or_create_collection("episodes", metadata={"hnsw:space": "cosine"})
        self.db.execute("DELETE FROM lessons")
        self.db.execute("DELETE FROM templates")
        self.db.commit()
