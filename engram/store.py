"""SQLite-backed memory store with hybrid retrieval that learns from use.

Retrieval blends BM25 keyword relevance with vector similarity. Every memory also carries
a use count and a last-used timestamp: memories that keep getting recalled rank higher;
memories nobody asks for decay and are pruned first.
"""

from __future__ import annotations

import math
import re
import sqlite3
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from .embed import Embedder, HashEmbedder, cosine, from_blob, normalize, to_blob

_TOKEN = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset(
    "a an and are as at be but by for from has have i in is it its my of on or "
    "that the this to was were will with you your".split()
)

# BM25 constants (standard defaults).
K1 = 1.5
B = 0.75

HALF_LIFE_DAYS = 30.0
DUPLICATE_THRESHOLD = 0.8

MODES = ("hybrid", "bm25", "vector")
# In hybrid mode, the share of the relevance score that comes from keyword matching.
KEYWORD_WEIGHT = 0.5
# Vector matches weaker than this are treated as unrelated.
MIN_SIMILARITY = 0.25


def _stem(token: str) -> str:
    # Light suffix stripping so "deploy" matches "deploys" / "deployed" / "deploying".
    for suffix in ("ing", "ed", "es", "s"):
        if token.endswith(suffix) and len(token) - len(suffix) >= 3 and not token.endswith("ss"):
            return token[: -len(suffix)]
    return token


def tokenize(text: str) -> list[str]:
    return [_stem(t) for t in _TOKEN.findall(text.lower()) if t not in _STOPWORDS]


@dataclass(frozen=True)
class Memory:
    id: int
    text: str
    kind: str
    importance: float
    uses: int
    created_at: float
    last_used: float


@dataclass(frozen=True)
class Hit:
    memory: Memory
    score: float


_COLUMNS = "id, text, kind, importance, uses, created_at, last_used"


class MemoryStore:
    def __init__(self, path: str | Path = "engram.db", embedder: Embedder | None = None):
        self.embedder = embedder or HashEmbedder()
        self.db = sqlite3.connect(str(path), check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.execute(
            """CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text TEXT NOT NULL,
                kind TEXT NOT NULL DEFAULT 'fact',
                importance REAL NOT NULL DEFAULT 1.0,
                uses INTEGER NOT NULL DEFAULT 0,
                created_at REAL NOT NULL,
                last_used REAL NOT NULL,
                embedding BLOB
            )"""
        )
        # Databases created before vector search have no embedding column.
        columns = {row["name"] for row in self.db.execute("PRAGMA table_info(memories)")}
        if "embedding" not in columns:
            self.db.execute("ALTER TABLE memories ADD COLUMN embedding BLOB")
        self.db.commit()

    def close(self) -> None:
        self.db.close()

    def all(self) -> list[Memory]:
        rows = self.db.execute(f"SELECT {_COLUMNS} FROM memories ORDER BY id").fetchall()
        return [Memory(**dict(r)) for r in rows]

    def add(self, text: str, kind: str = "fact", importance: float = 1.0,
            now: float | None = None) -> Memory:
        """Store a memory. A near-duplicate reinforces the existing one instead."""
        text = text.strip()
        if not text:
            raise ValueError("memory text is empty")
        now = time.time() if now is None else now

        tokens = set(tokenize(text))
        for existing in self.all():
            other = set(tokenize(existing.text))
            union = tokens | other
            if union and len(tokens & other) / len(union) >= DUPLICATE_THRESHOLD:
                self._reinforce([existing.id], now)
                return self.get(existing.id)

        cur = self.db.execute(
            "INSERT INTO memories (text, kind, importance, created_at, last_used, embedding) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (text, kind, importance, now, now, to_blob(self._embed(text))),
        )
        self.db.commit()
        return self.get(cur.lastrowid)

    def get(self, memory_id: int) -> Memory:
        row = self.db.execute(f"SELECT {_COLUMNS} FROM memories WHERE id = ?",
                              (memory_id,)).fetchone()
        if row is None:
            raise KeyError(memory_id)
        return Memory(**dict(row))

    def forget(self, memory_id: int) -> bool:
        cur = self.db.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
        self.db.commit()
        return cur.rowcount > 0

    def search(self, query: str, k: int = 5, reinforce: bool = True,
               now: float | None = None, mode: str = "hybrid") -> list[Hit]:
        """Rank memories by relevance, boosted by use count, recency and importance.

        `mode` picks the relevance signal: "bm25" (keywords), "vector" (embedding
        similarity) or "hybrid" (both, each scaled to its best match and averaged).
        """
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        now = time.time() if now is None else now
        memories = self.all()
        if not memories or not query.strip():
            return []

        keyword = self._bm25(query, memories) if mode != "vector" else {}
        vector = self._similarity(query, memories) if mode != "bm25" else {}
        if mode == "hybrid":
            relevance = _blend(keyword, vector, KEYWORD_WEIGHT)
        else:
            relevance = keyword or vector

        by_id = {m.id: m for m in memories}
        hits = [Hit(by_id[i], score * self._strength(by_id[i], now))
                for i, score in relevance.items()]
        hits.sort(key=lambda h: h.score, reverse=True)
        hits = hits[:k]
        if reinforce and hits:
            self._reinforce([h.memory.id for h in hits], now)
        return hits

    def prune(self, keep: int, now: float | None = None) -> int:
        """Drop the weakest memories until at most `keep` remain. Returns how many were dropped."""
        now = time.time() if now is None else now
        memories = sorted(self.all(), key=lambda m: self._strength(m, now), reverse=True)
        doomed = memories[keep:]
        self.db.executemany("DELETE FROM memories WHERE id = ?", [(m.id,) for m in doomed])
        self.db.commit()
        return len(doomed)

    def _bm25(self, query: str, memories: list[Memory]) -> dict[int, float]:
        terms = tokenize(query)
        docs = [tokenize(m.text) for m in memories]
        avg_len = sum(len(d) for d in docs) / len(docs)
        doc_freq = Counter(t for d in docs for t in set(d))
        n = len(docs)

        scores = {}
        for memory, doc in zip(memories, docs):
            counts = Counter(doc)
            score = 0.0
            for term in terms:
                tf = counts.get(term, 0)
                if not tf:
                    continue
                idf = math.log(1 + (n - doc_freq[term] + 0.5) / (doc_freq[term] + 0.5))
                norm = tf + K1 * (1 - B + B * len(doc) / (avg_len or 1))
                score += idf * tf * (K1 + 1) / norm
            if score > 0:
                scores[memory.id] = score
        return scores

    def _similarity(self, query: str, memories: list[Memory]) -> dict[int, float]:
        target = self._embed(query)
        rows = self.db.execute("SELECT id, text, embedding FROM memories").fetchall()
        scores = {}
        for row in rows:
            blob = row["embedding"]
            if blob is None:
                # Row written before vector search existed: embed it now and keep the result.
                blob = to_blob(self._embed(row["text"]))
                self.db.execute("UPDATE memories SET embedding = ? WHERE id = ?", (blob, row["id"]))
            similarity = cosine(target, from_blob(blob))
            if similarity >= MIN_SIMILARITY:
                scores[row["id"]] = similarity
        self.db.commit()
        return scores

    def _embed(self, text: str):
        return normalize(self.embedder(text))

    @staticmethod
    def _strength(memory: Memory, now: float) -> float:
        age_days = max(0.0, now - memory.last_used) / 86400
        recency = 0.5 ** (age_days / HALF_LIFE_DAYS)
        return memory.importance * (1 + math.log1p(memory.uses)) * (0.5 + 0.5 * recency)

    def _reinforce(self, ids: list[int], now: float) -> None:
        self.db.executemany(
            "UPDATE memories SET uses = uses + 1, last_used = ? WHERE id = ?",
            [(now, i) for i in ids],
        )
        self.db.commit()


def _blend(keyword: dict[int, float], vector: dict[int, float], weight: float) -> dict[int, float]:
    """Scale each signal to its best match, then take a weighted average."""
    top_keyword = max(keyword.values(), default=0.0) or 1.0
    top_vector = max(vector.values(), default=0.0) or 1.0
    return {
        i: weight * keyword.get(i, 0.0) / top_keyword + (1 - weight) * vector.get(i, 0.0) / top_vector
        for i in keyword.keys() | vector.keys()
    }
