"""Dependency-free text embeddings from hashed character n-grams.

Each word is broken into overlapping character n-grams (as in fastText) and every n-gram is
hashed into one of `dim` buckets. Words that share most of their letters land close together,
so the vector search tolerates typos and inflections that exact keyword matching misses.
It does not capture meaning: "car" and "automobile" stay far apart. For that, pass any
function that maps text to a vector as `embedder=` when creating the store.
"""

from __future__ import annotations

import math
import re
import zlib
from array import array
from typing import Callable

Embedder = Callable[[str], "array[float]"]

_WORD = re.compile(r"[a-z0-9]+")


class HashEmbedder:
    def __init__(self, dim: int = 512, min_n: int = 3, max_n: int = 5):
        self.dim = dim
        self.min_n = min_n
        self.max_n = max_n

    def __call__(self, text: str) -> "array[float]":
        vector = [0.0] * self.dim
        for word in _WORD.findall(text.lower()):
            padded = f"<{word}>"
            grams = [padded]
            for n in range(self.min_n, self.max_n + 1):
                grams += [padded[i:i + n] for i in range(len(padded) - n + 1)]
            for gram in grams:
                # crc32 is stable across runs, unlike the built-in hash().
                h = zlib.crc32(gram.encode())
                vector[h % self.dim] += 1.0 if h & 0x80000000 else -1.0
        return normalize(vector)


def normalize(vector) -> "array[float]":
    norm = math.sqrt(sum(x * x for x in vector))
    return array("f", (x / norm for x in vector) if norm else vector)


def cosine(a, b) -> float:
    """Dot product of two already-normalised vectors."""
    return sum(x * y for x, y in zip(a, b))


def to_blob(vector) -> bytes:
    return array("f", vector).tobytes()


def from_blob(blob: bytes) -> "array[float]":
    vector = array("f")
    vector.frombytes(blob)
    return vector
