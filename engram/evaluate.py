"""Measure retrieval quality on a labelled benchmark.

    python -m engram.evaluate benchmarks/retrieval.json

Each query has exactly one correct memory. Reported per search mode and query kind:
recall@1, recall@3 and mean reciprocal rank (MRR, the average of 1/rank of the correct
memory, counting 0 when it is not in the top results).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from .store import MODES, MemoryStore

DEPTH = 10


@dataclass(frozen=True)
class Score:
    queries: int
    recall_at_1: float
    recall_at_3: float
    mrr: float


def rank_of(store: MemoryStore, query: str, expected_id: int, mode: str) -> int | None:
    hits = store.search(query, k=DEPTH, reinforce=False, now=0, mode=mode)
    for rank, hit in enumerate(hits, start=1):
        if hit.memory.id == expected_id:
            return rank
    return None


def evaluate(benchmark: dict, mode: str, embedder=None) -> dict[str, Score]:
    """Return a Score per query kind, plus "all"."""
    store = MemoryStore(":memory:", embedder=embedder)
    try:
        ids = [store.add(text, now=0).id for text in benchmark["memories"]]
        ranks: dict[str, list[int | None]] = defaultdict(list)
        for item in benchmark["queries"]:
            rank = rank_of(store, item["query"], ids[item["expect"]], mode)
            ranks[item["kind"]].append(rank)
            ranks["all"].append(rank)
    finally:
        store.close()

    return {
        kind: Score(
            queries=len(values),
            recall_at_1=sum(r == 1 for r in values) / len(values),
            recall_at_3=sum(r is not None and r <= 3 for r in values) / len(values),
            mrr=sum(1 / r for r in values if r) / len(values),
        )
        for kind, values in ranks.items()
    }


def render(results: dict[str, dict[str, Score]]) -> str:
    kinds = [k for k in next(iter(results.values())) if k != "all"] + ["all"]
    lines = ["| Query kind | Queries | Mode | Recall@1 | Recall@3 | MRR |", "|---|--:|---|--:|--:|--:|"]
    for kind in kinds:
        for mode, scores in results.items():
            s = scores[kind]
            lines.append(f"| {kind} | {s.queries} | {mode} | {s.recall_at_1:.0%} | "
                         f"{s.recall_at_3:.0%} | {s.mrr:.2f} |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="engram.evaluate")
    parser.add_argument("benchmark", type=Path)
    args = parser.parse_args(argv)
    benchmark = json.loads(args.benchmark.read_text(encoding="utf-8"))
    print(render({mode: evaluate(benchmark, mode) for mode in MODES}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
