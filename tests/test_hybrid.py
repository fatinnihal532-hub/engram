import json
import math
import sqlite3
import tempfile
import unittest
from pathlib import Path

from engram import MemoryStore
from engram.embed import HashEmbedder, cosine, from_blob, to_blob
from engram.evaluate import evaluate

BENCHMARK = Path(__file__).parent.parent / "benchmarks" / "retrieval.json"


class EmbedderTest(unittest.TestCase):
    def setUp(self):
        self.embed = HashEmbedder()

    def test_vectors_are_unit_length_and_deterministic(self):
        vector = self.embed("deployment pipeline")
        self.assertAlmostEqual(math.sqrt(sum(x * x for x in vector)), 1.0, places=5)
        self.assertEqual(list(vector), list(self.embed("deployment pipeline")))

    def test_misspelling_stays_close_and_unrelated_word_stays_far(self):
        word = self.embed("kubernetes")
        self.assertGreater(cosine(word, self.embed("kuberntes")), 0.5)
        self.assertLess(cosine(word, self.embed("invoice")), 0.2)

    def test_empty_text_gives_a_zero_vector(self):
        self.assertEqual(set(self.embed("")), {0.0})

    def test_blob_round_trip(self):
        vector = self.embed("round trip")
        self.assertEqual(list(from_blob(to_blob(vector))), list(vector))


class HybridSearchTest(unittest.TestCase):
    def setUp(self):
        self.store = MemoryStore(":memory:")
        self.addCleanup(self.store.close)
        self.target = self.store.add("The Kubernetes cluster has three worker nodes", now=0)
        self.store.add("Invoices are generated on the first day of each month", now=0)
        self.store.add("The user's dog is called Biscuit", now=0)

    def test_typo_is_missed_by_keywords_but_found_by_vectors(self):
        query = "kuberntes clustr"
        self.assertEqual(self.store.search(query, mode="bm25", now=0), [])
        for mode in ("vector", "hybrid"):
            hits = self.store.search(query, mode=mode, reinforce=False, now=0)
            self.assertEqual(hits[0].memory.id, self.target.id, mode)

    def test_unrelated_query_returns_nothing_in_every_mode(self):
        for mode in ("bm25", "vector", "hybrid"):
            self.assertEqual(self.store.search("quarterly revenue forecast", mode=mode, now=0), [], mode)

    def test_unknown_mode_is_rejected(self):
        with self.assertRaises(ValueError):
            self.store.search("anything", mode="psychic")

    def test_custom_embedder_is_used(self):
        # Two-dimensional toy embedder: anything mentioning "cat" points one way, the rest another.
        store = MemoryStore(":memory:", embedder=lambda t: [1.0, 0.0] if "cat" in t else [0.0, 1.0])
        self.addCleanup(store.close)
        cat = store.add("The cat sleeps on the radiator", now=0)
        store.add("The server sleeps at midnight", now=0)

        hits = store.search("cat", mode="vector", now=0)

        self.assertEqual([h.memory.id for h in hits], [cat.id])


class MigrationTest(unittest.TestCase):
    def test_database_from_before_vector_search_is_upgraded(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "old.db"
            db = sqlite3.connect(path)
            db.execute("""CREATE TABLE memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT, text TEXT NOT NULL,
                kind TEXT NOT NULL DEFAULT 'fact', importance REAL NOT NULL DEFAULT 1.0,
                uses INTEGER NOT NULL DEFAULT 0, created_at REAL NOT NULL, last_used REAL NOT NULL)""")
            db.execute("INSERT INTO memories (text, created_at, last_used) VALUES (?, 0, 0)",
                       ("The Kubernetes cluster has three worker nodes",))
            db.commit()
            db.close()

            store = MemoryStore(path)
            try:
                hits = store.search("kuberntes", mode="vector", now=0)
                self.assertEqual(len(hits), 1)
                stored = store.db.execute("SELECT embedding FROM memories").fetchone()[0]
                self.assertIsNotNone(stored)
            finally:
                store.close()


class BenchmarkTest(unittest.TestCase):
    """Guards the numbers quoted in the README against regressions."""

    @classmethod
    def setUpClass(cls):
        benchmark = json.loads(BENCHMARK.read_text(encoding="utf-8"))
        cls.scores = {mode: evaluate(benchmark, mode) for mode in ("bm25", "hybrid")}

    def test_every_mode_solves_exact_queries(self):
        for mode, scores in self.scores.items():
            self.assertEqual(scores["exact"].recall_at_1, 1.0, mode)

    def test_hybrid_beats_keywords_on_typos_and_overall(self):
        self.assertGreater(self.scores["hybrid"]["typo"].recall_at_1,
                           self.scores["bm25"]["typo"].recall_at_1 + 0.3)
        self.assertGreater(self.scores["hybrid"]["all"].mrr, self.scores["bm25"]["all"].mrr)

    def test_hybrid_is_never_worse_than_keywords_on_any_kind(self):
        for kind, score in self.scores["hybrid"].items():
            self.assertGreaterEqual(score.mrr, self.scores["bm25"][kind].mrr, kind)


if __name__ == "__main__":
    unittest.main()
