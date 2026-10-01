import unittest

from engram import MemoryStore

DAY = 86400.0


class MemoryStoreTest(unittest.TestCase):
    def setUp(self):
        self.store = MemoryStore(":memory:")
        self.addCleanup(self.store.close)

    def test_search_ranks_relevant_memory_first(self):
        self.store.add("The user prefers dark mode in every editor", now=0)
        self.store.add("The project deploys to Fly.io from the main branch", now=0)
        self.store.add("The user's dog is called Biscuit", now=0)

        hits = self.store.search("where does the project deploy", now=0)

        self.assertEqual(hits[0].memory.text, "The project deploys to Fly.io from the main branch")

    def test_search_matches_across_word_endings(self):
        self.store.add("The project deploys to Fly.io", now=0)
        self.assertEqual(len(self.store.search("deploy", now=0)), 1)
        self.assertEqual(len(self.store.search("deploying", now=0)), 1)

    def test_search_without_overlap_returns_nothing(self):
        self.store.add("The user prefers dark mode", now=0)
        self.assertEqual(self.store.search("kubernetes cluster", now=0), [])

    def test_near_duplicate_reinforces_instead_of_inserting(self):
        first = self.store.add("The user prefers tabs over spaces", now=0)
        second = self.store.add("the user prefers tabs over spaces.", now=10)

        self.assertEqual(first.id, second.id)
        self.assertEqual(second.uses, 1)
        self.assertEqual(len(self.store.all()), 1)

    def test_recalled_memories_outrank_unused_ones(self):
        used = self.store.add("Staging database runs Postgres 16", now=0)
        self.store.add("Production database runs Postgres 15", now=0)
        for _ in range(5):
            self.store.search("staging", now=0)

        hits = self.store.search("database postgres", reinforce=False, now=0)

        self.assertEqual(hits[0].memory.id, used.id)

    def test_peek_does_not_reinforce(self):
        memory = self.store.add("The API key rotates monthly", now=0)
        self.store.search("api key", reinforce=False, now=0)
        self.assertEqual(self.store.get(memory.id).uses, 0)

    def test_prune_drops_stale_memories_first(self):
        stale = self.store.add("Old sprint goal was the billing rewrite", now=0)
        fresh = self.store.add("Current sprint goal is the search index", now=200 * DAY)

        dropped = self.store.prune(keep=1, now=200 * DAY)

        self.assertEqual(dropped, 1)
        self.assertEqual([m.id for m in self.store.all()], [fresh.id])
        with self.assertRaises(KeyError):
            self.store.get(stale.id)

    def test_forget(self):
        memory = self.store.add("Temporary note", now=0)
        self.assertTrue(self.store.forget(memory.id))
        self.assertFalse(self.store.forget(memory.id))

    def test_empty_text_is_rejected(self):
        with self.assertRaises(ValueError):
            self.store.add("   ")


if __name__ == "__main__":
    unittest.main()
