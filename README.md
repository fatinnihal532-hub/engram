# engram

[![tests](https://github.com/fatinnihal532-hub/engram/actions/workflows/tests.yml/badge.svg)](https://github.com/fatinnihal532-hub/engram/actions/workflows/tests.yml)

Long-term memory for LLM agents that gets better the more it is used.

Most agent memory is a pile of notes with a search box. `engram` tracks how often each
memory is actually recalled: memories that keep proving useful rise in the ranking, and
ones nobody asks for fade and are pruned first.

- **Hybrid retrieval** – BM25 keyword search blended with vector similarity, so a misspelt
  or differently inflected query still finds the right memory.
- **Learns from use** – every recall reinforces the memories it returns.
- **Measured** – a labelled benchmark reports recall@k and MRR for each search mode.
- **MCP server** – plug the memory into Claude Code or any other Model Context Protocol client.
- **Zero dependencies** for the store, the embeddings and the MCP server: Python standard
  library only. The optional chat demo uses the Anthropic SDK.
- **Deduplicates and forgets** – near-identical facts strengthen the existing memory, and
  `prune` keeps only the strongest N.

## Benchmark

[benchmarks/retrieval.json](benchmarks/retrieval.json) holds 30 memories and 40 queries, each
with one correct answer, in four kinds: exact keywords, different word endings, typos, and
paraphrases that share no words with the memory.

```bash
python -m engram.evaluate benchmarks/retrieval.json
```

| Query kind | Queries | Mode | Recall@1 | Recall@3 | MRR |
|---|--:|---|--:|--:|--:|
| exact | 10 | bm25 | 100% | 100% | 1.00 |
| exact | 10 | vector | 100% | 100% | 1.00 |
| exact | 10 | **hybrid** | 100% | 100% | 1.00 |
| inflected | 10 | bm25 | 80% | 80% | 0.80 |
| inflected | 10 | vector | 80% | 80% | 0.80 |
| inflected | 10 | **hybrid** | 100% | 100% | 1.00 |
| typo | 10 | bm25 | 20% | 20% | 0.20 |
| typo | 10 | vector | 60% | 60% | 0.60 |
| typo | 10 | **hybrid** | 70% | 70% | 0.70 |
| paraphrase | 10 | bm25 | 10% | 20% | 0.15 |
| paraphrase | 10 | vector | 10% | 20% | 0.16 |
| paraphrase | 10 | **hybrid** | 10% | 30% | 0.18 |
| **all** | 40 | bm25 | 52% | 55% | 0.54 |
| **all** | 40 | vector | 62% | 65% | 0.64 |
| **all** | 40 | **hybrid** | 70% | 75% | 0.72 |

What this shows:

- Hybrid is the best mode overall and is never worse than keywords alone on any query kind.
- **Paraphrases are not solved.** The built-in embeddings compare spelling, not meaning, so
  "name of the pet" does not find "the user's dog is called Biscuit". Fixing that needs a
  learned embedding model, which you can plug in (see below).
- The benchmark is small and written by the author of the code, so treat the numbers as a
  comparison between modes, not as an absolute score.

## Use the store

```python
from engram import MemoryStore

store = MemoryStore("engram.db")
store.add("The user prefers dark mode", kind="preference")
store.add("The project deploys to Fly.io from main", importance=2.0)

for hit in store.search("where do we deploy?"):
    print(hit.score, hit.memory.text)
```

`search` takes `mode="hybrid"` (default), `"bm25"` or `"vector"`.

### Plugging in a real embedding model

Pass any function that turns text into a list of numbers:

```python
store = MemoryStore("engram.db", embedder=my_model.encode)
```

Use one embedder per database file: stored vectors are not recomputed when it changes.

## Use it from Claude Code (MCP)

```bash
claude mcp add engram -- python -m engram.mcp_server --db ~/engram.db
```

Run that from this folder, or install the package first. The server exposes three tools,
`remember`, `recall` and `forget`, over stdio. It is a from-scratch implementation of the
protocol's tool subset (`initialize`, `ping`, `tools/list`, `tools/call`) in
[engram/mcp_server.py](engram/mcp_server.py).

## Command line

```bash
python -m engram add "The project deploys to Fly.io from main"
python -m engram search "deploy" --mode hybrid
python -m engram list
python -m engram prune 100
```

## Chat demo

```bash
pip install -r requirements.txt
python -m engram chat
```

Needs `ANTHROPIC_API_KEY`. Claude decides when to call `remember`, `recall` and `forget`.
Tell it something in one session, quit, start a new one and ask about it.

## How ranking works

```
keyword   = BM25(query, memory) / best BM25 score for this query
vector    = cosine(query, memory) / best cosine for this query     (ignored below 0.25)
relevance = 0.5 × keyword + 0.5 × vector
score     = relevance × importance × (1 + ln(1 + uses)) × (0.5 + 0.5 × recency)
```

`recency` halves every 30 days since the memory was last used, so that factor never drops
below 0.5: old memories get weaker, but a strong match can still surface them.

The default embeddings ([engram/embed.py](engram/embed.py)) hash each word's character
3- to 5-grams into a 512-dimensional vector, the idea behind fastText's subword vectors.
Words that share most of their letters end up close together.

## Tests

```bash
python -m unittest discover -s tests
```

30 tests: the store, hybrid search, upgrading an older database file, the MCP server (in
process and as a real subprocess over stdio), and a guard that the benchmark numbers above
do not regress.

## Not verified

The chat demo ([engram/agent.py](engram/agent.py)) follows the Anthropic SDK's documented
tool-runner pattern but has not been run against the live API.
