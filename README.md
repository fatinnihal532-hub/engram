# engram

Long-term memory for LLM agents that gets better the more it is used.

Most agent memory is a pile of notes with a search box. `engram` tracks how often each
memory is actually recalled: memories that keep proving useful rise in the ranking, and
ones nobody asks for fade and are pruned first.

- **Zero dependencies for the store** – SQLite and BM25 from the Python standard library.
- **Learns from use** – every recall reinforces the memories it returns.
- **Deduplicates** – storing a near-identical fact strengthens the existing one.
- **Forgets on purpose** – `prune` keeps only the strongest N memories.
- **Claude agent included** – a chat loop where Claude decides what to remember, recall and forget.

## Install

```bash
pip install -r requirements.txt
```

Only the `chat` command needs the Anthropic SDK. Everything else runs on plain Python 3.10+.

## Use the store

```python
from engram import MemoryStore

store = MemoryStore("engram.db")
store.add("The user prefers dark mode", kind="preference")
store.add("The project deploys to Fly.io from main", importance=2.0)

for hit in store.search("where do we deploy?"):
    print(hit.score, hit.memory.text)
```

## Command line

```bash
python -m engram add "The project deploys to Fly.io from main"
python -m engram search "deploy"
python -m engram list
python -m engram prune 100
```

## Chat with memory

Set `ANTHROPIC_API_KEY`, then:

```bash
python -m engram chat
```

Tell it something in one session, quit, start a new session and ask about it.

## How ranking works

```
score = BM25(query, memory) × importance × (1 + ln(1 + uses)) × (0.5 + 0.5 × recency)
```

`recency` halves every 30 days since the memory was last used, so the recency factor never
drops below 0.5 – old memories get weaker but a strong keyword match can still surface them.

## Tests

```bash
python -m unittest discover -s tests
```
