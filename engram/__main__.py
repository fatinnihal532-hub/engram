"""Command line interface: python -m engram <command>"""

from __future__ import annotations

import argparse
import sys

from .store import MemoryStore


def cmd_chat(store: MemoryStore, _args: argparse.Namespace) -> int:
    try:
        import anthropic
    except ImportError:
        print("The chat command needs the Anthropic SDK: pip install anthropic", file=sys.stderr)
        return 1
    from .agent import MemoryAgent

    agent = MemoryAgent(store)
    print("Chatting with memory. Ctrl+C or an empty line to quit.")
    while True:
        try:
            line = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not line:
            return 0
        try:
            print(f"claude> {agent.send(line)}")
        except anthropic.AuthenticationError:
            print("No valid credentials. Set ANTHROPIC_API_KEY and try again.", file=sys.stderr)
            return 1
        except anthropic.RateLimitError:
            print("Rate limited. Wait a moment and try again.", file=sys.stderr)
        except anthropic.APIStatusError as e:
            print(f"API error {e.status_code}: {e.message}", file=sys.stderr)
        except anthropic.APIConnectionError:
            print("Network error. Check your connection.", file=sys.stderr)


def cmd_add(store: MemoryStore, args: argparse.Namespace) -> int:
    memory = store.add(args.text, kind=args.kind, importance=args.importance)
    print(f"#{memory.id} stored")
    return 0


def cmd_search(store: MemoryStore, args: argparse.Namespace) -> int:
    hits = store.search(args.query, k=args.limit, reinforce=not args.peek)
    for hit in hits:
        print(f"{hit.score:6.2f}  #{hit.memory.id} [{hit.memory.kind}] {hit.memory.text}")
    if not hits:
        print("No matching memories.")
    return 0


def cmd_list(store: MemoryStore, _args: argparse.Namespace) -> int:
    for m in store.all():
        print(f"#{m.id} [{m.kind}] uses={m.uses} {m.text}")
    return 0


def cmd_forget(store: MemoryStore, args: argparse.Namespace) -> int:
    if store.forget(args.id):
        print("Deleted.")
        return 0
    print(f"No memory with id {args.id}.", file=sys.stderr)
    return 1


def cmd_prune(store: MemoryStore, args: argparse.Namespace) -> int:
    print(f"Dropped {store.prune(args.keep)} memories.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="engram", description="Long-term memory for LLM agents.")
    parser.add_argument("--db", default="engram.db", help="path to the SQLite file")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("chat", help="chat with Claude using persistent memory").set_defaults(fn=cmd_chat)

    p = sub.add_parser("add", help="store a memory")
    p.add_argument("text")
    p.add_argument("--kind", default="fact")
    p.add_argument("--importance", type=float, default=1.0)
    p.set_defaults(fn=cmd_add)

    p = sub.add_parser("search", help="search memories")
    p.add_argument("query")
    p.add_argument("--limit", type=int, default=5)
    p.add_argument("--peek", action="store_true", help="do not reinforce the results")
    p.set_defaults(fn=cmd_search)

    sub.add_parser("list", help="list all memories").set_defaults(fn=cmd_list)

    p = sub.add_parser("forget", help="delete a memory by id")
    p.add_argument("id", type=int)
    p.set_defaults(fn=cmd_forget)

    p = sub.add_parser("prune", help="keep only the N strongest memories")
    p.add_argument("keep", type=int)
    p.set_defaults(fn=cmd_prune)

    args = parser.parse_args(argv)
    store = MemoryStore(args.db)
    try:
        return args.fn(store, args)
    finally:
        store.close()


if __name__ == "__main__":
    sys.exit(main())
