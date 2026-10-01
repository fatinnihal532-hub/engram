"""A Claude chat agent that reads and writes long-term memory through tools."""

from __future__ import annotations

import anthropic
from anthropic import beta_tool

from .store import MemoryStore

MODEL = "claude-opus-5-5"

SYSTEM = """You are an assistant with long-term memory that persists across conversations.

Before answering anything that could depend on what you know about the user or earlier \
sessions, call `recall` with a short search query. When the user tells you something worth \
keeping (a preference, a fact about them or their project, a decision, a correction), call \
`remember` with one self-contained sentence. If the user says a stored memory is wrong or \
asks you to drop it, call `forget` with its id. Do not store secrets such as passwords or \
API keys, and do not store small talk."""


def build_tools(store: MemoryStore) -> list:
    @beta_tool
    def remember(text: str, kind: str = "fact", importance: float = 1.0) -> str:
        """Save one fact to long-term memory.

        Args:
            text: A single self-contained sentence stating the fact.
            kind: One of "fact", "preference", "decision" or "correction".
            importance: Weight from 0.5 (minor) to 3.0 (critical). Default 1.0.
        """
        memory = store.add(text, kind=kind, importance=importance)
        return f"Stored as memory #{memory.id}."

    @beta_tool
    def recall(query: str, limit: int = 5) -> str:
        """Search long-term memory for facts relevant to a query.

        Args:
            query: Keywords describing what you want to know.
            limit: Maximum number of memories to return.
        """
        hits = store.search(query, k=limit)
        if not hits:
            return "No matching memories."
        return "\n".join(f"#{h.memory.id} [{h.memory.kind}] {h.memory.text}" for h in hits)

    @beta_tool
    def forget(memory_id: int) -> str:
        """Delete a memory that is wrong or no longer wanted.

        Args:
            memory_id: The numeric id shown by recall, without the # sign.
        """
        return "Deleted." if store.forget(memory_id) else f"No memory with id {memory_id}."

    return [remember, recall, forget]


class MemoryAgent:
    def __init__(self, store: MemoryStore, client: anthropic.Anthropic | None = None):
        self.client = client or anthropic.Anthropic()
        self.tools = build_tools(store)
        self.messages: list[dict] = []

    def send(self, user_message: str) -> str:
        self.messages.append({"role": "user", "content": user_message})
        runner = self.client.beta.messages.tool_runner(
            model=MODEL,
            max_tokens=16000,
            system=SYSTEM,
            tools=self.tools,
            messages=self.messages,
        )
        last = None
        for message in runner:
            last = message
            # The runner keeps its own history; mirror it so the next turn has full context.
            self.messages.append({"role": "assistant", "content": message.content})
            tool_response = runner.generate_tool_call_response()
            if tool_response is not None:
                self.messages.append(tool_response)

        if last is None:
            return ""
        if last.stop_reason == "refusal":
            return "[The model declined to answer this request.]"
        return "".join(b.text for b in last.content if b.type == "text")
