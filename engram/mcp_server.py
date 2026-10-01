"""Model Context Protocol server: exposes the memory store to any MCP client over stdio.

    python -m engram.mcp_server --db ~/engram.db

Implements the parts of MCP a tool server needs (initialize, ping, tools/list, tools/call)
as newline-delimited JSON-RPC 2.0, using only the standard library.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import IO

from .store import MemoryStore

PROTOCOL_VERSION = "2025-06-18"
SERVER_INFO = {"name": "engram", "version": "0.2.0"}

PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602

TOOLS = [
    {
        "name": "remember",
        "description": "Save one fact to long-term memory. Use a single self-contained sentence.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "The fact to store."},
                "kind": {"type": "string", "enum": ["fact", "preference", "decision", "correction"]},
                "importance": {"type": "number", "minimum": 0.5, "maximum": 3.0},
            },
            "required": ["text"],
        },
    },
    {
        "name": "recall",
        "description": "Search long-term memory for facts relevant to a query.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "limit": {"type": "integer", "minimum": 1, "maximum": 20},
            },
            "required": ["query"],
        },
    },
    {
        "name": "forget",
        "description": "Delete a memory by the numeric id shown by recall.",
        "inputSchema": {
            "type": "object",
            "properties": {"memory_id": {"type": "integer"}},
            "required": ["memory_id"],
        },
    },
]


class ToolError(Exception):
    """Raised for a failure the model should see and can recover from."""


def call_tool(store: MemoryStore, name: str, arguments: dict) -> str:
    if name == "remember":
        text = arguments.get("text")
        if not isinstance(text, str) or not text.strip():
            raise ToolError("`text` must be a non-empty string.")
        memory = store.add(text, kind=arguments.get("kind", "fact"),
                           importance=float(arguments.get("importance", 1.0)))
        return f"Stored as memory #{memory.id}."
    if name == "recall":
        query = arguments.get("query")
        if not isinstance(query, str):
            raise ToolError("`query` must be a string.")
        hits = store.search(query, k=int(arguments.get("limit", 5)))
        if not hits:
            return "No matching memories."
        return "\n".join(f"#{h.memory.id} [{h.memory.kind}] {h.memory.text}" for h in hits)
    if name == "forget":
        memory_id = arguments.get("memory_id")
        if not isinstance(memory_id, int):
            raise ToolError("`memory_id` must be an integer.")
        return "Deleted." if store.forget(memory_id) else f"No memory with id {memory_id}."
    raise KeyError(name)


def handle(store: MemoryStore, message: dict) -> dict | None:
    """Process one JSON-RPC message. Returns the response, or None for a notification."""
    request_id = message.get("id")
    method = message.get("method")
    is_notification = "id" not in message

    def result(value: dict) -> dict:
        return {"jsonrpc": "2.0", "id": request_id, "result": value}

    def error(code: int, text: str) -> dict | None:
        if is_notification:
            return None
        return {"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": text}}

    if not isinstance(method, str):
        return error(INVALID_REQUEST, "Missing method.")
    if is_notification:
        return None  # e.g. notifications/initialized; nothing to answer

    params = message.get("params") or {}
    if method == "initialize":
        return result({
            "protocolVersion": params.get("protocolVersion", PROTOCOL_VERSION),
            "capabilities": {"tools": {}},
            "serverInfo": SERVER_INFO,
        })
    if method == "ping":
        return result({})
    if method == "tools/list":
        return result({"tools": TOOLS})
    if method == "tools/call":
        name = params.get("name")
        try:
            text = call_tool(store, name, params.get("arguments") or {})
        except KeyError:
            return error(INVALID_PARAMS, f"Unknown tool: {name}")
        except (ToolError, ValueError) as e:
            # Tool failures go back as a result so the model can read them and retry.
            return result({"content": [{"type": "text", "text": str(e)}], "isError": True})
        return result({"content": [{"type": "text", "text": text}], "isError": False})
    return error(METHOD_NOT_FOUND, f"Method not found: {method}")


def serve(store: MemoryStore, stdin: IO[str] = sys.stdin, stdout: IO[str] = sys.stdout) -> None:
    for line in stdin:
        line = line.strip()
        if not line:
            continue
        try:
            message = json.loads(line)
        except json.JSONDecodeError:
            response = {"jsonrpc": "2.0", "id": None,
                        "error": {"code": PARSE_ERROR, "message": "Invalid JSON."}}
        else:
            if isinstance(message, dict):
                response = handle(store, message)
            else:
                response = {"jsonrpc": "2.0", "id": None,
                            "error": {"code": INVALID_REQUEST, "message": "Expected an object."}}
        if response is not None:
            stdout.write(json.dumps(response) + "\n")
            stdout.flush()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="engram.mcp_server")
    parser.add_argument("--db", default="engram.db", help="path to the SQLite file")
    args = parser.parse_args(argv)
    store = MemoryStore(args.db)
    try:
        serve(store)
    finally:
        store.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
