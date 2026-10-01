import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from engram import MemoryStore
from engram.mcp_server import METHOD_NOT_FOUND, PARSE_ERROR, handle, serve

ROOT = Path(__file__).parent.parent


def request(method, params=None, id=1):
    message = {"jsonrpc": "2.0", "id": id, "method": method}
    if params is not None:
        message["params"] = params
    return message


def call(name, arguments, id=1):
    return request("tools/call", {"name": name, "arguments": arguments}, id=id)


class HandleTest(unittest.TestCase):
    def setUp(self):
        self.store = MemoryStore(":memory:")
        self.addCleanup(self.store.close)

    def text(self, response):
        return response["result"]["content"][0]["text"]

    def test_initialize_echoes_the_client_protocol_version(self):
        response = handle(self.store, request("initialize", {"protocolVersion": "2024-11-05"}))
        self.assertEqual(response["result"]["protocolVersion"], "2024-11-05")
        self.assertIn("tools", response["result"]["capabilities"])
        self.assertEqual(response["result"]["serverInfo"]["name"], "engram")

    def test_notifications_get_no_response(self):
        self.assertIsNone(handle(self.store, {"jsonrpc": "2.0", "method": "notifications/initialized"}))

    def test_tools_list_has_valid_schemas(self):
        tools = handle(self.store, request("tools/list"))["result"]["tools"]
        self.assertEqual([t["name"] for t in tools], ["remember", "recall", "forget"])
        for tool in tools:
            self.assertEqual(tool["inputSchema"]["type"], "object")
            self.assertTrue(set(tool["inputSchema"]["required"]) <= set(tool["inputSchema"]["properties"]))

    def test_remember_recall_forget_round_trip(self):
        stored = handle(self.store, call("remember", {"text": "The staging database runs Postgres 16"}))
        self.assertEqual(self.text(stored), "Stored as memory #1.")

        found = handle(self.store, call("recall", {"query": "postgres"}))
        self.assertIn("#1 [fact] The staging database runs Postgres 16", self.text(found))
        self.assertFalse(found["result"]["isError"])

        self.assertEqual(self.text(handle(self.store, call("forget", {"memory_id": 1}))), "Deleted.")
        self.assertEqual(self.text(handle(self.store, call("recall", {"query": "postgres"}))),
                         "No matching memories.")

    def test_bad_arguments_come_back_as_a_tool_error_not_a_protocol_error(self):
        response = handle(self.store, call("remember", {"text": "   "}))
        self.assertTrue(response["result"]["isError"])
        self.assertNotIn("error", response)

    def test_unknown_tool_and_unknown_method_are_protocol_errors(self):
        self.assertIn("error", handle(self.store, call("teleport", {})))
        response = handle(self.store, request("resources/list"))
        self.assertEqual(response["error"]["code"], METHOD_NOT_FOUND)

    def test_response_carries_the_request_id(self):
        self.assertEqual(handle(self.store, request("ping", id="abc"))["id"], "abc")


class ServeTest(unittest.TestCase):
    def test_serve_survives_garbage_and_blank_lines(self):
        store = MemoryStore(":memory:")
        self.addCleanup(store.close)
        stdin = io.StringIO("not json\n\n" + json.dumps(request("ping", id=7)) + "\n")
        stdout = io.StringIO()

        serve(store, stdin, stdout)

        first, second = [json.loads(line) for line in stdout.getvalue().splitlines()]
        self.assertEqual(first["error"]["code"], PARSE_ERROR)
        self.assertEqual(second, {"jsonrpc": "2.0", "id": 7, "result": {}})

    def test_real_process_speaks_the_protocol_over_stdio(self):
        messages = [
            request("initialize", {"protocolVersion": "2025-06-18", "capabilities": {},
                                   "clientInfo": {"name": "test", "version": "0"}}, id=1),
            {"jsonrpc": "2.0", "method": "notifications/initialized"},
            call("remember", {"text": "The user prefers dark mode", "kind": "preference"}, id=2),
            call("recall", {"query": "dark mode"}, id=3),
        ]
        with tempfile.TemporaryDirectory() as folder:
            result = subprocess.run(
                [sys.executable, "-m", "engram.mcp_server", "--db", str(Path(folder) / "m.db")],
                input="".join(json.dumps(m) + "\n" for m in messages),
                capture_output=True, text=True, cwd=ROOT, timeout=30,
            )

        self.assertEqual(result.returncode, 0, result.stderr)
        responses = [json.loads(line) for line in result.stdout.splitlines()]
        self.assertEqual([r["id"] for r in responses], [1, 2, 3])
        self.assertIn("[preference] The user prefers dark mode",
                      responses[2]["result"]["content"][0]["text"])


if __name__ == "__main__":
    unittest.main()
