import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx
import pytest

from scene_generator.brief import SceneBrief
from scene_generator.checkpoint import State
from scene_generator.config import LLMConfig
from scene_generator.llm import LLM, LLMError
from scene_generator.llm_transport import strict_schema
from scene_generator.logging import Log
from scene_generator.models import RoomDesign

VALID = {"furnishing": "gallery", "centerpiece": "vase", "density": 2, "wall_art": True}


def chunk(content=None, finish=None, **extra):
    delta = {"content": content} if content is not None else {}
    return (
        b"data: "
        + json.dumps({"choices": [{"index": 0, "delta": {**delta, **extra}, "finish_reason": finish}]}).encode()
        + b"\r\n\r\n"
    )


def answer(content=None):
    return chunk(json.dumps(VALID) if content is None else content) + chunk(finish="stop")


class Body(httpx.SyncByteStream):
    def __init__(self, parts, failure=None):
        self.parts, self.failure, self.closed = parts, failure, False

    def __iter__(self):
        yield from self.parts
        if self.failure:
            raise self.failure("private provider text")

    def close(self):
        self.closed = True


@pytest.fixture
def make_client(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "private-test-key")
    clients = []

    def create(handler, **config):
        state = State(tmp_path / "state.sqlite")
        cfg = LLMConfig(requests_per_minute=600000, **config)
        llm = LLM(cfg, state, "test", tmp_path / "cache", httpx.MockTransport(handler), Log(tmp_path / "events.jsonl"))
        clients.append(llm)
        return llm

    yield create
    for llm in clients:
        llm.close()
        llm.state.close()


def test_sse_boundaries_usage_and_zero_token_cache(make_client, tmp_path):
    usage = b'data: {"choices":[],"usage":{"prompt_tokens":17,"completion_tokens":20,"total_tokens":37}}\n\n'
    data = (
        b": heartbeat\r\n\r\n" + chunk(reasoning_content="private reasoning") + answer() + usage + b"data: [DONE]\n\n"
    )
    body = Body([data[i : i + 7] for i in range(0, len(data), 7)])
    client = make_client(
        lambda _: httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=body), retries=0
    )
    assert client.request(RoomDesign, {}, "s") == RoomDesign.model_validate(VALID)
    assert body.closed
    client.request(RoomDesign, {}, "s")
    rows = client.state.rows("SELECT * FROM llm_calls")
    assert rows[0]["total_tokens"] == 37 and rows[1]["cache_hit"] == 1 and rows[1]["total_tokens"] == 0
    log = (tmp_path / "events.jsonl").read_text()
    assert "reasoning_chars" in log and "private reasoning" not in log and "private-test-key" not in log


@pytest.mark.parametrize("failure", [httpx.RemoteProtocolError, httpx.ReadTimeout, None])
def test_partial_stream_retried_without_changing_prompt(make_client, monkeypatch, failure):
    monkeypatch.setattr("scene_generator.llm.time.sleep", lambda _: None)
    calls, bodies = [], []

    def handle(request):
        calls.append(json.loads(request.content))
        body = Body([chunk('{"furnishing":')], failure) if len(calls) == 1 else Body([answer()])
        bodies.append(body)
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=body)

    client = make_client(handle, retries=1)
    assert client.request(RoomDesign, {}, "s") == RoomDesign.model_validate(VALID)
    assert len(calls) == 2 and calls[0] == calls[1]
    assert all(b.closed for b in bodies)


@pytest.mark.parametrize("valid", [True, False])
def test_disconnect_after_stop_requires_valid_complete_answer(make_client, valid):
    body = Body([answer() if valid else answer('{"bad":true}')], httpx.RemoteProtocolError)
    client = make_client(
        lambda _: httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=body), retries=0
    )
    if valid:
        assert client.request(RoomDesign, {}, "s").density == 2
    else:
        with pytest.raises(LLMError, match="ValidationError"):
            client.request(RoomDesign, {}, "s")
        assert not list(client.cache.glob("*.json"))
    assert body.closed


@pytest.mark.parametrize("reason", ["length", "content_filter"])
def test_terminal_generation_failures_do_not_retry(make_client, reason):
    calls = []

    def handle(request):
        calls.append(request)
        return httpx.Response(
            200, headers={"content-type": "text/event-stream"}, stream=Body([chunk("{}"), chunk(finish=reason)])
        )

    client = make_client(handle, retries=3)
    with pytest.raises(LLMError):
        client.request(RoomDesign, {}, "s")
    assert len(calls) == 1 and not list(client.cache.glob("*.json"))


@pytest.mark.parametrize("error", [httpx.PoolTimeout, httpx.LocalProtocolError, httpx.UnsupportedProtocol])
def test_local_http_failures_are_not_retried(make_client, error):
    calls = []

    def handle(request):
        calls.append(request)
        raise error("private provider text")

    client = make_client(handle, retries=3)
    with pytest.raises(LLMError, match="local HTTP") as caught:
        client.request(RoomDesign, {}, "s")
    assert "private provider text" not in str(caught.value)
    assert len(calls) == 1


@pytest.mark.parametrize("status,attempts", [(429, 2), (408, 2), (503, 2), (400, 1), (401, 1), (501, 1)])
def test_http_retry_classification(make_client, monkeypatch, status, attempts):
    delays = []
    monkeypatch.setattr("scene_generator.llm.time.sleep", delays.append)
    calls = []

    def handle(request):
        calls.append(request)
        if len(calls) == 1:
            return httpx.Response(status, headers={"retry-after": "2"}, text="private provider body")
        return httpx.Response(
            200, json={"choices": [{"finish_reason": "stop", "message": {"content": json.dumps(VALID)}}]}
        )

    client = make_client(handle, retries=1)
    if attempts == 2:
        assert client.request(RoomDesign, {}, "s").density == 2
        assert 2 in delays
    else:
        with pytest.raises(LLMError, match=str(status)):
            client.request(RoomDesign, {}, "s")
    assert len(calls) == attempts


def test_serialization_and_cache_write_errors_are_local(make_client, monkeypatch):
    calls = []

    def handle(request):
        calls.append(request)
        return httpx.Response(
            200, json={"choices": [{"finish_reason": "stop", "message": {"content": json.dumps(VALID)}}]}
        )

    client = make_client(handle, retries=3)
    with pytest.raises(TypeError):
        client.request(RoomDesign, {"bad": object()}, "s")
    assert not calls

    def broken_write(*args):
        raise ValueError("local cache error")

    monkeypatch.setattr("scene_generator.llm.write_json", broken_write)
    with pytest.raises(ValueError, match="local cache error"):
        client.request(RoomDesign, {}, "s")
    assert len(calls) == 1


def test_output_settings_invalidate_cache_and_cap_can_be_disabled(make_client):
    calls = []

    def handle(request):
        calls.append(json.loads(request.content))
        return httpx.Response(
            200, json={"choices": [{"finish_reason": "stop", "message": {"content": json.dumps(VALID)}}]}
        )

    client = make_client(handle, retries=0)
    client.request(RoomDesign, {}, "s")
    client.config.temperature = 0.9
    client.request(RoomDesign, {}, "s")
    client.config.max_tokens = None
    client.config.stream = False
    client.request(RoomDesign, {}, "s")
    assert len(calls) == 3 and "max_tokens" not in calls[-1] and "stream_options" not in calls[-1]


def test_strict_schema_preserves_named_fields_and_normalizes_tuples():
    schema = strict_schema(SceneBrief.model_json_schema())
    assert "title" in schema["properties"] and "title" in schema["required"]

    def check(node):
        if isinstance(node, list):
            for value in node:
                check(value)
        elif isinstance(node, dict):
            assert "prefixItems" not in node
            if node.get("type") == "object":
                assert node["additionalProperties"] is False
                assert set(node["required"]) == set(node["properties"])
            for value in node.values():
                check(value)

    check(schema)
    from scene_generator.recipes import ObjectRecipe

    check(strict_schema(ObjectRecipe.model_json_schema()))


def test_heartbeat_cannot_extend_total_deadline(make_client):
    class Heartbeat(httpx.SyncByteStream):
        def __iter__(self):
            for _ in range(10):
                time.sleep(0.01)
                yield b": keep-alive\n\n"

    client = make_client(
        lambda _: httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=Heartbeat()),
        retries=3,
        total_timeout=0.025,
    )
    with pytest.raises(LLMError, match="total_deadline"):
        client.request(RoomDesign, {}, "s")
    assert len(client.state.rows("SELECT * FROM llm_calls")) == 1


def test_byte_limit_and_cancellation_do_not_retry(make_client):
    client = make_client(lambda _: httpx.Response(200, content=b" " * 2048), retries=3, max_response_bytes=1024)
    with pytest.raises(LLMError, match="byte_limit"):
        client.request(RoomDesign, {}, "s")
    assert len(client.state.rows("SELECT * FROM llm_calls")) == 1

    def interrupted(_):
        raise KeyboardInterrupt()

    client = make_client(interrupted, retries=3)
    with pytest.raises(KeyboardInterrupt):
        client.request(RoomDesign, {}, "s")


def test_real_http_stream_outlives_idle_timeout(tmp_path, monkeypatch):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            pieces = [chunk(reasoning_content="private")] * 5 + [answer(), b"data: [DONE]\n\n"]
            for piece in pieces:
                self.wfile.write(piece)
                self.wfile.flush()
                time.sleep(0.04)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    monkeypatch.setenv("OPENAI_BASE_URL", f"http://127.0.0.1:{server.server_port}")
    state = State(tmp_path / "state.sqlite")
    client = LLM(LLMConfig(timeout=0.15, total_timeout=2, retries=0), state, "test", tmp_path / "cache")
    try:
        start = time.monotonic()
        assert client.request(RoomDesign, {}, "s").density == 2
        assert time.monotonic() - start > 0.15
    finally:
        client.close()
        state.close()
        server.shutdown()
        server.server_close()
        thread.join()
