import httpx
import pytest

from scene_generator.checkpoint import State
from scene_generator.config import LLMConfig
from scene_generator.llm import LLM, LLMError
from scene_generator.logging import Log
from scene_generator.models import RoomDesign

VALID = {"furnishing": "gallery", "centerpiece": "vase", "density": 2, "wall_art": True}


def test_typed_retry_usage_and_cache(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-secret")
    state = State(tmp_path / "state.sqlite")
    calls = []

    def handler(request):
        import json

        calls.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"content": json.dumps({**VALID, "density": 99} if len(calls) == 1 else VALID)},
                    }
                ],
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 20,
                    "total_tokens": 120,
                    "prompt_tokens_details": {"cached_tokens": 40},
                },
            },
        )

    cfg = LLMConfig(
        mode="live", requests_per_minute=60000, retries=1, input_cost_per_million=1, output_cost_per_million=2
    )
    client = LLM(
        cfg, state, "run", tmp_path / "cache", httpx.MockTransport(handler), log=Log(tmp_path / "events.jsonl")
    )
    try:
        a = client.request(RoomDesign, {"room": "a"}, "scene", "component")
        b = client.request(RoomDesign, {"room": "a"}, "scene", "component")
        assert a == b
        assert len(calls) == 2
        assert calls[0]["temperature"] == 0.2
        assert calls[0]["max_tokens"] == cfg.max_tokens
        assert "validation_feedback" in calls[1]["messages"][1]["content"]
        rows = state.rows("SELECT * FROM llm_calls")
        assert sum(r["total_tokens"] for r in rows) == 240
        assert rows[-1]["cache_hit"] == 1 and rows[-1]["total_tokens"] == 0
        assert "sk-test-secret" not in str(rows)
        output = capsys.readouterr().err
        assert "[llm_request]" in output and "[llm_retry]" in output
        assert "[llm_success]" in output and "[llm_cache_hit]" in output
        assert "less_than_equal" in output and "density" in output
        assert "sk-test-secret" not in output
        assert "centerpiece" not in output
        assert "llm_error" in (tmp_path / "events.jsonl").read_text()
    finally:
        client.close()
        state.close()


def test_http_failure_redacts_provider_response(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-private-key")
    state = State(tmp_path / "state.sqlite")
    client = LLM(
        LLMConfig(mode="live"),
        state,
        "run",
        tmp_path,
        httpx.MockTransport(lambda _: httpx.Response(401, text="sk-private-key")),
    )
    try:
        with pytest.raises(LLMError, match="401") as exc:
            client.request(RoomDesign, {}, "scene")
        assert "sk-private-key" not in str(exc.value)
        assert len(state.rows("SELECT * FROM llm_calls")) == 1
    finally:
        client.close()
        state.close()


@pytest.mark.parametrize("encoded", [False, True])
@pytest.mark.parametrize("envelope_key", ["final", "final{", "answer"])
def test_final_envelope_is_validated_and_cached(tmp_path, monkeypatch, encoded, envelope_key):
    import json

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    state = State(tmp_path / "state.sqlite")
    answer = json.dumps(VALID) if encoded else VALID
    client = LLM(
        LLMConfig(retries=0),
        state,
        "run",
        tmp_path / "cache",
        httpx.MockTransport(
            lambda _: httpx.Response(
                200,
                json={
                    "choices": [{"finish_reason": "stop", "message": {"content": json.dumps({envelope_key: answer})}}]
                },
            )
        ),
    )
    try:
        result = client.request(RoomDesign, {}, "scene")
        assert result == RoomDesign.model_validate(VALID)
        assert client.request(RoomDesign, {}, "scene") == result
        assert state.rows("SELECT * FROM llm_calls")[-1]["cache_hit"] == 1
    finally:
        client.close()
        state.close()


@pytest.mark.parametrize("answer", [{"final": {**VALID, "density": 99}}, {"final": VALID, "extra": True}])
def test_final_envelope_does_not_bypass_contract(tmp_path, monkeypatch, answer):
    import json

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    state = State(tmp_path / "state.sqlite")
    client = LLM(
        LLMConfig(retries=0),
        state,
        "run",
        tmp_path / "cache",
        httpx.MockTransport(
            lambda _: httpx.Response(
                200, json={"choices": [{"finish_reason": "stop", "message": {"content": json.dumps(answer)}}]}
            )
        ),
    )
    try:
        with pytest.raises(LLMError, match="ValidationError"):
            client.request(RoomDesign, {}, "scene")
        assert not list((tmp_path / "cache").glob("*.json"))
    finally:
        client.close()
        state.close()


@pytest.mark.parametrize("wrapped", [False, True])
def test_fenced_json_still_enforces_schema(tmp_path, monkeypatch, wrapped):
    import json

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    state = State(tmp_path / "state.sqlite")
    content = "```json\n" + json.dumps(VALID) + "\n```"
    if wrapped:
        content = json.dumps({"final": content})
    client = LLM(
        LLMConfig(retries=0),
        state,
        "run",
        tmp_path / "cache",
        httpx.MockTransport(
            lambda _: httpx.Response(
                200, json={"choices": [{"finish_reason": "stop", "message": {"content": content}}]}
            )
        ),
    )
    try:
        assert client.request(RoomDesign, {}, "scene") == RoomDesign.model_validate(VALID)
    finally:
        client.close()
        state.close()


def test_incompatible_json_mode_retries_without_response_format(tmp_path, monkeypatch):
    import json

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    state = State(tmp_path / "state.sqlite")
    calls = []

    def handler(request):
        calls.append(json.loads(request.content))
        answer = {"broken wrapper": {}, "other broken key": {}} if len(calls) == 1 else VALID
        return httpx.Response(
            200, json={"choices": [{"finish_reason": "stop", "message": {"content": json.dumps(answer)}}]}
        )

    client = LLM(
        LLMConfig(retries=1, requests_per_minute=60000), state, "run", tmp_path / "cache", httpx.MockTransport(handler)
    )
    try:
        assert client.request(RoomDesign, {}, "scene") == RoomDesign.model_validate(VALID)
        assert calls[0]["response_format"] == {"type": "json_object"}
        assert "response_format" not in calls[1]
    finally:
        client.close()
        state.close()
