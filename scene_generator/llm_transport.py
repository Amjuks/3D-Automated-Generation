"""Bounded Chat Completions decoding. Provider text never enters diagnostics."""

import json
import time

import httpx


class ResponseError(ValueError):
    def __init__(self, reason, retryable=True):
        super().__init__(reason)
        self.reason, self.retryable = reason, retryable


def strict_schema(value):
    """Convert Pydantic defaults and homogeneous tuples to the strict wire dialect.

    Pydantic still validates the original, stronger schema after decoding.
    """
    if isinstance(value, list):
        return [strict_schema(v) for v in value]
    if not isinstance(value, dict):
        return value
    result = {
        k: {name: strict_schema(child) for name, child in v.items()}
        if k in {"properties", "$defs"}
        else strict_schema(v)
        for k, v in value.items()
        if k not in {"default", "title"}
    }
    if "prefixItems" in result:
        items = result.pop("prefixItems")
        if not items or any(item != items[0] for item in items):
            raise ValueError("strict output does not support heterogeneous tuples")
        result["items"] = items[0]
    if result.get("type") == "object":
        result["required"] = list(result.get("properties", {}))
        result["additionalProperties"] = False
    return result


def completion(data):
    if not isinstance(data, dict):
        raise ResponseError("invalid_completion_envelope")
    choices = data.get("choices")
    if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
        raise ResponseError("invalid_completion_choices")
    choice = choices[0]
    message = choice.get("message")
    if not isinstance(message, dict):
        raise ResponseError("invalid_completion_message")
    if message.get("refusal") or choice.get("finish_reason") == "content_filter":
        raise ResponseError("refused_response", retryable=False)
    if choice.get("finish_reason") == "length":
        raise ResponseError("output_token_limit_reached; adjust max_tokens or reduce requested detail", retryable=False)
    if choice.get("finish_reason") != "stop":
        raise ResponseError("incomplete_completion")
    if not isinstance(message.get("content"), str):
        raise ResponseError("invalid_completion_content")
    return message["content"]


def receive(client, url, payload, headers, config, metrics, emit):
    start = time.monotonic()
    content = []
    finish = None
    usage = {}
    received = 0
    last_progress = start

    def event(data):
        nonlocal finish, usage
        if not isinstance(data, dict) or data.get("error"):
            raise ResponseError("invalid_stream_event")
        if isinstance(data.get("usage"), dict):
            usage = data["usage"]
            metrics["usage"] = usage
        choices = data.get("choices", [])
        if not isinstance(choices, list):
            raise ResponseError("invalid_stream_choices")
        for choice in choices:
            if not isinstance(choice, dict) or choice.get("index", 0) != 0:
                raise ResponseError("invalid_stream_choice")
            delta = choice.get("delta")
            if not isinstance(delta, dict):
                raise ResponseError("invalid_stream_delta")
            if delta.get("refusal") or choice.get("finish_reason") == "content_filter":
                raise ResponseError("refused_response", retryable=False)
            reasoning = delta.get("reasoning_content", delta.get("reasoning"))
            if isinstance(reasoning, str):
                metrics["reasoning_chars"] = metrics.get("reasoning_chars", 0) + len(reasoning)
            text = delta.get("content")
            if text is not None:
                if not isinstance(text, str) or (finish is not None and text):
                    raise ResponseError("invalid_stream_content")
                if text and "first_content_seconds" not in metrics:
                    metrics["first_content_seconds"] = round(time.monotonic() - start, 3)
                    emit("llm_first_content", **public_metrics(metrics))
                content.append(text)
                metrics["content_chars"] = metrics.get("content_chars", 0) + len(text)
            if choice.get("finish_reason") is not None:
                finish = choice["finish_reason"]
                metrics["finish_reason"] = finish if finish in {"stop", "length", "content_filter"} else "other"

    def blocks(response):
        nonlocal received, last_progress
        for block in response.iter_bytes():
            now = time.monotonic()
            if now - start > config.total_timeout:
                raise ResponseError("total_deadline_exceeded", retryable=False)
            if "first_byte_seconds" not in metrics:
                metrics["first_byte_seconds"] = round(now - start, 3)
            received += len(block)
            metrics["response_bytes"] = received
            if received > config.max_response_bytes:
                raise ResponseError("response_byte_limit_exceeded", retryable=False)
            if now - last_progress >= 30:
                emit("llm_progress", **public_metrics(metrics))
                last_progress = now
            yield block

    # Streaming the HTTP body also bounds buffered-provider responses and exposes
    # whether a disconnect occurred before headers, in a body, or after completion.
    metrics["phase"] = "response_headers"
    with client.stream("POST", url, json=payload, headers=headers) as response:
        metrics["headers_seconds"] = round(time.monotonic() - start, 3)
        metrics["http_status"] = response.status_code
        metrics["retry_after"] = response.headers.get("retry-after", "")
        response.raise_for_status()
        metrics["phase"] = "response_body"
        is_sse = "text/event-stream" in response.headers.get("content-type", "").lower()
        metrics["stream_received"] = is_sse
        emit("llm_response_headers", **public_metrics(metrics))
        if not is_sse:
            try:
                data = json.loads(b"".join(blocks(response)))
            except (json.JSONDecodeError, UnicodeDecodeError):
                raise ResponseError("invalid_response_json") from None
            if isinstance(data, dict) and isinstance(data.get("usage"), dict):
                metrics["usage"] = data["usage"]
            return completion(data)

        buffer = b""
        lines = []
        done = False
        try:
            for block in blocks(response):
                buffer += block
                while b"\n" in buffer:
                    line, buffer = buffer.split(b"\n", 1)
                    line = line.rstrip(b"\r")
                    if line.startswith(b"data:"):
                        lines.append(line[5:].lstrip(b" "))
                    elif not line and lines:
                        raw = b"\n".join(lines)
                        lines = []
                        if raw == b"[DONE]":
                            done = True
                            break
                        try:
                            event(json.loads(raw))
                        except (json.JSONDecodeError, UnicodeDecodeError):
                            raise ResponseError("invalid_stream_json") from None
                if done:
                    break
        except (httpx.RemoteProtocolError, httpx.ReadError, httpx.ReadTimeout):
            # A terminal stop is an application-level completion. Some gateways
            # drop the connection before the optional usage/[DONE] trailer.
            # Only the fully schema-validated answer may be accepted by the caller.
            if finish != "stop":
                raise
            metrics["trailer_interrupted"] = True
        if finish not in {"stop", "length", "content_filter"}:
            raise ResponseError("incomplete_stream")
        metrics["stream_done"] = done
        return completion({"choices": [{"finish_reason": finish, "message": {"content": "".join(content)}}]})


def public_metrics(metrics):
    # Headers and provider usage are untrusted; only fixed diagnostic fields escape.
    allowed = {
        "phase",
        "headers_seconds",
        "http_status",
        "stream_received",
        "first_byte_seconds",
        "first_content_seconds",
        "response_bytes",
        "content_chars",
        "finish_reason",
        "stream_done",
        "trailer_interrupted",
        "reasoning_chars",
    }
    return {k: v for k, v in metrics.items() if k in allowed}
