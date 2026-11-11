"""OpenAI-compatible request/response/stream contracts."""

from __future__ import annotations

import json
from typing import Any

import pytest
from hypothesis import given
from hypothesis import strategies as st
from pydantic import ValidationError

from high_throughput_vllm_deployment.schemas.enums import FinishReason
from high_throughput_vllm_deployment.schemas.inference import (
    SSE_DONE_EVENT,
    AssistantMessage,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    CompletionRequest,
    CompletionResponse,
    DeltaMessage,
    StreamChoice,
    StreamChunk,
    Usage,
)

MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
USAGE: dict[str, int] = {"prompt_tokens": 5, "completion_tokens": 7, "total_tokens": 12}


# --------------------------------------------------------------------------- #
# Requests                                                                    #
# --------------------------------------------------------------------------- #


def test_completion_request_defaults() -> None:
    request = CompletionRequest(model=MODEL, prompt="Hello")
    assert (request.temperature, request.top_p, request.max_tokens, request.n) == (
        1.0,
        1.0,
        None,
        1,
    )
    assert request.stream is False
    assert request.stop is None


@pytest.mark.parametrize("prompt", ["Hello", ["a", "b"]])
def test_completion_request_accepts_string_or_batch(prompt: str | list[str]) -> None:
    assert CompletionRequest.model_validate({"model": MODEL, "prompt": prompt}).prompt == prompt


@pytest.mark.parametrize("prompt", ["", "   ", [], ["ok", ""], [1, 2], None])
def test_completion_request_rejects_empty_or_malformed_prompts(prompt: object) -> None:
    with pytest.raises(ValidationError, match="prompt"):
        CompletionRequest.model_validate({"model": MODEL, "prompt": prompt})


def test_chat_request_parses_messages_from_json() -> None:
    raw = json.dumps(
        {
            "model": MODEL,
            "messages": [
                {"role": "system", "content": "be terse"},
                {"role": "user", "content": "hi", "name": "alice"},
            ],
            "temperature": 0,
            "max_tokens": 64,
            "stream": True,
            "stop": ["\n\n"],
            "seed": 7,
        }
    )
    request = ChatCompletionRequest.model_validate_json(raw)
    assert [m.role for m in request.messages] == ["system", "user"]
    assert request.temperature == 0.0
    assert request.stop == ["\n\n"]


@pytest.mark.parametrize(
    "messages",
    [
        [],
        [{"role": "robot", "content": "x"}],
        [{"role": "user"}],
        [{"role": "user", "content": "x", "extra": 1}],
    ],
)
def test_chat_request_rejects_invalid_messages(messages: list[dict[str, Any]]) -> None:
    with pytest.raises(ValidationError, match="messages"):
        ChatCompletionRequest.model_validate({"model": MODEL, "messages": messages})


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("temperature", -0.01),
        ("temperature", 2.01),
        ("temperature", float("nan")),
        ("temperature", "0.7"),
        ("top_p", 0.0),
        ("top_p", 1.01),
        ("max_tokens", 0),
        ("max_tokens", 1.5),
        ("n", 0),
        ("n", 129),
        ("stream", "true"),
        ("stop", ""),
        ("stop", [""]),
        ("stop", []),
        ("seed", True),
        ("model", ""),
        ("logprobs", 5),
    ],
)
def test_sampling_parameters_are_validated(field: str, value: object) -> None:
    with pytest.raises(ValidationError, match=field):
        CompletionRequest.model_validate({"model": MODEL, "prompt": "x", field: value})


@pytest.mark.parametrize(
    ("field", "value"),
    [("temperature", 0), ("temperature", 2.0), ("top_p", 1.0), ("top_p", 1e-9), ("max_tokens", 1)],
)
def test_sampling_parameter_bounds_are_inclusive_where_specified(field: str, value: object) -> None:
    request = CompletionRequest.model_validate({"model": MODEL, "prompt": "x", field: value})
    assert getattr(request, field) == value


# --------------------------------------------------------------------------- #
# Responses                                                                   #
# --------------------------------------------------------------------------- #


def _completion_payload(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "id": "cmpl-1",
        "object": "text_completion",
        "created": 1_700_000_000,
        "model": MODEL,
        "choices": [{"index": 0, "text": "world", "finish_reason": "length"}],
        "usage": USAGE,
    }
    return payload | overrides


def _chat_payload(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "created": 1_700_000_000,
        "model": MODEL,
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}
        ],
        "usage": USAGE,
    }
    return payload | overrides


def test_completion_response_parses_vllm_shaped_dict() -> None:
    response = CompletionResponse.model_validate(_completion_payload())
    assert response.choices[0].finish_reason is FinishReason.LENGTH
    assert response.usage.total_tokens == 12
    assert CompletionResponse.model_validate_json(response.model_dump_json()) == response


def test_chat_response_parses_vllm_shaped_dict() -> None:
    response = ChatCompletionResponse.model_validate(_chat_payload())
    assert response.choices[0].message.content == "hi"
    assert response.choices[0].finish_reason is FinishReason.STOP
    assert ChatCompletionResponse.model_validate_json(response.model_dump_json()) == response


@pytest.mark.parametrize(
    "overrides",
    [
        {"object": "chat.completion"},
        {"id": ""},
        {"created": -1},
        {"choices": []},
        {"choices": [{"index": 0, "text": "a"}, {"index": 0, "text": "b"}]},
        {"choices": [{"index": 0, "text": "a", "finish_reason": "eos"}]},
        {"usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 3}},
    ],
)
def test_completion_response_rejects_contract_violations(overrides: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        CompletionResponse.model_validate(_completion_payload(**overrides))


@pytest.mark.parametrize(
    "overrides",
    [
        {"object": "text_completion"},
        {"choices": [{"index": 0, "message": {"role": "user", "content": "x"}}]},
        {
            "choices": [
                {"index": 1, "message": {"role": "assistant", "content": "a"}},
                {"index": 1, "message": {"role": "assistant", "content": "b"}},
            ]
        },
    ],
)
def test_chat_response_rejects_contract_violations(overrides: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        ChatCompletionResponse.model_validate(_chat_payload(**overrides))


@given(prompt=st.integers(min_value=0, max_value=10**9), completion=st.integers(0, 10**9))
def test_usage_total_must_equal_sum(prompt: int, completion: int) -> None:
    usage = Usage(
        prompt_tokens=prompt, completion_tokens=completion, total_tokens=prompt + completion
    )
    assert Usage.model_validate_json(usage.model_dump_json()) == usage
    with pytest.raises(ValidationError, match="total_tokens"):
        Usage(
            prompt_tokens=prompt, completion_tokens=completion, total_tokens=prompt + completion + 1
        )


# --------------------------------------------------------------------------- #
# Streaming                                                                   #
# --------------------------------------------------------------------------- #


def _chunk(**overrides: Any) -> StreamChunk:
    payload: dict[str, Any] = {
        "id": "chatcmpl-1",
        "created": 1_700_000_000,
        "model": MODEL,
        "choices": [{"index": 0, "delta": {"content": "tok"}, "finish_reason": None}],
    }
    return StreamChunk.model_validate(payload | overrides)


def test_stream_chunk_defaults_and_sse_encoding() -> None:
    chunk = _chunk()
    assert chunk.object == "chat.completion.chunk"
    event = chunk.to_sse()
    assert event.startswith("data: {") and event.endswith("}\n\n")
    assert json.loads(event.removeprefix("data: "))["choices"][0]["finish_reason"] is None


def test_stream_chunk_round_trips_through_sse() -> None:
    chunk = _chunk(choices=[{"index": 0, "delta": {"role": "assistant"}, "finish_reason": "stop"}])
    assert StreamChunk.from_sse_line(chunk.to_sse()) == chunk


def test_usage_only_final_chunk_is_allowed() -> None:
    chunk = _chunk(choices=[], usage=USAGE)
    assert chunk.usage is not None and chunk.usage.total_tokens == 12


@pytest.mark.parametrize(
    "overrides",
    [
        {"choices": []},
        {"object": "chat.completion"},
        {"choices": [{"index": 0, "delta": {}}, {"index": 0, "delta": {}}]},
        {"choices": [{"index": 0, "delta": {"role": "robot"}}]},
        {"choices": [{"index": -1, "delta": {}}]},
    ],
)
def test_stream_chunk_rejects_contract_violations(overrides: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        _chunk(**overrides)


@pytest.mark.parametrize("line", [SSE_DONE_EVENT, "data: [DONE]", "data:[DONE]\n"])
def test_done_sentinel_decodes_to_none(line: str) -> None:
    assert StreamChunk.from_sse_line(line) is None


@pytest.mark.parametrize("line", ["", ": keep-alive", "event: message", "{}"])
def test_non_data_lines_are_rejected(line: str) -> None:
    with pytest.raises(ValueError, match="not an SSE data line"):
        StreamChunk.from_sse_line(line)


def test_malformed_data_payload_is_rejected() -> None:
    with pytest.raises(ValidationError):
        StreamChunk.from_sse_line('data: {"id": "x"}')


@given(
    content=st.text(max_size=64),
    index=st.integers(min_value=0, max_value=127),
    reason=st.none() | st.sampled_from(FinishReason),
)
def test_any_delta_round_trips_through_sse(
    content: str, index: int, reason: FinishReason | None
) -> None:
    chunk = StreamChunk(
        id="chatcmpl-x",
        created=0,
        model=MODEL,
        choices=[
            StreamChoice(index=index, delta=DeltaMessage(content=content), finish_reason=reason)
        ],
    )
    assert StreamChunk.from_sse_line(chunk.to_sse()) == chunk


def test_chat_message_name_must_not_be_blank() -> None:
    with pytest.raises(ValidationError, match="name"):
        ChatMessage(role="user", content="x", name=" ")


# --------------------------------------------------------------------------- #
# vLLM response extensions                                                    #
# --------------------------------------------------------------------------- #


def test_responses_accept_vllm_extension_fields() -> None:
    logprobs = {
        "content": [{"token": "hi", "logprob": -0.1, "bytes": [104, 105], "top_logprobs": []}]
    }
    chat = ChatCompletionResponse.model_validate(
        _chat_payload(
            system_fingerprint="fp-abc",
            choices=[
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "reasoning_content": "thinking",
                        "tool_calls": [{"id": "call-1", "type": "function"}],
                    },
                    "logprobs": logprobs,
                    "finish_reason": "stop",
                    "stop_reason": 128009,
                }
            ],
        )
    )
    choice = chat.choices[0]
    assert chat.system_fingerprint == "fp-abc"
    assert choice.logprobs == logprobs
    assert choice.stop_reason == 128009
    assert choice.message.content is None
    assert choice.message.tool_calls == [{"id": "call-1", "type": "function"}]

    completion = CompletionResponse.model_validate(
        _completion_payload(
            choices=[
                {"index": 0, "text": "x", "stop_reason": "</s>", "logprobs": {"tokens": ["x"]}}
            ]
        )
    )
    assert completion.choices[0].stop_reason == "</s>"
    assert completion.system_fingerprint is None


def test_assistant_message_defaults() -> None:
    message = AssistantMessage()
    assert (message.role, message.content, message.tool_calls) == ("assistant", None, [])


@pytest.mark.parametrize(
    "choice",
    [
        {"index": 0, "text": "x", "stop_reason": 1.5},
        {"index": 0, "text": "x", "stop_reason": True},
        {"index": 0, "text": "x", "token_ids": [1, 2]},
    ],
)
def test_extension_fields_stay_strict(choice: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        CompletionResponse.model_validate(_completion_payload(choices=[choice]))


def test_stream_choice_accepts_reasoning_and_stop_reason() -> None:
    chunk = _chunk(
        choices=[
            {
                "index": 0,
                "delta": {"reasoning_content": "hmm"},
                "finish_reason": "stop",
                "stop_reason": "\n\n",
            }
        ]
    )
    assert chunk.choices[0].delta.reasoning_content == "hmm"
    assert chunk.choices[0].stop_reason == "\n\n"
