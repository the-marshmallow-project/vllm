# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU checks for zero-token chat through the real FastAPI route.

The renderer and engine are boundary spies, and the request model fixture
provides the fields needed by the serving methods without importing vLLM.
"""

import ast
import asyncio
import json
from http import HTTPStatus
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, Mock

import httpx
import pytest
from fastapi import APIRouter, FastAPI, Request

SOURCE = Path(__file__).resolve().parents[4]


def load_nodes(relative, names, namespace, owner=None):
    tree = ast.parse((SOURCE / relative).read_text())
    nodes = tree.body
    if owner is not None:
        nodes = next(node for node in nodes if getattr(node, "name", "") == owner).body
    selected = [node for node in nodes if getattr(node, "name", "") in names]
    assert len(selected) == len(names)
    exec(compile(ast.Module(selected, type_ignores=[]), relative, "exec"), namespace)


@pytest.fixture
def api():
    ns = dict(globals())
    for relative in (
        "vllm/entrypoints/serve/engine/protocol.py",
        "vllm/entrypoints/serve/utils/api_utils.py",
        "vllm/entrypoints/serve/middleware/authenticate.py",
        "vllm/entrypoints/openai/chat_completion/protocol.py",
        "vllm/entrypoints/openai/chat_completion/api_router.py",
        "vllm/entrypoints/openai/chat_completion/serving.py",
    ):
        imports = []
        for node in ast.parse((SOURCE / relative).read_text()).body:
            module = (
                node.module
                if isinstance(node, ast.ImportFrom)
                else node.names[0].name
                if isinstance(node, ast.Import)
                else ""
            )
            if module and module.split(".")[0] in {
                "asyncio",
                "collections",
                "functools",
                "hashlib",
                "os",
                "secrets",
                "time",
                "typing",
                "pydantic",
                "fastapi",
                "starlette",
            }:
                imports.append(node)
        exec(compile(ast.Module(imports, type_ignores=[]), relative, "exec"), ns)
    ns.update(
        logger=SimpleNamespace(debug=lambda *a: None, error=Mock()),
        _REMOVED_GUIDED_FIELDS=frozenset(),
        RequestOutput=Any,
        DeltaMessage=Any,
        FunctionCall=Any,
        Logprob=Any,
        OpenAIAnnotation=Any,
        OpenAIChatCompletionAudio=Any,
        PerRequestMetrics=Any,
        ToolCall=Any,
        WeightVersionSpan=Any,
        _T=Any,
        ConversationMessage=Any,
        EngineInput=Any,
        CompletionRequest=Any,
        ResponsesRequest=Any,
        OpenAIServingChat=Any,
        RequestOutputKind=SimpleNamespace(FINAL_ONLY="final"),
        SamplingParams=lambda **kw: SimpleNamespace(**kw),
        RequestResponseMetadata=lambda **kw: SimpleNamespace(**kw),
    )
    load_nodes(
        "vllm/entrypoints/serve/engine/protocol.py",
        {
            "OpenAIBaseModel",
            "ErrorInfo",
            "ErrorResponse",
            "PromptTokenUsageInfo",
            "CompletionTokenUsageInfo",
            "UsageInfo",
        },
        ns,
    )

    class ChatCompletionRequest(ns["OpenAIBaseModel"]):
        messages: list[dict[str, Any]]
        model: str | None = None
        request_id: str = "fixture"
        session_id: str | None = None
        vllm_xargs: dict | None = None
        priority: int = 0
        cache_salt: str | None = None
        tools: list | None = None
        chat_template_kwargs: dict | None = None
        add_generation_prompt: bool = True
        max_completion_tokens: int | None = None
        max_tokens: int | None = None
        n: int | None = 1
        stream: bool | None = False
        use_beam_search: bool = False
        min_tokens: int = 0
        logprobs: bool | None = False
        top_logprobs: int | None = 0
        prompt_logprobs: int | None = None
        logprob_token_ids: list[int] | None = None
        echo: bool = False
        kv_transfer_params: dict | None = None
        ec_transfer_params: dict | None = None
        return_token_ids: bool | None = None
        return_prompt_text: bool | None = None

    ns["ChatCompletionRequest"] = ChatCompletionRequest
    load_nodes(
        "vllm/exceptions.py",
        {"VLLMError", "VLLMClientError", "VLLMValidationError"},
        ns,
    )
    load_nodes(
        "vllm/entrypoints/openai/chat_completion/protocol.py",
        {
            "ChatMessage",
            "ChatCompletionLogProb",
            "ChatCompletionLogProbsContent",
            "ChatCompletionLogProbs",
            "ChatCompletionResponseChoice",
            "ChatCompletionResponse",
            "ChatCompletionResponseStreamChoice",
            "ChatCompletionStreamResponse",
        },
        ns,
    )
    load_nodes(
        "vllm/entrypoints/openai/chat_completion/serving.py",
        {"_make_prompt_tokens_details"},
        ns,
    )
    serving_methods = {
        "create_chat_completion",
        "_create_chat_completion",
        "_create_chat_prefill",
        "_create_chat_message",
        "get_chat_request_role",
        "render_chat_request",
    }
    load_nodes(
        "vllm/entrypoints/openai/chat_completion/serving.py",
        serving_methods,
        ns,
        owner="OpenAIServingChat",
    )
    helper_names = {
        "_get_session_id",
        "_get_session_id_from_headers",
        "_get_priority",
        "_get_data_parallel_rank",
        "_with_kv_transfer_rejection_cleanup",
    }
    load_nodes(
        "vllm/entrypoints/generate/base/serving.py",
        helper_names,
        ns,
        owner="GenerateBaseServing",
    )
    ns.update(SESSION_ID_HEADER="X-Session-ID", PRIORITY_HEADER="X-Vllm-Priority")
    helpers = type("GenerateBaseServing", (), {name: ns[name] for name in helper_names})
    ns["GenerateBaseServing"] = helpers
    serving_type = type(
        "PrefillServing",
        (helpers,),
        {name: ns[name] for name in serving_methods},
    )
    serving = serving_type()
    serving.has_kv_connector = False
    serving.response_role = "assistant"
    serving.system_fingerprint = None
    serving.renderer = SimpleNamespace(tokenizer=object())
    serving.parser_cls = None
    serving._effective_chat_template_kwargs = lambda request: {}
    serving.input_processor = SimpleNamespace(validate_prefill_only_config=Mock())
    serving._check_model = AsyncMock(return_value=None)
    serving._preflight = Mock()
    serving._maybe_get_adapters = Mock(return_value=None)
    serving._base_request_id = lambda raw, identifier: identifier
    serving._extract_prompt_len = lambda prompt: len(prompt["prompt_token_ids"])
    serving._get_trace_headers = AsyncMock(return_value={"traceparent": "fixture"})
    serving.models = SimpleNamespace(model_name=lambda lora: "dev-model")
    serving.create_error_response = (
        lambda message,
        err_type="BadRequestError",
        status_code=HTTPStatus.BAD_REQUEST: ns["ErrorResponse"](
            error=ns["ErrorInfo"](message=message, type=err_type, code=int(status_code))
        )
    )
    prompt = {"type": "token", "prompt_token_ids": [3, 5, 8], "cache_salt": "private"}
    serving.online_renderer = SimpleNamespace(
        render_chat=AsyncMock(return_value=([], [prompt]))
    )
    terminal = SimpleNamespace(
        finished=True,
        outputs=[SimpleNamespace(token_ids=[], text="", finish_reason="stop")],
        num_cached_tokens=2,
        num_cache_creation_tokens=1,
        prompt_token_ids=prompt["prompt_token_ids"],
        prompt="hello",
    )
    calls = []
    events = []

    async def generate(*args, **kwargs):
        calls.append((args, kwargs))
        try:
            yield terminal
        finally:
            events.append("closed")

    serving.engine_client = SimpleNamespace(generate=generate)
    return SimpleNamespace(
        ns=ns,
        serving=serving,
        prompt=prompt,
        terminal=terminal,
        calls=calls,
        events=events,
    )


def body(api, **fields):
    fields.setdefault("max_completion_tokens", 0)
    return api.ns["ChatCompletionRequest"](
        messages=[{"role": "user", "content": "hello"}], **fields
    )


@pytest.mark.asyncio
async def test_prefill_uses_chat_renderer_and_returns_numeric_zero_output_ack(api):
    request = body(
        api,
        tools=[{"type": "function"}],
        cache_salt="private",
        chat_template_kwargs={"enable_thinking": False},
    )
    result = await api.serving.create_chat_completion(request)
    assert result.object == "chat.completion"
    assert len(result.choices) == 1
    choice = result.choices[0]
    assert (choice.message.role, choice.message.content) == ("assistant", "")
    assert choice.finish_reason == "length"
    assert result.usage.model_dump(exclude_none=True) == {
        "prompt_tokens": 3,
        "total_tokens": 3,
        "completion_tokens": 0,
        "prompt_tokens_details": {"cached_tokens": 2, "created_cache_tokens": 1},
    }
    api.serving.online_renderer.render_chat.assert_awaited_once_with(request)
    args, _ = api.calls[0]
    assert args[0] is api.prompt
    assert args[1].prefill_only and not args[1].detokenize
    assert not args[1].watermarking and args[1].output_kind == "final"
    assert api.events == ["closed"]


@pytest.mark.asyncio
async def test_prefill_preserves_session_priority_and_trace_headers(api):
    raw = Request(
        {
            "type": "http",
            "headers": [
                (b"x-session-id", b"session"),
                (b"x-vllm-priority", b"4"),
                (b"x-data-parallel-rank", b"0"),
            ],
        }
    )
    result = await api.serving.create_chat_completion(body(api), raw)
    _, kwargs = api.calls[0]
    assert kwargs == dict(
        lora_request=None,
        trace_headers={"traceparent": "fixture"},
        priority=4,
        data_parallel_rank=0,
        session_id="session",
    )
    assert raw.state.request_metadata.final_usage_info == result.usage


@pytest.mark.parametrize(
    "fields",
    [
        {"stream": True},
        {"n": 2},
        {"use_beam_search": True},
        {"min_tokens": 1},
        {"logprobs": True},
        {"top_logprobs": 1},
        {"prompt_logprobs": 0},
        {"logprob_token_ids": [1]},
        {"echo": True},
        {"kv_transfer_params": {"do_remote_prefill": True}},
        {"ec_transfer_params": {}},
    ],
)
@pytest.mark.asyncio
async def test_prefill_rejects_generation_or_remote_transfer_requests(api, fields):
    result = await api.serving.create_chat_completion(body(api, **fields))
    assert result.error.code == 400
    api.serving.online_renderer.render_chat.assert_not_awaited()
    assert not api.calls


@pytest.mark.asyncio
async def test_unsupported_config_never_renders_or_enqueues(api):
    api.serving.input_processor.validate_prefill_only_config.side_effect = api.ns[
        "VLLMValidationError"
    ]("unsupported")
    assert (await api.serving.create_chat_completion(body(api))).error.code == 400
    api.serving.online_renderer.render_chat.assert_not_awaited()
    assert not api.calls


@pytest.mark.asyncio
async def test_model_or_render_rejection_never_enqueues(api):
    rejected = api.serving.create_error_response("unknown model")
    api.serving._check_model.return_value = rejected
    assert await api.serving.create_chat_completion(body(api)) is rejected
    api.serving.online_renderer.render_chat.assert_not_awaited()
    assert not api.calls


@pytest.mark.asyncio
async def test_render_rejection_never_enqueues(api):
    rejected = api.serving.create_error_response("invalid prompt")
    api.serving.online_renderer.render_chat.return_value = rejected
    assert await api.serving.create_chat_completion(body(api)) is rejected
    assert not api.calls


@pytest.mark.parametrize("failure", ["token", "abort", "unfinished", "empty"])
@pytest.mark.asyncio
async def test_prefill_never_reports_success_for_generated_or_incomplete_output(
    api, failure
):
    if failure == "token":
        api.terminal.outputs[0].token_ids = [42]
    elif failure == "abort":
        api.terminal.outputs[0].finish_reason = "abort"
    elif failure == "unfinished":
        api.terminal.finished = False
    else:
        api.terminal.outputs = []
    result = await api.serving.create_chat_completion(body(api))
    assert result.error.code == 500
    assert api.events == ["closed"]


def make_app(api):
    ns = api.ns
    ns.update(_ERROR_REF={"$ref": "#/components/schemas/ErrorResponse"})
    load_nodes(
        "vllm/entrypoints/serve/utils/api_utils.py",
        {
            "listen_for_disconnect",
            "with_cancellation",
            "load_aware_call",
            "decrement_server_load",
            "validate_json_request",
            "_sse_event_schema",
            "json_or_sse_responses",
        },
        ns,
    )
    ns.update(
        router=APIRouter(),
        metrics_header=lambda _: {},
        ENDPOINT_LOAD_METRICS_FORMAT_HEADER_LABEL="endpoint-load-metrics-format",
    )
    load_nodes(
        "vllm/entrypoints/openai/chat_completion/api_router.py",
        {"chat", "create_chat_completion"},
        ns,
    )
    ns["GUARDED_PREFIX"] = ("/v1", "/v2", "/inference", "/cohere")
    load_nodes(
        "vllm/entrypoints/serve/middleware/authenticate.py",
        {"AuthenticationMiddleware"},
        ns,
    )
    app = FastAPI()
    app.state.openai_serving_chat = api.serving
    app.state.enable_server_load_tracking = True
    app.include_router(ns["router"])
    app.add_middleware(ns["AuthenticationMiddleware"], tokens=["local-fixture"])
    return app


@pytest.mark.parametrize(
    "fields",
    [
        {"max_completion_tokens": 0},
        {"max_completion_tokens": 0, "max_tokens": 16},
        {"max_tokens": 0},
        {"max_completion_tokens": None, "max_tokens": 0},
    ],
)
@pytest.mark.asyncio
async def test_real_chat_route_authentication_and_zero_token_response(api, fields):
    app = make_app(api)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://local"
    ) as client:
        payload = {"messages": [{"role": "user", "content": "hello"}], **fields}
        assert (
            await client.post("/v1/chat/completions", json=payload)
        ).status_code == 401
        response = await client.post(
            "/v1/chat/completions",
            json=payload,
            headers={"Authorization": "Bearer local-fixture"},
        )
    assert response.status_code == 200
    assert response.json()["usage"]["completion_tokens"] == 0
    assert response.json()["choices"][0]["message"]["content"] == ""
    assert app.state.server_load_metrics == 0
    assert len(api.calls) == 1 and api.calls[0][0][1].prefill_only


@pytest.mark.parametrize(
    "fields",
    [
        {},
        {"max_completion_tokens": None},
        {"max_completion_tokens": 1},
        {"max_completion_tokens": 1, "max_tokens": 0},
        {"max_tokens": 1},
    ],
)
@pytest.mark.asyncio
async def test_regular_chat_requests_do_not_enter_prefill(api, fields):
    request = api.ns["ChatCompletionRequest"](messages=[], **fields)
    rejection = api.serving.create_error_response("stop at render boundary")
    api.serving.online_renderer.render_chat.return_value = rejection

    assert await api.serving.create_chat_completion(request) is rejection
    api.serving.input_processor.validate_prefill_only_config.assert_not_called()
    assert not api.calls


def test_dedicated_prefill_route_is_removed():
    tree = ast.parse(
        (SOURCE / "vllm/entrypoints/openai/chat_completion/api_router.py").read_text()
    )
    assert not any(
        isinstance(node, ast.Constant) and node.value == "/v1/chat/prefill"
        for node in ast.walk(tree)
    )


@pytest.mark.asyncio
async def test_disconnect_cancels_engine_iteration_and_closes_generator(api):
    started = asyncio.Event()
    closed = asyncio.Event()

    async def wait_for_engine(*args, **kwargs):
        try:
            started.set()
            await asyncio.Event().wait()
            yield api.terminal
        finally:
            closed.set()

    api.serving.engine_client.generate = wait_for_engine
    task = asyncio.create_task(api.serving.create_chat_completion(body(api)))
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert closed.is_set()


@pytest.mark.asyncio
async def test_asgi_disconnect_cancels_engine_and_releases_load_counter(api):
    started = asyncio.Event()
    closed = asyncio.Event()

    async def wait_for_engine(*args, **kwargs):
        try:
            started.set()
            await asyncio.Event().wait()
            yield api.terminal
        finally:
            closed.set()

    api.serving.engine_client.generate = wait_for_engine
    app = make_app(api)
    sent_body = False

    async def receive():
        nonlocal sent_body
        if not sent_body:
            sent_body = True
            return {
                "type": "http.request",
                "body": json.dumps(
                    {"messages": [], "max_completion_tokens": 0}
                ).encode(),
                "more_body": False,
            }
        await started.wait()
        return {"type": "http.disconnect"}

    messages = []

    async def send(message):
        messages.append(message)

    scope = {
        "type": "http",
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/v1/chat/completions",
        "query_string": b"",
        "root_path": "",
        "headers": [
            (b"content-type", b"application/json"),
            (b"authorization", b"Bearer local-fixture"),
        ],
    }
    await asyncio.wait_for(app(scope, receive, send), timeout=2)
    await asyncio.wait_for(closed.wait(), timeout=2)
    assert app.state.server_load_metrics == 0
