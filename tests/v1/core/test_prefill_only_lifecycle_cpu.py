# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Actual engine lifecycle methods, without model/runtime imports or a GPU.

The contract is a completed prompt, retained reusable KV, and a terminal empty
completion. In particular scheduling a chunk is not evidence it has executed.
Only cache/coordinator and inference outputs are fakes; production lifecycle,
request status, transport structures, output handling and timing code are used.
"""

from __future__ import annotations

import ast
import copy
import enum
import logging
import math
import os
import time
from collections import defaultdict, deque
from collections.abc import Sequence
from contextlib import nullcontext
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from types import SimpleNamespace as NS
from typing import Any, Generic, TypeVar, cast, overload
from unittest.mock import Mock

import msgspec
import pytest
from pydantic import BeforeValidator
from pydantic.dataclasses import dataclass as pydantic_dataclass

SOURCE = Path(__file__).resolve().parents[3]


def load(namespace, path, names, methods=None):
    nodes = []
    for node in ast.parse((SOURCE / path).read_text()).body:
        name = getattr(node, "name", None)
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            name = getattr(node.targets[0], "id", None)
        if name not in names:
            continue
        if methods and isinstance(node, ast.ClassDef):
            node.body = [n for n in node.body if getattr(n, "name", None) in methods]
        nodes.append(node)
    assert len(nodes) == len(names), (path, names)
    tree = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            *nodes,
        ],
        type_ignores=[],
    )
    exec(
        compile(ast.fix_missing_locations(tree), str(SOURCE / path), "exec"), namespace
    )


class ValidationError(ValueError):
    def __init__(self, message, **kwargs):
        super().__init__(message)


@pytest.fixture(scope="module")
def engine():
    ns = dict(
        __name__=__name__,
        enum=enum,
        Enum=enum.Enum,
        IntEnum=enum.IntEnum,
        msgspec=msgspec,
        PydanticMsgspecMixin=type("PydanticMsgspecMixin", (), {}),
        dataclass=pydantic_dataclass,
        field=field,
        BeforeValidator=BeforeValidator,
        Annotated=__import__("typing").Annotated,
        Any=Any,
        cached_property=cached_property,
        copy=copy,
        math=math,
        envs=NS(
            VLLM_MAX_N_SEQUENCES=16384,
            VLLM_COMPUTE_NANS_IN_LOGITS=False,
            VLLM_BATCH_INVARIANT=False,
        ),
        VLLMValidationError=ValidationError,
        logger=logging.getLogger(__name__),
        time=time,
        os=os,
        deque=deque,
        defaultdict=defaultdict,
        cast=cast,
        T=TypeVar("T"),
        Generic=Generic,
        Sequence=Sequence,
        overload=overload,
        SchedulerInterface=object,
        record_function_or_nullcontext=lambda name: nullcontext(),
        remove_all=lambda items, removed: [r for r in items if r not in removed],
        length_from_prompt_token_ids_or_embeds=lambda tokens, embeds: len(tokens),
        StructuredOutputRequest=NS(from_sampling_params=lambda params: None),
        gc_untrack=__import__(
            "vllm.utils.gc_utils", fromlist=["gc_untrack"]
        ).gc_untrack,
        TokenBuffer=__import__(
            "vllm.v1.token_buffer", fromlist=["TokenBuffer"]
        ).TokenBuffer,
    )
    load(
        ns,
        "vllm/sampling_params.py",
        {
            "_verify_num_sequences",
            "validate_thinking_token_budget",
            "SamplingType",
            "RequestOutputKind",
            "StructuredOutputsParams",
            "RepetitionDetectionParams",
            "SamplingParams",
            "_SAMPLING_EPS",
            "_MAX_TEMP",
            "ThinkingTokenBudget",
        },
    )
    ns["dataclass"] = dataclass
    load(
        ns,
        "vllm/v1/engine/__init__.py",
        {
            "FinishReason",
            "FINISH_REASON_STRINGS",
            "EngineCoreOutput",
            "EngineCoreOutputs",
            "EngineCoreEventType",
        },
    )
    load(ns, "vllm/v1/utils.py", {"ConstantList"})
    load(
        ns,
        "vllm/v1/metrics/stats.py",
        {
            "PrefillStats",
            "RequestStateStats",
            "PromptTokenStats",
            "FinishedRequestStats",
        },
    )
    load(
        ns,
        "vllm/v1/metrics/stats.py",
        {"IterationStats"},
        {
            "__init__",
            "_time_since",
            "update_from_output",
            "update_from_finished_request",
        },
    )
    load(ns, "vllm/v1/request.py", {"Request", "RequestStatus", "_FINISHED_REASON_MAP"})
    load(
        ns,
        "vllm/v1/core/sched/scheduler.py",
        {"Scheduler"},
        {
            "update_from_output",
            "_update_after_schedule",
            "_update_request_with_output",
            "_handle_stopped_request",
            "_free_request",
            "_free_blocks",
            "_request_blocks_can_be_freed",
            "_free_request_blocks",
            "finish_requests",
            "_reserve_prefill_lookahead",
            "_drafts_unused",
            "_get_encoder_input_updates",
            "_set_kv_fetch_stage",
            "get_request_counts",
            "update_draft_token_ids",
            "_preempt_request",
            "_unreserve_parked",
            "schedule",
        },
    )
    load(
        ns,
        "vllm/v1/core/sched/async_scheduler.py",
        {"AsyncScheduler"},
        {
            "_update_after_schedule",
            "_update_request_with_output",
        },
    )
    load(
        ns,
        "vllm/v1/engine/input_processor.py",
        {"InputProcessor"},
        {
            "validate_prefill_only_config",
            "_validate_prompt_len",
        },
    )
    load(ns, "vllm/outputs.py", {"CompletionOutput", "RequestOutput"})
    load(ns, "vllm/v1/core/sched/interface.py", {"PauseState"})
    ns["SchedulerOutput"] = lambda **kwargs: NS(
        has_structured_output_requests=False,
        pending_structured_output_tokens=False,
        **kwargs,
    )
    ns["PoolingOutput"] = type("PoolingOutput", (), {})
    load(
        ns,
        "vllm/v1/engine/output_processor.py",
        {"RequestState"},
        {
            "__init__",
            "from_new_request",
            "make_request_output",
            "_new_request_output",
            "_new_completion_output",
        },
    )
    load(
        ns,
        "vllm/v1/engine/output_processor.py",
        {"OutputProcessor"},
        {
            "process_outputs",
            "abort_requests",
            "_finish_request",
        },
    )
    ns["OutputProcessorOutput"] = NS
    return NS(**ns)


class Queue(list):
    def remove_requests(self, requests):
        self[:] = [r for r in self if r not in requests]

    def prepend_request(self, request):
        self.insert(0, request)


def request(engine, name="prepare", prefill=True):
    params = engine.SamplingParams(
        prefill_only=prefill,
        max_tokens=1,
        detokenize=False,
        output_kind=engine.RequestOutputKind.FINAL_ONLY,
    )
    result = engine.Request(name, list(range(8)), params, None)
    result.status = engine.RequestStatus.RUNNING
    return result


def scheduler(engine, requests):
    result = object.__new__(engine.AsyncScheduler)
    result.reply_first = False
    result.__dict__.update(
        requests={r.request_id: r for r in requests},
        running=list(requests),
        waiting=Queue(),
        kv_holding_waiting=Queue(),
        async_kv_waiting=Queue(),
        deferred_waiting=set(),
        vllm_config=NS(watermark_config=None),
        _parked_reservations={},
        _parked_reserved_blocks=0,
        _inflight_prefills=set(requests),
        _pending_encoder_input_ids={},
        _kv_fetch_stages=None,
        defer_block_free=False,
        perf_metrics=None,
        ec_connector=None,
        connector=None,
        aux_output_connector=None,
        is_mm_encoder_only=False,
        grammar_compile_error_reqs={},
        encoder_cache_mismatch_reqs=set(),
        return_sampling_mask=False,
        finished_req_ids=set(),
        finished_req_ids_dict=None,
        num_sampled_tokens_per_step=1,
        num_spec_tokens=3,
        num_prefill_lookahead=3,
        use_v2_model_runner=True,
        decode_stagger=1,
        current_step=0,
        log_stats=False,
        reset_preempted_req_ids=set(),
        max_model_len=16,
    )
    result.kv_cache_manager = Mock()
    result.kv_cache_manager.take_events.return_value = []
    result.kv_cache_manager.estimate_cached_tokens.return_value = 8
    result.encoder_cache_manager = Mock()
    result._connector_finished = Mock(return_value=(False, None))
    result.structured_output_manager = Mock()
    result.structured_output_manager.accept_tokens.return_value = True
    result.make_stats = Mock(return_value=None)
    return result


def step(**counts):
    return NS(
        num_scheduled_tokens=counts,
        total_num_scheduled_tokens=sum(counts.values()),
        scheduled_spec_decode_tokens={},
        has_structured_output_requests=False,
        pending_structured_output_tokens=False,
        num_spec_tokens_to_schedule=3,
        num_invalid_spec_tokens=None,
    )


def output(*names, tokens=None, kv=None):
    return NS(
        req_ids=list(names),
        req_id_to_index={name: i for i, name in enumerate(names)},
        sampled_token_ids=tokens or [[] for _ in names],
        logprobs=None,
        prompt_logprobs_dict={},
        prompt_token_id_logprobs_dict={},
        pooler_output=None,
        num_nans_in_logits=None,
        kv_connector_output=kv,
        ec_connector_output=None,
        cudagraph_stats=None,
        request_errors={},
    )


def test_explicit_intent_survives_msgpack_and_zero_is_still_invalid(engine):
    params = request(engine).sampling_params
    encoded = msgspec.msgpack.encode(params)
    decoded = msgspec.msgpack.decode(encoded)
    assert decoded["prefill_only"] is True
    assert decoded["max_tokens"] == 1
    assert not engine.SamplingParams(max_tokens=1).prefill_only
    with pytest.raises(ValidationError, match="at least 1"):
        engine.SamplingParams(max_tokens=0)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n": 2},
        {"logprobs": 0},
        {"prompt_logprobs": 0},
        {"prompt_logprob_token_ids": [[1]]},
        {"min_tokens": 1},
        {"trace_decode_token_ids": [1]},
        {"max_tokens": 2},
    ],
)
def test_prefill_rejects_generation_features(engine, kwargs):
    with pytest.raises(ValidationError, match="prefill_only requires"):
        engine.SamplingParams(prefill_only=True, **({"max_tokens": 1} | kwargs))


def test_chunked_async_prefill_waits_for_final_executed_chunk(engine):
    req = request(engine)
    sched = scheduler(engine, [req])
    first, second = step(prepare=3), step(prepare=5)
    sched._update_after_schedule(first)
    sched._update_after_schedule(second)
    assert req.num_computed_tokens == 8
    assert req.num_in_flight_tokens == 8
    assert req.num_output_placeholders == 0
    assert req.spec_token_ids == []
    assert sched.update_from_output(first, output("prepare")) == {}
    assert req.request_id in sched.requests
    final = sched.update_from_output(second, output("prepare"))[0].outputs[0]
    assert final.finish_reason == engine.FinishReason.STOP
    assert final.new_token_ids == []
    assert req.num_output_tokens == 0
    sched.kv_cache_manager.cache_blocks.assert_called_once_with(req, 8)
    sched.kv_cache_manager.free.assert_called_once_with(req)
    assert not sched.requests and not sched.running


def test_cached_prefix_executes_only_tail_then_acknowledges(engine):
    req = request(engine)
    req.num_computed_tokens = 7
    sched = scheduler(engine, [req])
    tail = step(prepare=1)
    sched._update_after_schedule(tail)
    assert req.num_in_flight_tokens == 1
    final = sched.update_from_output(tail, output("prepare"))[0].outputs[0]
    assert final.new_token_ids == [] and final.finished
    sched.kv_cache_manager.cache_blocks.assert_called_once_with(req, 8)


def test_stale_preempted_step_cannot_acknowledge_rescheduled_prompt(engine):
    req = request(engine)
    sched = scheduler(engine, [req])
    old = step(prepare=8)
    sched._update_after_schedule(old)
    sched._preempt_request(req, time.monotonic())
    req.status = engine.RequestStatus.RUNNING
    current = step(prepare=8)
    sched._update_after_schedule(current)
    assert sched.update_from_output(old, output("prepare")) == {}
    assert not req.is_finished()
    final = sched.update_from_output(current, output("prepare"))[0].outputs[0]
    assert final.finished and final.new_token_ids == []


def test_cancelled_inflight_prefill_frees_once_and_never_reports_success(engine):
    req = request(engine)
    sched = scheduler(engine, [req])
    pending = step(prepare=8)
    sched._update_after_schedule(pending)
    sched.finish_requests("prepare", engine.RequestStatus.FINISHED_ABORTED)
    assert sched.update_from_output(pending, output("prepare")) == {}
    sched.kv_cache_manager.free.assert_called_once_with(req)
    sched.kv_cache_manager.cache_blocks.assert_not_called()


def test_mixed_generation_keeps_placeholder_and_token_completion(engine):
    req, gen = request(engine), request(engine, "answer", prefill=False)
    sched = scheduler(engine, [req, gen])
    sched.__class__.__bases__[0]._update_request_with_output.__globals__[
        "check_stop"
    ] = lambda r, limit: (
        setattr(r, "status", engine.RequestStatus.FINISHED_LENGTH_CAPPED) or True
    )
    batch = step(prepare=8, answer=8)
    sched._update_after_schedule(batch)
    assert req.num_output_placeholders == 0
    assert gen.num_output_placeholders == 1
    results = sched.update_from_output(
        batch, output("prepare", "answer", tokens=[[], [42]])
    )[0].outputs
    assert [(r.request_id, r.new_token_ids) for r in results] == [
        ("prepare", []),
        ("answer", [42]),
    ]
    assert gen.num_output_tokens == 1 and req.num_output_tokens == 0


def test_prefill_has_no_draft_chunk_reservation_or_late_drafts(engine):
    req = request(engine)
    sched = scheduler(engine, [req])
    assert sched._reserve_prefill_lookahead(req, 0, 7) == 7
    req.spec_token_ids = [10]
    sched.update_draft_token_ids(NS(req_ids=["prepare"], draft_token_ids=[[4, 5]]))
    assert req.spec_token_ids == []
    sched.structured_output_manager.validate_tokens.assert_not_called()


def test_full_context_only_allowed_for_explicit_prefill(engine):
    proc = object.__new__(engine.InputProcessor)
    proc.skip_prompt_length_check = False
    proc.model_config = NS(max_model_len=8, runner_type="generate")
    proc.supports_mm_inputs = False
    proc._validate_prompt_len(8, "decoder", prefill_only=True)
    with pytest.raises(ValidationError, match="at least 1"):
        proc._validate_prompt_len(8, "decoder")
    with pytest.raises(ValidationError, match="longer"):
        proc._validate_prompt_len(9, "decoder", prefill_only=True)


def test_prefill_request_output_requires_no_detokenizer_or_logprobs(engine):
    params = request(engine).sampling_params
    req = NS(
        sampling_params=params,
        pooling_params=None,
        external_req_id="public",
        request_id="internal",
        lora_request=None,
        prompt_token_ids=list(range(8)),
        prompt_embeds=None,
        arrival_time=0.0,
        resumable=False,
    )
    state = engine.RequestState.from_new_request(
        None, req, None, None, 0, None, False, 5
    )
    assert state.detokenizer is None and state.logprobs_processor is None
    final = state.make_request_output([], None, engine.FinishReason.STOP, None)
    assert final.finished and final.outputs[0].token_ids == []
    assert final.outputs[0].text == "" and final.outputs[0].finish_reason == "stop"
    assert final.prompt_token_ids == list(range(8))
    assert final.prompt_token_id_logprobs is None


def test_prefill_cpu_scheduler_uses_full_context_without_draft_headroom(engine):
    req = request(engine)
    sched = scheduler(engine, [req])
    sched.__dict__.update(
        max_num_scheduled_tokens=8,
        scheduler_config=NS(
            max_num_batched_tokens=8,
            long_prefill_token_threshold=0,
            enable_chunked_prefill=True,
        ),
        vllm_config=NS(
            speculative_config=NS(max_num_new_slots_for_drafting=3),
            watermark_config=None,
        ),
        _pause_state=engine.PauseState.PAUSED_NEW,
        max_num_active_reqs=4,
        num_waiting_for_streaming_input=0,
        cache_aware_window=0,
        max_num_encoder_input_tokens=0,
        need_mamba_block_aligned_split=False,
        num_lookahead_tokens=3,
        lora_config=None,
        max_num_running_reqs=4,
        max_model_len=8,
        kv_cache_config=NS(kv_cache_groups=[object()]),
        dynamic_sd_lookup=None,
        model_uses_mrope=False,
        can_skip_drafts=True,
        local_prefill_interval=1,
        num_executed_steps=0,
    )
    sched.kv_cache_manager.allocate_slots.return_value = NS()
    sched.kv_cache_manager.take_kv_cache_block_copies.return_value = ([], [])
    sched.kv_cache_manager.get_num_common_prefix_blocks.return_value = [0]
    sched._make_cached_request_data = Mock(return_value=NS())
    sched._get_new_block_ids_to_zero = Mock(return_value=None)
    result = sched.schedule()
    assert result.num_scheduled_tokens == {"prepare": 8}
    assert result.scheduled_spec_decode_tokens == {}
    assert result.skip_drafts
    sched.kv_cache_manager.allocate_slots.assert_called_once_with(
        req, 8, num_lookahead_tokens=0
    )


def test_failed_offload_read_cannot_acknowledge_prefill(engine):
    req = request(engine)
    sched = scheduler(engine, [req])
    pending = step(prepare=8)
    sched._update_after_schedule(pending)
    sched._handle_failed_recving = Mock(return_value={"prepare"})
    sched._update_from_kv_xfer_finished = Mock()
    kv = NS(failed_recving={"prepare"}, invalid_block_ids=None, kv_connector_stats=None)
    sched.recompute_kv_load_failures = True
    assert sched.update_from_output(pending, output("prepare", kv=kv)) == {}
    assert not req.is_finished()
    sched.kv_cache_manager.cache_blocks.assert_not_called()
    sched.kv_cache_manager.free.assert_not_called()


def test_prefill_stats_count_prompt_and_time_without_a_fictitious_first_token(engine):
    stats = engine.IterationStats()
    req_stats = engine.RequestStateStats(arrival_time=time.time() - 1)
    prefill = engine.PrefillStats(num_prompt_tokens=8, num_computed_tokens=8)
    result = engine.EngineCoreOutput(
        request_id="prepare",
        new_token_ids=[],
        finish_reason=engine.FinishReason.STOP,
        prefill_stats=prefill,
    )
    now = time.monotonic()
    req_stats.scheduled_ts = now - 0.2
    stats.update_from_output(
        result, now, True, req_stats, Mock(), None, prefill_only=True
    )
    assert stats.prompt_token_stats.total == 8
    assert stats.num_generation_tokens == req_stats.num_generation_tokens == 0
    assert stats.time_to_first_tokens_iter == []
    assert req_stats.first_token_latency == 0
    stats.update_from_finished_request(
        engine.FinishReason.STOP, "prepare", 8, 1, req_stats
    )
    final = stats.finished_requests[0]
    assert final.prefill_time == pytest.approx(0.2)
    assert final.decode_time == 0 and final.num_generation_tokens == 0


def admission(engine):
    proc = object.__new__(engine.InputProcessor)
    proc.supports_mm_inputs = False
    proc.model_config = NS(
        is_encoder_decoder=False, is_diffusion=False, enable_prompt_embeds=False
    )
    proc.cache_config = NS(enable_prefix_caching=True, kv_sharing_fast_prefill=False)
    proc.speculative_config = None
    proc.vllm_config = NS(
        use_v2_model_runner=True,
        watermark_config=None,
        parallel_config=NS(
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            data_parallel_size=1,
            prefill_context_parallel_size=1,
            decode_context_parallel_size=1,
        ),
    )
    return proc


@pytest.mark.parametrize(
    "owner,name,value",
    [
        ("vllm_config", "use_v2_model_runner", False),
        ("model_config", "is_encoder_decoder", True),
        ("model_config", "enable_prompt_embeds", True),
        ("model_config", "is_diffusion", True),
        ("vllm_config", "watermark_config", object()),
        ("cache_config", "kv_sharing_fast_prefill", True),
        ("cache_config", "enable_prefix_caching", False),
        ("parallel", "tensor_parallel_size", 2),
        ("parallel", "pipeline_parallel_size", 2),
        ("parallel", "data_parallel_size", 2),
        ("parallel", "prefill_context_parallel_size", 2),
        ("parallel", "decode_context_parallel_size", 2),
    ],
)
def test_admission_rejects_unsupported_modes_before_core_queue(
    engine, owner, name, value
):
    proc = admission(engine)
    proc.validate_prefill_only_config()
    target = (
        proc.vllm_config.parallel_config
        if owner == "parallel"
        else getattr(proc, owner)
    )
    setattr(target, name, value)
    with pytest.raises(ValidationError, match="prefill_only requires"):
        proc.validate_prefill_only_config()


def test_admission_without_optin_still_requires_compatible_speculator(
    engine, monkeypatch
):
    proc = admission(engine)
    proc.validate_prefill_only_config()
    proc.speculative_config = NS(
        use_gemma4_mtp=lambda: True,
        draft_sample_method="greedy",
        enable_adaptive_verification=False,
    )
    monkeypatch.delenv("MALLOW_PROMPT_LOOKUP_MAX", raising=False)
    with pytest.raises(ValidationError, match="speculator"):
        proc.validate_prefill_only_config()
    monkeypatch.setenv("MALLOW_PROMPT_LOOKUP_MAX", "8")
    proc.validate_prefill_only_config()
    proc.speculative_config.enable_adaptive_verification = True
    with pytest.raises(ValidationError, match="speculator"):
        proc.validate_prefill_only_config()


def test_output_processor_delivers_terminal_empty_completion_and_removes_request(
    engine,
):
    req = request(engine)
    wire = NS(
        sampling_params=req.sampling_params,
        pooling_params=None,
        external_req_id="public",
        request_id="internal",
        lora_request=None,
        prompt_token_ids=list(range(8)),
        prompt_embeds=None,
        arrival_time=0.0,
        resumable=False,
    )
    queue = Mock()
    state = engine.RequestState.from_new_request(
        None, wire, None, None, 0, queue, False, 1
    )
    proc = object.__new__(engine.OutputProcessor)
    proc.__dict__.update(
        request_states={"internal": state},
        external_req_ids={"public": ["internal"]},
        parent_requests={},
        tracing_enabled=False,
        lora_states=Mock(),
    )
    proc._update_stats_from_output = Mock()
    proc._update_stats_from_finished = Mock()
    proc._update_admission_stats = Mock()
    core = engine.EngineCoreOutput(
        request_id="internal",
        new_token_ids=[],
        finish_reason=engine.FinishReason.STOP,
    )
    result = proc.process_outputs([core])
    final = queue.put.call_args.args[0]
    assert final.finished and final.outputs[0].token_ids == []
    assert not proc.request_states and not proc.external_req_ids
    assert result.reqs_to_abort == []


def test_output_processor_abort_is_an_empty_completion_not_pooling(engine):
    req = request(engine)
    wire = NS(
        sampling_params=req.sampling_params,
        pooling_params=None,
        external_req_id="public",
        request_id="internal",
        lora_request=None,
        prompt_token_ids=list(range(8)),
        prompt_embeds=None,
        arrival_time=0.0,
        resumable=False,
    )
    queue = Mock()
    state = engine.RequestState.from_new_request(
        None, wire, None, None, 0, queue, False, 1
    )
    proc = object.__new__(engine.OutputProcessor)
    proc.__dict__.update(
        request_states={"internal": state},
        external_req_ids={"public": ["internal"]},
        parent_requests={},
        lora_states=Mock(),
    )
    proc._update_admission_stats = Mock()
    assert proc.abort_requests(["public"], internal=False) == ["internal"]
    final = queue.put.call_args.args[0]
    assert final.finished and final.outputs[0].finish_reason == "abort"
    assert final.outputs[0].token_ids == []
