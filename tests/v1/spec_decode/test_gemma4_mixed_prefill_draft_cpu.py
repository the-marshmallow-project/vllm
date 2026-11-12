# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU checks of the actual compact Gemma inputs against full-query references.

The existing model tests require CUDA. These tests load the pure packing module
directly, use real CPU tensors, and compare to the inputs consumed by the
original full-query first pass. They do not validate CUDA kernels or weights.
"""

import ast
import importlib.util
import os
import sys
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tests.v1.worker import test_prefill_only_cpu as worker_tests

endpoint_native = worker_tests.native


@pytest.fixture(scope="module")
def packing():
    path = Path(__file__).resolve().parents[3] / ("vllm/v1/worker/gpu/spec_decode/gemma4/prefill.py")
    name = "_gemma4_compact_prefill_inputs_cpu"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def inputs():
    sizes = np.array([3, 6, 2, 4], dtype=np.int32)
    starts = np.concatenate(([0], np.cumsum(sizes))).astype(np.int32)
    state_rows = np.array([5, 2, 7, 1], dtype=np.intp)
    seq_lens = torch.tensor([103, 206, 302, 404], dtype=torch.int32)
    positions = torch.full((18,), 9999, dtype=torch.int64)
    for row, size in enumerate(sizes):
        start = int(starts[row])
        positions[start : start + size] = torch.arange(int(seq_lens[row]) - int(size), int(seq_lens[row]))
    batch = SimpleNamespace(
        num_reqs=4,
        num_reqs_after_padding=6,
        num_tokens=15,
        num_tokens_after_padding=18,
        idx_mapping_np=state_rows,
        idx_mapping=torch.from_numpy(state_rows.astype(np.int32)),
        num_scheduled_tokens=sizes,
        query_start_loc_np=starts,
        query_start_loc=torch.from_numpy(starts),
        seq_lens=seq_lens,
        seq_lens_cpu_upper_bound=torch.tensor([110, 220, 310, 420]),
        positions=positions,
        input_ids=torch.arange(18, dtype=torch.int32) + 100,
    )
    return SimpleNamespace(
        batch=batch,
        hidden=torch.arange(18 * 4, dtype=torch.float64).reshape(18, 4) / 100,
        sampled=torch.tensor([1, 3, 0, 0], dtype=torch.int32),
        rejected=torch.tensor([0, 2, 0, 1], dtype=torch.int32),
        last_sampled=torch.arange(8, dtype=torch.int32) * 100 + 1,
        next_prefill=torch.stack((torch.arange(8) * 10 + 2000, torch.arange(8) * 10 + 3000)).to(torch.int32),
    )


def select(packing, inputs, rows):
    return packing.select_gemma4_prefill_inputs(
        inputs.batch,
        np.asarray(rows, dtype=np.intp),
        inputs.hidden,
        inputs.sampled,
        inputs.rejected,
        inputs.last_sampled,
        inputs.next_prefill,
    )


@pytest.mark.parametrize("rows", [(1, 3), (3, 1), (1,), (0, 2)])
def test_compaction_preserves_request_order_and_last_valid_query(packing, inputs, rows):
    result = select(packing, inputs, rows)
    expected_indices = [int(inputs.batch.query_start_loc_np[row + 1]) - int(inputs.rejected[row]) - 1 for row in rows]
    expected_state_rows = [int(inputs.batch.idx_mapping_np[row]) for row in rows]
    expected_tokens = [
        int(inputs.last_sampled[state]) if inputs.sampled[row] > 0 else int(inputs.next_prefill[0, state])
        for row, state in zip(rows, expected_state_rows)
    ]
    assert result.row_indices.tolist() == list(rows)
    assert result.idx_mapping.tolist() == expected_state_rows
    assert result.last_token_indices.tolist() == expected_indices
    assert result.input_ids.tolist() == expected_tokens
    assert result.positions.tolist() == inputs.batch.positions[expected_indices].tolist()
    torch.testing.assert_close(result.hidden_states, inputs.hidden[expected_indices])
    assert result.target_seq_lens.tolist() == inputs.batch.seq_lens[list(rows)].tolist()
    assert result.seq_lens.tolist() == [int(inputs.batch.seq_lens[row]) - int(inputs.rejected[row]) for row in rows]
    assert result.seq_lens_cpu_upper_bound.tolist() == (inputs.batch.seq_lens_cpu_upper_bound[list(rows)].tolist())
    assert result.seq_lens_cpu_upper_bound.device.type == "cpu"
    assert result.num_sampled.tolist() == inputs.sampled[list(rows)].tolist()
    assert result.num_rejected.tolist() == inputs.rejected[list(rows)].tolist()


@pytest.mark.parametrize("rows", [(), (-1,), (4,), (1, 1)])
def test_invalid_selection_fails_before_gathering(packing, inputs, rows):
    with pytest.raises(ValueError, match="unique active request rows"):
        select(packing, inputs, rows)


def test_chunked_prefill_uses_first_lookahead_and_global_request_index(packing, inputs):
    result = select(packing, inputs, (3,))
    assert result.input_ids.tolist() == [2010]
    assert result.input_ids[0] != inputs.next_prefill[1, 1]
    assert result.input_ids[0] != inputs.next_prefill[0, 3]


def test_packing_does_not_modify_target_buffers_or_sampling_state(packing, inputs):
    tensors = [
        inputs.hidden,
        inputs.sampled,
        inputs.rejected,
        inputs.last_sampled,
        inputs.next_prefill,
        inputs.batch.input_ids,
        inputs.batch.positions,
        inputs.batch.seq_lens,
        inputs.batch.query_start_loc,
        inputs.batch.idx_mapping,
    ]
    before = [tensor.clone() for tensor in tensors]
    select(packing, inputs, (3, 1))
    for old, current in zip(before, tensors):
        torch.testing.assert_close(current, old, rtol=0, atol=0)


def test_preallocated_device_row_selector_is_reused(packing, inputs):
    rows = np.array([3, 1], dtype=np.intp)
    persistent_rows = torch.tensor(rows, dtype=torch.int64)
    result = packing.select_gemma4_prefill_inputs(
        inputs.batch,
        rows,
        inputs.hidden,
        inputs.sampled,
        inputs.rejected,
        inputs.last_sampled,
        inputs.next_prefill,
        row_indices=persistent_rows,
    )
    assert result.row_indices.data_ptr() == persistent_rows.data_ptr()
    assert result.idx_mapping.tolist() == [1, 2]


def frozen_query_output(hidden, tokens, positions, seq_lens, state_rows, keys, values, window=None):
    """Small independent frozen-KV attention oracle, not model inference."""
    h = hidden.clone()
    for layer in range(2):
        result = []
        for row, state in enumerate(state_rows):
            length = int(seq_lens[row])
            start = max(0, length - window) if window is not None else 0
            q = h[row] + tokens[row].to(h.dtype) / 1000 + positions[row] / 1000
            scores = keys[layer, state, start:length] @ q
            attended = torch.softmax(scores, dim=0) @ values[layer, state, start:length]
            result.append(torch.tanh(h[row] + attended))
        h = torch.stack(result)
    return h


@pytest.mark.parametrize("rows", [(1, 3), (3, 1), (0, 2)])
@pytest.mark.parametrize("window", [None, 2])
def test_one_query_matches_full_valid_queries_over_frozen_target_kv(packing, inputs, rows, window):
    generator = torch.Generator().manual_seed(73)
    keys = torch.randn(2, 8, 512, 4, dtype=torch.float64, generator=generator)
    values = torch.randn(2, 8, 512, 4, dtype=torch.float64, generator=generator)
    expected = []
    for row in rows:
        state = int(inputs.batch.idx_mapping_np[row])
        start = int(inputs.batch.query_start_loc_np[row])
        end = int(inputs.batch.query_start_loc_np[row + 1]) - int(inputs.rejected[row])
        next_token = inputs.last_sampled[state] if inputs.sampled[row] > 0 else inputs.next_prefill[0, state]
        shifted = torch.cat((inputs.batch.input_ids[start + 1 : end], next_token[None]))
        positions = inputs.batch.positions[start:end]
        full = frozen_query_output(
            inputs.hidden[start:end],
            shifted,
            positions,
            positions + 1,
            [state] * (end - start),
            keys,
            values,
            window,
        )
        expected.append(full[-1])
    packed = select(packing, inputs, rows)
    actual = frozen_query_output(
        packed.hidden_states,
        packed.input_ids,
        packed.positions,
        packed.seq_lens,
        packed.idx_mapping.tolist(),
        keys,
        values,
        window,
    )
    torch.testing.assert_close(actual, torch.stack(expected), rtol=1e-12, atol=1e-12)


class GraphMode(Enum):
    NONE = 0
    FULL = 1
    FULL_DECODE_ONLY = 2

    def decode_mode(self):
        return self.FULL if self != self.NONE else self.NONE


@pytest.mark.parametrize("dynamic,expected", [(False, [9]), (True, [9, 3])])
def test_native_decode_capture_uses_mtp_steps_and_not_lookup_slots(dynamic, expected):
    path = Path(__file__).resolve().parents[3] / (
        "vllm/v1/worker/gpu/spec_decode/target_dependent_ar/cudagraph_utils.py"
    )
    tree = ast.parse(path.read_text())
    tree.body = [
        ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
        *(node for node in tree.body if isinstance(node, ast.ClassDef)),
    ]

    @dataclass(frozen=True)
    class Descriptor:
        cg_mode: GraphMode
        num_reqs: int = 1
        num_tokens: int = 1
        max_query_len: int = 1

    class Manager:
        def capture(self, make_forward, desc):
            for descriptors in self._capture_descs.values():
                for descriptor in descriptors:
                    make_forward(descriptor, warmup=False)(descriptor.cg_mode)

    namespace = {
        "BatchExecutionDescriptor": Descriptor,
        "CudaGraphManager": Manager,
        "CUDAGraphMode": GraphMode,
        "dataclass": dataclass,
        "replace": replace,
        "build_dynamic_sd_schedule_lookup": lambda *a, **kw: [0, 1, 3, 10, 15],
        "prepare_inputs_to_capture": lambda *a, **kw: ({}, {}),
    }
    exec(compile(ast.fix_missing_locations(tree), str(path), "exec"), namespace)
    manager = namespace["SpeculatorCudaGraphManager"]()
    manager.vllm_config = SimpleNamespace(
        num_speculative_tokens=15,
        speculative_config=SimpleNamespace(num_speculative_tokens_per_batch_size={}),
    )
    manager.max_num_reqs = manager.decode_query_len = manager.dp_size = 1
    manager._capture_descs = {GraphMode.FULL: [Descriptor(GraphMode.FULL)]}
    captured_steps = []

    def capture(*args, num_speculative_steps):
        captured_steps.append(num_speculative_steps)

    manager.capture(
        capture,
        None,
        None,
        None,
        None,
        None,
        specialize_spec_tokens=dynamic,
        num_speculative_steps=9,
    )
    assert captured_steps == expected


@pytest.fixture
def compact_runtime(packing, inputs):
    """Run changed orchestration; only native inference/kernel APIs are spies."""
    root = Path(__file__).resolve().parents[3]
    path = root / "vllm/v1/worker/gpu/spec_decode/gemma4/speculator.py"
    node = next(
        n for n in ast.parse(path.read_text()).body if isinstance(n, ast.ClassDef) and n.name == "Gemma4Speculator"
    )

    class Parent:
        def capture(self):
            self.original_capture_calls += 1

        def init_cudagraph_manager(self, mode):
            self.prefill_cudagraph_manager = SimpleNamespace(kind="original")
            self.decode_cudagraph_manager = SimpleNamespace(kind="decode")

    class Graph:
        def __init__(self, config=None, device=None, mode=None, decode_query_len=1):
            self.mode = mode
            self.decode_query_len = decode_query_len
            self.use_breakable_cg = False
            self.owner = None
            self.capture_args = None
            self.capture_error = False
            self.replays = 0

        def capture(self, *args, **kwargs):
            self.capture_args = args
            if self.capture_error:
                raise RuntimeError("capture failure")
            args[0](1, 1, {}, {}, None, GraphMode.FULL, num_speculative_steps=15)

        def run_fullgraph(self, desc):
            assert self.owner.metadata_calls
            self.replays += 1
            self.owner._prefill(desc.num_reqs, desc.num_tokens, {}, {}, None, desc.cg_mode)

        def specialize_spec_tokens(self, desc, steps):
            self.specialized_steps = steps
            return desc

    def dispatch(manager, num_reqs, num_tokens, **kwargs):
        assert kwargs["uniform_token_count"] == 1
        mode = (
            GraphMode.FULL if manager is runtime.compact_prefill_cudagraph_manager and runtime.full else GraphMode.NONE
        )
        padded = 4 if mode == GraphMode.FULL else num_reqs
        return SimpleNamespace(cg_mode=mode, num_tokens=padded, num_reqs=padded), None

    def prepare_decode(drafts, original_seq, rejected, buffers, positions, *args, **kwargs):
        # Assert the real orchestration passes original S, rather than S-R again.
        expected = inputs.batch.seq_lens[runtime.rows]
        torch.testing.assert_close(original_seq, expected)
        torch.testing.assert_close(rejected, inputs.rejected[runtime.rows])
        assert kwargs["advance_draft_positions"] is False
        n = len(drafts)
        buffers.input_ids[:n].copy_(drafts)
        buffers.seq_lens[:n].copy_(original_seq - rejected)
        positions[:n].add_(1)

    namespace = {
        "TargetDependentARSpeculator": Parent,
        "CUDAGraphMode": GraphMode,
        "SpeculatorCudaGraphManager": Graph,
        "can_compact_gemma4_prefill": packing.can_compact_gemma4_prefill,
        "select_gemma4_prefill_inputs": packing.select_gemma4_prefill_inputs,
        "dispatch_cg_and_sync_dp": dispatch,
        "prepare_decode_inputs": prepare_decode,
        "os": os,
        "build_slot_mappings_by_layer": lambda slots, config: {"draft": slots},
        "torch": torch,
        "np": np,
        "Any": object,
    }
    tree = ast.Module(
        body=[
            ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
            node,
        ],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(tree), str(path), "exec"), namespace)
    runtime = namespace["Gemma4Speculator"]()
    runtime.max_num_reqs = 8
    runtime.max_model_len = 512
    runtime.num_speculative_steps = 9
    runtime.device = torch.device("cpu")
    runtime.dp_size = 1
    runtime.dp_rank = 0
    runtime.full = False
    runtime.rows = [1, 3]
    runtime.active_count = 2
    runtime.skip_prefill_draft = True
    runtime.vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(enable_prompt_embeds=False),
        parallel_config=SimpleNamespace(data_parallel_size=1),
    )
    runtime.original_capture_calls = 0
    runtime.metadata_calls = []
    runtime.prefill_calls = []
    runtime.prefill_postprocess = []
    runtime.decode_calls = []
    runtime.fail_at = None
    runtime.input_buffers = SimpleNamespace(
        input_ids=torch.full((32,), 777, dtype=torch.int32),
        positions=torch.full((32,), 777, dtype=torch.int64),
        seq_lens=torch.full((8,), 777, dtype=torch.int32),
        query_start_loc=torch.full((9,), 777, dtype=torch.int32),
        is_padding=torch.zeros(32, dtype=torch.bool),
    )
    runtime.target_input_buffers = object()
    runtime.hidden_states = torch.full((32, 4), 777, dtype=torch.float64)
    runtime.last_token_indices = torch.full((8,), 777, dtype=torch.int64)
    runtime.sample_src_positions = torch.zeros(8, dtype=torch.int64)
    runtime.current_draft_step = torch.tensor(777)
    runtime.idx_mapping = torch.full((8,), 777, dtype=torch.int32)
    runtime.temperature = torch.zeros(8)
    runtime.seeds = torch.zeros(8, dtype=torch.int64)
    runtime.draft_tokens = torch.full((8, 15), 777, dtype=torch.int64)
    runtime.compact_row_indices = torch.empty(8, dtype=torch.int64)
    runtime.model_state = SimpleNamespace(supports_mm_inputs=False, prompt_embeds_state=None)
    runtime.model = object()
    runtime.attn_groups = object()
    runtime.target_attn_groups = object()
    runtime.kv_cache_config = object()
    runtime.prefill_cudagraph_manager = SimpleNamespace(kind="original")
    runtime.compact_prefill_cudagraph_manager = Graph()
    runtime.compact_prefill_cudagraph_manager.owner = runtime
    runtime.decode_cudagraph_manager = Graph()
    runtime.supports_decode_metadata_update = True
    runtime.pcp_manager = None
    runtime.draft_logits = None

    class Tables:
        def __init__(self):
            self.input_block_tables = [torch.arange(32).reshape(8, 4) + n * 100 for n in range(2)]
            self.slot_mappings = torch.arange(64).reshape(2, 32)

        def gather_block_tables(self, mapping, num_reqs_padded):
            for table in self.input_block_tables:
                table.zero_()
                table[: len(mapping)] = mapping[:, None] * 100 + torch.arange(4)

        def compute_slot_mappings(self, mapping, starts, positions, padded):
            self.slot_mappings.fill_(-1)
            self.slot_mappings[:, : len(mapping)] = positions[: len(mapping)] + mapping * 1000
            return self.slot_mappings[:, :padded]

    runtime.block_tables = Tables()
    runtime.compact_block_table_backup = [torch.empty_like(t) for t in runtime.block_tables.input_block_tables]
    runtime.compact_slot_mapping_backup = torch.empty_like(runtime.block_tables.slot_mappings)
    runtime._prepare_eplb_forward = lambda n: None
    runtime.on_prefill_begin = lambda n: None
    runtime.on_prefill_end = lambda n: None
    runtime.on_multi_step_decode_begin = lambda n: None
    runtime.on_multi_step_decode_end = lambda n: None

    # Reuse the real per-request copy code so global RNG indices are tested too.
    path = root / "vllm/v1/worker/gpu/spec_decode/speculator.py"
    base = next(
        n for n in ast.parse(path.read_text()).body if isinstance(n, ast.ClassDef) and n.name == "DraftModelSpeculator"
    )
    method = next(n for n in base.body if isinstance(n, ast.FunctionDef) and n.name == "_copy_request_inputs")
    exec(
        compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"),
        namespace,
    )
    runtime._copy_request_inputs = lambda *args: namespace["_copy_request_inputs"](runtime, *args)

    def metadata(**kwargs):
        runtime.metadata_calls.append(kwargs)
        n = kwargs["num_reqs"]
        assert kwargs["num_query_per_req"] == 1 and kwargs["step"] == 0
        assert runtime.input_buffers.query_start_loc.tolist() == [
            0,
            1,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
        ]
        assert runtime.idx_mapping[n:].tolist() == [-1] * (8 - n)
        assert not runtime.input_buffers.seq_lens[n:].any()
        return {"step": 0}

    def run_model(padded, *args, **kwargs):
        n = runtime.active_count
        expected = select(packing, inputs, runtime.rows)
        torch.testing.assert_close(runtime.input_buffers.input_ids[:n], expected.input_ids)
        torch.testing.assert_close(runtime.input_buffers.positions[:n], expected.positions)
        torch.testing.assert_close(runtime.input_buffers.seq_lens[:n], expected.seq_lens)
        torch.testing.assert_close(runtime.hidden_states[:n], expected.hidden_states)
        assert not runtime.hidden_states[n:padded].any()
        assert not runtime.input_buffers.input_ids[n:padded].any()
        assert not runtime.input_buffers.positions[n:padded].any()
        assert runtime.current_draft_step == 0
        if runtime.fail_at == "prefill":
            raise RuntimeError("prefill failure")
        output = runtime.hidden_states[:padded] + 100
        return output, output

    def sample_draft(hidden, positions, mapping, *args):
        expected = select(packing, inputs, runtime.rows)
        torch.testing.assert_close(positions[:2], expected.positions + 1)
        assert mapping[:2].tolist() == expected.idx_mapping.tolist()
        assert all(index in (-1, *expected.idx_mapping.tolist()) for index in mapping)
        return torch.where(mapping >= 0, mapping * 100, 0).to(torch.int64)

    path = root / "vllm/v1/worker/gpu/spec_decode/target_dependent_ar/speculator.py"
    base = next(
        n
        for n in ast.parse(path.read_text()).body
        if isinstance(n, ast.ClassDef) and n.name == "TargetDependentARSpeculator"
    )
    method = next(n for n in base.body if isinstance(n, ast.FunctionDef) and n.name == "_prefill")
    exec(
        compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"),
        namespace,
    )
    native_prefill = namespace["_prefill"]

    def prefill(n, padded, *args, **kwargs):
        runtime.prefill_calls.append((n, padded))
        # FULL replay really uses the captured request count, including padding.
        result = native_prefill(runtime, n, padded, *args, **kwargs)
        runtime.prefill_postprocess.append(
            (
                runtime.hidden_states[:n].clone(),
                runtime.input_buffers.positions[:n].clone(),
            )
        )
        return result

    def decode(n, skip, desc, sync, bounds, steps):
        runtime.decode_calls.append(n)
        expected = select(packing, inputs, runtime.rows)
        torch.testing.assert_close(runtime.input_buffers.seq_lens[:n], expected.seq_lens)
        torch.testing.assert_close(runtime.sample_src_positions[:n], expected.positions + 2)
        if runtime.fail_at == "decode":
            raise RuntimeError("decode failure")
        assert runtime.decode_cudagraph_manager.specialized_steps == steps
        for step in range(1, steps):
            runtime.draft_tokens[:n, step] = runtime.idx_mapping[:n] * 100 + step

    runtime._build_uniform_attn_metadata = metadata
    runtime._run_model = run_model
    runtime.sample_draft = sample_draft
    runtime._prefill = prefill
    runtime._fused_multi_step_decode = decode
    runtime._multi_step_decode = decode
    return runtime, namespace


def run_compact(runtime, inputs, steps=9):
    return runtime.propose_for_active_requests(
        inputs.batch,
        np.asarray(runtime.rows, dtype=np.intp),
        inputs.hidden,
        inputs.sampled,
        inputs.rejected,
        inputs.last_sampled,
        inputs.next_prefill,
        torch.arange(8, dtype=torch.float32) / 10,
        torch.arange(8, dtype=torch.int64) * 101,
        steps,
    )


@pytest.mark.parametrize("full", [False, True])
@pytest.mark.parametrize("steps", [1, 9])
def test_actual_mixed_path_runs_only_selected_queries_and_preserves_buffers(compact_runtime, inputs, full, steps):
    runtime, _ = compact_runtime
    runtime.full = full
    runtime.num_speculative_steps = steps
    original_manager = runtime.prefill_cudagraph_manager
    tables = [t.clone() for t in runtime.block_tables.input_block_tables]
    slots = runtime.block_tables.slot_mappings.clone()
    pointers = [
        runtime.input_buffers.input_ids.data_ptr(),
        runtime.hidden_states.data_ptr(),
    ]
    for rows in ([1, 3], [3, 1]):
        runtime.rows = rows
        result = run_compact(runtime, inputs, steps)
        state_rows = inputs.batch.idx_mapping[list(rows)]
        expected = state_rows[:, None] * 100 + torch.arange(steps)
        torch.testing.assert_close(result[:, :steps], expected.to(torch.int64))
        assert result.shape == (2, 15)
        assert runtime.prefill_cudagraph_manager is original_manager
        assert runtime.idx_mapping[:2].tolist() == state_rows.tolist()
        torch.testing.assert_close(runtime.seeds, torch.arange(8) * 101)
        assert pointers == [
            runtime.input_buffers.input_ids.data_ptr(),
            runtime.hidden_states.data_ptr(),
        ]
        for actual, saved in zip(runtime.block_tables.input_block_tables, tables):
            torch.testing.assert_close(actual, saved, rtol=0, atol=0)
        torch.testing.assert_close(runtime.block_tables.slot_mappings, slots, rtol=0, atol=0)
    assert runtime.prefill_calls == [(4, 4) if full else (2, 2)] * 2
    assert runtime.decode_calls == ([2, 2] if steps > 1 else [])
    assert runtime.compact_prefill_cudagraph_manager.replays == (2 if full else 0)


@pytest.mark.parametrize("full", [False, True])
def test_compact_attention_preserves_target_sliding_window_null_blocks(
    compact_runtime, inputs, full
):
    runtime, _ = compact_runtime
    runtime.full = full
    full_table, sliding_table = runtime.block_tables.input_block_tables
    for table in (full_table, sliding_table):
        table[:inputs.batch.num_reqs] = (
            inputs.batch.idx_mapping[:, None] * 100 + torch.arange(table.shape[1])
        )
    # Native target gather masks these expired pages to the null block.
    sliding_table[1, :1] = 0
    sliding_table[3, :2] = 0
    pointers = [table.data_ptr() for table in (full_table, sliding_table)]
    snapshots = [table.clone() for table in (full_table, sliding_table)]
    run_model = runtime._run_model

    def check_compact_tables(*args, **kwargs):
        for table, snapshot, pointer in zip(
            runtime.block_tables.input_block_tables, snapshots, pointers, strict=True
        ):
            assert table.data_ptr() == pointer
            torch.testing.assert_close(table[:2], snapshot[runtime.rows])
            assert not table[2:].any()
        return run_model(*args, **kwargs)

    runtime._run_model = check_compact_tables
    run_compact(runtime, inputs)
    for table, snapshot in zip(
        runtime.block_tables.input_block_tables, snapshots, strict=True
    ):
        torch.testing.assert_close(table, snapshot)


@pytest.mark.parametrize("failure", ["prefill", "decode"])
def test_compact_failure_restores_shared_target_tables_and_manager(compact_runtime, inputs, failure):
    runtime, _ = compact_runtime
    runtime.fail_at = failure
    manager = runtime.prefill_cudagraph_manager
    tables = [t.clone() for t in runtime.block_tables.input_block_tables]
    slots = runtime.block_tables.slot_mappings.clone()
    with pytest.raises(RuntimeError, match=failure):
        run_compact(runtime, inputs)
    assert runtime.prefill_cudagraph_manager is manager
    for actual, saved in zip(runtime.block_tables.input_block_tables, tables):
        torch.testing.assert_close(actual, saved, rtol=0, atol=0)
    torch.testing.assert_close(runtime.block_tables.slot_mappings, slots, rtol=0, atol=0)


@pytest.mark.parametrize("fail", [False, True])
def test_compact_capture_uses_own_buffers_and_restores_original_manager(compact_runtime, fail):
    runtime, _ = compact_runtime
    original = runtime.prefill_cudagraph_manager
    manager = runtime.compact_prefill_cudagraph_manager
    manager.capture_error = fail
    runtime._prefill = lambda n, padded, *args: runtime.prefill_calls.append((n, padded))
    if fail:
        with pytest.raises(RuntimeError, match="capture failure"):
            runtime.capture()
    else:
        runtime.capture()
    assert runtime.original_capture_calls == 1
    assert runtime.prefill_calls == ([] if fail else [(1, 1)])
    assert manager.capture_args[2] is runtime.input_buffers
    assert manager.capture_args[2] is not runtime.target_input_buffers
    assert manager.capture_args[4] is runtime.attn_groups
    assert manager.capture_args[4] is not runtime.target_attn_groups
    assert runtime.prefill_cudagraph_manager is original
    assert not runtime.last_token_indices.any()
    assert not runtime.idx_mapping.any()
    assert runtime.current_draft_step == 0


@pytest.mark.parametrize(
    "enabled,mode,expected",
    [
        (True, GraphMode.FULL, GraphMode.FULL_DECODE_ONLY),
        (True, GraphMode.NONE, GraphMode.NONE),
        (False, GraphMode.FULL, None),
    ],
)
def test_compact_graph_manager_is_separate_and_opt_in(compact_runtime, enabled, mode, expected):
    runtime, _ = compact_runtime
    runtime.skip_prefill_draft = enabled
    runtime.init_cudagraph_manager(mode)
    assert runtime.prefill_cudagraph_manager.kind == "original"
    assert runtime.decode_cudagraph_manager.kind == "decode"
    manager = runtime.compact_prefill_cudagraph_manager
    if expected is None:
        assert manager is None
    else:
        assert manager.mode == expected
        assert manager.decode_query_len == 1
        assert manager is not runtime.prefill_cudagraph_manager


@pytest.mark.parametrize(
    "state,config,expected",
    [
        (None, SimpleNamespace(enable_prompt_embeds=False), False),
        (SimpleNamespace(), SimpleNamespace(enable_prompt_embeds=False), False),
        (
            SimpleNamespace(supports_mm_inputs=True, prompt_embeds_state=None),
            SimpleNamespace(enable_prompt_embeds=False),
            False,
        ),
        (
            SimpleNamespace(supports_mm_inputs=False, prompt_embeds_state=object()),
            SimpleNamespace(enable_prompt_embeds=False),
            False,
        ),
        (SimpleNamespace(supports_mm_inputs=False), None, False),
        (
            SimpleNamespace(supports_mm_inputs=False),
            SimpleNamespace(enable_prompt_embeds=True),
            False,
        ),
        (SimpleNamespace(supports_mm_inputs=False), SimpleNamespace(), False),
        (
            SimpleNamespace(supports_mm_inputs=False, prompt_embeds_state=None),
            SimpleNamespace(enable_prompt_embeds=False),
            True,
        ),
    ],
)
def test_only_known_text_state_can_compact(packing, state, config, expected):
    assert packing.can_compact_gemma4_prefill(state, config) is expected


def test_padded_full_replay_postprocessing_keeps_live_results_valid(compact_runtime, inputs):
    runtime, _ = compact_runtime
    runtime.full = True
    result = run_compact(runtime, inputs)
    assert runtime.prefill_calls == [(4, 4)]
    assert runtime.idx_mapping.tolist() == [2, 1, -1, -1, -1, -1, -1, -1]
    assert not runtime.input_buffers.seq_lens[2:].any()
    # The native _prefill gathers last_token_indices for all captured rows.
    # Padded indices are zero, so padded feedback is a copy of row zero; it
    # need not remain zero after replay because the invalid rows are ignored.
    hidden, positions = runtime.prefill_postprocess[0]
    torch.testing.assert_close(hidden[2:4], hidden[:1].expand(2, -1))
    assert positions[2:4].tolist() == [203, 203]
    assert result[:, :9].tolist() == [
        [200 + step for step in range(9)],
        [100 + step for step in range(9)],
    ]


@pytest.mark.parametrize("full", [False, True])
@pytest.mark.parametrize("budget", [0, 1, 3, 9, 10, 15])
@pytest.mark.parametrize("explicit_prefill", [False, True])
def test_public_mixed_proposal_excludes_prefill_from_model_and_lookup_then_scatters(
    compact_runtime, inputs, full, budget, explicit_prefill
):
    runtime, namespace = compact_runtime
    runtime.full = full
    path = Path(__file__).resolve().parents[3] / ("vllm/v1/worker/gpu/spec_decode/gemma4/prompt_lookup.py")
    node = next(
        n
        for n in ast.parse(path.read_text()).body
        if isinstance(n, ast.ClassDef) and n.name == "Gemma4PromptLookupSpeculator"
    )
    lookup_calls = []

    def lookup(tokens, lengths, mapping, drafts, out, extended, sampled, rejected, *args):
        lookup_calls.append(mapping.tolist())
        assert mapping.tolist() == [2, 1]
        assert len(drafts) == len(out) == 2
        assert sampled.tolist() == [3, 0]
        assert rejected.tolist() == [2, 1]
        steps = min(9, budget)
        assert args[1] == steps
        out[:, :steps].copy_(drafts[:, :steps])
        out[:, steps:].copy_(drafts[:, steps - 1 : steps].expand(-1, 15 - steps))
        extended[mapping.to(torch.int64)] = 1

    namespace.update(
        Gemma4Speculator=type(runtime),
        prompt_lookup=lookup,
        LOG_EVERY=4096,
    )
    tree = ast.Module(
        body=[
            ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
            node,
        ],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(tree), str(path), "exec"), namespace)
    cls = namespace["Gemma4PromptLookupSpeculator"]
    proposal = cls.__new__(cls)
    # Sharing the fixture's state dictionary also keeps inference spies bound
    # to the same persistent tensors while the public subclass dispatches.
    proposal.__dict__ = runtime.__dict__
    proposal.num_slots = 15
    proposal.skip_single_token_draft = not explicit_prefill
    proposal.compact_prefill_cudagraph_manager.owner = proposal
    proposal.prompt_len_np = np.arange(8, dtype=np.int32) * 100
    proposal.max_seq_len_np = proposal.prompt_len_np + 128
    proposal.max_seq_len_np[[5, 7]] = proposal.prompt_len_np[[5, 7]] + int(not explicit_prefill)
    proposal.prefill_only_np = np.zeros(8, dtype=bool)
    proposal.prefill_only_np[[5, 7]] = explicit_prefill
    proposal.draft_watermarker = None
    proposal.token_ids = torch.zeros(8, 512, dtype=torch.int32)
    proposal.total_len = torch.zeros(8, dtype=torch.int32)
    proposal.lookup_drafts = torch.full((8, 15), 999, dtype=torch.int64)
    proposal.full_batch_drafts = torch.full((8, 15), 999, dtype=torch.int64)
    proposal.lookup_extended = torch.ones(8, dtype=torch.int32)
    proposal.lookup_stats = torch.zeros(6, dtype=torch.int64)
    proposal.lookup_min = 1
    proposal.lookup_max = 8
    proposal.prefill_req_indices = torch.empty(8, dtype=torch.int64)
    proposal.steps = 0
    result = proposal.propose(
        inputs.batch,
        {},
        {},
        inputs.hidden,
        None,
        inputs.sampled,
        inputs.rejected,
        inputs.last_sampled,
        inputs.next_prefill,
        torch.arange(8, dtype=torch.float32),
        torch.arange(8, dtype=torch.int64) * 101,
        num_speculative_tokens=budget,
    )
    assert result.shape == (4, 15)
    if budget == 0:
        assert torch.all(result == -1)
        assert not runtime.prefill_calls and not runtime.decode_calls
        assert not lookup_calls and proposal.steps == 0
        return
    steps = min(9, budget)
    assert not result[[0, 2]].any()
    assert result[1, :budget].tolist() == [200 + min(step, steps - 1) for step in range(budget)]
    assert result[3, :budget].tolist() == [100 + min(step, steps - 1) for step in range(budget)]
    assert lookup_calls == [[2, 1]]
    assert proposal.lookup_extended[[5, 7]].tolist() == [0, 0]
    assert proposal.steps == 1
    assert proposal.prefill_calls == [(4, 4) if full else (2, 2)]


@pytest.fixture(scope="module")
def completion_native(endpoint_native):
    worker_tests.load_nodes("vllm/v1/outputs.py", {"LogprobsLists", "SamplingMaskLists"}, endpoint_native)
    return endpoint_native


@pytest.mark.parametrize("seed", range(16))
@pytest.mark.parametrize("mask_slots", [1, 16])
def test_mixed_sampler_matches_full_reference_for_ordinary_replies(completion_native, seed, mask_slots):
    ns = completion_native
    rng = np.random.default_rng(600 + seed)
    count = int(rng.integers(2, 12))
    slots = rng.choice(32, size=count, replace=False)
    warm = np.zeros(32, dtype=bool)
    warm[rng.choice(slots, size=rng.integers(1, count), replace=False)] = True
    queries = rng.integers(1, 18, size=count)
    logit_counts = rng.integers(1, np.minimum(queries, mask_slots) + 1)
    logit_counts[warm[slots]] = 1
    if mask_slots > 1:
        ordinary = np.flatnonzero(~warm[slots])[0]
        queries[ordinary] = max(queries[ordinary], 2)
        logit_counts[ordinary] = max(logit_counts[ordinary], 2)
    query = np.concatenate(([0], np.cumsum(queries))).astype(np.int32)
    cu = np.concatenate(([0], np.cumsum(logit_counts))).astype(np.int32)
    tokens = int(query[-1])
    padded = tokens + int(rng.integers(1, 8))
    seq_lens = torch.from_numpy(queries + rng.integers(0, 60, size=count))
    unfinished = torch.from_numpy(rng.integers(0, 2, size=count).astype(bool))
    prefill_lengths = seq_lens + unfinished * 8
    positions = torch.from_numpy(
        np.concatenate(
            [np.arange(int(seq_lens[i]) - q, int(seq_lens[i])) for i, q in enumerate(queries)]
            + [np.full(padded - tokens, -1)]
        )
    )
    original = ns["InputBatch"](
        req_ids=[str(slot) for slot in slots],
        num_reqs=count,
        num_reqs_after_padding=count + 1,
        idx_mapping=torch.from_numpy(slots),
        idx_mapping_np=slots,
        expanded_idx_mapping=torch.from_numpy(np.repeat(slots, logit_counts)),
        expanded_local_pos=torch.from_numpy(np.concatenate([np.arange(n) for n in logit_counts])),
        num_scheduled_tokens=queries,
        num_tokens=tokens,
        num_tokens_after_padding=padded,
        num_draft_tokens=int(cu[-1]) - count,
        num_draft_tokens_per_req=logit_counts - 1,
        query_start_loc=torch.tensor([*query, tokens]),
        query_start_loc_np=np.append(query, tokens),
        seq_lens=seq_lens,
        seq_lens_cpu_upper_bound=seq_lens + 2,
        dcp_local_seq_lens=None,
        num_computed_tokens_np=seq_lens.numpy() - queries,
        prefill_len_np=prefill_lengths.numpy(),
        num_computed_prefill_tokens_np=seq_lens.numpy() - queries,
        is_prefilling_np=unfinished.numpy(),
        has_prefill=bool(unfinished.any()),
        decode_graph_eligible=False,
        input_ids=torch.from_numpy(rng.integers(0, 127, size=padded)),
        positions=positions,
        is_padding=positions < 0,
        logits_indices=torch.from_numpy(
            np.concatenate([np.arange(query[i + 1] - n, query[i + 1]) for i, n in enumerate(logit_counts)])
        ),
        cu_num_logits=torch.from_numpy(cu),
        cu_num_logits_np=cu,
        has_structured_output_reqs=False,
        prompt_lens=prefill_lengths,
    )
    hidden = torch.from_numpy(rng.integers(0, 100, size=(padded, 2)))
    row_seed = torch.from_numpy(rng.integers(0, 1000, size=32))

    def sample_reference(batch, states):
        # A deterministic sampler oracle keyed by global slot and token position.
        projected = states[batch.logits_indices].sum(1)
        position = batch.positions[batch.logits_indices]
        token = batch.input_ids[batch.logits_indices]
        score = projected + position * 7 + token * 3
        choices = (score + row_seed[batch.expanded_idx_mapping] + batch.expanded_local_pos) % 127
        output = torch.zeros((batch.num_reqs, mask_slots), dtype=torch.int64)
        counts = torch.zeros(batch.num_reqs, dtype=torch.int32)
        rejected = torch.zeros_like(counts)
        for i, (lo, hi) in enumerate(zip(batch.cu_num_logits_np[:-1], batch.cu_num_logits_np[1:])):
            if batch.seq_lens[i] >= batch.prefill_len_np[i]:
                counts[i] = int(choices[lo]) % (hi - lo) + 1
                rejected[i] = hi - lo - counts[i]
                output[i, : counts[i]] = choices[lo : lo + counts[i]]
        logprobs = ns["LogprobsTensors"](
            choices[:, None],
            score[:, None].float(),
            choices,
            batch.cu_num_logits_np.tolist() if mask_slots > 1 else None,
        )
        flat = output.flatten()
        active = (torch.arange(mask_slots)[None, :] < counts[:, None]).flatten()
        supports = active.int() * (1 + (flat % 2 == 1).int())
        packed = torch.zeros((output.numel(), 16), dtype=torch.uint8)
        wider = torch.nonzero(supports == 2).flatten()
        for value in (flat[wider], (flat[wider] + 1) % 127):
            packed[wider, value // 8] |= (1 << (value % 8)).to(torch.uint8)
        masks = ns["SamplingMaskTensors"](flat[:, None].int(), packed, supports, 127, mask_slots)
        return ns["SamplerOutput"](output, logprobs, None, counts, rejected, masks)

    baseline = sample_reference(original, hidden)
    compact, compact_hidden, rows = ns["completion_inputs"](original, hidden, warm)
    compact_output = sample_reference(compact, compact_hidden)
    result = ns["scatter_completion_output"](compact_output, rows, count, hidden.device, compact.cu_num_logits_np)
    np.testing.assert_array_equal(rows, np.flatnonzero(~warm[slots]))
    assert not result.num_sampled[warm[slots]].any()
    assert not result.num_rejected[warm[slots]].any()
    assert not result.sampled_token_ids[warm[slots]].any()
    assert result.sampling_mask_tensors.rows_per_request == mask_slots
    assert not result.sampling_mask_tensors.counts.reshape(count, mask_slots)[warm[slots]].any()
    expected_masks = baseline.sampling_mask_tensors.tolists(baseline.num_sampled.numpy())
    actual_masks = result.sampling_mask_tensors.tolists(result.num_sampled.numpy())
    expected_logprobs = baseline.logprobs_tensors.tolists()
    actual_logprobs = result.logprobs_tensors.tolists()
    for row in rows:
        sampled = int(baseline.num_sampled[row])
        assert result.num_sampled[row] == sampled
        assert result.num_rejected[row] == baseline.num_rejected[row]
        torch.testing.assert_close(result.sampled_token_ids[row], baseline.sampled_token_ids[row])
        actual_lp = actual_logprobs.slice_request(row, sampled)
        expected_lp = expected_logprobs.slice_request(row, sampled)
        np.testing.assert_array_equal(actual_lp.logprobs, expected_lp.logprobs)
        np.testing.assert_array_equal(actual_lp.logprob_token_ids, expected_lp.logprob_token_ids)
        assert actual_masks.slice_request(row, sampled).to_nested_list() == (
            expected_masks.slice_request(row, sampled).to_nested_list()
        )
