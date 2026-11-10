# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU execution of native batching/runner methods, without CUDA imports.

These tests guard the zero-output contract and ordinary-row routing. CUDA
attention, sampler kernels and graph numerical equivalence need dev GPU evals.
"""

import ast
from contextlib import nullcontext
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import NamedTuple
from unittest.mock import Mock

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]


def load_nodes(path, names, namespace):
    nodes = [
        node
        for node in ast.parse((ROOT / path).read_text()).body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names
    ]
    future = ast.ImportFrom(
        module="__future__", names=[ast.alias(name="annotations")], level=0
    )
    module = ast.fix_missing_locations(
        ast.Module(body=[future, *nodes], type_ignores=[])
    )
    exec(compile(module, str(ROOT / path), "exec"), namespace)


@pytest.fixture(scope="module")
def native():
    ns = dict(
        torch=torch,
        np=np,
        dataclass=dataclass,
        replace=replace,
        NamedTuple=NamedTuple,
    )
    load_nodes("vllm/v1/outputs.py", {"LogprobsTensors"}, ns)
    load_nodes(
        "vllm/v1/worker/gpu/sample/output.py",
        {"SamplerOutput", "SamplingMaskTensors"},
        ns,
    )
    load_nodes("vllm/v1/worker/gpu/input_batch.py", {"InputBatch"}, ns)
    load_nodes(
        "vllm/v1/worker/gpu/prefill_only.py",
        {"completion_inputs", "scatter_completion_output"},
        ns,
    )
    return ns


def batch(native, query_counts=(3, 5, 2), logits_counts=(1, 3, 1)):
    query = np.concatenate(([0], np.cumsum(query_counts))).astype(np.int32)
    cu = np.concatenate(([0], np.cumsum(logits_counts))).astype(np.int32)
    indices = np.array([4, 1, 3])
    logits = np.concatenate(
        [
            np.arange(query[i + 1] - count, query[i + 1])
            for i, count in enumerate(logits_counts)
        ]
    )
    return native["InputBatch"](
        req_ids=["warm", "complete", "other"],
        num_reqs=3,
        num_reqs_after_padding=4,
        idx_mapping=torch.tensor(indices),
        idx_mapping_np=indices,
        expanded_idx_mapping=torch.tensor(np.repeat(indices, logits_counts)),
        expanded_local_pos=torch.tensor(
            np.concatenate([np.arange(count) for count in logits_counts])
        ),
        num_scheduled_tokens=np.array(query_counts),
        num_tokens=int(query[-1]),
        num_tokens_after_padding=12,
        num_draft_tokens=sum(logits_counts) - 3,
        num_draft_tokens_per_req=np.array(logits_counts) - 1,
        query_start_loc=torch.tensor([*query, query[-1]]),
        query_start_loc_np=np.array([*query, query[-1]]),
        seq_lens=torch.tensor([20, 42, 15, 0], dtype=torch.int32),
        seq_lens_cpu_upper_bound=torch.tensor([20, 42, 15, 0]),
        dcp_local_seq_lens=None,
        num_computed_tokens_np=np.array([17, 37, 13]),
        prefill_len_np=np.array([30, 40, 15]),
        num_computed_prefill_tokens_np=np.array([17, 37, 13]),
        is_prefilling_np=np.array([True, True, True]),
        has_prefill=True,
        decode_graph_eligible=False,
        input_ids=torch.arange(12, dtype=torch.int32) + 100,
        positions=torch.arange(12, dtype=torch.int64) + 30,
        is_padding=torch.tensor([False] * 10 + [True] * 2),
        logits_indices=torch.tensor(logits),
        cu_num_logits=torch.tensor(cu),
        cu_num_logits_np=cu,
        has_structured_output_reqs=False,
        prompt_lens=torch.tensor([30, 40, 15, 0]),
    )


@pytest.mark.parametrize("warm_slots", [(), (4,), (1,), (4, 3), (4, 1, 3)])
def test_selection_projects_only_completion_tokens_and_preserves_rng_indices(
    native, warm_slots
):
    original = batch(native)
    hidden = torch.arange(24).reshape(12, 2)
    flags = np.zeros(5, dtype=bool)
    flags[list(warm_slots)] = True
    selected, states, rows = native["completion_inputs"](original, hidden, flags)
    expected_rows = [i for i, slot in enumerate([4, 1, 3]) if slot not in warm_slots]
    if not warm_slots:
        assert selected is original and states is hidden and rows is None
        return
    assert rows.tolist() == expected_rows
    if not expected_rows:
        assert selected is None and states.numel() == 0
        return
    expected_tokens = [
        token
        for i in expected_rows
        for token in range(
            original.query_start_loc_np[i], original.query_start_loc_np[i + 1]
        )
    ]
    assert selected.idx_mapping.tolist() == original.idx_mapping[expected_rows].tolist()
    assert states.tolist() == hidden[expected_tokens].tolist()
    assert selected.positions.tolist() == original.positions[expected_tokens].tolist()
    projected = states[selected.logits_indices]
    expected_logits = [
        int(original.logits_indices[j])
        for i in expected_rows
        for j in range(original.cu_num_logits_np[i], original.cu_num_logits_np[i + 1])
    ]
    assert projected.tolist() == hidden[expected_logits].tolist()
    assert selected.expanded_idx_mapping.tolist() == [
        int(original.idx_mapping[i])
        for i in expected_rows
        for _ in range(original.cu_num_logits_np[i + 1] - original.cu_num_logits_np[i])
    ]
    assert selected.query_start_loc[-1] == len(expected_tokens)
    assert original.query_start_loc.tolist() == [0, 3, 8, 10, 10]


@pytest.mark.parametrize("logit_counts", [(1, 1), (3, 2)])
def test_scatter_retains_completion_outputs_logprob_boundaries_and_masks(
    native, logit_counts
):
    count = sum(logit_counts)
    cu = np.concatenate(([0], np.cumsum(logit_counts)))
    scores = torch.arange(count * 2).reshape(count, 2)
    logprobs = native["LogprobsTensors"](
        scores,
        scores.float(),
        torch.arange(count),
        None if count == 2 else cu.tolist(),
    )
    masks = native["SamplingMaskTensors"](
        torch.tensor([[4, 5], [6, 7]]),
        torch.tensor([[1], [2]], dtype=torch.uint8),
        torch.tensor([2, 2]),
        8,
    )
    output = native["SamplerOutput"](
        torch.tensor([[9, 8, 7], [6, 5, 4]]),
        logprobs,
        torch.arange(count, dtype=torch.int32),
        torch.tensor([3, 1], dtype=torch.int32),
        torch.tensor([0, 1], dtype=torch.int32),
        masks,
    )
    result = native["scatter_completion_output"](
        output, np.array([1, 3]), 4, torch.device("cpu"), cu
    )
    assert result.sampled_token_ids.tolist() == [[0] * 3, [9, 8, 7], [0] * 3, [6, 5, 4]]
    assert result.num_sampled.tolist() == [0, 3, 0, 1]
    assert result.num_rejected.tolist() == [0, 0, 0, 1]
    assert result.logprobs_tensors.logprobs is logprobs.logprobs
    assert result.logprobs_tensors.cu_num_generated_tokens == [
        0,
        0,
        logit_counts[0],
        logit_counts[0],
        count,
    ]
    assert result.sampling_mask_tensors.counts.tolist() == [0, 2, 0, 2]
    assert result.num_nans.tolist() == [
        0,
        sum(range(logit_counts[0])),
        0,
        sum(range(logit_counts[0], count)),
    ]


@pytest.mark.parametrize("warm_slots", [(4, 1, 3), (4,), ()])
def test_native_sample_tokens_skips_logits_sampling_and_prompt_logprobs_for_warm_rows(
    native, warm_slots
):
    ns = dict(native)
    source = ast.parse((ROOT / "vllm/v1/worker/gpu/model_runner.py").read_text())
    cls = next(
        n
        for n in source.body
        if isinstance(n, ast.ClassDef) and n.name == "GPUModelRunner"
    )
    methods = [
        n
        for n in cls.body
        if isinstance(n, ast.FunctionDef) and n.name in {"sample", "sample_tokens"}
    ]
    ns.update(
        pcp=SimpleNamespace(maybe_restore_pcp_for_sampling=lambda p, h, b: (h, b)),
        AsyncOutput=lambda **kw: SimpleNamespace(**kw),
        ModelRunnerOutput=lambda **kw: SimpleNamespace(**kw),
        DraftModelSpeculator=type("DraftModelSpeculator", (), {}),
        GPUWatermarkSampler=type("GPUWatermarkSampler", (), {}),
        use_workspace_lane=lambda _: nullcontext(),
        step_eplb_after=lambda: lambda function: function,
    )
    wrapped = ast.fix_missing_locations(
        ast.Module(
            body=[
                ast.ImportFrom(
                    module="__future__", names=[ast.alias(name="annotations")], level=0
                ),
                ast.ClassDef(
                    name="Runner",
                    bases=[],
                    keywords=[],
                    body=methods,
                    decorator_list=[],
                ),
            ],
            type_ignores=[],
        )
    )
    exec(compile(wrapped, str(ROOT / "vllm/v1/worker/gpu/model_runner.py"), "exec"), ns)
    runner = ns["Runner"]()
    original = batch(native, logits_counts=(1, 1, 1))
    hidden = torch.arange(24).reshape(12, 2).float()
    flags = np.zeros(5, dtype=bool)
    flags[list(warm_slots)] = True
    runner.execute_model_state = SimpleNamespace(
        input_batch=original,
        attn_metadata={},
        slot_mappings_by_layer={},
        hidden_states=hidden,
        aux_hidden_states=None,
        dp_sync_state=None,
        finished_req_ids=set(),
        ec_connector_output=None,
        cudagraph_stats=None,
        num_spec_tokens_to_schedule=0,
    )
    runner.is_last_pp_rank = True
    runner.pcp_manager = runner.pp_handler = runner.batch_sharder = None
    runner.aux_output_connector = None
    runner.speculator = runner.rejection_sampler = runner.adaptive_verification = None
    runner.main_stream = runner.output_copy_stream = None
    runner.check_ep_fault = False
    runner.num_speculative_steps = 0
    runner.req_states = SimpleNamespace(
        prefill_only=flags,
        all_token_ids=SimpleNamespace(gpu=torch.zeros(5, 50)),
        num_computed_tokens=SimpleNamespace(gpu=torch.zeros(5)),
        prompt_len=SimpleNamespace(np=np.full(5, 30)),
    )
    projection = Mock(side_effect=lambda values: values)
    runner.model = SimpleNamespace(compute_logits=projection)
    runner.model_state = SimpleNamespace(compute_sample_logits=projection)

    def sample(logits, selected):
        count = selected.num_reqs
        return native["SamplerOutput"](
            logits[:, :1].to(torch.int64),
            None,
            None,
            torch.ones(count, dtype=torch.int32),
            torch.zeros(count, dtype=torch.int32),
        )

    runner.sampler = Mock(side_effect=sample)
    prompt_worker = Mock(return_value={})
    token_id_prompt_worker = Mock(return_value={})
    runner.prompt_logprobs_worker = SimpleNamespace(
        compute_prompt_logprobs=prompt_worker,
        compute_prompt_token_id_logprobs=token_id_prompt_worker,
    )
    runner.postprocess_sampled = Mock()
    runner.kv_connector = SimpleNamespace(post_forward=Mock(return_value="kv"))
    output = runner.sample_tokens(None)
    expected = [i for i, slot in enumerate([4, 1, 3]) if slot not in warm_slots]
    assert output.num_sampled_tokens.tolist() == [int(i in expected) for i in range(3)]
    if expected:
        selected = runner.sampler.call_args.args[1]
        assert selected.req_ids == [original.req_ids[i] for i in expected]
        assert (
            projection.call_args.args[0].tolist()
            == hidden[original.logits_indices[expected]].tolist()
        )
        assert prompt_worker.call_args.args[2].req_ids == selected.req_ids
        assert token_id_prompt_worker.call_args.args[2].req_ids == selected.req_ids
        expected_tokens = (
            sum(original.num_scheduled_tokens[expected]) if warm_slots else len(hidden)
        )
        assert prompt_worker.call_args.args[1].shape[0] == expected_tokens
    else:
        projection.assert_not_called()
        runner.sampler.assert_not_called()
        prompt_worker.assert_not_called()
        token_id_prompt_worker.assert_not_called()
    assert runner.postprocess_sampled.call_args.args[0] is original.idx_mapping
    assert runner.postprocess_sampled.call_args.args[4] is original.query_start_loc
    assert output.model_runner_output.kv_connector_output == "kv"


def test_empty_output_and_ordinary_scatter_identity(native):
    output = native["scatter_completion_output"](
        None, np.array([], dtype=int), 3, torch.device("cpu")
    )
    assert output.num_sampled.tolist() == [0, 0, 0]
    assert output.num_rejected.tolist() == [0, 0, 0]
    assert not output.sampled_token_ids.any()
    assert (
        native["scatter_completion_output"](output, None, 3, torch.device("cpu"))
        is output
    )


def test_native_request_slot_reuse_clears_prefill_intent(native):
    class Buffer:
        def __init__(self, shape, dtype, **kwargs):
            self.gpu = torch.zeros(shape, dtype=dtype)
            self.np = self.gpu.numpy()

        def stage_write_elem(self, index, value):
            self.gpu[index] = value

        def stage_write(self, index, offset, values):
            self.gpu[index, offset : offset + len(values)] = torch.tensor(values)

    ns = dict(native, StagedWriteTensor=Buffer, UvaBackedTensor=Buffer)
    load_nodes("vllm/v1/worker/gpu/states.py", {"RequestState"}, ns)
    states = ns["RequestState"](2, 20, 10, 3, 50, torch.device("cpu"))
    states.add_request("warm", 4, [1, 2, 3, 4], 0, 1, prefill_only=True)
    slot = states.req_id_to_index["warm"]
    assert states.prefill_only[slot]
    states.draft_tokens[slot].fill_(99)
    states.remove_request("warm")
    states.add_request("ordinary", 3, [5, 6, 7], 0, 10)
    assert states.req_id_to_index["ordinary"] == slot
    assert not states.prefill_only[slot]
    assert not states.draft_tokens[slot].any()
