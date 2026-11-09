# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU control-flow tests for the opt-in one-token proposal bypass.

The existing prompt-lookup suite requires CUDA. Execute the actual class with
its inference parent and lookup kernel replaced by spies, so these tests check
routing and tensor-state changes without loading the CUDA serving stack.
"""

import ast
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import numpy as np
import pytest
import torch


@pytest.fixture
def make_speculator(monkeypatch):
    path = Path(__file__).resolve().parents[3] / (
        "vllm/v1/worker/gpu/spec_decode/gemma4/prompt_lookup.py"
    )
    cls = next(
        node
        for node in ast.parse(path.read_text()).body
        if isinstance(node, ast.ClassDef)
        and node.name == "Gemma4PromptLookupSpeculator"
    )

    class Parent:
        def __init__(self, config, device):
            self.vllm_config = config
            self.model_state = SimpleNamespace(
                supports_mm_inputs=False, prompt_embeds_state=None
            )
            self.max_num_reqs = 4
            self.num_speculative_steps = 15
            self.speculative_config = config.speculative_config
            self.enable_adaptive_verification = config.adaptive
            self.pcp_manager = None
            self.draft_watermarker = None
            self.draft_tokens = torch.full((4, 15), 7, device=device)
            self.calls = []
            self.subset_calls = []
            self.compact_row_indices = torch.empty(4, dtype=torch.int64)
            self.prefill_req_indices = torch.empty(4, dtype=torch.int64)
            self.idx_mapping = torch.empty(4, dtype=torch.int64)

        def propose(self, *args, **kwargs):
            self.calls.append((args, kwargs))
            self.before_proposal = self.draft_tokens.clone()
            self.draft_tokens[: args[0].num_reqs].fill_(7)
            return self.draft_tokens[: args[0].num_reqs]

        def propose_for_active_requests(self, batch, rows, *args):
            self.subset_calls.append(rows.copy())
            self.subset_steps = args[-1]
            count = len(rows)
            self.compact_row_indices[:count].copy_(torch.from_numpy(rows))
            self.idx_mapping[:count].copy_(
                batch.idx_mapping.index_select(0, self.compact_row_indices[:count])
            )
            self.draft_tokens[:count].fill_(7)
            return self.draft_tokens[:count]

    lookup = Mock(side_effect=lambda *args: args[4].fill_(13))
    namespace = {
        "Gemma4Speculator": Parent,
        "torch": torch,
        "np": np,
        "os": os,
        "STATS": tuple(range(6)),
        "logger": Mock(),
        "prompt_lookup": lookup,
        "LOG_EVERY": 4096,
        "Any": Any,
    }
    helper_path = path.with_name("prefill.py")
    policy = next(
        node
        for node in ast.parse(helper_path.read_text()).body
        if isinstance(node, ast.FunctionDef)
        and node.name == "can_compact_gemma4_prefill"
    )
    exec(
        compile(ast.Module(body=[policy], type_ignores=[]), str(helper_path), "exec"),
        namespace,
    )
    exec(compile(ast.Module(body=[cls], type_ignores=[]), str(path), "exec"), namespace)

    def make(
        limits=(1, 1, 1, 1),
        indices=(0, 1),
        flag="1",
        attach=True,
        parallel=None,
        adaptive=False,
    ):
        for key, value in {
            "MALLOW_MTP_DRAFT_STEPS": "9",
            "MALLOW_PROMPT_LOOKUP_MAX": "8",
            "MALLOW_PROMPT_LOOKUP_MIN": "1",
        }.items():
            monkeypatch.setenv(key, value)
        if flag is None:
            monkeypatch.delenv("MALLOW_SKIP_PREFILL_DRAFT", raising=False)
        else:
            monkeypatch.setenv("MALLOW_SKIP_PREFILL_DRAFT", flag)
        topology = dict.fromkeys(
            (
                "tensor_parallel_size",
                "pipeline_parallel_size",
                "data_parallel_size",
                "prefill_context_parallel_size",
                "decode_context_parallel_size",
            ),
            1,
        )
        topology.update(parallel or {})
        config = SimpleNamespace(
            parallel_config=SimpleNamespace(**topology),
            model_config=SimpleNamespace(enable_prompt_embeds=False),
            adaptive=adaptive,
            speculative_config=SimpleNamespace(draft_sample_method="greedy"),
        )
        spec = namespace["Gemma4PromptLookupSpeculator"](config, torch.device("cpu"))
        prompt_len = np.array([100, 200, 300, 400], dtype=np.int32)
        states = SimpleNamespace(
            all_token_ids=SimpleNamespace(gpu=torch.zeros(4, 1)),
            total_len=SimpleNamespace(gpu=torch.zeros(4)),
            prompt_len=SimpleNamespace(np=prompt_len),
            max_seq_len=prompt_len + np.array(limits, dtype=np.int32),
        )
        if attach:
            spec.attach_request_states(states)
        spec.lookup_drafts.fill_(99)
        spec.lookup_extended.fill_(1)
        batch = SimpleNamespace(
            num_reqs=len(indices),
            idx_mapping_np=np.array(indices, dtype=np.int64),
            idx_mapping=torch.tensor(indices, dtype=torch.int64),
            positions=torch.zeros(len(indices), dtype=torch.int64),
        )
        return spec, batch, states, lookup

    return make


def propose(spec, batch, sampled=1, **kwargs):
    per_req = torch.full((batch.num_reqs,), sampled, dtype=torch.int32)
    return spec.propose(
        batch,
        {},
        {},
        torch.zeros(batch.num_reqs, 1),
        None,
        per_req,
        torch.zeros_like(per_req),
        torch.zeros(4, 1),
        torch.zeros(1, 4),
        torch.ones(4),
        torch.zeros(4),
        **kwargs,
    )


@pytest.mark.parametrize("indices", [(0,), (3, 1)])
def test_one_token_batch_skips_mtp_and_lookup_and_clears_drafts(
    make_speculator, indices
):
    spec, batch, _, lookup = make_speculator(indices=indices)
    result = propose(spec, batch)
    assert result.shape == (len(indices), 15)
    assert not result.any() and not spec.draft_tokens[: len(indices)].any()
    assert spec.lookup_extended.tolist() == [0 if i in indices else 1 for i in range(4)]
    assert spec.calls == [] and spec.steps == 0
    lookup.assert_not_called()


def test_explicit_prefill_bypasses_drafting_independently_of_output_sentinel(
    make_speculator,
):
    spec, batch, states, lookup = make_speculator(limits=(20, 20, 20, 20))
    states.prefill_only = np.array([True, False, False, False])
    spec.attach_request_states(states)
    result = propose(spec, batch, sampled=0)
    assert spec.subset_calls[0].tolist() == [1]
    assert result.tolist() == [[0] * 15, [13] * 15]
    assert lookup.call_args.args[2].tolist() == [1]


def test_endpoint_opt_in_enables_required_draft_compaction(
    make_speculator, monkeypatch
):
    monkeypatch.setenv("MALLOW_ENABLE_PREFILL_ENDPOINT", "1")
    spec, batch, _, lookup = make_speculator(flag=None)
    assert not propose(spec, batch).any()
    assert spec.calls == []
    lookup.assert_not_called()


def test_explicit_prefill_never_falls_back_to_unsupported_proposal(make_speculator):
    spec, batch, states, lookup = make_speculator(flag=None)
    states.prefill_only = np.array([True, False, False, False])
    spec.attach_request_states(states)
    with pytest.raises(RuntimeError, match="cannot enter draft proposal"):
        propose(spec, batch, sampled=0)
    assert spec.calls == []
    lookup.assert_not_called()


@pytest.mark.parametrize(
    "options,kwargs,uses_lookup",
    [
        ({"flag": None}, {}, True),
        ({"flag": "0"}, {}, True),
        ({"flag": "true"}, {}, True),
        ({"limits": (2, 2, 1, 1)}, {}, True),
        ({"attach": False}, {}, False),
        ({}, {"dummy_run": True}, False),
        ({}, {"is_profile": True}, False),
        ({"adaptive": True}, {}, True),
        ({}, {"dp_sync_state": object()}, True),
        ({}, {"mm_inputs": ([], torch.zeros(2))}, True),
        *[
            ({"parallel": {name: 2}}, {}, True)
            for name in (
                "tensor_parallel_size",
                "pipeline_parallel_size",
                "data_parallel_size",
                "prefill_context_parallel_size",
                "decode_context_parallel_size",
            )
        ],
        ({"indices": (-1, 1)}, {}, True),
        ({"indices": (4, 1)}, {}, True),
        ({"indices": ()}, {}, True),
    ],
)
def test_ordinary_and_unsupported_batches_keep_original_path(
    make_speculator,
    options,
    kwargs,
    uses_lookup,
):
    spec, batch, _, lookup = make_speculator(**options)
    result = propose(spec, batch, **kwargs)
    assert len(spec.calls) == 1
    assert spec.calls[0][0][0] is batch
    assert result.tolist() == [[13 if uses_lookup else 7] * 15] * batch.num_reqs
    assert lookup.call_count == int(uses_lookup)


@pytest.mark.parametrize("attribute", ["pcp_manager", "draft_watermarker"])
def test_late_bound_optional_features_keep_original_path(make_speculator, attribute):
    spec, batch, _, lookup = make_speculator()
    setattr(spec, attribute, object())
    propose(spec, batch)
    assert len(spec.calls) == 1
    lookup.assert_called_once()


@pytest.mark.parametrize("attribute", ["prompt_len_np", "max_seq_len_np"])
def test_short_cpu_metadata_keeps_original_path(make_speculator, attribute):
    spec, batch, _, lookup = make_speculator()
    setattr(spec, attribute, getattr(spec, attribute)[:1])
    propose(spec, batch)
    assert len(spec.calls) == 1
    lookup.assert_called_once()


def test_incomplete_cpu_mapping_keeps_original_path(make_speculator):
    spec, batch, _, lookup = make_speculator(indices=(0,))
    batch.num_reqs = 2
    propose(spec, batch)
    assert len(spec.calls) == 1
    lookup.assert_called_once()


def test_chunked_prefill_and_reused_slot_do_not_leave_stale_state(make_speculator):
    spec, batch, states, lookup = make_speculator(indices=(3,))
    # Non-final and final prompt chunks cannot need drafts for max_tokens=1.
    for sampled in (0, 0, 1):
        spec.draft_tokens.fill_(8)
        result = propose(spec, batch, sampled=sampled)
        assert not result.any() and spec.calls == []
    # RequestState updates these same CPU arrays when the slot is reused.
    states.max_seq_len[3] = states.prompt_len.np[3] + 128
    result = propose(spec, batch)
    assert result.tolist() == [[13] * 15]
    assert not spec.before_proposal[0].any()
    assert len(spec.calls) == 1 and lookup.call_count == 1


def test_mixed_batch_only_drafts_normal_requests_and_restores_output_order(
    make_speculator,
):
    spec, batch, _, lookup = make_speculator(
        limits=(128, 1, 128, 1), indices=(3, 0, 1, 2)
    )
    result = propose(spec, batch)
    assert spec.calls == []
    assert len(spec.subset_calls) == 1
    assert spec.subset_calls[0].tolist() == [1, 3]
    assert result.tolist() == [[0] * 15, [13] * 15, [0] * 15, [13] * 15]
    assert lookup.call_args.args[2].tolist() == [0, 2]
    assert spec.lookup_extended.tolist() == [1, 0, 1, 0]
    assert spec.steps == 1


@pytest.mark.parametrize("budget,mtp_steps", [(1, 1), (3, 3), (15, 9)])
def test_dynamic_draft_budget_reaches_native_and_compact_paths(
    make_speculator, budget, mtp_steps
):
    spec, batch, _, lookup = make_speculator(limits=(1, 128, 1, 1))
    propose(spec, batch, num_speculative_tokens=budget)
    assert spec.subset_steps == mtp_steps
    assert lookup.call_args.args[9] == mtp_steps

    spec, batch, _, lookup = make_speculator(limits=(128, 128, 1, 1))
    propose(spec, batch, num_speculative_tokens=budget)
    assert spec.calls[0][1]["num_speculative_tokens"] == mtp_steps
    assert lookup.call_args.args[9] == mtp_steps


@pytest.mark.parametrize("flag", ["0", "1"])
def test_zero_dynamic_budget_skips_lookup_and_drafting(make_speculator, flag):
    spec, batch, _, lookup = make_speculator(limits=(1, 128, 1, 1), flag=flag)
    result = propose(spec, batch, num_speculative_tokens=0)
    assert result.tolist() == [[-1] * 15] * 2
    assert spec.lookup_extended.tolist() == [0, 0, 1, 1]
    assert not spec.calls and not spec.subset_calls
    lookup.assert_not_called()


def test_zero_dynamic_budget_keeps_distributed_base_path(make_speculator):
    spec, batch, _, lookup = make_speculator()
    sync = object()
    propose(spec, batch, dp_sync_state=sync, num_speculative_tokens=0)
    assert spec.calls[0][1]["dp_sync_state"] is sync
    assert spec.calls[0][1]["num_speculative_tokens"] == 0
    lookup.assert_not_called()


@pytest.mark.parametrize(
    "unsupported",
    [
        "missing_state",
        "multimodal",
        "prompt_state",
        "prompt_config",
        "multiaxis_positions",
        "fast_prefill",
    ],
)
def test_mixed_compaction_requires_known_text_only_token_inputs(
    make_speculator,
    unsupported,
):
    spec, batch, _, lookup = make_speculator(limits=(1, 128, 1, 1))
    if unsupported == "missing_state":
        spec.model_state = None
    elif unsupported == "multimodal":
        spec.model_state.supports_mm_inputs = True
    elif unsupported == "prompt_state":
        spec.model_state.prompt_embeds_state = object()
    elif unsupported == "prompt_config":
        spec.vllm_config.model_config.enable_prompt_embeds = True
    elif unsupported == "multiaxis_positions":
        batch.positions = torch.zeros(2, 2)
    else:
        batch.fast_prefill = object()
    result = propose(spec, batch)
    assert len(spec.calls) == 1 and not spec.subset_calls
    assert result.tolist() == [[13] * 15] * batch.num_reqs
    lookup.assert_called_once()
