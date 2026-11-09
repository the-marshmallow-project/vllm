# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Keep cache warming requests out of output-token computation."""

from __future__ import annotations

import os
from dataclasses import replace
from typing import TYPE_CHECKING

import numpy as np
import torch

from vllm.v1.outputs import LogprobsTensors
from vllm.v1.worker.gpu.sample.output import SamplerOutput, SamplingMaskTensors

if TYPE_CHECKING:
    from vllm.v1.worker.gpu.input_batch import InputBatch


def completion_inputs(
    input_batch: InputBatch,
    hidden_states: torch.Tensor,
    prefill_only: np.ndarray,
) -> tuple[InputBatch | None, torch.Tensor, np.ndarray | None]:
    """Select complete query spans while retaining global request/RNG indices.

    A None row selection preserves the ordinary serving path by identity. An
    empty selection never calls the logits projection or sampler. All selection
    decisions use CPU request metadata, without reading device counters.
    """
    mask = prefill_only[input_batch.idx_mapping_np]
    if not np.any(mask):
        return input_batch, hidden_states, None
    rows = np.flatnonzero(~mask)
    if not len(rows):
        return None, hidden_states[:0], rows

    device = hidden_states.device
    row_indices = torch.as_tensor(rows, device=device)
    query = input_batch.query_start_loc_np
    query_counts = np.diff(query[: input_batch.num_reqs + 1])[rows]
    compact_query = np.concatenate(([0], np.cumsum(query_counts))).astype(np.int32)
    tokens = np.concatenate([np.arange(query[i], query[i + 1]) for i in rows])
    token_indices = torch.as_tensor(tokens, device=device)
    cu = input_batch.cu_num_logits_np
    logit_counts = np.diff(cu)[rows]
    compact_cu = np.concatenate(([0], np.cumsum(logit_counts))).astype(np.int32)
    logit_rows = np.concatenate([np.arange(cu[i], cu[i + 1]) for i in rows])
    logit_indices = torch.as_tensor(logit_rows, device=device)
    offsets = torch.as_tensor(
        np.repeat(query[rows] - compact_query[:-1], logit_counts), device=device
    )
    drafts = input_batch.num_draft_tokens_per_req
    compact = replace(
        input_batch,
        req_ids=[input_batch.req_ids[i] for i in rows],
        num_reqs=len(rows),
        num_reqs_after_padding=len(rows),
        idx_mapping=input_batch.idx_mapping.index_select(0, row_indices),
        idx_mapping_np=input_batch.idx_mapping_np[rows],
        expanded_idx_mapping=input_batch.expanded_idx_mapping.index_select(
            0, logit_indices
        ),
        expanded_local_pos=input_batch.expanded_local_pos.index_select(
            0, logit_indices
        ),
        num_scheduled_tokens=input_batch.num_scheduled_tokens[rows],
        num_tokens=len(tokens),
        num_tokens_after_padding=len(tokens),
        num_draft_tokens=int(drafts[rows].sum()) if drafts is not None else 0,
        num_draft_tokens_per_req=drafts[rows] if drafts is not None else None,
        query_start_loc=torch.as_tensor(compact_query, device=device),
        query_start_loc_np=compact_query,
        seq_lens=input_batch.seq_lens.index_select(0, row_indices),
        seq_lens_cpu_upper_bound=input_batch.seq_lens_cpu_upper_bound[rows],
        num_computed_tokens_np=input_batch.num_computed_tokens_np[rows],
        prefill_len_np=input_batch.prefill_len_np[rows],
        num_computed_prefill_tokens_np=input_batch.num_computed_prefill_tokens_np[rows],
        is_prefilling_np=input_batch.is_prefilling_np[rows],
        has_prefill=bool(input_batch.is_prefilling_np[rows].any()),
        input_ids=input_batch.input_ids.index_select(0, token_indices),
        positions=input_batch.positions.index_select(0, token_indices),
        is_padding=input_batch.is_padding.index_select(0, token_indices),
        logits_indices=input_batch.logits_indices.index_select(0, logit_indices)
        - offsets,
        cu_num_logits=torch.as_tensor(compact_cu, device=device),
        cu_num_logits_np=compact_cu,
        prompt_lens=(
            input_batch.prompt_lens.index_select(0, row_indices)
            if input_batch.prompt_lens is not None
            else None
        ),
        prefill_runs_as_decode_np=(
            input_batch.prefill_runs_as_decode_np[rows]
            if input_batch.prefill_runs_as_decode_np is not None
            else None
        ),
        fast_prefill=None,
    )
    return compact, hidden_states.index_select(0, token_indices), rows


def scatter_completion_output(
    output: SamplerOutput | None,
    rows: np.ndarray | None,
    num_reqs: int,
    device: torch.device,
    cu_num_logits: np.ndarray | None = None,
) -> SamplerOutput:
    """Restore original request order, with no output for cache warming rows."""
    if rows is None:
        assert output is not None
        return output
    if output is None:
        return SamplerOutput(
            sampled_token_ids=torch.zeros(
                (num_reqs, 1), dtype=torch.int64, device=device
            ),
            logprobs_tensors=None,
            num_nans=None,
            num_sampled=torch.zeros(num_reqs, dtype=torch.int32, device=device),
            num_rejected=torch.zeros(num_reqs, dtype=torch.int32, device=device),
        )
    indices = torch.as_tensor(rows, device=device)

    def scatter(tensor):
        if tensor is None:
            return None
        result = tensor.new_zeros((num_reqs, *tensor.shape[1:]))
        return result.index_copy_(0, indices, tensor)

    logprobs = output.logprobs_tensors
    if logprobs is not None:
        # Adaptive verification and TP batch sharding use device-side offsets.
        # The endpoint rejects both configurations.
        assert logprobs.cu_num_generated_tokens_tensor is None
        old_cu = logprobs.cu_num_generated_tokens
        lengths = (
            np.ones(len(rows), dtype=np.int64) if old_cu is None else np.diff(old_cu)
        )
        full_lengths = np.zeros(num_reqs, dtype=np.int64)
        full_lengths[rows] = lengths
        full_cu = np.concatenate(([0], np.cumsum(full_lengths))).tolist()
        logprobs = LogprobsTensors(
            logprobs.logprob_token_ids,
            logprobs.logprobs,
            logprobs.selected_token_ranks,
            full_cu,
        )
    masks = output.sampling_mask_tensors
    if masks is not None:
        slots = masks.rows_per_request

        def scatter_mask_slots(tensor):
            by_request = tensor.unflatten(0, (len(rows), slots))
            return scatter(by_request).flatten(0, 1)

        masks = SamplingMaskTensors(
            scatter_mask_slots(masks.token_ids),
            scatter_mask_slots(masks.packed_mask),
            scatter_mask_slots(masks.counts),
            masks.vocab_size,
            slots,
        )
    num_nans = output.num_nans
    if num_nans is not None and num_nans.shape[0] != len(rows):
        assert cu_num_logits is not None
        cu = torch.as_tensor(cu_num_logits, device=device)
        prefix = torch.cat((num_nans.new_zeros(1), num_nans.cumsum(0)))
        num_nans = prefix[cu[1:]] - prefix[cu[:-1]]
    return SamplerOutput(
        sampled_token_ids=scatter(output.sampled_token_ids),
        logprobs_tensors=logprobs,
        num_nans=scatter(num_nans),
        num_sampled=scatter(output.num_sampled),
        num_rejected=scatter(output.num_rejected),
        sampling_mask_tensors=masks,
    )


def check_prefill_only_support(runner) -> None:
    """Backup invariant; input admission rejects these modes before queuing."""
    from vllm.v1.worker.gpu.sample.sampler import Sampler
    from vllm.v1.worker.gpu.spec_decode.gemma4.prefill import (
        can_compact_gemma4_prefill,
    )
    from vllm.v1.worker.gpu.spec_decode.gemma4.prompt_lookup import (
        Gemma4PromptLookupSpeculator,
    )

    parallel = runner.vllm_config.parallel_config
    supported = (
        os.environ.get("MALLOW_ENABLE_PREFILL_ENDPOINT") == "1"
        and all(
            getattr(parallel, name) == 1
            for name in (
                "tensor_parallel_size",
                "data_parallel_size",
                "pipeline_parallel_size",
                "prefill_context_parallel_size",
                "decode_context_parallel_size",
            )
        )
        and can_compact_gemma4_prefill(runner.model_state, runner.model_config)
        and type(runner.sampler) is Sampler
        and runner.adaptive_verification is None
        and runner.fast_prefill is None
        and not runner.use_aux_hidden_state_outputs
        and (
            (runner.speculator is None and runner.num_speculative_steps == 0)
            or (
                isinstance(runner.speculator, Gemma4PromptLookupSpeculator)
                and runner.speculator.skip_prefill_draft
                and runner.speculator.draft_watermarker is None
            )
        )
    )
    if not supported:
        raise ValueError(
            "prefill_only is unsupported by this model runner configuration"
        )
