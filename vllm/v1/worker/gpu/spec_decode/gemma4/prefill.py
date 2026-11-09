# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Select independent Q-only draft queries without reading GPU values on CPU."""

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch


def can_compact_gemma4_prefill(model_state: Any, model_config: Any) -> bool:
    """Require the actual text-only state, not the model's multimodal label."""
    return (
        model_state is not None
        and getattr(model_state, "supports_mm_inputs", True) is False
        and getattr(model_state, "prompt_embeds_state", None) is None
        and model_config is not None
        and getattr(model_config, "enable_prompt_embeds", True) is False
    )


@dataclass(frozen=True)
class CompactPrefillInputs:
    row_indices: torch.Tensor
    idx_mapping: torch.Tensor
    last_token_indices: torch.Tensor
    input_ids: torch.Tensor
    positions: torch.Tensor
    hidden_states: torch.Tensor
    seq_lens: torch.Tensor
    target_seq_lens: torch.Tensor
    seq_lens_cpu_upper_bound: torch.Tensor
    num_sampled: torch.Tensor
    num_rejected: torch.Tensor


def select_gemma4_prefill_inputs(
    input_batch: Any,
    active_rows: np.ndarray,
    last_hidden_states: torch.Tensor,
    num_sampled: torch.Tensor,
    num_rejected: torch.Tensor,
    last_sampled: torch.Tensor,
    next_prefill_tokens: torch.Tensor,
    row_indices: torch.Tensor | None = None,
) -> CompactPrefillInputs:
    """Keep the last valid query of each selected request against frozen target KV.

    The rejection sampler guarantees at least one valid query per request.
    Rejection counts and token selection stay on the device. Sequence-length
    bounds remain CPU upper bounds rather than forcing a device synchronization.
    """
    if (
        active_rows.ndim != 1
        or active_rows.dtype.kind not in "iu"
        or not len(active_rows)
        or np.any(active_rows < 0)
        or np.any(active_rows >= input_batch.num_reqs)
        or len(np.unique(active_rows)) != len(active_rows)
    ):
        raise ValueError("Expected unique active request rows within the batch")
    if input_batch.seq_lens_cpu_upper_bound.device.type != "cpu":
        raise ValueError("Sequence-length upper bounds must remain on CPU")
    cpu_rows = torch.from_numpy(active_rows.astype(np.int64, copy=False))
    rows = (
        cpu_rows.to(input_batch.idx_mapping.device, non_blocking=True)
        if row_indices is None
        else row_indices
    )
    idx_mapping = input_batch.idx_mapping.index_select(0, rows)
    selected_sampled = num_sampled.index_select(0, rows)
    selected_rejected = num_rejected.index_select(0, rows)
    ends = input_batch.query_start_loc.index_select(0, rows + 1)
    last_indices = (ends - selected_rejected - 1).to(torch.int64)
    target_seq_lens = input_batch.seq_lens.index_select(0, rows)
    physical_indices = idx_mapping.to(torch.int64)
    input_ids = torch.where(
        selected_sampled > 0,
        last_sampled.reshape(-1).index_select(0, physical_indices),
        next_prefill_tokens.reshape(-1).index_select(0, physical_indices),
    ).to(input_batch.input_ids.dtype)
    return CompactPrefillInputs(
        row_indices=rows,
        idx_mapping=idx_mapping,
        last_token_indices=last_indices,
        input_ids=input_ids,
        positions=input_batch.positions.index_select(0, last_indices),
        hidden_states=last_hidden_states.index_select(0, last_indices),
        seq_lens=target_seq_lens - selected_rejected,
        target_seq_lens=target_seq_lens,
        seq_lens_cpu_upper_bound=input_batch.seq_lens_cpu_upper_bound.index_select(
            0, cpu_rows
        ),
        num_sampled=selected_sampled,
        num_rejected=selected_rejected,
    )
