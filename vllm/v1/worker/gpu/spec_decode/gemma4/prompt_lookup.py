# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Gemma 4 MTP drafting extended by prompt lookup.

The target verifies `num_speculative_tokens` slots per request, but the MTP
drafter runs only `MALLOW_MTP_DRAFT_STEPS` passes and fills the first slots, as
it would alone. Prompt lookup fills the rest: a Triton kernel appends MTP's
draft to the request's tokens (prompt and output), finds the latest earlier
occurrence of the longest trailing n-gram of that sequence, from
`MALLOW_PROMPT_LOOKUP_MAX` tokens down to `MALLOW_PROMPT_LOOKUP_MIN`, and copies
the tokens that followed it. Where no occurrence exists, the slots repeat MTP's
last token.

The lookup never changes MTP's tokens, so it only adds tokens on steps that
accept MTP's whole draft. The extra slots are verified whatever they hold, so a
wrong guess costs nothing more than padding, verification keeps outputs
lossless, and every batch keeps its shape for CUDA graphs.

Off unless `MALLOW_PROMPT_LOOKUP_MAX` is set: the speculator then drafts as the
plain Gemma 4 MTP speculator.

Zero-token prefill bypasses proposals on a single rank.
`MALLOW_SKIP_PREFILL_DRAFT=1` also bypasses proposals for one-token requests.
Mixed text batches draft only their remaining requests using
independent Q-only queries against the target's frozen KV cache. Unsupported
input modes and dummy/profile runs retain their original proposal path.
"""

import os

import numpy as np
import torch

from vllm.logger import init_logger
from vllm.triton_utils import tl, triton
from vllm.v1.worker.gpu.spec_decode.gemma4.prefill import can_compact_gemma4_prefill
from vllm.v1.worker.gpu.spec_decode.gemma4.speculator import Gemma4Speculator

logger = init_logger(__name__)

# Draft steps between two log lines of the lookup's statistics; reading them
# synchronizes with the GPU.
LOG_EVERY = 4096
# Verified drafts and the draft tokens they accepted, those the lookup extended
# and their tokens accepted past MTP's, and lookups run and found.
STATS = ("verified", "accepted", "extended", "extension_accepted", "lookups", "found")


@triton.jit
def _prompt_lookup_kernel(
    token_ids_ptr,
    token_ids_stride,
    total_len_ptr,
    idx_mapping_ptr,
    draft_ptr,
    draft_stride,
    out_ptr,
    out_stride,
    extended_ptr,
    num_sampled_ptr,
    num_rejected_ptr,
    stats_ptr,
    MIN_N: tl.constexpr,
    MAX_N: tl.constexpr,
    M: tl.constexpr,
    K: tl.constexpr,
    K_PAD: tl.constexpr,
    BLOCK: tl.constexpr,
):
    batch = tl.program_id(0)
    slots = tl.arange(0, K_PAD)
    in_draft = slots < K
    mtp = tl.load(draft_ptr + batch * draft_stride + slots, mask=slots < M, other=0)
    last = tl.load(draft_ptr + batch * draft_stride + M - 1)
    padded = tl.where(slots < M, mtp, last)
    req = tl.load(idx_mapping_ptr + batch)
    if req < 0:
        tl.store(out_ptr + batch * out_stride + slots, padded, mask=in_draft)
        return

    # Account for the draft just verified: a step that verified all K slots
    # sampled its accepted tokens plus one and rejected the rest.
    num_sampled = tl.load(num_sampled_ptr + batch)
    verified = num_sampled + tl.load(num_rejected_ptr + batch) == K + 1
    accepted = (num_sampled - 1).to(tl.int64)
    was_extended = tl.load(extended_ptr + req) != 0
    one = tl.full((), 1, tl.int64)
    tl.atomic_add(stats_ptr + 0, one, mask=verified)
    tl.atomic_add(stats_ptr + 1, accepted, mask=verified)
    tl.atomic_add(stats_ptr + 2, one, mask=verified & was_extended)
    extension = tl.maximum(accepted - M, 0)
    tl.atomic_add(stats_ptr + 3, extension, mask=verified & was_extended)

    # The sequence to continue is the request's tokens followed by MTP's draft.
    # Candidates are positions `end` of the request's tokens right after an
    # n-gram equal to that sequence's tail: the longest match wins, then the
    # latest position.
    length = tl.load(total_len_ptr + req)
    row = token_ids_ptr + req.to(tl.int64) * token_ids_stride
    best = tl.full((), -1, tl.int64)
    offsets = tl.arange(0, BLOCK)
    for start in range(0, length, BLOCK):
        end = start + offsets
        alive = (end >= MIN_N) & (end < length)
        matched = tl.zeros((BLOCK,), tl.int64)
        for i in tl.static_range(MAX_N):
            token = tl.load(
                row + end - 1 - i, mask=alive & (end - 1 - i >= 0), other=-1
            ).to(tl.int64)
            if i < M:
                tail = tl.load(draft_ptr + batch * draft_stride + M - 1 - i)
            else:
                tail = tl.load(
                    row + length + M - 1 - i,
                    mask=length + M - 1 - i >= 0,
                    other=-2,
                ).to(tl.int64)
            alive = alive & (token == tail)
            matched = tl.where(alive, i + 1, matched)
        score = tl.where(matched >= MIN_N, matched * 2147483648 + end, -1)
        best = tl.maximum(best, tl.max(score, axis=0))

    found = best >= 0
    match_end = tl.where(found, best % 2147483648, 0)
    available = tl.where(found, tl.minimum(length - match_end, K - M), 0)
    copied = (slots >= M) & (slots < M + available)
    continuation = tl.load(
        row + match_end + slots - M, mask=in_draft & copied, other=0
    ).to(tl.int64)
    tl.store(
        out_ptr + batch * out_stride + slots,
        tl.where(copied, continuation, padded),
        mask=in_draft,
    )
    tl.store(extended_ptr + req, (available > 0).to(tl.int32))
    tl.atomic_add(stats_ptr + 4, one)
    tl.atomic_add(stats_ptr + 5, found.to(tl.int64))


def prompt_lookup(
    token_ids: torch.Tensor,
    total_len: torch.Tensor,
    idx_mapping: torch.Tensor,
    draft: torch.Tensor,
    out: torch.Tensor,
    extended: torch.Tensor,
    num_sampled: torch.Tensor,
    num_rejected: torch.Tensor,
    stats: torch.Tensor,
    mtp_steps: int,
    min_n: int,
    max_n: int,
) -> None:
    """Write into `out` each request's draft: MTP's first `mtp_steps` tokens,
    then the lookup's continuation.

    Args:
        token_ids: [max_num_reqs, max_model_len] int32, every request's tokens.
        total_len: [max_num_reqs] int32, the number of tokens in each row.
        idx_mapping: [num_reqs] int32, batch index to request row; negative
            rows are skipped.
        draft: [num_reqs, K] int64, MTP's draft in its first `mtp_steps`
            columns.
        out: [num_reqs, K] int64.
        extended: [max_num_reqs] int32, whether the lookup extended each
            request's last draft; updated in place.
        num_sampled: [num_reqs] int32, tokens sampled when the previous draft
            was verified.
        num_rejected: [num_reqs] int32, its rejected draft slots.
        stats: [len(STATS)] int64 counters, added to.
    """
    num_reqs, k = draft.shape
    if num_reqs == 0:
        return
    _prompt_lookup_kernel[(num_reqs,)](
        token_ids,
        token_ids.stride(0),
        total_len,
        idx_mapping,
        draft,
        draft.stride(0),
        out,
        out.stride(0),
        extended,
        num_sampled,
        num_rejected,
        stats,
        MIN_N=min_n,
        MAX_N=max_n,
        M=mtp_steps,
        K=k,
        K_PAD=triton.next_power_of_2(k),
        BLOCK=1024,
        num_warps=4,
    )


class Gemma4PromptLookupSpeculator(Gemma4Speculator):
    """Gemma 4 MTP drafting the first slots of each draft, prompt lookup the
    rest."""

    def __init__(self, vllm_config, device: torch.device, req_states=None):
        super().__init__(vllm_config, device)
        self.lookup_enabled = bool(os.environ.get("MALLOW_PROMPT_LOOKUP_MAX"))
        if not self.lookup_enabled:
            return
        # The base class sized its draft buffer for every slot; MTP itself runs
        # fewer passes.
        self.num_slots = self.num_speculative_steps
        self.num_speculative_steps = int(os.environ["MALLOW_MTP_DRAFT_STEPS"])
        self.lookup_max = int(os.environ["MALLOW_PROMPT_LOOKUP_MAX"])
        self.lookup_min = int(os.environ.get("MALLOW_PROMPT_LOOKUP_MIN", 2))
        if not 1 < self.num_speculative_steps < self.num_slots:
            raise ValueError(
                "need 1 < MALLOW_MTP_DRAFT_STEPS < num_speculative_tokens="
                f"{self.num_slots}, got {self.num_speculative_steps}"
            )
        if not 1 <= self.lookup_min <= self.lookup_max:
            raise ValueError(
                "need 1 <= MALLOW_PROMPT_LOOKUP_MIN <= MALLOW_PROMPT_LOOKUP_MAX, "
                f"got {self.lookup_min} and {self.lookup_max}"
            )
        if self.speculative_config.draft_sample_method != "greedy":
            raise ValueError(
                "prompt lookup proposes one-hot drafts; it needs "
                "draft_sample_method='greedy'"
            )
        self.lookup_drafts = torch.zeros_like(self.draft_tokens)
        self.lookup_extended = torch.zeros(
            self.max_num_reqs, dtype=torch.int32, device=device
        )
        self.lookup_stats = torch.zeros(len(STATS), dtype=torch.int64, device=device)
        self.token_ids: torch.Tensor | None = None
        self.total_len: torch.Tensor | None = None
        self.prompt_len_np: np.ndarray | None = None
        self.max_seq_len_np: np.ndarray | None = None
        self.prefill_only_np: np.ndarray | None = None
        parallel = vllm_config.parallel_config
        self.skip_single_token_draft = (
            os.environ.get("MALLOW_SKIP_PREFILL_DRAFT") == "1"
        )
        self.skip_prefill_draft = (
            parallel.tensor_parallel_size == 1
            and parallel.pipeline_parallel_size == 1
            and parallel.data_parallel_size == 1
            and parallel.prefill_context_parallel_size == 1
            and parallel.decode_context_parallel_size == 1
            and not self.enable_adaptive_verification
        )
        self.full_batch_drafts = (
            torch.zeros_like(self.draft_tokens) if self.skip_prefill_draft else None
        )
        self.steps = 0
        logger.info(
            "Gemma4 MTP with prompt lookup: MTP drafts %d of %d slots, lookup "
            "n-grams of %d to %d tokens",
            self.num_speculative_steps,
            self.num_slots,
            self.lookup_min,
            self.lookup_max,
        )
        if req_states is not None:
            self.attach_request_states(req_states)

    def init_cudagraph_manager(self, cudagraph_mode) -> None:
        """Capture the draft prefill for the tokens the target verifies per
        request, one more than the slots, and the draft decode loop for MTP's
        own passes."""
        if not self.lookup_enabled:
            super().init_cudagraph_manager(cudagraph_mode)
            return
        steps, self.num_speculative_steps = self.num_speculative_steps, self.num_slots
        try:
            super().init_cudagraph_manager(cudagraph_mode)
        finally:
            self.num_speculative_steps = steps

    def attach_request_states(self, req_states) -> None:
        """Read each request's tokens from the model runner's request states,
        which hold a step's sampled tokens before the runner calls `propose`."""
        self.token_ids = req_states.all_token_ids.gpu
        self.total_len = req_states.total_len.gpu
        self.prompt_len_np = req_states.prompt_len.np
        self.max_seq_len_np = req_states.max_seq_len
        self.prefill_only_np = getattr(req_states, "prefill_only", None)

    @torch.inference_mode()
    def propose(
        self,
        input_batch,
        attn_metadata,
        slot_mappings,
        last_hidden_states,
        aux_hidden_states,
        num_sampled: torch.Tensor,
        num_rejected: torch.Tensor,
        last_sampled,
        next_prefill_tokens,
        temperature,
        seeds,
        dp_sync_state=None,
        dummy_run: bool = False,
        skip_attn_for_dummy_run: bool = False,
        mm_inputs=None,
        is_profile: bool = False,
        num_speculative_tokens: int | None = None,
    ) -> torch.Tensor:
        """Each request's draft over every slot; the arguments are the base
        class's. `num_speculative_tokens` counts the slots the target verifies,
        of which MTP drafts at most its own passes."""
        if not self.lookup_enabled:
            return super().propose(
                input_batch,
                attn_metadata,
                slot_mappings,
                last_hidden_states,
                aux_hidden_states,
                num_sampled,
                num_rejected,
                last_sampled,
                next_prefill_tokens,
                temperature,
                seeds,
                dp_sync_state=dp_sync_state,
                dummy_run=dummy_run,
                skip_attn_for_dummy_run=skip_attn_for_dummy_run,
                mm_inputs=mm_inputs,
                is_profile=is_profile,
                num_speculative_tokens=num_speculative_tokens,
            )
        if num_speculative_tokens is None:
            num_speculative_tokens = self.num_slots
        mtp_steps = min(num_speculative_tokens, self.num_speculative_steps)
        if (
            mtp_steps == 0
            and not dummy_run
            and not is_profile
            and self.vllm_config.parallel_config.data_parallel_size == 1
            and dp_sync_state is None
            and self.pcp_manager is None
        ):
            self.lookup_extended[input_batch.idx_mapping[: input_batch.num_reqs]] = 0
            return self.lookup_drafts[: input_batch.num_reqs].fill_(-1)

        active_rows = None
        draft = None
        prefill_only_np = getattr(self, "prefill_only_np", None)
        explicit_prefill = (
            prefill_only_np is not None
            and not dummy_run
            and not is_profile
            and np.any(prefill_only_np[input_batch.idx_mapping_np])
        )
        if (
            self.skip_prefill_draft
            and (explicit_prefill or self.skip_single_token_draft)
            and not dummy_run
            and not is_profile
            and dp_sync_state is None
            and self.pcp_manager is None
            and self.draft_watermarker is None
            and self.prompt_len_np is not None
            and self.max_seq_len_np is not None
            and self.token_ids is not None
            and self.total_len is not None
            and input_batch.num_reqs > 0
            and mm_inputs is None
            and not aux_hidden_states
        ):
            indices = input_batch.idx_mapping_np[: input_batch.num_reqs]
            if (
                len(indices) == input_batch.num_reqs
                and indices.dtype.kind in "iu"
                and np.all(indices >= 0)
                and np.all(indices < len(self.prompt_len_np))
                and np.all(indices < len(self.max_seq_len_np))
            ):
                skip_draft = (
                    prefill_only_np[indices]
                    if prefill_only_np is not None
                    else np.zeros(len(indices), dtype=bool)
                )
                if self.skip_single_token_draft:
                    skip_draft |= (
                        self.max_seq_len_np[indices] - self.prompt_len_np[indices] == 1
                    )
                if np.all(skip_draft):
                    out = self.lookup_drafts[: input_batch.num_reqs]
                    out.zero_()
                    self.draft_tokens[: input_batch.num_reqs].zero_()
                    self.lookup_extended[
                        input_batch.idx_mapping[: input_batch.num_reqs]
                    ] = 0
                    return out
                if (
                    np.any(skip_draft)
                    and can_compact_gemma4_prefill(
                        getattr(self, "model_state", None),
                        self.vllm_config.model_config,
                    )
                    and input_batch.positions.ndim == 1
                    and getattr(input_batch, "fast_prefill", None) is None
                ):
                    active_rows = np.flatnonzero(~skip_draft)
                    draft = self.propose_for_active_requests(
                        input_batch,
                        active_rows,
                        last_hidden_states,
                        num_sampled,
                        num_rejected,
                        last_sampled,
                        next_prefill_tokens,
                        temperature,
                        seeds,
                        mtp_steps,
                    )
                    prefill_indices = indices[skip_draft]
                    self.prefill_req_indices[: len(prefill_indices)].copy_(
                        torch.from_numpy(prefill_indices), non_blocking=True
                    )
                    self.lookup_extended.index_fill_(
                        0, self.prefill_req_indices[: len(prefill_indices)], 0
                    )
        if draft is None:
            if explicit_prefill:
                raise RuntimeError("prefill_only requests cannot enter draft proposal")
            draft = super().propose(
                input_batch,
                attn_metadata,
                slot_mappings,
                last_hidden_states,
                aux_hidden_states,
                num_sampled,
                num_rejected,
                last_sampled,
                next_prefill_tokens,
                temperature,
                seeds,
                dp_sync_state=dp_sync_state,
                dummy_run=dummy_run,
                skip_attn_for_dummy_run=skip_attn_for_dummy_run,
                mm_inputs=mm_inputs,
                is_profile=is_profile,
                num_speculative_tokens=mtp_steps,
            )
        if dummy_run or is_profile or mtp_steps == 0 or self.token_ids is None:
            return draft
        assert self.total_len is not None
        num_reqs = draft.shape[0]
        out = self.lookup_drafts[:num_reqs]
        idx_mapping = input_batch.idx_mapping[:num_reqs]
        if active_rows is not None:
            rows = self.compact_row_indices[:num_reqs]
            idx_mapping = self.idx_mapping[:num_reqs]
            num_sampled = num_sampled.index_select(0, rows)
            num_rejected = num_rejected.index_select(0, rows)
        prompt_lookup(
            self.token_ids,
            self.total_len,
            idx_mapping,
            draft,
            out,
            self.lookup_extended,
            num_sampled[:num_reqs],
            num_rejected[:num_reqs],
            self.lookup_stats,
            mtp_steps,
            self.lookup_min,
            self.lookup_max,
        )
        self.steps += 1
        if self.steps % LOG_EVERY == 0:
            self._log_stats()
        if active_rows is not None:
            assert self.full_batch_drafts is not None
            full_out = self.full_batch_drafts[: input_batch.num_reqs]
            full_out.zero_()
            full_out.index_copy_(0, self.compact_row_indices[:num_reqs], out)
            return full_out
        return out

    def _log_stats(self) -> None:
        stats = dict(zip(STATS, self.lookup_stats.tolist(), strict=True))
        logger.info(
            "Gemma4 prompt lookup: %d verified drafts accepted %.2f tokens each; "
            "the lookup extended %.1f%% of them, which accepted %.2f tokens past "
            "MTP's; %.1f%% of %d lookups found a match",
            stats["verified"],
            stats["accepted"] / max(stats["verified"], 1),
            100 * stats["extended"] / max(stats["verified"], 1),
            stats["extension_accepted"] / max(stats["extended"], 1),
            100 * stats["found"] / max(stats["lookups"], 1),
            stats["lookups"],
        )
