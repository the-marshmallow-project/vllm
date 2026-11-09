# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Gemma4 MTP (Multi-Token Prediction) speculator for speculative decoding.

The Gemma4 assistant model runs all decoder layers per draft step
(producing one token), and all its attention layers share KV cache
with the target model via cross-model KV sharing.
"""

from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn

from vllm.compilation.backends import set_model_tag
from vllm.config import VllmConfig, replace
from vllm.config.compilation import CUDAGraphMode
from vllm.distributed.parallel_state import get_pp_group
from vllm.logger import init_logger
from vllm.model_executor.model_loader import get_model
from vllm.model_executor.model_loader.utils import get_draft_load_config
from vllm.v1.worker.gpu.attn_utils import build_slot_mappings_by_layer
from vllm.v1.worker.gpu.dp_utils import dispatch_cg_and_sync_dp
from vllm.v1.worker.gpu.spec_decode.gemma4.prefill import (
    can_compact_gemma4_prefill,
    select_gemma4_prefill_inputs,
)
from vllm.v1.worker.gpu.spec_decode.autoregressive.cudagraph_utils import (
    SpeculatorCudaGraphManager,
)
from vllm.v1.worker.gpu.spec_decode.target_dependent_ar.speculator import (
    TargetDependentARSpeculator,
    _prepare_decode_inputs_kernel,
)

logger = init_logger(__name__)


def _copy_target_kv_scales(attn: nn.Module, target_attn: nn.Module) -> None:
    """Copy target KV scales while preserving their tensor representation.

    Default attention scales are scalar buffers, while some quantization
    methods replace them with length-one or per-head parameters. Preserve the
    draft layer's registration type so the shared KV cache is interpreted with
    the target's values and shapes without aliasing target parameters.
    """
    for scale_name in ("_k_scale", "_v_scale"):
        draft_scale = getattr(attn, scale_name)
        target_scale = getattr(target_attn, scale_name)
        scale = target_scale.detach().clone()
        if isinstance(draft_scale, nn.Parameter):
            scale = nn.Parameter(scale, requires_grad=False)
        setattr(attn, scale_name, scale)
    for scale_name in ("_k_scale_float", "_v_scale_float"):
        setattr(attn, scale_name, getattr(target_attn, scale_name))
    for scale_name in ("_k_scale_cpu", "_v_scale_cpu"):
        getattr(attn, scale_name).copy_(getattr(target_attn, scale_name))


class Gemma4Speculator(TargetDependentARSpeculator):
    def set_attn(
        self,
        model_state,
        kv_cache_config,
        block_tables,
        target_input_buffers,
        target_attn_groups,
    ) -> None:
        super().set_attn(
            model_state,
            kv_cache_config,
            block_tables,
            target_input_buffers,
            target_attn_groups,
        )
        if getattr(self, "skip_prefill_draft", False) and can_compact_gemma4_prefill(
            self.model_state, self.vllm_config.model_config
        ):
            self.compact_row_indices = torch.empty(self.max_num_reqs, dtype=torch.int64, device=self.device)
            self.prefill_req_indices = torch.empty_like(self.compact_row_indices)
            self.compact_block_table_backup = [torch.empty_like(table) for table in block_tables.input_block_tables]
            self.compact_slot_mapping_backup = torch.empty_like(block_tables.slot_mappings)

    def init_cudagraph_manager(self, cudagraph_mode: CUDAGraphMode) -> None:
        super().init_cudagraph_manager(cudagraph_mode)
        compact_mode = (
            CUDAGraphMode.FULL_DECODE_ONLY if cudagraph_mode.decode_mode() == CUDAGraphMode.FULL else CUDAGraphMode.NONE
        )
        self.compact_prefill_cudagraph_manager = (
            SpeculatorCudaGraphManager(
                self.vllm_config,
                self.device,
                compact_mode,
                decode_query_lens={1: self.max_num_reqs},
            )
            if getattr(self, "skip_prefill_draft", False)
            and can_compact_gemma4_prefill(getattr(self, "model_state", None), self.vllm_config.model_config)
            else None
        )

    def capture(self) -> None:
        super().capture()
        manager = getattr(self, "compact_prefill_cudagraph_manager", None)
        if manager is None:
            return
        self.last_token_indices.copy_(torch.arange(self.max_num_reqs, device=self.device))
        self.idx_mapping.zero_()
        self.current_draft_step.zero_()
        original_manager = self.prefill_cudagraph_manager
        self.prefill_cudagraph_manager = manager
        try:
            if manager.use_breakable_cg:
                manager.init_breakable_cg_runner(self.model)
            self.on_prefill_begin(self.max_num_reqs)
            try:

                def capture_prefill(*args, num_speculative_steps: int) -> None:
                    self._prefill(*args)

                manager.capture(
                    capture_prefill,
                    self.model_state,
                    self.input_buffers,
                    self.block_tables,
                    self.attn_groups,
                    self.kv_cache_config,
                    progress_bar_desc="Capturing compact Gemma4 prefill CUDA graphs",
                )
            finally:
                self.on_prefill_end(self.max_num_reqs)
        finally:
            self.prefill_cudagraph_manager = original_manager
            self.last_token_indices.zero_()
            self.idx_mapping.zero_()
            self.current_draft_step.zero_()

    def propose_for_active_requests(
        self,
        input_batch,
        active_rows: np.ndarray,
        last_hidden_states: torch.Tensor,
        num_sampled: torch.Tensor,
        num_rejected: torch.Tensor,
        last_sampled: torch.Tensor,
        next_prefill_tokens: torch.Tensor,
        temperature: torch.Tensor,
        seeds: torch.Tensor,
        num_speculative_steps: int,
    ) -> torch.Tensor:
        """Draft only selected requests; Gemma's Q-only queries share frozen KV."""
        num_reqs = len(active_rows)
        self.compact_row_indices[:num_reqs].copy_(torch.from_numpy(active_rows), non_blocking=True)
        packed = select_gemma4_prefill_inputs(
            input_batch,
            active_rows,
            last_hidden_states,
            num_sampled,
            num_rejected,
            last_sampled,
            next_prefill_tokens,
            row_indices=self.compact_row_indices[:num_reqs],
        )
        self._copy_request_inputs(num_reqs, packed.idx_mapping, temperature, seeds)
        self.input_buffers.input_ids[:num_reqs].copy_(packed.input_ids)
        self.input_buffers.positions[:num_reqs].copy_(packed.positions)
        self.input_buffers.positions[num_reqs : self.max_num_reqs].zero_()
        self.input_buffers.seq_lens[:num_reqs].copy_(packed.seq_lens)
        self.input_buffers.seq_lens[num_reqs:].zero_()
        self.input_buffers.query_start_loc[: num_reqs + 1].copy_(torch.arange(num_reqs + 1, device=self.device))
        self.input_buffers.query_start_loc[num_reqs + 1 :].fill_(num_reqs)
        self.hidden_states[:num_reqs].copy_(packed.hidden_states)
        self.hidden_states[num_reqs : self.max_num_reqs].zero_()
        self.last_token_indices[:num_reqs].copy_(torch.arange(num_reqs, device=self.device))
        self.last_token_indices[num_reqs:].zero_()
        self.current_draft_step.zero_()
        self.draft_max_seq_len = min(
            int(packed.seq_lens_cpu_upper_bound.max()) + num_speculative_steps,
            self.max_model_len,
        )
        original_manager = self.prefill_cudagraph_manager
        for backup, table in zip(
            self.compact_block_table_backup,
            self.block_tables.input_block_tables,
            strict=True,
        ):
            backup.copy_(table)
        self.compact_slot_mapping_backup.copy_(self.block_tables.slot_mappings)
        try:
            self.prefill_cudagraph_manager = self.compact_prefill_cudagraph_manager
            # Keep the target's sliding-window null blocks and graph buffer addresses.
            for backup, table in zip(
                self.compact_block_table_backup,
                self.block_tables.input_block_tables,
                strict=True,
            ):
                torch.index_select(
                    backup, 0, packed.row_indices, out=table[:num_reqs]
                )
                table[num_reqs:].zero_()
            batch_desc, _ = dispatch_cg_and_sync_dp(
                self.prefill_cudagraph_manager,
                num_reqs,
                num_reqs,
                uniform_token_count=1,
                dp_size=self.dp_size,
                dp_rank=self.dp_rank,
            )
            self.hidden_states[num_reqs : batch_desc.num_tokens].zero_()
            self.input_buffers.input_ids[num_reqs : batch_desc.num_tokens].zero_()
            self.input_buffers.positions[num_reqs : batch_desc.num_tokens].zero_()
            self.input_buffers.is_padding[:num_reqs].fill_(False)
            self.input_buffers.is_padding[num_reqs : batch_desc.num_tokens].fill_(True)
            self._prepare_eplb_forward(num_reqs)
            self.on_prefill_begin(num_reqs)
            try:
                slot_mappings = self.block_tables.compute_slot_mappings(
                    packed.idx_mapping,
                    self.input_buffers.query_start_loc[: num_reqs + 1],
                    self.input_buffers.positions[:num_reqs],
                    batch_desc.num_tokens,
                )
                metadata = self._build_uniform_attn_metadata(
                    batch_desc=batch_desc,
                    num_reqs=num_reqs,
                    num_query_per_req=1,
                    seq_lens_cpu_upper_bound=packed.seq_lens_cpu_upper_bound,
                    step=0,
                )
                if batch_desc.cg_mode == CUDAGraphMode.FULL:
                    assert self.prefill_cudagraph_manager is not None
                    self.prefill_cudagraph_manager.run_fullgraph(batch_desc)
                else:
                    self._prefill(
                        num_reqs,
                        batch_desc.num_tokens,
                        metadata,
                        build_slot_mappings_by_layer(slot_mappings, self.kv_cache_config),
                        num_tokens_across_dp=None,
                        cudagraph_runtime_mode=batch_desc.cg_mode,
                    )
            finally:
                self.on_prefill_end(num_reqs)
            if num_speculative_steps > 1:
                self.hidden_states[num_reqs : self.max_num_reqs].zero_()
                self.input_buffers.positions[num_reqs : self.max_num_reqs].zero_()
                self.sample_src_positions[num_reqs:].zero_()
                _prepare_decode_inputs_kernel[(num_reqs + 1,)](
                    self.draft_tokens[:num_reqs, 0],
                    self.draft_tokens.stride(0),
                    packed.target_seq_lens,
                    packed.num_rejected,
                    self.input_buffers.input_ids,
                    self.input_buffers.positions,
                    self.sample_src_positions,
                    self.input_buffers.query_start_loc,
                    self.input_buffers.seq_lens,
                    self.max_model_len,
                    self.max_num_reqs,
                    BLOCK_SIZE=1024,
                    ADVANCE_DRAFT_POSITIONS=False,
                )
                batch_desc, _ = dispatch_cg_and_sync_dp(
                    self.decode_cudagraph_manager,
                    num_reqs,
                    num_reqs,
                    uniform_token_count=1,
                    dp_size=self.dp_size,
                    dp_rank=self.dp_rank,
                )
                if self.decode_cudagraph_manager is not None:
                    batch_desc = self.decode_cudagraph_manager.specialize_spec_tokens(batch_desc, num_speculative_steps)
                self.on_multi_step_decode_begin(num_reqs)
                decode_fn = (
                    self._run_fused_multi_step_decode
                    if self.supports_decode_metadata_update
                    else self._multi_step_decode
                )
                try:
                    decode_fn(
                        num_reqs,
                        False,
                        batch_desc,
                        None,
                        packed.seq_lens_cpu_upper_bound,
                        num_speculative_steps,
                    )
                finally:
                    self.on_multi_step_decode_end(num_reqs)
            return self.draft_tokens[:num_reqs]
        finally:
            self.prefill_cudagraph_manager = original_manager
            for backup, table in zip(
                self.compact_block_table_backup,
                self.block_tables.input_block_tables,
                strict=True,
            ):
                table.copy_(backup)
            self.block_tables.slot_mappings.copy_(self.compact_slot_mapping_backup)

    @property
    def advance_draft_positions(self) -> bool:
        # Gemma4 MTP is Q-only and reads K/V from the target's existing cache.
        # No new KV slots are written, so positions and seq_lens stay fixed.
        return False

    def load_draft_model(
        self,
        target_model: nn.Module,
        target_attn_layer_names: set[str],
    ) -> nn.Module:
        draft_vllm_config = self._create_draft_vllm_config()
        with set_model_tag("eagle_head"):
            draft_model = get_model(
                vllm_config=draft_vllm_config,
                model_config=self.speculative_config.draft_model_config,
                load_config=get_draft_load_config(draft_vllm_config),
            )
        self._setup_gemma4_kv_sharing(draft_model, target_attn_layer_names)
        self._share_embeddings(draft_model, target_model)
        return draft_model

    def _create_draft_vllm_config(self) -> VllmConfig:
        """Preserve the target's forced TRITON_ATTN backend for draft layers.

        Gemma4 forces TRITON_ATTN due to heterogeneous head dimensions
        (head_dim=256 sliding, global_head_dim=512 full). The base class
        resets attention_config.backend to None for draft models, causing
        sliding layers to fall back to FLASH_ATTN which cannot handle
        KV-shared cache. Override to carry the target's backend through.
        """
        draft_model_config = self.speculative_config.draft_model_config
        draft_vllm_config = replace(
            self.vllm_config,
            model_config=draft_model_config,
        )
        target_backend = self.vllm_config.attention_config.backend
        if target_backend is not None:
            draft_vllm_config = replace(
                draft_vllm_config,
                attention_config=replace(
                    draft_vllm_config.attention_config,
                    backend=target_backend,
                ),
            )
        return draft_vllm_config

    def _setup_gemma4_kv_sharing(
        self,
        model: nn.Module,
        target_attn_layer_names: set[str],
    ) -> None:
        """Wire draft layers to share KV and KV scales with the target model.

        Each draft decoder layer is mapped to the last non-KV-shared
        target layer of the same attention type (sliding or full).
        """
        draft_config = self.speculative_config.draft_model_config.hf_config
        draft_text_config = draft_config.get_text_config()
        target_config = self.vllm_config.model_config.hf_config
        target_text_config = target_config.get_text_config()
        target_layer_types = getattr(target_text_config, "layer_types", [])

        if not (hasattr(model, "model") and hasattr(model.model, "layers")):
            return

        target_num_kv_shared = getattr(target_text_config, "num_kv_shared_layers", 0)
        target_num_layers = getattr(target_text_config, "num_hidden_layers", len(target_layer_types))
        num_non_shared = target_num_layers - target_num_kv_shared
        type_to_target_indices: dict[str, list[int]] = defaultdict(list)
        for idx, lt in enumerate(target_layer_types[:num_non_shared]):
            type_to_target_indices[lt].append(idx)

        target_prefix = "model.layers"
        for name in target_attn_layer_names:
            if ".layers." in name:
                target_prefix = name.split(".layers.")[0] + ".layers"
                break

        draft_layer_types = getattr(draft_text_config, "layer_types", [])
        for draft_idx, layer in enumerate(model.model.layers):
            if not hasattr(layer, "self_attn"):
                continue
            attn = getattr(layer.self_attn, "attn", None)
            if attn is None:
                continue

            draft_layer_type = draft_layer_types[draft_idx] if draft_idx < len(draft_layer_types) else "full_attention"
            candidates = type_to_target_indices.get(draft_layer_type, [])
            if not candidates:
                logger.warning(
                    "No target layer of type '%s' for draft layer %d",
                    draft_layer_type,
                    draft_idx,
                )
                continue

            target_idx = candidates[-1]
            target_layer_name = f"{target_prefix}.{target_idx}.self_attn.attn"
            attn.kv_sharing_target_layer_name = target_layer_name

            # KV-cache sharing aliases the cache tensor during allocation, but
            # the quantization scales live on the Attention modules themselves.
            # The BF16 draft model has no quantization config, so its K/V scales
            # otherwise remain at the default 1.0 while it reads the target's
            # calibrated FP8 cache.
            target_attn = self.vllm_config.compilation_config.static_forward_context[target_layer_name]
            _copy_target_kv_scales(attn, target_attn)

            logger.info(
                "Gemma4 MTP: draft layer %d (%s) -> %s",
                draft_idx,
                draft_layer_type,
                target_layer_name,
            )

    def _share_embeddings(
        self,
        draft_model: nn.Module,
        target_model: nn.Module,
    ) -> None:
        target_language_model = (
            target_model.get_language_model() if hasattr(target_model, "get_language_model") else target_model
        )
        if get_pp_group().world_size == 1:
            target_embed = getattr(target_language_model.model, "embed_tokens", None)
            if target_embed is not None:
                del draft_model.model.embed_tokens
                draft_model.model.embed_tokens = target_embed
