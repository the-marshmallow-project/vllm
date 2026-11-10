# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import TYPE_CHECKING

from vllm.v1.worker.gpu.spec_decode.mtp.config import MTPSpeculatorConfig

if TYPE_CHECKING:
    from vllm.config import SpeculativeConfig


class Gemma4MTPSpeculatorConfig(MTPSpeculatorConfig):
    speculator_class = (
        "vllm.v1.worker.gpu.spec_decode.gemma4.prompt_lookup."
        "Gemma4PromptLookupSpeculator"
    )

    @classmethod
    def apply_options(cls, spec: "SpeculativeConfig") -> None:
        super().apply_options(spec)
        # draft_lm_head_quantization quantizes the drafter's own head. Drafters
        # with ordered embeddings score centroid-selected rows of its weight
        # directly instead of running the head.
        hf_config = spec.draft_model_config.hf_config
        if spec.draft_lm_head_quantization is not None and getattr(
            hf_config, "use_ordered_embeddings", False
        ):
            raise ValueError(
                "draft_lm_head_quantization is not supported for Gemma 4 MTP "
                "drafters with ordered embeddings."
            )
