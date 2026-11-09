# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Read-only prefix lookup and positional cache hits across storage pools."""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_utils import (
    BlockHash,
    KVCacheBlock,
)

if TYPE_CHECKING:
    from vllm.v1.core.kv_cache_coordinator import KVCacheCoordinator


class CacheLookup(Protocol):
    hash_block_size: int
    null_block: KVCacheBlock

    def get_cached_block(
        self, block_hash: BlockHash, kv_cache_group_ids: list[int]
    ) -> list[KVCacheBlock] | None: ...


class JointCacheLookup:
    """Prefer GPU independently for each group, memoizing physical lookups."""

    def __init__(self, gpu: BlockPool, cpu: BlockPool):
        self.gpu = gpu
        self.cpu = cpu
        self.hash_block_size = gpu.hash_block_size
        self.null_block = gpu.null_block
        self._hits: dict[tuple[BlockHash, int], KVCacheBlock | None] = {}

    def get_cached_block(
        self, block_hash: BlockHash, kv_cache_group_ids: list[int]
    ) -> list[KVCacheBlock] | None:
        result = []
        for group in kv_cache_group_ids:
            key = (block_hash, group)
            if key not in self._hits:
                hit = self.gpu.get_cached_block(block_hash, [group])
                if hit is None:
                    hit = self.cpu.get_cached_block(block_hash, [group])
                self._hits[key] = hit[0] if hit is not None else None
            block = self._hits[key]
            if block is None:
                return None
            result.append(block)
        return result


@dataclass(frozen=True, slots=True)
class CacheHitBlock:
    block: KVCacheBlock
    is_cpu: bool = False


@dataclass(kw_only=True)
class JointCacheHit:
    """A reconciled prefix; null blocks denote attention padding, not loads.

    ``num_gpu_prefix_tokens`` is a boundary, not the total GPU reuse count.
    Scheduling uses the boundary and its extension; allocation uses positions.
    Admission accepts the complete match or defers the request.
    """

    blocks: tuple[list[CacheHitBlock], ...]
    num_computed_tokens: int
    num_gpu_prefix_tokens: int
    shared_prefix_boundary: int
    num_cpu_tokens: int
    lookup: JointCacheLookup
    pinned: bool = False
    _pin_blocks: tuple[list[KVCacheBlock], list[KVCacheBlock]] | None = field(
        default=None, init=False, repr=False, compare=False
    )

    @classmethod
    def find(
        cls,
        coordinator: "KVCacheCoordinator",
        lookup: JointCacheLookup,
        block_hashes: list[BlockHash],
        max_length: int,
        scheduler_block_size: int,
    ) -> "JointCacheHit":
        blocks, length, uncached = coordinator.find_longest_cache_hit(
            block_hashes, max_length, lookup=lookup
        )
        while length % scheduler_block_size and any(
            block.pool is not None and block.pool() is lookup.cpu
            for group in blocks
            for block in group
        ):
            blocks, length, uncached = coordinator.find_longest_cache_hit(
                block_hashes,
                length // scheduler_block_size * scheduler_block_size,
                lookup=lookup,
            )
        hits: list[list[CacheHitBlock]] = []
        gpu_blocks: list[KVCacheBlock] = []
        cpu_blocks: list[KVCacheBlock] = []
        null_hit = CacheHitBlock(lookup.null_block)
        gpu_prefix_tokens = length
        cpu_hash_blocks: set[int] = set()
        hash_block_size = lookup.hash_block_size
        for group, manager in zip(
            blocks, coordinator.single_type_managers, strict=True
        ):
            group_hits: list[CacheHitBlock] = []
            for index, block in enumerate(group):
                if block is lookup.null_block:
                    group_hits.append(null_hit)
                    continue
                is_cpu = block.pool is not None and block.pool() is lookup.cpu
                group_hits.append(CacheHitBlock(block, is_cpu))
                if not block.is_null:
                    (cpu_blocks if is_cpu else gpu_blocks).append(block)
                if is_cpu:
                    start = index * manager.block_size
                    end = min(start + manager.block_size, length)
                    gpu_prefix_tokens = min(gpu_prefix_tokens, start)
                    cpu_hash_blocks.update(
                        range(start // hash_block_size, end // hash_block_size)
                    )
            hits.append(group_hits)
        hit = cls(
            blocks=tuple(hits),
            num_computed_tokens=length,
            num_gpu_prefix_tokens=(
                gpu_prefix_tokens // scheduler_block_size * scheduler_block_size
                if cpu_hash_blocks
                else length
            ),
            num_cpu_tokens=len(cpu_hash_blocks) * hash_block_size,
            shared_prefix_boundary=length + uncached if uncached else 0,
            lookup=lookup,
        )
        hit._pin_blocks = (gpu_blocks, cpu_blocks)
        return hit

    def _get_pin_blocks(self) -> tuple[list[KVCacheBlock], list[KVCacheBlock]]:
        if self._pin_blocks is None:
            # Keep a reference for every non-null position, even for repeated blocks.
            gpu_blocks: list[KVCacheBlock] = []
            cpu_blocks: list[KVCacheBlock] = []
            for group in self.blocks:
                for hit in group:
                    if not hit.block.is_null:
                        (cpu_blocks if hit.is_cpu else gpu_blocks).append(hit.block)
            self._pin_blocks = (gpu_blocks, cpu_blocks)
        return self._pin_blocks

    @property
    def num_gpu_tokens(self) -> int:
        return self.num_computed_tokens - self.num_cpu_tokens

    @property
    def needs_load(self) -> bool:
        return self.num_cpu_tokens > 0

    def pin(self) -> None:
        assert not self.pinned
        for pool, blocks in zip(
            (self.lookup.gpu, self.lookup.cpu), self._get_pin_blocks(), strict=True
        ):
            pool.touch(blocks)
        self.pinned = True

    def release(self) -> None:
        if not self.pinned:
            return
        for pool, blocks in zip(
            (self.lookup.gpu, self.lookup.cpu), self._get_pin_blocks(), strict=True
        ):
            pool.free_blocks(blocks)
        self.pinned = False

    def allocation_blocks(self) -> tuple[list[KVCacheBlock], ...]:
        """Use unowned placeholders only for the allocator's capacity check."""
        return tuple(
            [KVCacheBlock(block_id=-1) if hit.is_cpu else hit.block for hit in group]
            for group in self.blocks
        )
