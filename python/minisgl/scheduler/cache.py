from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from minisgl.kvcache import BaseCacheHandle, create_cache_manager

if TYPE_CHECKING:
    from .utils import PendingReq


# 向上暴露三个核心接口：match_req, allocate, free_and_cache_finished_req
# 向下将前缀复用的具体逻辑委托给 RadixCacheManager，自身仅维护一个朴素的物理空闲页池 _free_slots
class CacheManager:
    def __init__(self, device: torch.device, num_pages: int, type: str):
        # TODO: support page_size > 1
        self._free_slots = torch.arange(num_pages, dtype=torch.int32, device=device)    # 仅负责记录当前空闲的物理页
        self.device = device
        self.manager = create_cache_manager(device=device, type=type)   # 负责前缀的匹配、插入、驱逐决策，以及前缀与物理页的映射关系
        self.num_pages = num_pages

    def _free(self, indices: torch.Tensor) -> None:
        if len(indices) > 0:
            self._free_slots = torch.cat([self._free_slots, indices])

    def match_req(self, req: PendingReq):   # 前缀匹配
        input_len = req.input_len
        assert input_len > 0, "Input length must be greater than 0."
        return self.manager.match_prefix(req.input_ids[: input_len - 1])    # 调用 RadixCacheManager match_prefix 方法，返回命中句柄（含命中长度）和对应物理页索引

    @property
    def available_size(self) -> int:
        return self.manager.size_info.evictable_size + len(self._free_slots)

    def lock(self, handle: BaseCacheHandle) -> None:
        self.manager.lock_handle(handle, unlock=False)

    def unlock(self, handle: BaseCacheHandle) -> None:
        self.manager.lock_handle(handle, unlock=True)

    def allocate(self, needed_len: int) -> torch.Tensor:    # 页分配
        if needed_len <= (free_len := len(self._free_slots)):
            allocated = self._free_slots[:needed_len]   # 从 _free_slots 取空闲页
            self._free_slots = self._free_slots[needed_len:]
            return allocated

        # 触发 RadixCacheManager 的驱逐逻辑（按 LRU 和引用计数规则）
        # NOTE: len(evicted) + free_len >= needed_len
        evicted = self.manager.evict(needed_len - free_len)
        merged = torch.cat([self._free_slots, evicted])
        assert len(merged) >= needed_len, "Eviction did not free enough space."

        allocated = merged[:needed_len]
        self._free_slots = merged[needed_len:]
        return allocated

    # 请求结束后，并非直接释放所有 KV 页，而是将其前缀写入 Radix Tree，转化为可复用资源。
    # 再回收重叠部分的页到空闲池，解锁旧句柄允许后续驱逐。
    def free_and_cache_finished_req(
        self,
        old_handle: BaseCacheHandle,
        input_ids: torch.Tensor,
        indices: torch.Tensor,
    ) -> None:
        in_cache_len = self.manager.insert_prefix(input_ids, indices)   # 这次插入后，有多少 token 的前缀被纳入 cache 管理
        self._free(indices[old_handle.cached_len : in_cache_len])   # old_handle.cached_len 是这个请求在开始执行时已经命中的那段前缀长度，它来自历史缓存
        self.unlock(old_handle)

    def check_integrity(self) -> None:
        self.manager.check_integrity()
        if len(self._free_slots) + self.manager.size_info.total_size != self.num_pages:
            raise RuntimeError(
                "CacheManager integrity check failed:"
                f" free_slots({len(self._free_slots)}) +"
                f" total_size({self.manager.size_info.total_size}) != num_pages({self.num_pages})"
            )
