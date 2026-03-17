from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Tuple

import torch
from minisgl.core import Batch, Req
from minisgl.utils import init_logger

from .utils import PendingReq

if TYPE_CHECKING:
    from minisgl.kvcache import BaseCacheHandle
    from minisgl.message import UserMsg

    from .cache import CacheManager
    from .decode import DecodeManager
    from .table import TableManager

logger = init_logger(__name__)


class ChunkedReq(Req):
    def append_host(self, next_token: torch.Tensor) -> None:
        raise NotImplementedError("ChunkedReq should be sampled")

    def can_decode(self) -> bool:
        return False

# PrefillAdder 先把 token 内容准备好？？？
# Scheduler 统一分配 KV 页并回填 page_table
@dataclass
class PrefillAdder:
    token_budget: int # 本轮 prefill 还能处理多少 token
    reserved_size: int  # 预留给 decode 的空间
    cache_manager: CacheManager # prefix 匹配、lock/unlock、available_size
    table_manager: TableManager # 分配请求表槽位并写 token poll/page table？？？

    # 先检查可行性，再 allocate table 写入命中的前缀
    def _try_allocate_one(self, req: PendingReq) -> Tuple[BaseCacheHandle, int] | None:
        if self.table_manager.available_size == 0:
            return None

        handle, match_indices = self.cache_manager.match_req(req)   # 命中了多长的前缀（handle.cached_len），以及命中前缀对应的 KV pages 在哪里（match_indices）
        cached_len = handle.cached_len
        # TODO: better estimate policy
        # 在把请求塞进 batch 之前，先估算“这轮算上它之后，KV pages 够不够”，不够就先不收这个请求
        extend_len = req.input_len - cached_len
        estimated_len = extend_len + req.output_len

        if estimated_len + self.reserved_size > self.cache_manager.available_size:  # reserved_size: 把正在 decode 的 in-flight token 也加上
            return None
        self.cache_manager.lock(handle) # 将匹配到的前缀 lock，同时也会更新 available_size（因为 lock 可能会把一些之前 evictable 的节点转成 protected）
        if estimated_len + self.reserved_size > self.cache_manager.available_size:
            return self.cache_manager.unlock(handle)

        # 将可以复用的 ids 写入 page table 
        table_idx = self.table_manager.allocate()
        if cached_len > 0:  # NOTE: set the cached part
            device_ids = self.table_manager.token_pool[table_idx][:cached_len]
            page_entry = self.table_manager.page_table[table_idx][:cached_len]
            device_ids.copy_(req.input_ids[:cached_len].pin_memory(), non_blocking=True) # 将 token ids 从 CPU 拷贝到 GPU 的 token_pool 中
            page_entry.copy_(match_indices) # 在 page_table 中写入匹配前缀的 Page Index

        return handle, table_idx

    # 把请求加入当前 Batch，必要时切成 chunk
    def _add_one_req(
        self,
        pending_req: PendingReq,
        cache_handle: BaseCacheHandle,
        table_idx: int,
        cached_len: int,
    ) -> Req:
        remain_len = pending_req.input_len - cached_len # 计算还需要 prefill 的 token
        chunk_size = min(self.token_budget, remain_len) # 受限于 token_budget，可能 chunk
        is_chunked = chunk_size < remain_len
        
        CLS = ChunkedReq if is_chunked else Req
        self.token_budget -= chunk_size
        self.reserved_size += remain_len + pending_req.output_len
        # NOTE: update the tokens ids only; new pages will be allocated in the scheduler
        _slice = slice(cached_len, cached_len + chunk_size)
        device_ids = self.table_manager.token_pool[table_idx][_slice] # 拿到 token_pool 里这个 slot 对应的目标写入区间
        device_ids.copy_(pending_req.input_ids[_slice].pin_memory(), non_blocking=True) # 把 token ids 从 CPU 的 input_ids 拷贝入 GPU 的 token_pool
        return CLS(
            input_ids=pending_req.input_ids[: cached_len + chunk_size],
            table_idx=table_idx,
            cached_len=cached_len,
            output_len=pending_req.output_len,
            uid=pending_req.uid,
            cache_handle=cache_handle,
            sampling_params=pending_req.sampling_params,
        )

    # 尝试将一个 pending 请求加入本轮 prefill batch
    # 1. 先查找 Radix 是否命中（_try_allocate_one），命中部分就返回这部分前缀对应的物理 Slot Indices，同时 lock
    # 2. 
    def try_add_one(self, pending_req: PendingReq) -> Req | None:
        if self.token_budget <= 0:
            return None

        if chunked_req := pending_req.chunked_req: # 如果已经被切过
            return self._add_one_req(
                pending_req=pending_req,
                cache_handle=chunked_req.cache_handle,
                table_idx=chunked_req.table_idx,
                cached_len=chunked_req.cached_len,
            )

        if resource := self._try_allocate_one(pending_req):
            cache_handle, table_idx = resource
            return self._add_one_req(
                pending_req=pending_req,
                cache_handle=cache_handle,
                table_idx=table_idx,
                cached_len=cache_handle.cached_len,
            )

        return None


@dataclass
class PrefillManager:
    cache_manager: CacheManager
    table_manager: TableManager
    decode_manager: DecodeManager
    pending_list: List[PendingReq] = field(default_factory=list)

    def add_one_req(self, req: UserMsg) -> None:
        self.pending_list.append(PendingReq(req.uid, req.input_ids, req.sampling_params))

    def schedule_next_batch(self, prefill_budget: int) -> Batch | None:
        if len(self.pending_list) == 0:
            return None

        # estimated offset due to in-flight decode
        adder = PrefillAdder(
            token_budget=prefill_budget, # 本轮 prefill 的 token 预算 = max_extend_tokens
            reserved_size=self.decode_manager.inflight_tokens, # 给正在 decode 的 token 预留空间
            cache_manager=self.cache_manager,
            table_manager=self.table_manager,
        )
        reqs: List[Req] = []
        chunked_list: List[PendingReq] = []
        for pending_req in self.pending_list: # 遍历 pending_list 里的请求
            if req := adder.try_add_one(pending_req):
                pending_req.chunked_req = None
                if isinstance(req, ChunkedReq): # 如果返回的是 ChunkedReq，说明这个请求被切分了
                    pending_req.chunked_req = req
                    chunked_list.append(pending_req)
                reqs.append(req)
            else:
                break  # We cannot add more requests
        if len(reqs) == 0:
            return None
        self.pending_list = chunked_list + self.pending_list[len(reqs) :]
        return Batch(reqs=reqs, phase="prefill")

    @property
    def runnable(self) -> bool:
        return len(self.pending_list) > 0
