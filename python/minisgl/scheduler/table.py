import torch


class TableManager:
    def __init__(self, max_running_reqs: int, page_table: torch.Tensor) -> None:
        self._max_running_reqs = max_running_reqs
        self._free_slots = list(range(max_running_reqs)) # 可用槽位列表，整数列表，每一个 Req 对应一个槽位
        # logical KV page table, shape = (max_running_reqs, max_input_len)
        # 行索引是请求 ID，列索引是 token 的位置，存储的是这个位置对应的 KV page id（物理页索引）
        # 后续会在 PrefillAdder 中根据 Radix 匹配将可复用的 Page id 写入，新到的 Token 则在 Scheduler=>CacheManager 中分配 Page id
        # kernel store_kv 会按 page id 将算出来的 kvcache 存入 GPU；AttentionBackend 会根据这个表拼接出 kvcache 用以计算注意力
        self.page_table = page_table

        # NOTE: dummy request also use this pool to get the input ids, so we need to
        # make sure the token pool is initialized with valid values (token_id = 0).
        # 一个二维矩阵，行索引是请求 ID；列索引是 token 的位置，存储的是 token ids
        # shape = (max_running_reqs, max_input_len)
        self.token_pool = torch.zeros_like(page_table, dtype=torch.int32)

    @property
    def available_size(self) -> int:
        return len(self._free_slots)

    def allocate(self) -> int:
        return self._free_slots.pop()

    def free(self, slot: int) -> None:
        self._free_slots.append(slot)
