# ------------------------------------------------------------------------------
# Original Code developed by Jang-Hyun Kim
# Licensed under The MIT License
# GitHub Repository: https://github.com/snu-mllab/KVzip
# ------------------------------------------------------------------------------
from typing import Optional, Tuple

import copy
import torch
from attention.score import KVScore
from transformers import DynamicCache
from transformers.cache_utils import DynamicLayer
from attention.am_cache import AMCacheMixin

class RetainCache(AMCacheMixin, DynamicCache, KVScore):
    """ KV cache that subsamples KV at each attention module while retaining the full KV in memory.
        This cache enables evaluation across multiple compression ratios with a single prefill.
    """

    def __init__(self, model, evict_range: Tuple[int, int]):
        DynamicCache.__init__(self)
        self._seen_tokens = 0
        self.device = next(model.parameters()).device
        self.dtype = next(model.parameters()).dtype
        self.n_layers = model.config.num_hidden_layers
        self.n_heads = model.config.num_attention_heads
        self.n_heads_kv = model.config.num_key_value_heads
        self.n_group_kv = self.n_heads // self.n_heads_kv

        self.start_idx, self.end_idx = evict_range
        self.ctx_len = self.end_idx - self.start_idx
        self.sink = self.start_idx
        self.prefill_ids = None
        self.ctx_ids = None
        self.get_score = False  # indicator for KV scoring
        self.pruned = False

        self.valid_pad = torch.ones((1, self.n_heads_kv, self.start_idx),
                                    dtype=bool,
                                    device=self.device)

        self._init_am()

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        # Physically compact AM layers no longer expose the logical context
        # length through ``layer.keys.shape[-2]``.  Generation and RoPE must
        # still advance from the original prefill position.
        return int(self._seen_tokens)

    def update(
            self,
            key_states: torch.Tensor,
            value_states: torch.Tensor,
            layer_idx: int,
            cache_kwargs=dict(),
    ):
        """ Update KV cache and return
        """
        if layer_idx == 0:
            seen_token = cache_kwargs.get("seen_token", key_states.shape[-2])
            self._seen_tokens += seen_token

        # Update the cache
        if len(self.layers) <= layer_idx:
            layer = DynamicLayer()
            layer.lazy_initialization(key_states)
            layer.keys = key_states
            layer.values = value_states
            self.layers.append(layer)
        else:
            self.layers[layer_idx].keys = torch.cat([self.layers[layer_idx].keys, key_states], dim=-2)
            self.layers[layer_idx].values = torch.cat([self.layers[layer_idx].values, value_states],
                                                    dim=-2)

        return self.layers[layer_idx].keys, self.layers[layer_idx].values

    def slice(self, seen_token_prev: int):
        """ Evict KV of qeuries and generated tokens from the cache (for the reuse of the context cache)
        """
        assert len(self.layers[0].keys.shape) == 4, "Cache at each layer should be 4D tensor"
        # Reference-style AM chunk views retain the original logical cache
        # length for RoPE while physically omitting the other article chunks.
        # Convert the logical slice cursor back to the physical tensor cursor.
        physical_seen_token_prev = (
            seen_token_prev - int(getattr(self, "_am_physical_token_offset", 0))
        )
        if physical_seen_token_prev < 0:
            raise ValueError(
                f"invalid physical cache cursor {physical_seen_token_prev} "
                f"for logical cursor {seen_token_prev}"
            )
        compact_lengths = getattr(self, "_am_compact_base_lengths", None)
        for i in range(self.n_layers):
            compact_base_len = (
                compact_lengths[i]
                if compact_lengths is not None
                else None
            )
            restore_len = (
                int(compact_base_len)
                if compact_base_len is not None
                else physical_seen_token_prev
            )
            self.layers[i].keys = self.layers[i].keys[:, :, :restore_len]
            self.layers[i].values = self.layers[i].values[:, :, :restore_len]
        self._seen_tokens = seen_token_prev

    def _mem(self):
        """ Returns the memory usage of the cache in GB.
        """
        mem = self.n_layers * self.layers[0].keys.numel() * self.layers[0].keys.element_size()
        mem *= 2  # key + value
        return round(mem / 10**9, 1)

    def prune(self, ratio: float, level: str = "pair"):
        """Select KVzip pairs, keeping full backing tensors for repeated ratios."""
        if not 0 <= ratio <= 1:
            raise ValueError("ratio must be in [0, 1]")
        if level == "pair":
            self.valid, thres = self._threshold(self.score, ratio)
        elif level == "pair-uniform":
            self.valid, thres = self._threshold_uniform(self.score, ratio)
        else:
            raise ValueError(f"Unsupported KVzip level: {level}")
        assert self.valid.size(-1) == self.ctx_len
        actual = self.valid.float().mean().item()
        self.pruned = True
        print(f"ratio {actual:.4f} ({level}), threshold {thres:.4f}")
        return thres, actual

    def _get_valid(self, layer_idx: int, n_seq: int):
        """ obtain full mask for the given keys (retain system prompt and queries)
        """
        valid = torch.cat([self.valid_pad, self.valid[layer_idx]], dim=-1)  # sys prompt + context

        size = list(valid.shape)
        size[-1] = n_seq - valid.shape[-1]
        ones = torch.ones(size, device=valid.device, dtype=bool)
        valid = torch.cat([valid, ones], dim=-1)  # sys prompt + context + query ...

        return valid

    def prepare(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
    ):
        """ Subsample KV and flatten features for var_len FlashAttention
        """
        bsz, n_heads_q, q_len, dim = query_states.shape
        valid = self._get_valid(layer_idx, key_states.size(2))

        # prepare queries
        query_states = query_states.view(bsz, self.n_heads_kv, self.n_group_kv, q_len, dim)
        query_states = query_states.transpose(2, 3).contiguous().view(
            -1, self.n_group_kv, dim)  # bsz x head x seq, group, dim
        cu_seqlens_q = q_len * torch.arange(
            self.n_heads_kv + 1, dtype=torch.int32, device=self.device)

        # prepare keys/values
        key_states = key_states.view(-1, 1, dim)[valid.view(-1)]  # bsz x head x seq, dim
        value_states = value_states.view(-1, 1, dim)[valid.view(-1)]

        lens_k_head = valid.sum(-1).squeeze()
        cu_seqlens_k = lens_k_head.cumsum(0).int()
        cu_seqlens_k = torch.cat(
            [torch.tensor([0], dtype=torch.int32, device=self.device), cu_seqlens_k])

        info = {
            "cu_len_q": cu_seqlens_q,
            "cu_len_k": cu_seqlens_k,
            "max_len_q": q_len,
            "max_len_k": lens_k_head.max()
        }

        return query_states, key_states, value_states, info

def fork_cache(cache):
    """Share immutable K/V storage, with separate mutable cache wrappers."""
    if type(cache) is not RetainCache or cache.has_beta:
        raise ValueError("KVzip fork requires a plain, full-context RetainCache")
    forked = copy.copy(cache)
    forked.layers = [copy.copy(layer) for layer in cache.layers]
    for name in ("info", "_query_buffer", "am_beta", "am_ctx_indices", "am_layer_compacted"):
        value = getattr(cache, name, None)
        if isinstance(value, (dict, list)):
            setattr(forked, name, copy.copy(value))
    valid = getattr(cache, "valid", None)
    if isinstance(valid, torch.Tensor):
        forked.valid = valid.clone()
    elif isinstance(valid, list):
        forked.valid = [v.clone() for v in valid]
    return forked


class EvictCache(RetainCache):
    """Packed KVzip context for single-sequence inference after patch fitting.

    Construct from a pruned RetainCache. Only selected context K/V are copied;
    no reference to the full cache is kept. Heads may retain different lengths.
    Query/answer K/V live in separate tails, so repeated questions can discard
    their tails without copying or changing the compressed context.
    """

    @torch.no_grad()
    def __init__(self, cache):
        if (type(cache) is not RetainCache or not cache.pruned or cache.has_beta
                or cache.get_score or cache.collect_queries):
            raise ValueError("EvictCache requires a pruned, plain KVzip RetainCache")
        if len(cache.layers) != cache.n_layers or any(
            layer.keys.ndim != 4 or layer.keys.shape[0] != 1
            or layer.keys.shape != layer.values.shape
            or layer.keys.shape[-2] != cache.get_seq_length()
            for layer in cache.layers
        ):
            raise ValueError("EvictCache supports a populated single-sequence cache")
        self.__dict__.update(cache.__dict__)
        self.layers = []
        self.query_layers = [DynamicLayer() for _ in cache.layers]
        self.head_lengths = []
        self.cu_lengths = []
        self.base_seen_tokens = cache.get_seq_length()
        self.cu_heads = torch.arange(self.n_heads_kv + 1, dtype=torch.int32,
                                    device=self.device)
        for i, source in enumerate(cache.layers):
            valid = cache._get_valid(i, source.keys.shape[-2])
            lengths = valid.sum(-1).squeeze(0).to(torch.int32)
            self.head_lengths.append(tuple(lengths.tolist()))
            self.cu_lengths.append(torch.cat([lengths.new_zeros(1), lengths.cumsum(0)]).int())
            dim = source.keys.shape[-1]
            layer = DynamicLayer()
            layer.lazy_initialization(source.keys)
            layer.keys = source.keys.reshape(-1, dim)[valid.reshape(-1)]
            layer.values = source.values.reshape(-1, dim)[valid.reshape(-1)]
            self.layers.append(layer)
        # Selection is now encoded in packed tensors and head lengths.
        self.__dict__.pop('score', None)
        self.__dict__.pop('valid', None)

    def prune(self, *args, **kwargs):
        raise RuntimeError("Choose retention on RetainCache before constructing EvictCache")

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        if key_states.shape[0] != 1:
            raise ValueError("EvictCache supports one sequence (no beam search)")
        if layer_idx == 0:
            self._seen_tokens += (cache_kwargs or {}).get('seen_token', key_states.shape[-2])
        tail_keys, tail_values = self.query_layers[layer_idx].update(
            key_states, value_states, cache_kwargs)
        lengths = self.head_lengths[layer_idx]
        layer = self.layers[layer_idx]

        def append(packed, tail):
            return torch.cat([part for head, base in enumerate(packed.split(lengths))
                              for part in (base, tail[0, head])], dim=0)

        # Combined tensors are temporary inputs to the current attention call.
        return append(layer.keys, tail_keys), append(layer.values, tail_values)

    def prepare(self, query_states, key_states, value_states, layer_idx):
        _, _, q_len, dim = query_states.shape
        query_states = query_states.reshape(1, self.n_heads_kv, self.n_group_kv, q_len, dim)
        query_states = query_states.transpose(2, 3).reshape(-1, self.n_group_kv, dim)
        tail_len = self.query_layers[layer_idx].get_seq_length()
        info = dict(cu_len_q=q_len * self.cu_heads,
                    cu_len_k=self.cu_lengths[layer_idx] + tail_len * self.cu_heads,
                    max_len_q=q_len,
                    max_len_k=max(self.head_lengths[layer_idx]) + tail_len)
        return query_states, key_states.view(-1, 1, dim), value_states.view(-1, 1, dim), info

    def get_mask_sizes(self, cache_position, layer_idx):
        # Positions advance along the original document, not the packed length.
        length = self.base_seen_tokens + self.query_layers[layer_idx].get_seq_length()
        return length + cache_position.shape[0], 0

    def slice(self, seen_token_prev):
        keep = seen_token_prev - self.base_seen_tokens
        if not 0 <= keep <= self._seen_tokens - self.base_seen_tokens:
            raise ValueError("Cannot slice before the compressed context or beyond the query tail")
        for i, tail in enumerate(self.query_layers):
            if keep == 0:
                self.query_layers[i] = DynamicLayer()
            else:
                tail.keys = tail.keys[:, :, :keep].clone()
                tail.values = tail.values[:, :, :keep].clone()
        self._seen_tokens = seen_token_prev

    def _mem(self):
        tensors = [tensor for layer in self.layers + self.query_layers
                   for tensor in (layer.keys, layer.values) if tensor is not None]
        return round(sum(t.numel() * t.element_size() for t in tensors) / 10**9, 1)
