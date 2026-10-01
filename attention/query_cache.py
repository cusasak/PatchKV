import copy

import torch
from transformers import GenerationConfig
from transformers.cache_utils import DynamicLayer

from attention.kvcache import RetainCache


class QueryCache(RetainCache):
    def __init__(self, context):
        if not self.supports(context):
            raise ValueError("QueryCache requires a populated plain RetainCache")
        self.__dict__.update(context.__dict__)
        self.layers = [copy.copy(layer) for layer in context.layers]
        self.query_layers = [DynamicLayer() for _ in self.layers]

    @staticmethod
    def supports(context):
        # AM may rewrite values and scoring may update shared metadata. Keep
        # those paths, quantized/static caches and empty caches unchanged.
        if type(context) is not RetainCache:
            return False
        if any(bool(getattr(context, attr, False)) for attr in (
            "has_beta", "collect_queries", "get_score", "_am_physical_token_offset",
        )):
            return False
        layers = context.layers
        return bool(layers) and len(layers) == context.n_layers and all(
            layer.keys is not None and layer.values is not None
            and layer.keys.ndim == 4 and layer.keys.shape[0] == 1
            and layer.keys.shape == layer.values.shape
            and layer.keys.shape[-2] == context._seen_tokens
            for layer in layers
        )

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        if layer_idx == 0:
            self._seen_tokens += (cache_kwargs or {}).get(
                "seen_token", key_states.shape[-2]
            )
        query_keys, query_values = self.query_layers[layer_idx].update(
            key_states, value_states, cache_kwargs
        )
        context = self.layers[layer_idx]
        return (
            torch.cat([context.keys, query_keys], dim=-2),
            torch.cat([context.values, query_values], dim=-2),
        )

    def get_mask_sizes(self, cache_position, layer_idx):
        # HF asks for each layer's size before appending the current query.
        length = self.layers[layer_idx].keys.shape[-2]
        length += self.query_layers[layer_idx].get_seq_length()
        return length + cache_position.shape[0], 0


def fork_generation_cache(context, model_generation_config=None, **generation_kwargs):
    if not QueryCache.supports(context):
        return context
    config = copy.copy(
        generation_kwargs.get("generation_config")
        or model_generation_config
        or GenerationConfig()
    )
    for key, value in generation_kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
    mode = config.get_generation_mode(generation_kwargs.get("assistant_model"))
    if mode not in ("greedy_search", "sample") or config.num_return_sequences != 1:
        return context
    if not config.use_cache or generation_kwargs.get("custom_generate") is not None:
        return context
    return QueryCache(context)
