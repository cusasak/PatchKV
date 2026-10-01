import copy
from typing import List, Sequence, Tuple

import torch


ChunkRange = Tuple[int, int]


def fixed_chunk_ranges(length: int, chunk_size: int) -> List[ChunkRange]:
    length = int(length)
    chunk_size = int(chunk_size)
    if length < 0:
        raise ValueError(f"length must be non-negative, got {length}")
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    return [
        (start, min(start + chunk_size, length))
        for start in range(0, length, chunk_size)
    ]


def _clear_am_state(cache) -> None:
    cache.am_beta = None
    cache.am_ctx_indices = None
    cache.am_layer_compacted = None
    cache.has_beta = False
    cache.pruned = False
    cache.am_queries = None
    cache.am_query_subsample = None
    cache._query_buffer = None
    cache.collect_queries = False
    if hasattr(cache, "valid"):
        cache.valid = None


def _shallow_fork_cache_layers(base_cache):
    forked = copy.copy(base_cache)
    forked.layers = []
    for source_layer in base_cache.layers:
        layer = copy.copy(source_layer)
        layer.keys = source_layer.keys
        layer.values = source_layer.values
        forked.layers.append(layer)
    forked._am_shared_value_layers = [
        True for _ in range(int(base_cache.n_layers))
    ]
    forked._am_compact_base_lengths = [
        None for _ in range(int(base_cache.n_layers))
    ]
    return forked


def fork_reference_chunk_student(base_cache):
    student = _shallow_fork_cache_layers(base_cache)
    student._am_compact_on_merge = False
    _clear_am_state(student)
    return student


def fork_compact_am_destination(base_cache):
    destination = _shallow_fork_cache_layers(base_cache)
    destination._am_compact_on_merge = True
    _clear_am_state(destination)
    return destination


def compact_am_layer_storage(cache, layer_idx: int) -> int:
    layer_idx = int(layer_idx)
    if not cache.should_use_am(layer_idx):
        raise ValueError(
            f"cannot compact inactive AM layer {layer_idx}"
        )
    compact_lengths = getattr(cache, "_am_compact_base_lengths", None)
    if compact_lengths is None:
        compact_lengths = [None for _ in range(int(cache.n_layers))]
        cache._am_compact_base_lengths = compact_lengths
    if compact_lengths[layer_idx] is not None:
        return int(compact_lengths[layer_idx])

    layer = cache.layers[layer_idx]
    sink = int(cache.sink)
    ctx_end = sink + int(cache.ctx_len)
    physical_len = int(layer.keys.shape[2])
    if ctx_end > physical_len:
        raise ValueError(
            "AM context exceeds physical cache storage: "
            f"sink={sink}, ctx_len={cache.ctx_len}, physical={physical_len}"
        )

    indices = cache.am_ctx_indices[layer_idx]
    n_heads, selected = indices.shape
    dim = int(layer.keys.shape[-1])
    gather_idx = (indices + sink).view(1, n_heads, selected, 1)
    gather_idx = gather_idx.expand(1, n_heads, selected, dim)
    selected_keys = torch.gather(layer.keys, 2, gather_idx)
    selected_values = torch.gather(layer.values, 2, gather_idx)

    layer.keys = torch.cat(
        [layer.keys[:, :, :sink], selected_keys, layer.keys[:, :, ctx_end:]],
        dim=2,
    ).contiguous()
    layer.values = torch.cat(
        [
            layer.values[:, :, :sink],
            selected_values,
            layer.values[:, :, ctx_end:],
        ],
        dim=2,
    ).contiguous()
    compact_lengths[layer_idx] = int(layer.keys.shape[2])
    shared_values = getattr(cache, "_am_shared_value_layers", None)
    if shared_values is not None:
        shared_values[layer_idx] = False
    return int(compact_lengths[layer_idx])


def clone_reference_chunk_cache(
    base_cache,
    ctx_ids: torch.Tensor,
    article_start: int,
    article_end: int,
    chunk_start: int,
    chunk_end: int,
):
    article_start = int(article_start)
    article_end = int(article_end)
    chunk_start = int(chunk_start)
    chunk_end = int(chunk_end)
    ctx_len = int(ctx_ids.shape[1])

    if not (0 <= article_start <= article_end <= ctx_len):
        raise ValueError(
            f"invalid article range [{article_start}, {article_end}) for ctx_len={ctx_len}"
        )
    article_len = article_end - article_start
    if not (0 <= chunk_start < chunk_end <= article_len):
        raise ValueError(
            f"invalid chunk range [{chunk_start}, {chunk_end}) for article_len={article_len}"
        )

    chunk_cache = _shallow_fork_cache_layers(base_cache)
    base_sink = int(base_cache.sink)
    article_abs_start = base_sink + article_start
    article_abs_end = base_sink + article_end
    chunk_abs_start = article_abs_start + chunk_start
    chunk_abs_end = article_abs_start + chunk_end

    original_physical_len = int(base_cache.layers[0].keys.shape[2])
    keep_slices = (
        (0, article_abs_start),
        (chunk_abs_start, chunk_abs_end),
        (article_abs_end, original_physical_len),
    )
    for layer in chunk_cache.layers:
        layer.keys = torch.cat(
            [layer.keys[:, :, start:end, :] for start, end in keep_slices],
            dim=2,
        ).contiguous()
        layer.values = torch.cat(
            [layer.values[:, :, start:end, :] for start, end in keep_slices],
            dim=2,
        ).contiguous()

    physical_len = int(chunk_cache.layers[0].keys.shape[2])
    chunk_cache._am_physical_token_offset = original_physical_len - physical_len
    chunk_cache.sink = article_abs_start
    chunk_cache.start_idx = article_abs_start
    chunk_cache.ctx_len = chunk_end - chunk_start
    chunk_cache.end_idx = chunk_cache.sink + chunk_cache.ctx_len
    chunk_cache.ctx_ids = ctx_ids[
        :, article_start + chunk_start:article_start + chunk_end
    ]
    chunk_cache._am_reference_range = (article_start + chunk_start, article_start + chunk_end)
    chunk_cache.prefill_ids = base_cache.prefill_ids
    chunk_cache._seen_tokens = base_cache._seen_tokens
    _clear_am_state(chunk_cache)
    return chunk_cache


def configure_destination_article(
    cache,
    base_sink: int,
    article_start: int,
    article_end: int,
) -> None:
    article_start = int(article_start)
    article_end = int(article_end)
    cache.sink = int(base_sink) + article_start
    cache.ctx_len = article_end - article_start
    _clear_am_state(cache)
    cache._init_am_state()


def merge_reference_chunk_layer(
    destination,
    chunk_caches: Sequence,
    chunk_ranges: Sequence[ChunkRange],
    layer_idx: int,
) -> float:
    if len(chunk_caches) != len(chunk_ranges):
        raise ValueError(
            f"chunk cache/range mismatch: {len(chunk_caches)} != "
            f"{len(chunk_ranges)}"
        )
    if not chunk_caches:
        raise ValueError("cannot merge an empty chunk list")
    layer_idx = int(layer_idx)
    if not 0 <= layer_idx < int(destination.n_layers):
        raise IndexError(f"destination layer index out of range: {layer_idx}")
    if (
        destination.am_beta is None
        or destination.am_ctx_indices is None
        or destination.am_layer_compacted is None
    ):
        raise ValueError(
            "destination AM state is not configured; call "
            "configure_destination_article first"
        )

    compact_on_merge = bool(
        getattr(destination, "_am_compact_on_merge", False)
    )
    if compact_on_merge:
        destination.layers[layer_idx].values = (
            destination.layers[layer_idx].values.clone()
        )
        shared_values = getattr(
            destination, "_am_shared_value_layers", None
        )
        if shared_values is not None:
            shared_values[layer_idx] = False

    indices_by_head = []
    beta_by_head = []
    valid_by_head = []

    for head_idx in range(destination.n_heads_kv):
        head_indices = []
        head_beta = []

        for chunk_cache, (chunk_start, _) in zip(chunk_caches, chunk_ranges):
            if not chunk_cache.should_use_am(layer_idx):
                raise ValueError(
                    f"chunk cache is not compacted at layer {layer_idx}"
                )
            local_idx = chunk_cache.am_ctx_indices[layer_idx][head_idx]
            local_beta = chunk_cache.am_beta[layer_idx][0, head_idx]
            # Defensive support for padded non-uniform heads.
            keep = torch.isfinite(local_beta)
            local_idx = local_idx[keep]
            local_beta = local_beta[keep]
            if local_idx.numel() == 0:
                continue

            global_idx = local_idx + int(chunk_start)
            head_indices.append(global_idx)
            head_beta.append(local_beta)

            source_positions = chunk_cache.sink + local_idx
            target_positions = destination.sink + global_idx
            destination.layers[layer_idx].values[
                0, head_idx, target_positions
            ] = chunk_cache.layers[layer_idx].values[
                0, head_idx, source_positions
            ]

        if head_indices:
            merged_idx = torch.cat(head_indices, dim=0)
            merged_beta = torch.cat(head_beta, dim=0)
            order = torch.argsort(merged_idx)
            merged_idx = merged_idx[order]
            merged_beta = merged_beta[order]
        else:
            merged_idx = torch.empty(
                0, dtype=torch.long, device=destination.device
            )
            merged_beta = torch.empty(
                0, dtype=destination.dtype, device=destination.device
            )

        valid = torch.zeros(
            destination.ctx_len,
            dtype=torch.bool,
            device=destination.device,
        )
        valid[merged_idx] = True
        indices_by_head.append(merged_idx)
        beta_by_head.append(merged_beta)
        valid_by_head.append(valid)

    max_t = max(indices.numel() for indices in indices_by_head)
    for head_idx in range(destination.n_heads_kv):
        pad = max_t - indices_by_head[head_idx].numel()
        if pad <= 0:
            continue
        indices_by_head[head_idx] = torch.cat(
            [
                indices_by_head[head_idx],
                torch.zeros(
                    pad, dtype=torch.long, device=destination.device
                ),
            ],
            dim=0,
        )
        beta_by_head[head_idx] = torch.cat(
            [
                beta_by_head[head_idx],
                torch.full(
                    (pad,),
                    float("-inf"),
                    dtype=beta_by_head[head_idx].dtype,
                    device=destination.device,
                ),
            ],
            dim=0,
        )

    destination.valid[layer_idx] = torch.stack(
        valid_by_head, dim=0
    ).unsqueeze(0)
    destination.am_ctx_indices[layer_idx] = torch.stack(
        indices_by_head, dim=0
    )
    destination.am_beta[layer_idx] = torch.stack(
        beta_by_head, dim=0
    ).unsqueeze(0)
    destination.am_layer_compacted[layer_idx] = True
    destination.has_beta = True
    destination.pruned = False

    if compact_on_merge:
        compact_am_layer_storage(destination, layer_idx)

    kept = sum(int(valid.sum().item()) for valid in valid_by_head)
    total = destination.ctx_len * destination.n_heads_kv
    return kept / max(total, 1)


def merge_reference_chunk_caches(
    destination,
    chunk_caches: Sequence,
    chunk_ranges: Sequence[ChunkRange],
    *,
    base_sink: int,
    article_start: int,
    article_end: int,
) -> float:
    if len(chunk_caches) != len(chunk_ranges):
        raise ValueError(
            f"chunk cache/range mismatch: {len(chunk_caches)} != {len(chunk_ranges)}"
        )
    if not chunk_caches:
        raise ValueError("cannot merge an empty chunk list")

    configure_destination_article(
        destination,
        base_sink=base_sink,
        article_start=article_start,
        article_end=article_end,
    )

    for layer_idx in range(destination.n_layers):
        merge_reference_chunk_layer(
            destination,
            chunk_caches,
            chunk_ranges,
            layer_idx,
        )

    kept = sum(int(valid.sum().item()) for valid in destination.valid)
    total = destination.ctx_len * destination.n_heads_kv * destination.n_layers
    return kept / max(total, 1)
