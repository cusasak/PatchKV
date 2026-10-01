from __future__ import annotations
import contextlib
import dataclasses
import math
import time
from typing import Dict, List, Sequence, Tuple, Union
import torch
from tqdm import tqdm
from attention.kvcache import RetainCache

PatchSequence = Union[Tuple[torch.Tensor, torch.Tensor],
                      Tuple[torch.Tensor, torch.Tensor, bool]]

@dataclasses.dataclass(frozen=True)
class CacheLayerSnapshot:
    layer_idx: int
    keys: torch.Tensor
    values: torch.Tensor
    seen_tokens: int

@dataclasses.dataclass
class _PreparedChunk:
    teacher_hidden: torch.Tensor
    student_hidden: torch.Tensor
    teacher_meta: Dict[str, torch.Tensor]
    student_meta: Dict[str, torch.Tensor]

def snapshot_cache_tensors(cache):
    return {
        "layers": [
            (getattr(layer, "keys", None), getattr(layer, "values", None))
            for layer in getattr(cache, "layers", [])
        ],
        "seen_tokens": int(cache._seen_tokens),
        "start_idx": int(cache.start_idx),
        "end_idx": int(cache.end_idx),
    }

def restore_cache_tensors(cache, snapshot) -> None:
    if len(cache.layers) != len(snapshot["layers"]):
        raise RuntimeError("cache layer count changed during patching")
    for layer, (keys, values) in zip(cache.layers, snapshot["layers"]):
        layer.keys = keys
        layer.values = values
    cache._seen_tokens = snapshot["seen_tokens"]
    cache.start_idx = snapshot["start_idx"]
    cache.end_idx = snapshot["end_idx"]

def snapshot_cache_layer(cache, layer_idx: int) -> CacheLayerSnapshot:
    if type(cache) is not RetainCache:
        raise TypeError(
            "exact-layer cleanup supports RetainCache only; "
            f"got {type(cache).__name__}"
        )
    layer_idx = int(layer_idx)
    if not 0 <= layer_idx < len(cache.layers):
        raise IndexError(f"cache layer index out of range: {layer_idx}")
    layer = cache.layers[layer_idx]
    return CacheLayerSnapshot(
        layer_idx=layer_idx,
        keys=layer.keys,
        values=layer.values,
        seen_tokens=int(cache._seen_tokens),
    )

def restore_cache_layer(cache, snapshot: CacheLayerSnapshot) -> None:
    layer = cache.layers[snapshot.layer_idx]
    layer.keys = snapshot.keys
    layer.values = snapshot.values
    cache._seen_tokens = snapshot.seen_tokens

@contextlib.contextmanager
def _tf32_statistics():
    previous = torch.get_float32_matmul_precision()
    try:
        torch.set_float32_matmul_precision("high")
        yield
    finally:
        torch.set_float32_matmul_precision(previous)

def _meta_from_state(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {
        "position_ids": state["position_ids"],
        "causal_mask": state["causal_mask"],
        "cache_position": state["cache_position"],
        "position_embeddings": state["position_embeddings"],
    }

def _unpack_patch_sequence(
    sequence: PatchSequence,
) -> Tuple[torch.Tensor, torch.Tensor, bool]:
    if len(sequence) == 2:
        prefill_ids, forward_ids = sequence
        return prefill_ids, forward_ids, False
    if len(sequence) == 3:
        prefill_ids, forward_ids, is_study = sequence
        return prefill_ids, forward_ids, bool(is_study)
    raise ValueError(
        "patch sequences must be (prefill_ids, forward_ids) or "
        "(prefill_ids, forward_ids, is_study)"
    )

def _prepare_chunks(model, teacher_kv, student_kv, sequences: Sequence[PatchSequence]):
    teacher_start, teacher_end = int(teacher_kv.start_idx), int(teacher_kv.end_idx)
    student_start, student_end = int(student_kv.start_idx), int(student_kv.end_idx)
    prepared: List[_PreparedChunk] = []
    teacher_kv.end_idx = 0
    student_kv.end_idx = 0
    try:
        for sequence in sequences:
            prefill_ids, forward_ids, is_study = _unpack_patch_sequence(
                sequence
            )
            if is_study:
                # Unlike repeat chunks, every self-study Q+A attends to the
                # same complete context.  Restart each question by
                # rewinding only the compressed student's active window.
                student_kv.start_idx = student_start
            teacher_kv.end_idx = teacher_kv.start_idx + int(prefill_ids.shape[1])
            student_kv.end_idx = student_kv.start_idx + int(prefill_ids.shape[1])
            teacher_state = prepare_forward_state(model, forward_ids, teacher_kv)
            student_state = prepare_forward_state(model, forward_ids, student_kv)
            prepared.append(
                _PreparedChunk(
                    teacher_hidden=teacher_state["hidden_states"].detach().clone(),
                    student_hidden=student_state["hidden_states"].detach().clone(),
                    teacher_meta=_meta_from_state(teacher_state),
                    student_meta=_meta_from_state(student_state),
                )
            )
            if not is_study:
                # Repeat reconstruction walks through disjoint context chunks.
                student_kv.start_idx = student_kv.end_idx
    finally:
        teacher_kv.start_idx, teacher_kv.end_idx = teacher_start, teacher_end
        student_kv.start_idx, student_kv.end_idx = student_start, student_end
    return prepared


class RidgeAccumulator:
    def __init__(self, d_out, d_in, device):
        self.r = torch.zeros(d_out, d_in, device=device, dtype=torch.float32)
        self.g = torch.zeros(d_in, d_in, device=device, dtype=torch.float32)
        self.n_rows = 0
        self.solve_fallbacks = 0

    def add_rows(self, target_rows, h_rows):
        if target_rows.ndim != 2 or h_rows.ndim != 2 or target_rows.shape[0] != h_rows.shape[0]:
            raise ValueError("Ridge inputs must be matrices with matching row counts")
        if target_rows.dtype != torch.float32 or h_rows.dtype != torch.float32:
            raise TypeError("Ridge rows must be FP32")
        with _tf32_statistics():
            self.r.addmm_(target_rows.T, h_rows)
            self.g.addmm_(h_rows.T, h_rows)
        self.n_rows += h_rows.shape[0]

    def solve(self, lam):
        if not self.n_rows:
            raise ValueError("Cannot fit a patch without reference tokens")
        gram = 0.5 * (self.g + self.g.T)
        # Weighted ridge scaling.
        scale = ((gram * gram).sum() / torch.trace(gram).clamp_min(1e-12)).clamp_min(1e-12)
        alpha = lam * scale
        self.g.copy_(gram)
        self.g.diagonal().add_(alpha)
        del gram
        if not torch.isfinite(self.g).all() or not torch.isfinite(self.r).all():
            raise FloatingPointError("Nonfinite patch statistics")
        try:
            factor = torch.linalg.cholesky(self.g)
            patch = torch.cholesky_solve(self.r.T, factor).T
        except torch.linalg.LinAlgError:
            self.solve_fallbacks += 1
            patch = torch.linalg.solve(self.g, self.r.T).T
        if not torch.isfinite(patch).all():
            raise FloatingPointError("Nonfinite ridge solution")
        return patch, float(alpha)


def _block_forward(model, block, hidden, meta, kv):
    snapshot = snapshot_cache_layer(kv, block.self_attn.layer_idx)
    try:
        return block_forward(
            block, hidden, attention_mask=meta["causal_mask"],
            position_ids=meta["position_ids"], past_key_values=kv,
            cache_position=meta["cache_position"],
            position_embeddings=meta["position_embeddings"],
        )
    finally:
        restore_cache_layer(kv, snapshot)


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.inference_mode()
def sequential_patch(model, teacher_kv, sequences: Sequence[PatchSequence],
                     *, student_kv, lam=1e-3):
    if not math.isfinite(lam) or lam < 0:
        raise ValueError("lam must be finite and nonnegative")
    if student_kv is teacher_kv:
        raise ValueError("Teacher and student caches must be distinct")
    if not sequences:
        raise ValueError("No patch references supplied")
    device = model.model.model.layers[0].mlp.down_proj.weight.device
    teacher_snapshot = snapshot_cache_tensors(teacher_kv)
    student_snapshot = snapshot_cache_tensors(student_kv)
    _sync(device)
    start = time.monotonic()
    rows, alphas, fallbacks = [], [], []
    try:
        prepared = _prepare_chunks(model, teacher_kv, student_kv, sequences)
        with tqdm(model.model.model.layers, desc="Patch construction", unit="layer") as layers:
            for block in layers:
                weight = block.mlp.down_proj.weight
                accumulator = RidgeAccumulator(*weight.shape, weight.device)
                for chunk in prepared:
                    teacher_out, _, _ = _block_forward(model, block, chunk.teacher_hidden,
                                                       chunk.teacher_meta, teacher_kv)
                    student_out, student_h, _ = _block_forward(model, block, chunk.student_hidden,
                                                               chunk.student_meta, student_kv)
                    accumulator.add_rows((teacher_out-student_out).squeeze(0).float(),
                                         student_h.squeeze(0).float())
                    chunk.teacher_hidden = teacher_out.detach()
                    del student_out, student_h, teacher_out
                patch, alpha = accumulator.solve(lam)
                weight.add_(patch.to(weight.dtype))
                for chunk in prepared:
                    student_out, _, _ = _block_forward(model, block, chunk.student_hidden,
                                                       chunk.student_meta, student_kv)
                    chunk.student_hidden = student_out.detach()
                rows.append(accumulator.n_rows)
                alphas.append(alpha)
                fallbacks.append(accumulator.solve_fallbacks)
                del patch, accumulator
    finally:
        restore_cache_tensors(teacher_kv, teacher_snapshot)
        restore_cache_tensors(student_kv, student_snapshot)
    _sync(device)
    print(f"[Patch] Construction time: {time.monotonic()-start:.2f} seconds")
    return dict(method="sequential", activation_policy="recompute",
                stats_precision="tf32", solve_precision="fp32",
                ridge_scale="weighted", seconds=time.monotonic()-start,
                rows_per_layer=rows, ridge_alpha=alphas, solve_fallbacks=fallbacks)


def prepare_forward_state(model, input_ids: torch.Tensor, kv):
    """Prepare inputs for a custom layer-by-layer forward pass."""
    from transformers.models.llama.modeling_llama import create_causal_mask
    inputs_embeds = model.model.model.embed_tokens(input_ids)
    past_seen_tokens = kv.get_seq_length()
    cache_position = torch.arange(
        past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
    )
    position_ids = cache_position.unsqueeze(0)
    causal_mask = create_causal_mask(
        config=model.model.model.config,
        input_embeds=inputs_embeds,
        attention_mask=None,
        cache_position=cache_position,
        past_key_values=kv,
    )
    position_embeddings = model.model.model.rotary_emb(inputs_embeds, position_ids)
    return {
        "hidden_states": inputs_embeds,
        "position_ids": position_ids,
        "causal_mask": causal_mask,
        "cache_position": cache_position,
        "position_embeddings": position_embeddings,
    }


def block_forward(block, hidden_states, attention_mask, position_ids,
                  past_key_values, cache_position, position_embeddings):
    """Custom forward through a single transformer block, returning intermediate states.
    Note: assumes Qwen2.5/LLaMA block structure (input_layernorm → self_attn → post_attention_layernorm → gate/up/down MLP).

    Returns:
        hidden_states: block output (MLP output + residual)
        mlp_inter_h: MLP intermediate (gate * up, before down_proj) — this is 'h'
        residual: post-attention state (before MLP) — this is 'v' for the patch objective
    """
    residual = hidden_states
    hidden_states = block.input_layernorm(hidden_states)
    hidden_states, _ = block.self_attn(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        use_cache=False,
        cache_position=cache_position,
        position_embeddings=position_embeddings,
    )
    hidden_states = hidden_states + residual

    residual = hidden_states  # post-attention state (before MLP)
    hidden_states = block.post_attention_layernorm(hidden_states)
    mlp_inter_h = block.mlp.act_fn(block.mlp.gate_proj(hidden_states)) * block.mlp.up_proj(hidden_states)
    hidden_states = block.mlp.down_proj(mlp_inter_h) + residual

    return hidden_states, mlp_inter_h, residual


class PatchKV:
    def __init__(self, model):
        self.model = model
        self.original_weights = {
            (i, 'down'): layer.mlp.down_proj.weight.detach().cpu().clone()
            for i, layer in enumerate(model.model.model.layers)
        }

    def apply(self, teacher, student, references, lam=1e-3):
        return sequential_patch(self.model, teacher, references, student_kv=student, lam=lam)

    @torch.no_grad()
    def reset(self):
        for (i, _), weight in self.original_weights.items():
            self.model.model.model.layers[i].mlp.down_proj.weight.copy_(weight)
