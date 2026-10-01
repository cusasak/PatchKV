# Attention Matching, adapted from compaction/algorithms/base.py and highest_attention_keys.py.
import math
from typing import List, Optional, Tuple
import torch

def _ridge_cholesky_solve(M: torch.Tensor, y: torch.Tensor, lam) -> torch.Tensor:
    def solve(matrix, target):
        if matrix.shape[0] < matrix.shape[1]:
            gram = matrix @ matrix.T
            rhs = target
        else:
            gram = matrix.T @ matrix
            rhs = matrix.T @ target
        gram = 0.5 * (gram + gram.T)
        gram.diagonal().add_(lam)
        factor = torch.linalg.cholesky(gram)
        vector = rhs.ndim == 1
        value = torch.cholesky_solve(rhs.unsqueeze(1) if vector else rhs, factor)
        if vector:
            value = value.squeeze(1)
        if matrix.shape[0] < matrix.shape[1]:
            value = matrix.T @ value
        if not torch.isfinite(value).all():
            raise FloatingPointError('Non-finite AM ridge solution')
        return value

    try:
        return solve(M, y)
    except (torch.linalg.LinAlgError, FloatingPointError):
        if M.dtype == torch.float64 or not torch.isfinite(M).all() or not torch.isfinite(y).all():
            raise
        print(f'AM ridge Cholesky rescue: rebuilding in float64 with unchanged lambda={float(lam):g}', flush=True)
        value = solve(M.double(), y.double()).to(M.dtype)
        if not torch.isfinite(value).all():
            raise FloatingPointError('Non-finite AM ridge solution after dtype conversion')
        return value


def _nnls_solve(M, y):
    """Fit positive attention masses with two projected-gradient steps."""
    lower, upper = math.exp(-3), math.exp(3)
    try:
        B = torch.linalg.lstsq(M, y.unsqueeze(1), driver='gels').solution.squeeze(1)
        if not torch.isfinite(B).all():
            raise FloatingPointError('Non-finite NNLS solution')
    except (RuntimeError, FloatingPointError):
        B = _ridge_cholesky_solve(M, y, 1e-6)
    B.clamp_(lower, upper)

    u = torch.randn(M.shape[1], device=M.device, dtype=M.dtype)
    u = u / (u.norm() + 1e-12)
    for _ in range(3):
        v = M @ u
        if v.norm() == 0:
            break
        v = v / v.norm()
        u = M.T @ v
        if u.norm() == 0:
            break
        u = u / u.norm()
    sigma = (u @ (M.T @ (M @ u))).sqrt().clamp_min(1e-6)
    eta = 1.0 / (sigma ** 2).clamp_min(1e-6)
    for _ in range(2):
        B = B - eta * (M.T @ (M @ B - y))
        B.clamp_(lower, upper)
    if not torch.isfinite(B).all():
        raise FloatingPointError('Non-finite AM projected-gradient solution')
    return B


def solve_c2(queries, C1, beta, Y):
    """Fit values to the teacher attention output, retaining numerical rescue."""
    logits = (queries @ C1.T).float() * (1.0 / C1.shape[1]) ** .5 + beta.float()
    exp_scores = torch.exp(logits - logits.max(dim=1, keepdim=True)[0])
    X = exp_scores / exp_scores.sum(dim=1, keepdim=True)
    try:
        values = torch.linalg.lstsq(X, Y, driver='gels').solution
        if not torch.isfinite(values).all():
            raise FloatingPointError('Non-finite C2 solution')
    except (RuntimeError, FloatingPointError):
        values = _ridge_cholesky_solve(X, Y, 1e-6)
    if not torch.isfinite(values).all():
        raise FloatingPointError('Non-finite C2 solution')
    return values.to(C1.dtype)


def am_compress_head(queries, K, V, t):
    """Select keys by RMS attention, fit beta, then fit replacement values."""
    t = min(t, K.shape[0])
    if t == 0:
        empty = K.new_empty((0, K.shape[1]))
        return empty, K.new_empty((0,), dtype=torch.float32), empty, K.new_empty((0,), dtype=torch.long)
    logits = (queries @ K.T).float() * (1.0 / K.shape[1]) ** .5
    exp_scores = torch.exp(logits - logits.max(dim=1, keepdim=True)[0])
    target = exp_scores.sum(dim=1)
    attention = exp_scores / exp_scores.sum(dim=1, keepdim=True)
    output = attention @ V.float()
    scores = torch.sqrt((attention ** 2).mean(dim=0))
    indices = torch.topk(scores, t, largest=True).indices
    beta = torch.log(_nnls_solve(exp_scores[:, indices], target))
    C1 = K[indices]
    return C1, beta, solve_c2(queries, C1, beta, output), indices


from tqdm import tqdm
from attention.am_chunking import (
    fixed_chunk_ranges, clone_reference_chunk_cache, fork_reference_chunk_student,
    fork_compact_am_destination, merge_reference_chunk_caches,
)


def am_score_config(args):
    return dict(am_query_source=args.am_query_source,
                repeat_chunk_size=args.repeat_chunk_size,
                am_prefill_chunk_size=args.am_prefill_chunk_size,
                am_max_queries_per_kv_head=args.am_max_queries_per_kv_head,
                include_repeat_prefill=args.include_repeat_prefill,
                am_quiet=True)


class AMPreparation:
    """Reference tokens/queries shared only across ratios of this context.

    On-policy extraction reuses the same generated tokens and the same sampled
    query indices after upstream cache layers have been compacted. It runs with
    original model weights; patching starts only after construction finishes.
    """
    def __init__(self, model, teacher, args, head_budget=None):
        from patching.references import am_references
        self.model, self.teacher, self.args = model, teacher, args
        self.config = am_score_config(args)
        self.ranges = (fixed_chunk_ranges(teacher.ctx_len, args.am_chunk_size)
                       if args.am_chunk_size else None)
        self.specs = []
        ranges = self.ranges or [(0, teacher.ctx_len)]
        for start, end in ranges:
            base = (clone_reference_chunk_cache(teacher, teacher.ctx_ids, 0,
                    teacher.ctx_len, start, end) if self.ranges else teacher)
            base.am_head_budget = head_budget
            sequences = None
            if args.am_query_source == 'self_study':
                sequences = am_references(model, base, include_prefill=args.include_repeat_prefill)
            collect_queries(model, base, base.ctx_ids, self.config, self_study_sequences=sequences)
            queries = list(base.am_queries)
            if args.am_on_policy and args.am_query_source == 'self_study':
                queries = [q if i == 0 else None for i, q in enumerate(queries)]
            self.specs.append(dict(base=base, sequences=sequences, queries=queries,
                                   subsample=base.am_query_subsample))
            base.am_queries = None
            base.am_query_subsample = None

    def _compact(self, student, spec, ratio):
        args = self.args
        on_policy = args.am_on_policy and args.am_query_source == 'self_study'
        if on_policy:
            for layer_idx in range(student.n_layers):
                if layer_idx == 0:
                    queries = spec['queries']
                else:
                    collect_queries(self.model,
                        student, spec['base'].ctx_ids, self.config,
                        max_layer=layer_idx, self_study_sequences=spec['sequences'],
                        query_subsample=spec['subsample'])
                    queries = [q if i == layer_idx else None
                               for i, q in enumerate(student.am_queries)]
                _, actual = student.prune_am(ratio, queries, layer_indices=[layer_idx], verbose=False)
                student.am_queries = None
        else:
            _, actual = student.prune_am(ratio, spec['queries'], verbose=False)
        return actual

    def compress(self, ratio):
        destination = fork_compact_am_destination(self.teacher)
        if self.ranges is None:
            actual = self._compact(destination, self.specs[0], ratio)
        else:
            students = []
            for spec in self.specs:
                student = fork_reference_chunk_student(spec['base'])
                self._compact(student, spec, ratio)
                students.append(student)
            actual = merge_reference_chunk_caches(
                destination, students, self.ranges, base_sink=self.teacher.sink,
                article_start=0, article_end=self.teacher.ctx_len)
        return destination, actual


def _forward_chunk_extending(model, input_ids: torch.Tensor, kv, max_layer: Optional[int]):
    """Append a chunk, stopping at max_layer; the caller restores the cache."""
    n_layers = len(model.model.model.layers)
    if max_layer is None or max_layer >= n_layers - 1:
        model.model.model(input_ids, past_key_values=kv)
        return

    class _EarlyExit(Exception):
        pass

    def _abort_hook(module, args, kwargs):
        raise _EarlyExit()

    target_layer = model.model.model.layers[max_layer + 1]
    handle = target_layer.register_forward_pre_hook(_abort_hook, with_kwargs=True)
    try:
        try:
            model.model.model(input_ids, past_key_values=kv)
        except _EarlyExit:
            pass
    finally:
        handle.remove()


@torch.inference_mode()
def collect_queries(
    model,
    kv,
    ctx_ids,
    score_config,
    max_layer: Optional[int] = None,
    self_study_sequences: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
    query_subsample=None,
):
    """AM: collect diverse training queries for later prune_am calls.

    When ``max_layer`` is provided, the forward stops after layer ``max_layer``
    — queries for deeper layers are not collected (and not needed when
    on-policy prune is only updating that single layer).

    ``self_study_sequences`` supplies text generated once from the full
    base cache. Frozen-AM construction passes it explicitly so a cloned
    student cache never regenerates different self-study questions.
    """
    am_query_source = score_config.get("am_query_source", "repeat")
    am_quiet = bool(score_config.get("am_quiet", False))
    if am_query_source == "repeat":
        query_sequences = [
            repeat_ids_p for _, repeat_ids_p in model.self_task(
                ctx_ids, chunk_size=score_config["repeat_chunk_size"],
            )[0]
        ]
    elif am_query_source == "self_study":
        if self_study_sequences is None:
            raise ValueError("AM self-study requires references from the full-cache teacher")
        query_sequences = [ids for _, ids in self_study_sequences]
    else:
        raise ValueError(f"Unknown AM query source: {am_query_source}")

    kv.init_query_collection()
    kv.collect_queries = True

    prefill_chunk_size = int(score_config.get("am_prefill_chunk_size", 4096) or 0)

    for input_ids in tqdm(
        query_sequences,
        desc=f"AM query collection ({am_query_source})",
        disable=am_quiet,
    ):
        seen_token_prev = kv._seen_tokens
        layer_tensors = [
            (layer.keys, layer.values) for layer in kv.layers
        ]
        try:
            if prefill_chunk_size > 0:
                chunks = input_ids.split(prefill_chunk_size, dim=1)
            else:
                chunks = [input_ids]
            for chunk in chunks:
                _forward_chunk_extending(model, chunk, kv, max_layer)
        finally:
            # Restore the exact context tensor objects.  ``slice`` would
            # leave views into the last query-append concatenation alive,
            # which is especially expensive for layerwise AM query
            # extraction over many reference chunks.
            for layer, (keys, values) in zip(kv.layers, layer_tensors):
                layer.keys = keys
                layer.values = values
            kv._seen_tokens = seen_token_prev

    kv.collect_queries = False
    kv.am_queries = kv.get_collected_queries()
    kv.clear_query_buffer()

    # Cap queries-per-KV-head. Without this, long contexts blow up because
    # every head's lstsq in am_compress_head scales with n_queries × t.
    # Use the SAME index set across every layer and every KV head. During
    # on-policy extraction, reuse the original pass's index set instead of
    # drawing a fresh sample at each layer.
    max_q = int(score_config.get("am_max_queries_per_kv_head", 0) or 0)
    reference_q = next((q for q in kv.am_queries if q is not None), None)
    if reference_q is not None:
        n_total = int(reference_q.shape[1])
        if query_subsample is not None:
            expected_total = int(query_subsample["source_length"])
            if n_total != expected_total:
                raise RuntimeError(
                    "On-policy AM query count changed before subsampling: "
                    f"original={expected_total}, re-extracted={n_total}. "
                    "The same self-study suffix sequences must be reused."
                )
            stored_indices = query_subsample["indices"]
            idx = (
                None
                if stored_indices is None
                else stored_indices.to(reference_q.device)
            )
        elif max_q > 0 and n_total > max_q:
            idx = torch.randperm(
                n_total, device=reference_q.device
            )[:max_q].sort().values
            query_subsample = {
                "source_length": n_total,
                "indices": idx.detach().clone(),
            }
        else:
            idx = None
            query_subsample = {
                "source_length": n_total,
                "indices": None,
            }

        if idx is not None:
            for layer_idx, queries in enumerate(kv.am_queries):
                if queries is None:
                    continue
                if int(queries.shape[1]) != n_total:
                    raise RuntimeError(
                        "AM query count differs across layers before "
                        f"subsampling: layer={layer_idx}, expected={n_total}, "
                        f"actual={queries.shape[1]}"
                    )
                kv.am_queries[layer_idx] = queries[:, idx, :]
        kv.am_query_subsample = query_subsample

    if not am_quiet:
        per_head = kv.am_queries[0].shape[1] if kv.am_queries[0] is not None else 0
        print(f"# AM: collected queries for {kv.n_layers} layers, "
              f"{per_head} queries/head")
