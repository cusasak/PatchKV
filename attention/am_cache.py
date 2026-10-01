import torch
from attention.am import am_compress_head
from attention.am_budget import compute_head_budget


class AMCacheMixin:
    def _init_am(self):
        # AM attributes
        self._query_buffer = None
        self.collect_queries = False
        self.am_beta = None
        self.am_layer_compacted = None
        self.has_beta = False
        self.am_queries = None
        self.am_head_budget = None
        # Original-query subsample reused by every on-policy layer. The
        # reference implementation records both its attention-space and
        # KV-space samples and reapplies them during re-extraction; our query
        # collector is already in KV space, so one stored sample is sufficient.
        self.am_query_subsample = None
        # Optional copy-on-write / physically compact AM storage.  Ordinary
        # RetainCache instances leave these disabled.  Reference-chunk
        # construction can shallow-fork the immutable context tensors, clone a
        # layer's V only when C2 is written, and replace completed layers with
        # [sink | selected C1/C2 | suffix] storage.
        self._am_shared_value_layers = [False for _ in range(self.n_layers)]
        self._am_compact_base_lengths = [None for _ in range(self.n_layers)]
        self._am_compact_on_merge = False

    def init_query_collection(self):
        self._query_buffer = [[] for _ in range(self.n_layers)]


    def store_query(self, query_states: torch.Tensor, layer_idx: int):
        if self._query_buffer is None:
            return
        q = query_states[0]  # (n_heads, q_len, dim)
        if q.shape[0] != self.n_heads_kv:
            q = q.view(self.n_heads_kv, self.n_group_kv, -1, q.shape[-1])
            q = q.reshape(self.n_heads_kv, -1, q.shape[-1])
        self._query_buffer[layer_idx].append(q.detach())


    def get_collected_queries(self):
        result = []
        for layer_idx in range(self.n_layers):
            if self._query_buffer[layer_idx]:
                result.append(torch.cat(self._query_buffer[layer_idx], dim=1))
            else:
                result.append(None)
        return result


    def clear_query_buffer(self):
        self._query_buffer = None
        self.collect_queries = False


    def _init_am_state(self):
        self.valid = [
            torch.ones((1, self.n_heads_kv, self.ctx_len), dtype=torch.bool, device=self.device)
            for _ in range(self.n_layers)
        ]
        self.am_beta = [
            torch.zeros((1, self.n_heads_kv, self.ctx_len), dtype=torch.float32, device=self.device)
            for _ in range(self.n_layers)
        ]
        self.am_ctx_indices = [None for _ in range(self.n_layers)]
        self.am_layer_compacted = [False for _ in range(self.n_layers)]


    def should_use_am(self, layer_idx: int) -> bool:
        return bool(
            self.has_beta
            and self.am_layer_compacted is not None
            and 0 <= layer_idx < len(self.am_layer_compacted)
            and self.am_layer_compacted[layer_idx]
        )


    def prune_am(self, ratio, queries_per_layer, layer_indices=None, verbose=True):
        if not 0 <= ratio <= 1:
            raise ValueError('ratio must be in [0, 1]')
        t_total = max(1, int(self.ctx_len * ratio))
        if self.am_beta is None or self.am_layer_compacted is None or layer_indices is None:
            self._init_am_state()
        target_layers = range(self.n_layers) if layer_indices is None else layer_indices
        for layer_idx in target_layers:
            shared = getattr(self, '_am_shared_value_layers', None)
            if shared is not None and shared[layer_idx]:
                self.layers[layer_idx].values = self.layers[layer_idx].values.clone()
                shared[layer_idx] = False
            elif self.layers[layer_idx].values.is_inference():
                self.layers[layer_idx].values = self.layers[layer_idx].values.clone()
            K = self.layers[layer_idx].keys[0, :, self.sink:self.sink + self.ctx_len]
            V = self.layers[layer_idx].values[0, :, self.sink:self.sink + self.ctx_len]
            beta_heads, valid_heads, indices_heads = [], [], []
            for h in range(self.n_heads_kv):
                budget = t_total if self.am_head_budget is None else compute_head_budget(
                    self.am_head_budget, layer_idx, h, target_ratio=ratio,
                    context_len=self.ctx_len, num_layers=self.n_layers, num_kv_heads=self.n_heads_kv)
                _, beta, values, indices = am_compress_head(
                    queries_per_layer[layer_idx][h], K[h], V[h], min(budget, self.ctx_len))
                indices, order = torch.sort(indices)
                beta = beta[order].to(self.dtype)
                valid = torch.zeros(self.ctx_len, dtype=torch.bool, device=self.device)
                valid[indices] = True
                V[h, indices] = values[order]
                beta_heads.append(beta)
                valid_heads.append(valid)
                indices_heads.append(indices)

            max_t = max((i.numel() for i in indices_heads), default=0) or t_total
            for h in range(self.n_heads_kv):
                pad = max_t - indices_heads[h].numel()
                if pad:
                    indices_heads[h] = torch.cat([indices_heads[h], torch.zeros(pad, dtype=torch.long, device=self.device)])
                    beta_heads[h] = torch.cat([beta_heads[h], torch.full((pad,), float('-inf'), dtype=self.dtype, device=self.device)])
            self.valid[layer_idx] = torch.stack(valid_heads).unsqueeze(0)
            self.am_beta[layer_idx] = torch.stack(beta_heads).unsqueeze(0)
            self.am_ctx_indices[layer_idx] = torch.stack(indices_heads)
            self.am_layer_compacted[layer_idx] = True
            self.has_beta = True

        removed = sum((~v).float().sum().item() for v in self.valid)
        actual = 1 - removed / (self.ctx_len * self.n_heads_kv * self.n_layers)
        self.pruned = False
        self.has_beta = any(self.am_layer_compacted)
        if verbose:
            print(f'AM ratio {actual:.4f}')
        return 0., actual

    def prepare_am(self, query_states, key_states, value_states, layer_idx):
        bsz, n_heads_kv, seq_len, dim = key_states.shape
        sink = self.sink
        ctx_len = self.ctx_len
        compact_lengths = getattr(self, "_am_compact_base_lengths", None)
        compact_base_len = (
            compact_lengths[layer_idx]
            if compact_lengths is not None
            else None
        )
        if compact_base_len is not None:
            compact_base_len = int(compact_base_len)
            t_ctx = int(self.am_ctx_indices[layer_idx].shape[1])
            suffix_len = compact_base_len - int(sink) - t_ctx
            q_len = seq_len - compact_base_len
            if suffix_len < 0 or q_len < 0:
                raise RuntimeError(
                    "invalid compact AM physical layout: "
                    f"base={compact_base_len}, sink={sink}, selected={t_ctx}, "
                    f"seq={seq_len}"
                )
            am_beta = self.am_beta[layer_idx]
            beta_dtype = am_beta.dtype
            zeros_sink = torch.zeros(
                1, n_heads_kv, sink,
                dtype=beta_dtype, device=self.device,
            )
            zeros_tail = torch.zeros(
                1, n_heads_kv, suffix_len + q_len,
                dtype=beta_dtype, device=self.device,
            )
            beta_full = torch.cat(
                [zeros_sink, am_beta, zeros_tail], dim=2
            )
            return query_states, key_states, value_states, beta_full

        q_len = seq_len - sink - ctx_len
        ctx_indices = self.am_ctx_indices[layer_idx]        # (H, t_ctx) on device
        t_ctx = ctx_indices.size(1)
        t_total = sink + t_ctx + q_len                      # Python int, no sync

        sink_idx = torch.arange(sink, device=self.device, dtype=torch.long)
        q_idx = torch.arange(sink + ctx_len, seq_len, device=self.device, dtype=torch.long)

        head_ctx = ctx_indices + sink                       # (H, t_ctx)
        sink_head = sink_idx.unsqueeze(0).expand(n_heads_kv, -1)
        q_head = q_idx.unsqueeze(0).expand(n_heads_kv, -1)
        full_idx = torch.cat([sink_head, head_ctx, q_head], dim=1)   # (H, t_total)

        gather_idx = full_idx.view(1, n_heads_kv, t_total, 1).expand(bsz, -1, -1, dim)
        key_out = torch.gather(key_states, dim=2, index=gather_idx)   # (bsz, H, t_total, dim)
        value_out = torch.gather(value_states, dim=2, index=gather_idx)

        # beta_full = [zeros(sink) | am_beta | zeros(q_len)]
        am_beta = self.am_beta[layer_idx]                   # (1, H, t_ctx)
        beta_dtype = am_beta.dtype
        zeros_sink = torch.zeros(1, n_heads_kv, sink, dtype=beta_dtype, device=self.device)
        zeros_q = torch.zeros(1, n_heads_kv, q_len, dtype=beta_dtype, device=self.device)
        beta_full = torch.cat([zeros_sink, am_beta, zeros_q], dim=2)

        return query_states, key_out, value_out, beta_full
