# Adapted from snu-mllab/KVzip (MIT); see LICENSE.
import math
import torch
import torch.nn as nn
from typing import List, Union, Optional


class KVScore:
    def _update_score(self, layer_idx: int, score: torch.Tensor):
        self.score[layer_idx] = torch.cat([self.score[layer_idx], score.to(torch.float32)], dim=-1)

    def _get_score(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: Optional[torch.Tensor],
        layer_idx: int,
    ):
        bsz, _, q_len, head_dim = query_states.shape
        num_kv = key_states.size(1)

        query_states = query_states.view(bsz, num_kv, -1, q_len, head_dim)
        key_states = torch.cat(
            [
                key_states[:, :, :self.sink],
                key_states[:, :, self.start_idx:self.end_idx],
                key_states[:, :, -q_len:],
            ],
            dim=2,
        )

        key_states = key_states.unsqueeze(2).transpose(-2, -1).contiguous()
        ctx_len = self.end_idx - self.start_idx

        attn_weights = torch.matmul(query_states, key_states) / math.sqrt(head_dim)
        self._mask_causal(attn_weights, q_len)

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        attn_weights = attn_weights[..., self.sink:self.sink + ctx_len]

        score = attn_weights.float().amax(dim=(-3, -2))
        self._update_score(layer_idx, score)

    def _make_mask(self, attn_weights: torch.Tensor, window_size: int):
        """Define causal mask shared across layers."""
        mask = torch.full((window_size, window_size),
                          torch.finfo(attn_weights.dtype).min,
                          device=attn_weights.device)
        mask_cond = torch.arange(mask.size(-1), device=attn_weights.device)
        mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
        self.causal_mask_score = mask[None, None, None, :, :]

    def _mask_causal(self, attn_weights: torch.Tensor, window_size: int):
        """Apply causal masking."""
        if self.causal_mask_score is None:
            self._make_mask(attn_weights, window_size)
        elif self.causal_mask_score.size(-1) != window_size:
            self._make_mask(attn_weights, window_size)

        attn_weights[..., -window_size:, -window_size:] += self.causal_mask_score

    def _threshold(self, score: Union[torch.Tensor, List[torch.Tensor]], ratio: float):
        if type(score) == list:
            score = torch.stack(score, dim=0)
        if ratio < 1:
            score_sort = torch.sort(score.reshape(-1), descending=True).values
            n = max(int(len(score_sort) * ratio) - 1, 0)
            thres = score_sort[n].item()
            valids = torch.where(score > thres, True, False).bool()
        else:
            valids = torch.ones_like(score, dtype=bool)
            thres = 0.

        return valids, thres

    def _threshold_uniform(self, scores: Union[torch.Tensor, List[torch.Tensor]], ratio: float):
        valids = []
        for score in scores:
            if ratio < 1:
                n_seq = score.size(-1)
                k = int(n_seq * ratio)
                if k <= 0:
                    valid = torch.zeros_like(score, dtype=bool)
                else:
                    _, topk_indices = torch.topk(score, k, dim=-1)
                    valid = torch.zeros_like(score, dtype=bool)
                    valid.scatter_(-1, topk_indices, True)
            else:
                valid = torch.ones_like(score, dtype=bool)
            valids.append(valid)

        valids = torch.stack(valids)
        return valids, 0

    def init_score(self):
        self.get_score = True
        self.causal_mask_score = None
        self.score = [torch.zeros((1, self.n_heads_kv, 0), dtype=torch.float32,
                                  device=self.device) for _ in range(self.n_layers)]
