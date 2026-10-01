# ------------------------------------------------------------------------------
# Code modified from transformers.models.llama.modeling_llama.LlamaAttention.forward
# ------------------------------------------------------------------------------
import math
import torch
import torch.nn.functional as F
from typing import Optional, Tuple
from transformers.utils import logging
from transformers.cache_utils import Cache
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
from transformers.models.qwen3.modeling_qwen3 import Qwen3Attention
from transformers.modeling_flash_attention_utils import _flash_attention_forward, FlashAttentionKwargs
from transformers.processing_utils import Unpack

from flash_attn import flash_attn_varlen_func

logger = logging.get_logger(__name__)


def llama_qwen_attn_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    attention_mask: Optional[torch.Tensor],
    past_key_values: Optional[Cache] = None,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs: Unpack[FlashAttentionKwargs],
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:

    bsz, q_len, _ = hidden_states.size()
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    if isinstance(self, Qwen3Attention):
        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    else:
        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_values is not None:
        # sin and cos are specific to RoPE models; cache_position needed for the static cache
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx,
                                                         cache_kwargs)

    dropout_rate = self.attention_dropout if self.training else 0.0

    # AM: collect training queries
    if getattr(past_key_values, "collect_queries", False):
        past_key_values.store_query(query_states, self.layer_idx)

    #### Updated #############################################################
    if getattr(past_key_values, "get_score", None):  # calculate KV importance
        past_key_values._get_score(query_states, key_states, value_states, self.layer_idx)

    if hasattr(past_key_values, "should_use_am"):
        use_am_layer = past_key_values.should_use_am(self.layer_idx)
    else:
        use_am_layer = getattr(past_key_values, "has_beta", False)

    if use_am_layer:  # AM attention with beta
        q_am, k_am, v_am, beta_am = past_key_values.prepare_am(
            query_states, key_states, value_states, self.layer_idx)
        n_groups = self.num_key_value_groups
        k_exp = k_am.repeat_interleave(n_groups, dim=1)
        v_exp = v_am.repeat_interleave(n_groups, dim=1)
        beta_exp = beta_am.repeat_interleave(n_groups, dim=1)
        # SDPA path — beta goes into the additive attn_mask alongside the
        # causal block for the query portion. Matches compaction's mask
        # construction and lets PyTorch dispatch to its mem-efficient /
        # flash backends where supported.
        mask_dtype = q_am.dtype
        mask = beta_exp.to(mask_dtype).unsqueeze(2)  # (B, H, 1, K) broadcast over Q
        if self.is_causal and q_len > 1:
            neg_inf = torch.finfo(mask_dtype).min
            causal_block = torch.triu(
                torch.full((q_len, q_len), neg_inf, dtype=mask_dtype, device=mask.device),
                diagonal=1,
            )
            mask = mask.expand(-1, -1, q_len, -1).contiguous()
            mask[..., -q_len:] = mask[..., -q_len:] + causal_block
        attn_output = F.scaled_dot_product_attention(
            q_am, k_exp, v_exp,
            attn_mask=mask,
            dropout_p=dropout_rate,
            scale=1.0 / math.sqrt(self.head_dim),
            is_causal=False,
        )
        attn_output = attn_output.transpose(1, 2)  # (bsz, q_len, n_heads, dim)

    elif getattr(past_key_values, "pruned", None):  # attention with pruned cache
        query_states, key_states, value_states, info = past_key_values.prepare(
            query_states, key_states, value_states, self.layer_idx)

        # bsz x head x seq, group, dim
        attn_output = flash_attn_varlen_func(
            query_states,
            key_states,
            value_states,
            cu_seqlens_q=info["cu_len_q"],
            cu_seqlens_k=info["cu_len_k"],
            max_seqlen_q=info["max_len_q"],
            max_seqlen_k=info["max_len_k"],
            dropout_p=dropout_rate,
            causal=True,
        )
        attn_output = attn_output.view(bsz, self.config.num_key_value_heads, q_len,
                                       self.num_key_value_groups, self.head_dim).transpose(1, 2)

    else:
        query_states = query_states.transpose(1, 2)  # bsz, seq, head, dim
        key_states = key_states.transpose(1, 2)  # bsz, seq, head_kv, dim
        value_states = value_states.transpose(1, 2)

        attn_output = _flash_attention_forward(
            query_states,
            key_states,
            value_states,
            None,  # attention_mask
            q_len,
            dropout=dropout_rate,
            sliding_window=getattr(self, "sliding_window", None),
            is_causal=self.is_causal,
        )  # bsz, seq, head, dim
    ###################################################################

    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)

    attn_weights = None
    return attn_output, attn_weights
