# ------------------------------------------------------------------------------
# Original Code developed by Jang-Hyun Kim
# GitHub Repository: https://github.com/snu-mllab/KVzip
# ------------------------------------------------------------------------------
# Adapted from snu-mllab/KVzip; see LICENSE.
import torch
from typing import List, Union, Optional, Dict, Any
from tqdm import tqdm
from transformers import Qwen3ForCausalLM
from attention.kvcache import RetainCache
from attention.query_cache import fork_generation_cache
from model.load import load_model
from model.template import template

def chunk_fn(ctx_ids: torch.Tensor, chunk_size: int) -> List[torch.Tensor]:
    """ Chunk tokens
    """
    ctx_len = ctx_ids.shape[1]
    if ctx_len > chunk_size:
        print(f"Total context length: {ctx_len}")
        chunk_num = (ctx_len - 1) // chunk_size + 1
        print(f"chunk inputs, size: {chunk_size} (num {chunk_num})")

        input_ids = []
        for i in range(chunk_num):
            start = i * chunk_size
            end = (i + 1) * chunk_size
            a_ids = ctx_ids[:, start:end]
            if a_ids.shape[1] == 0:
                continue
            input_ids.append(a_ids)
    else:
        input_ids = [ctx_ids]

    return input_ids

class ModelKVzip:

    def __init__(self, model_name: str):
        self.model, self.tokenizer = load_model(model_name)
        self.name, self.dtype = self.model.name, self.model.dtype
        self.device, self.config = self.model.device, self.model.config
        self.gen_kwargs = dict(do_sample=False, temperature=1.0, top_p=1,
                               top_k=None, max_new_tokens=512)
        if isinstance(self.model, Qwen3ForCausalLM):
            self.gen_kwargs.update(cache_implementation=None, use_model_defaults=False,
                                   eos_token_id=151645)
        self.set_chat_template()

    def encode(self, text: str) -> torch.Tensor:
        """ Encode text into tokens
        """
        return self.tokenizer.encode(text, add_special_tokens=False, return_tensors="pt").to(self.device)

    def decode(self, input_ids: torch.Tensor) -> str:
        """ Decode tokens into text
        """
        if len(input_ids.shape) == 2:
            input_ids = input_ids[0]
        return self.tokenizer.decode(input_ids)

    def set_chat_template(self, task: str = "qa", enable_thinking: bool = False):
        prefix, postfix = template(self.name, task, enable_thinking=enable_thinking)
        self.sys_prompt_ids, self.postfix_ids = self.encode(prefix), self.encode(postfix)
        # Self-study always disables thinking, independently of evaluation.
        _, study_postfix = template(self.name, task, enable_thinking=False)
        self.self_study_postfix_ids = self.encode(study_postfix)

    def apply_template(self, query: str) -> torch.Tensor:
        query = f"\n\n{query.strip()}"
        query_ids = torch.cat([self.encode(query), self.postfix_ids], dim=1)
        return query_ids

    def __call__(
        self,
        input_ids: torch.Tensor,
        kv: RetainCache,
        update_cache: bool = False,
        return_logits: bool = False,
        *args,
        **kwargs,
    ):
        """ Compute Transformer forward pass
            In default, we do not update the KV cache with the newly given inputs.
            Set update_cache = True to enable the update.
        """
        seen_token_prev = kv._seen_tokens

        if return_logits:
            outputs = self.model(input_ids, past_key_values=kv, *args, **kwargs)
        else:
            _ = self.model.model(input_ids, past_key_values=kv, *args, **kwargs)
            outputs = None

        if not update_cache:
            kv.slice(seen_token_prev)
        return outputs

    def _init_kv(self, kv=None, evict_range=(0, 0)):
        return RetainCache(self.model, evict_range) if kv is None else kv

    @torch.inference_mode()
    def prefill(self, ctx_ids: Union[str, torch.Tensor], prefill_chunk_size: int = 16000,
                do_score=True, score_config=None):
        if isinstance(ctx_ids, str):
            ctx_ids = self.encode(ctx_ids)
        prefill_ids = torch.cat([self.sys_prompt_ids, ctx_ids], dim=1)
        kv = self._init_kv(evict_range=(self.sys_prompt_ids.shape[1], prefill_ids.shape[1]))
        kv.ctx_ids, kv.prefill_ids = ctx_ids, prefill_ids
        for ids in tqdm(chunk_fn(prefill_ids, prefill_chunk_size), desc="Prefill"):
            self(ids, kv, update_cache=True)
        if do_score:
            self.scoring(kv, ctx_ids, score_config=score_config)
        return kv

    @torch.inference_mode()
    def _score_with_repeat_queries(
        self,
        kv: RetainCache,
        ctx_ids: torch.Tensor,
        score_config: Dict[str, Any],
    ):
        kv.init_score()
        start_idx_tmp = kv.start_idx

        kv.end_idx = 0
        input_ids, _ = self.self_task(ctx_ids, chunk_size=score_config["repeat_chunk_size"])
        for prefill_ids_p, repeat_ids_p in tqdm(input_ids, desc="Importance scoring"):
            kv.end_idx = kv.start_idx + prefill_ids_p.shape[1]
            self.__call__(repeat_ids_p, kv, update_cache=False)
            kv.start_idx = kv.end_idx

        kv.start_idx = start_idx_tmp
        assert kv.score[0].shape[-1] == kv.ctx_len

    def self_task(
        self,
        ctx_ids: torch.Tensor,
        chunk_size: int = 2000,
        prev_postfix_size=8,
    ) -> List[torch.Tensor]:
        """ Prepare chunked inputs for KV importance scoring with context reconstruction
            return: List[torch.Tensor]
        """
        chunked_inputs = chunk_fn(ctx_ids, chunk_size)

        input_ids = []
        for i, a_ids in enumerate(chunked_inputs):
            if i == 0:
                prompt = f"\n\nRepeat the previous context exactly."
                q_ids = self.encode(prompt)
            else:
                prompt = f"\n\nRepeat the part of the previous context exactly, starting with "
                q_ids = self.encode(prompt)
                postfix_prev = chunked_inputs[i - 1][:, -prev_postfix_size:]
                q_ids = torch.cat([q_ids, postfix_prev], dim=1)

            input_ids.append((a_ids, torch.cat([q_ids, self.postfix_ids, a_ids], dim=1)))

        return input_ids, q_ids.shape[1]

    @torch.inference_mode()
    def generate(
        self,
        query: Union[str, torch.Tensor],
        kv: Optional[RetainCache] = None,
        update_cache: bool = False,
        gen_kwargs_override: Optional[Dict[str, Any]] = None,
    ) -> str:
        """ Obtain a model response to the query
            In default, we evict KV of query and generated answer after the generation by kv.slice (for multi-query evaluation).
            Set update_cache = True to enable multi-turn generation.
        """
        kv = self._init_kv(kv=kv)
        seen_token_prev = kv._seen_tokens

        input_ids = query
        if type(query) == str:
            input_ids = self.encode(query)
        if kv.prefill_ids is not None:
            # Huggingface Transformers model.generate requires full input tokens when using KV caches.
            # The inputs will be spliced to only contain new tokens as input[:, -kv.get_seq_length():].
            input_ids = torch.cat([kv.prefill_ids, input_ids], dim=1)

        gen_kwargs = dict(self.gen_kwargs)
        if gen_kwargs_override is not None:
            gen_kwargs.update(gen_kwargs_override)
        generation_kv = kv
        if not update_cache:
            generation_kv = fork_generation_cache(
                kv, getattr(self.model, "generation_config", None), **gen_kwargs
            )
        try:
            output = self.model.generate(input_ids, past_key_values=generation_kv, **gen_kwargs)
        finally:
            if not update_cache and generation_kv is kv:
                kv.slice(seen_token_prev)
        a_ids = output[:, len(input_ids[0]):-1]  # parse response
        a = self.decode(a_ids)

        if update_cache:
            kv.prefill_ids = torch.cat([input_ids, a_ids], dim=1)
        return a

    def scoring(self, kv, ctx_ids, score_config=None):
        config = dict(repeat_chunk_size=2000)
        config.update(score_config or {})
        self._score_with_repeat_queries(kv, ctx_ids, config)
        kv.get_score = False
