import transformers
from attention.attn import llama_qwen_attn_forward


def replace_attn(model_id):
    name = model_id.lower()
    if 'llama' in name:
        cls = transformers.models.llama.modeling_llama.LlamaAttention
    elif 'qwen2.5' in name:
        cls = transformers.models.qwen2.modeling_qwen2.Qwen2Attention
    elif 'qwen3' in name:
        cls = transformers.models.qwen3.modeling_qwen3.Qwen3Attention
    else:
        raise ValueError(f'Unsupported model family: {model_id}')
    cls.forward = llama_qwen_attn_forward
