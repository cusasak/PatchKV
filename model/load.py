from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM


def get_model_id(name):
    if name == 'llama3.1-8b':
        return 'meta-llama/Llama-3.1-8B-Instruct'
    if name.startswith('qwen2.5-'):
        return f'Qwen/Qwen2.5-{name.split("-")[-1].upper()}-Instruct-1M'
    if name.startswith('qwen3-'):
        return f'Qwen/Qwen3-{name.split("-")[-1].upper()}'
    return name


def load_model(model_name):
    from model.monkeypatch import replace_attn
    model_id = get_model_id(model_name)
    config = AutoConfig.from_pretrained(model_id)
    if config.model_type not in {'llama', 'qwen2', 'qwen3'}:
        raise ValueError(f'Unsupported model type: {config.model_type}')
    replace_attn(model_id)
    if config.model_type == 'qwen3':
        config.rope_scaling = dict(rope_type='yarn', factor=4., original_max_position_embeddings=32768)
        config.max_position_embeddings = 131072
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype='auto', device_map='auto',
        attn_implementation='flash_attention_2', config=config)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if config.model_type == 'llama':
        model.generation_config.pad_token_id = tokenizer.pad_token_id = 128004
    model.eval()
    model.name = model_id.split('/')[-1]
    print(f'Load {model_id} with {model.dtype}')
    return model, tokenizer
