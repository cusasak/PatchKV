def template(model_name, task, enable_thinking: bool = False):
    model_name = model_name.lower()

    if "llama" in model_name:
        # borrowed from https://www.llama.com/docs/model-cards-and-prompt-formats/llama3_1
        prefix = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        prefix += "You are a helpful assistant<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"

        postfix = "\n\n<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"

    elif model_name.startswith("qwen"):
        # https://github.com/QwenLM/Qwen3
        prefix = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        prefix += "<|im_start|>user\n"

        postfix = "<|im_end|>\n<|im_start|>assistant\n"
        # For Qwen3-*, KVzip's default skips the thinking stage by appending
        # an empty <think>…</think> block right after the assistant marker.
        # With enable_thinking=True we omit that and let the model generate
        # its own <think>…</think> freely.
        if "qwen3-" in model_name and not enable_thinking:
            postfix += "<think>\n\n</think>\n\n"

    else:
        raise ValueError(f"Unsupported chat template: {model_name}")

    if task.startswith("gsm"):
        prefix += "Given the context, answer to the following reasoning question.\n\n"
    else:
        prefix += "Given the context, answer to the following question or request without explanation.\n\n"

    return prefix, postfix
