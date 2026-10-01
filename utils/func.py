import random
from typing import Optional

import torch


def seed_for_sample(base_seed: Optional[int], data_idx: int) -> Optional[int]:
    if base_seed is None:
        return None
    sample_seed = (int(base_seed) + int(data_idx)) % (2**63 - 1)
    random.seed(sample_seed)
    torch.manual_seed(sample_seed)
    return sample_seed


def set_gen_length(dataname, model=None):
    if dataname == "needle" or "_mf" in dataname:
        max_len = 32
    elif dataname == "squad" or "summary" in dataname:
        max_len = 256
    elif dataname == "gsm" or "repoqa" in dataname:
        max_len = 512
    else:
        max_len = 96
    if model is not None:
        model.gen_kwargs["max_new_tokens"] = max_len
    return max_len
