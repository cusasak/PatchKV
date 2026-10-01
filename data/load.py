from pathlib import Path
from datasets import Dataset, load_dataset

SCBENCH_DATASETS = (
    "scbench_kv", "scbench_prefix_suffix", "scbench_repoqa", "scbench_summary",
    "scbench_qa_eng", "scbench_choice_eng", "scbench_mf",
)
DATASETS = ("squad", "gsm", "needle", *SCBENCH_DATASETS,
            *(name + suffix for name in ("scbench_kv", "scbench_prefix_suffix",
                "scbench_repoqa", "scbench_summary", "scbench_mf")
              for suffix in ("_tiny", "_short")))


def load_dataset_all(name, tokenizer, n_data=100):
    if name == "squad":
        return load_squad(n_data)
    if name == "gsm":
        return load_gsm(tokenizer, n_data)
    if name == "needle":
        return load_niah(tokenizer)
    if name in DATASETS:
        return load_scbench(name)
    raise ValueError(f"Unsupported dataset: {name}")

def load_squad(n_data):
    data = load_dataset('rajpurkar/squad', split='train')

    pool = dict()
    dataset = {"context": [], "question": [], "answers": []}
    for d in data:
        # aggregate qa pairs for the shared context
        ans = _first_text(d["answers"]["text"])
        if d["context"] not in pool:
            # Stop at the next context so every question of the last selected
            # context is retained, without returning a partial extra context.
            if len(pool) >= n_data:
                break
            pool[d["context"]] = len(dataset["context"])
            dataset["context"].append(d["context"])
            dataset["question"].append([d["question"]])
            dataset["answers"].append([ans])
        else:
            idx = pool[d["context"]]
            assert dataset["context"][idx] == d["context"]
            dataset["question"][idx].append(d["question"])
            dataset["answers"][idx].append(ans)

    dataset = Dataset.from_dict(dataset)
    return dataset

def _first_text(x):
    if isinstance(x, (list, tuple)):
        return str(x[0]) if len(x) > 0 else ""
    return str(x)

def load_niah(tokenizer, max_len=8000):
    dataset = []
    from data.needle import NeedleHaystackData

    for context_len in [500, 2000, max_len]:
        needle = NeedleHaystackData(tokenizer,
                                    haystack_dir=str(Path(__file__).parent / "needle" / "PaulGrahamEssays"),
                                    context_lengths=[context_len],
                                    final_context_length_buffer=0)

        for depth in [i * 10 for i in range(11)]:
            data = needle.generate_context_qa(context_len, depth)
            dataset.append(data)

    return dataset

def load_gsm(tokenizer, n_data):
    dataset_full = load_dataset('openai/gsm8k', 'main', split="test")

    dataset = []
    for data in dataset_full:
        st = data['question'].split(". ")

        data["context"] = ". ".join(st[:-1]).strip() + "."
        l = len(tokenizer.encode(data["context"], add_special_tokens=False))
        if l < 72:  # pass short context
            continue

        data["question"] = [st[-1].strip()]
        data["answers"] = [data["answer"]]
        dataset.append(data)

        if len(dataset) == n_data:
            break

    return dataset

def load_scbench(name):
    samples = load_dataset('Jang-Hyun/SCBench-preprocessed',
                           data_files=f'{name}.parquet', split='train')

    dataset = []
    for data in samples:
        d = {}
        d["context"] = data["prompts"][0]
        d["question"] = data["prompts"][1:]  # evaluate every question independently
        d["answers"] = []
        for gt in data["ground_truth"]:
            if isinstance(gt, list):
                gt = ", ".join(gt)
            else:
                gt = str(gt)
            d["answers"].append(gt)

        if "repoqa" in name:
            d["repoqa_refs"] = {
                "lang": data["lang"],
                "repo": data["repo"],
                "func_name": list(data["func_name"]),
                "ground_truth": list(data["ground_truth"]),
            }

        dataset.append(d)

    return dataset
