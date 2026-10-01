# AM head budgets

AM here uses uniform head budgets by default. Supply `--am_head_budget` to use a
model-specific allocation. Run the commands below from the repository root.

## Precomputed budget

[`am_head_budget_qwen2.5-7b.json`](am_head_budget_qwen2.5-7b.json) is the budget
from earlier AM experiments for **Qwen/Qwen2.5-7B-Instruct-1M** (28 layers × 4 KV
heads). It is not applied automatically or transferable to other checkpoints.

```bash
CUDA_VISIBLE_DEVICES=0 python eval.py -m qwen2.5-7b -d squad --baseline am \
  --am_head_budget configs/am_head_budget_qwen2.5-7b.json \
  --patch --patch_data repeat-context
```

## Compute a budget for another model

Use [`scripts/calibrate_am_budget.py`](../scripts/calibrate_am_budget.py):

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/calibrate_am_budget.py \
  -m qwen3-4b -d squad --output configs/am_head_budget_qwen3-4b.json
```

Defaults use SQuAD contexts **100–109**, all their questions, and **self-study +
whole-context repeat** references generated with Hugging Face. Calibration varies
one head at a time over **51 retentions from 0 to 0.5**, with the other heads fixed
at a uniform **0.05** baseline. The resulting budget optimizes mean predicted
answer log-perplexity across target retentions **0.01, 0.02, 0.05, and 0.1**.

Generated answers, references, and completed head curves are saved beside the
output in `<stem>.calibration/`. Rerun the same command to resume; use another
output path when changing settings. Run
`python scripts/calibrate_am_budget.py --help` for all options.

Use the generated JSON with the same checkpoint:

```bash
CUDA_VISIBLE_DEVICES=0 python eval.py -m qwen3-4b -d squad --baseline am \
  --am_head_budget configs/am_head_budget_qwen3-4b.json \
  --patch --patch_data repeat-context
```

Budget JSON maps every `L{layer}H{kv_head}` to a finite, nonnegative share of the
total KV budget. Shares sum to 1; they are not per-head retention ratios. The
loader validates the head dimensions and sum before evaluation.
