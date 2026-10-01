#!/usr/bin/env bash
# Run AM alone, then repeat-context, self-study, and joint patches.
# Example: MODEL=llama3.1-8b DATA=squad bash scripts/am.sh --num 1 --ratios 0.1
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."

common=(
    -m "${MODEL:-qwen2.5-7b}" -d "${DATA:-squad}"
    --baseline am
    --chunk_length 5000 --lam 0.001
    "$@"
)

python eval.py "${common[@]}"
python eval.py "${common[@]}" --patch --patch_data repeat-context
python eval.py "${common[@]}" --patch --patch_data self_study
python eval.py "${common[@]}" --patch --patch_data joint
