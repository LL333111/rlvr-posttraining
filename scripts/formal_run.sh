#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

bash scripts/prepare_data.sh
bash scripts/evaluate_base.sh
bash scripts/train_sft.sh
bash scripts/train_grpo.sh
bash scripts/train_continued_sft.sh

for checkpoint in sft continued_sft grpo; do
  for benchmark in gsm8k svamp; do
    python -m src.evaluation.evaluator \
      --config configs/eval.yaml \
      --checkpoint-key "$checkpoint" \
      --benchmark "$benchmark"
  done
done

bash scripts/analyze.sh
echo "FORMAL RUN COMPLETE"
