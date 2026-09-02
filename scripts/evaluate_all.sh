#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

for checkpoint in base sft continued_sft grpo; do
  for benchmark in gsm8k svamp; do
    python -m src.evaluation.evaluator \
      --config configs/eval.yaml \
      --checkpoint-key "$checkpoint" \
      --benchmark "$benchmark"
  done
done
