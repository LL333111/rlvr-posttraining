#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

test -s checkpoints/smoke/sft/adapter_config.json
test -s data/processed/gsm8k_rl.jsonl

# Preserve the completed base evaluation and SFT checkpoint. Remove only the
# failed GRPO branch and downstream smoke artifacts before resuming.
rm -rf checkpoints/smoke/grpo results/smoke/grpo generations/smoke/grpo
rm -rf checkpoints/smoke/continued_sft results/smoke/continued_sft
rm -f results/smoke/evaluation/sft_*.json
rm -f results/smoke/evaluation/continued_sft_*.json
rm -f results/smoke/evaluation/grpo_*.json
rm -f generations/smoke/evaluation/sft_*.jsonl
rm -f generations/smoke/evaluation/continued_sft_*.jsonl
rm -f generations/smoke/evaluation/grpo_*.jsonl

python -m src.training.grpo --config configs/smoke/grpo.yaml
python scripts/verify_weight_change.py \
  checkpoints/smoke/sft \
  checkpoints/smoke/grpo \
  --metadata results/smoke/grpo/run_metadata.json
python -m src.training.continued_sft --config configs/smoke/continued_sft.yaml

for checkpoint in sft continued_sft grpo; do
  for benchmark in gsm8k svamp; do
    python -m src.evaluation.evaluator \
      --config configs/smoke/eval.yaml \
      --checkpoint-key "$checkpoint" \
      --benchmark "$benchmark"
  done
done

test -s generations/smoke/grpo/rollouts.jsonl
test -s results/smoke/grpo/reward_groups.jsonl
echo "SMOKE PASS: resumed GRPO update -> continued SFT -> held-out evaluation"
