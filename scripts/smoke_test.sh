#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

python -m unittest discover -s tests -v
python -m src.data.prepare_gsm8k --sft-size 20 --rl-size 10 --validation-size 10 --seed 42
python -m src.data.prepare_svamp

python -m src.evaluation.evaluator --config configs/smoke/eval.yaml --checkpoint-key base --benchmark gsm8k
python -m src.evaluation.evaluator --config configs/smoke/eval.yaml --checkpoint-key base --benchmark svamp
python -m src.training.sft --config configs/smoke/sft.yaml
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
echo "SMOKE PASS: data -> SFT -> GRPO update -> continued SFT -> held-out evaluation"
