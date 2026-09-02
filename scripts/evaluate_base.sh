#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

python -m src.evaluation.evaluator --config configs/eval.yaml --checkpoint-key base --benchmark gsm8k
python -m src.evaluation.evaluator --config configs/eval.yaml --checkpoint-key base --benchmark svamp
