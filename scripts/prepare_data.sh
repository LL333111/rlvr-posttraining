#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

python -m src.data.prepare_gsm8k --sft-size 3000 --rl-size 1000 --validation-size 500 --seed 42
python -m src.data.prepare_svamp
