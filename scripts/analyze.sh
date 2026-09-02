#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

python -m src.evaluation.checkpoint_sweep --branch continued_sft --limit 200
python -m src.evaluation.checkpoint_sweep --branch grpo --limit 200
python -m src.analysis.training_dynamics
python -m src.analysis.compute_analysis
python -m src.analysis.error_analysis --examples 20
