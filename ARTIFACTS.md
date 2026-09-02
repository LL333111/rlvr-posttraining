# Artifact inventory

This repository is the public, GitHub-ready record of the experiment. It keeps
the code, frozen configuration, exact data split manifest, aggregate metrics,
example-level predictions, GRPO rollouts, statistical analyses, figures, and
written report under ordinary Git.

## Included in Git

- data/processed/splits.json: exact source indices, seed, dataset fingerprints,
  and split sizes used by the formal run.
- results/: formal summaries, run metadata, paired statistics, compute
  accounting, reward dynamics, and reviewed failure cases.
- generations/: all formal example-level predictions and the 8,000 GRPO
  training rollouts used by the analysis.
- figures/, README.md, and REPORT.md: generated plots and the final
  interpretation.
- src/, scripts/, configs/, and tests/: the complete reproducible pipeline.

Processed dataset rows are intentionally not duplicated in Git. They can be
recreated from the public datasets and the checked-in split manifest:

    bash scripts/prepare_data.sh

## Separately preserved archives

Large or redundant runtime artifacts are kept outside Git history:

| Archive | Size | SHA-256 | Contents |
|---|---:|---|---|
| rlvr-final-adapters.tar.gz | 204 MB | e68f587e12574ed9357099fe568dd9b1d597cd9a883729a9dd4f01e6c3286640 | Final SFT, GRPO, and continued-SFT LoRA adapters, tokenizer metadata, and training arguments |
| rlvr-formal-results.tar.gz | 3.5 MB | 393116e33c7383a4d1d85c7a130bdc23aaee7239b98e5c075d97827a8e68a4e0 | Full formal-run backup, including the console log and generated analyses |
| rlvr-data-provenance.tar.gz | 1.2 MB | c1e1de564121b28758fa18da3f517711afe633d3bd9854a4a53662a099fa431b | Prepared rows plus the exact split manifest |

The adapter archive should be published as a release asset or on a model
artifact host rather than committed to the repository. The other two archives
are backups: the important result files and exact split manifest are already
represented in Git.

After extracting the adapter archive beside this repository, the final adapters
appear at:

- checkpoints/sft
- checkpoints/grpo
- checkpoints/continued_sft

The upstream Qwen/Qwen2.5-1.5B-Instruct weights and intermediate optimizer
checkpoints are not preserved. The base model is public and downloaded by the
pipeline; final adapters are sufficient for inference. Intermediate optimizer
state is not required to reproduce the reported results or rerun the formal
experiment from its frozen inputs.
