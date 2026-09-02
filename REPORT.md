# Experimental Report

> Complete this report only from formal-run artifacts. Do not copy smoke-test metrics here.

## Research question

Under a comparable additional GPU-time budget, does verifier-guided GRPO improve held-out mathematical reasoning more than continued supervised fine-tuning?

## Frozen setup

| Item | Value |
|---|---|
| Base model | Qwen2.5-1.5B-Instruct |
| SFT / RL / validation split seed | 42 |
| GRPO generations per prompt | 4 |
| Primary reward | normalized exact answer correctness, 0/1 |
| Primary evaluation | greedy exact match |
| Benchmarks | GSM8K official test; SVAMP transfer |
| Git commit | pending |
| GPU | pending |

## Main results

| Model | GSM8K | SVAMP | Extra GPU hours |
|---|---:|---:|---:|
| Base | pending | pending | — |
| Reasoning SFT | pending | pending | — |
| Continued SFT | pending | pending | pending |
| GRPO / RLVR | pending | pending | pending |

Artifact sources:

- `results/evaluation/*.json`
- `results/continued_sft/run_metadata.json`
- `results/grpo/run_metadata.json`

## Compute comparison

Paste the generated `results/analysis/compute_table.md` here and state the percentage mismatch in additional GPU-hours. Use “comparable,” not “same,” unless the measurements support exact equality.

## Training dynamics

### Reward curve

Insert `figures/grpo_reward_curve.png` and describe trend, instability, and plateaus without treating training reward as held-out capability.

### Relative signal quality

Insert `figures/reward_group_composition.png` and report all-wrong, mixed, and all-correct fractions. Discuss how much training generated non-zero within-group variance.

### Completion behavior

Report mean completion length overall and separately for correct and incorrect rollouts. Check whether reward increases can be explained by formatting or length changes.

## Performance versus budget

Insert `figures/performance_vs_gpu_time.png`. State clearly that the curve uses the frozen 200-example diagnostic subset while the headline table uses full benchmarks.

## Manual failure analysis

Review `results/analysis/failure_review.csv`, fill at least 20 categories and notes, then summarize counts here.

| Category | Count | Representative IDs |
|---|---:|---|
| Arithmetic error | pending | pending |
| Wrong strategy | pending | pending |
| Correct reasoning / wrong extraction | pending | pending |
| Incomplete reasoning | pending | pending |
| Overlong reasoning | pending | pending |
| Answer-format error | pending | pending |
| RL-specific degeneration | pending | pending |
| Successful corrected reasoning | pending | pending |

Include both success and failure cases. Quote only short response excerpts and retain IDs so each claim is traceable to raw JSONL.

## Conclusion

Choose only the conclusion supported by held-out results:

1. GRPO provided an additional improvement beyond comparable extra supervised training.
2. GRPO showed no clear efficiency advantage in this regime.
3. Continued supervised training was more effective for this model/data/compute regime.

If the difference is small, run the key comparison with a second seed before claiming superiority.

## Limitations

- One small model and one primary training domain.
- GPU-hours are a practical proxy, not hardware-independent FLOPs.
- Different supervision sources: labeled solutions versus prompts plus verifier feedback.
- Final-answer correctness does not validate intermediate reasoning.
- One primary seed unless the close-result policy triggers a confirmation run.
