from __future__ import annotations

import argparse
import os
import statistics
from pathlib import Path
from typing import Any

from src.rewards.metrics import reward_group_counts
from src.rewards.verifier import verify_answer
from src.training.callbacks import JsonlLogCallback, WallClockBudgetCallback
from src.training.common import (
    ensure_trainable_adapter,
    grpo_dataset,
    load_tokenizer,
    load_trainable_adapter,
    require_cuda,
    validate_prompt_lengths,
)
from src.training.run_utils import prepare_run, train_and_record, wall_seconds_for_gpu_hours
from src.utils.config import load_config
from src.utils.logging import append_jsonl
from src.utils.reproducibility import set_seed


def completion_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts = []
        for message in value:
            if isinstance(message, dict):
                parts.append(str(message.get("content", "")))
            else:
                parts.append(str(message))
        return "\n".join(parts)
    return str(value)


def make_reward_function(
    *,
    generations_per_prompt: int,
    reward_groups_file: Path,
    rollouts_file: Path,
    tokenizer: Any,
):
    def verifiable_correctness_reward(
        completions: list[Any],
        ground_truth: list[Any],
        id: list[Any] | None = None,
        trainer_state: Any = None,
        **_: Any,
    ) -> list[float]:
        results = [
            verify_answer(completion_text(completion), target)
            for completion, target in zip(completions, ground_truth, strict=True)
        ]
        rewards = [result.reward for result in results]
        if os.environ.get("RANK", "0") == "0":
            step = int(getattr(trainer_state, "global_step", -1))
            groups = reward_group_counts(rewards, generations_per_prompt)
            append_jsonl(
                reward_groups_file,
                {
                    "step": step,
                    "mean_reward": statistics.fmean(rewards) if rewards else 0.0,
                    "reward_std": statistics.pstdev(rewards) if len(rewards) > 1 else 0.0,
                    **groups,
                },
            )
            identifiers = id or [None] * len(results)
            for identifier, completion, target, result in zip(
                identifiers, completions, ground_truth, results, strict=True
            ):
                append_jsonl(
                    rollouts_file,
                    {
                        "step": step,
                        "id": identifier,
                        "completion": completion_text(completion),
                        "completion_tokens": len(
                            tokenizer.encode(completion_text(completion), add_special_tokens=False)
                        ),
                        "raw_ground_truth": str(target),
                        **result.to_dict(),
                    },
                )
        return rewards

    return verifiable_correctness_reward


def run(config_path: Path) -> None:
    from trl import GRPOConfig, GRPOTrainer

    config = load_config(config_path)
    require_cuda()
    model_cfg, training, grpo_cfg = config["model"], config["training"], config["grpo"]
    source = Path(config["source_checkpoint"])
    set_seed(int(training["seed"]))
    prepare_run(config_path, training["output_dir"], training["results_dir"])

    tokenizer = load_tokenizer(
        model_cfg["name"], model_cfg.get("trust_remote_code", False), model_cfg.get("revision")
    )
    tokenizer.padding_side = "left"
    model = load_trainable_adapter(model_cfg, source)
    ensure_trainable_adapter(model)
    train_data = grpo_dataset(config["data"]["train_file"])
    validate_prompt_lengths(train_data, tokenizer, int(grpo_cfg["max_prompt_length"]))

    reward_groups = Path(config["logging"]["reward_groups_file"])
    rollouts = Path(config["logging"].get("rollouts_file", "generations/grpo/rollouts.jsonl"))
    reward_function = make_reward_function(
        generations_per_prompt=int(grpo_cfg["generations_per_prompt"]),
        reward_groups_file=reward_groups,
        rollouts_file=rollouts,
        tokenizer=tokenizer,
    )
    budget = WallClockBudgetCallback(
        wall_seconds_for_gpu_hours(float(grpo_cfg["max_gpu_hours"]))
    )
    log_callback = JsonlLogCallback(Path(training["results_dir"]) / "train_log.jsonl")
    args = GRPOConfig(
        output_dir=training["output_dir"],
        learning_rate=float(training["learning_rate"]),
        lr_scheduler_type=training["lr_scheduler_type"],
        optim=training["optimizer"],
        warmup_ratio=float(training["warmup_ratio"]),
        per_device_train_batch_size=int(training["per_device_train_batch_size"]),
        gradient_accumulation_steps=int(training["gradient_accumulation_steps"]),
        max_steps=int(training["max_steps"]),
        logging_steps=int(training["logging_steps"]),
        save_strategy="steps",
        save_steps=int(training["save_steps"]),
        save_total_limit=int(training["save_total_limit"]),
        max_grad_norm=float(training["max_grad_norm"]),
        gradient_checkpointing=bool(training["gradient_checkpointing"]),
        bf16=model_cfg.get("dtype") == "bfloat16",
        fp16=model_cfg.get("dtype") == "float16",
        tf32=bool(training["tf32"]),
        seed=int(training["seed"]),
        data_seed=int(training["seed"]),
        report_to="none",
        num_generations=int(grpo_cfg["generations_per_prompt"]),
        max_completion_length=int(grpo_cfg["max_completion_length"]),
        temperature=float(grpo_cfg["temperature"]),
        beta=float(grpo_cfg["beta"]),
        loss_type=grpo_cfg["loss_type"],
        scale_rewards=grpo_cfg["scale_rewards"],
        epsilon=float(grpo_cfg["epsilon"]),
        num_iterations=int(grpo_cfg["num_iterations"]),
        mask_truncated_completions=bool(grpo_cfg["mask_truncated_completions"]),
        use_vllm=bool(grpo_cfg["use_vllm"]),
        log_completions=bool(config["logging"]["save_generations"]),
        num_completions_to_print=0,
        include_num_input_tokens_seen="all",
    )
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_function,
        args=args,
        train_dataset=train_data,
        processing_class=tokenizer,
        callbacks=[budget, log_callback],
    )
    train_and_record(
        trainer,
        output_dir=training["output_dir"],
        results_dir=training["results_dir"],
        config_path=config_path,
        source_checkpoint=source,
        budget_callback=budget,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/grpo.yaml"))
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
