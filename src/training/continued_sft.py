from __future__ import annotations

import argparse
from pathlib import Path

from src.training.callbacks import JsonlLogCallback, WallClockBudgetCallback
from src.training.common import (
    ensure_trainable_adapter,
    load_tokenizer,
    load_trainable_adapter,
    require_cuda,
    sft_dataset,
)
from src.training.run_utils import matched_wall_seconds, prepare_run, train_and_record
from src.utils.config import load_config
from src.utils.reproducibility import set_seed


def run(config_path: Path) -> None:
    from trl import SFTConfig, SFTTrainer

    config = load_config(config_path)
    require_cuda()
    model_cfg, data_cfg, compute, training = (
        config["model"],
        config["data"],
        config["compute"],
        config["training"],
    )
    source = Path(config["source_checkpoint"])
    set_seed(int(training["seed"]))
    prepare_run(config_path, training["output_dir"], training["results_dir"])
    tokenizer = load_tokenizer(
        model_cfg["name"], model_cfg.get("trust_remote_code", False), model_cfg.get("revision")
    )
    model = load_trainable_adapter(model_cfg, source)
    ensure_trainable_adapter(model)
    budget_seconds = matched_wall_seconds(
        compute["match_metadata"], float(compute["fallback_max_gpu_hours"])
    )
    budget = WallClockBudgetCallback(budget_seconds)
    log_callback = JsonlLogCallback(Path(training["results_dir"]) / "train_log.jsonl")
    args = SFTConfig(
        output_dir=training["output_dir"],
        max_steps=int(training["max_steps"]),
        learning_rate=float(training["learning_rate"]),
        lr_scheduler_type=training["lr_scheduler_type"],
        optim=training["optimizer"],
        warmup_steps=int(training["warmup_steps"]),
        per_device_train_batch_size=int(training["per_device_train_batch_size"]),
        per_device_eval_batch_size=int(training["per_device_eval_batch_size"]),
        gradient_accumulation_steps=int(training["gradient_accumulation_steps"]),
        max_length=int(training["max_length"]),
        weight_decay=float(training["weight_decay"]),
        max_grad_norm=float(training["max_grad_norm"]),
        logging_steps=int(training["logging_steps"]),
        eval_strategy="steps",
        eval_steps=int(training["eval_steps"]),
        save_strategy="steps",
        save_steps=int(training["save_steps"]),
        save_total_limit=int(training["save_total_limit"]),
        gradient_checkpointing=bool(training["gradient_checkpointing"]),
        bf16=model_cfg.get("dtype") == "bfloat16",
        fp16=model_cfg.get("dtype") == "float16",
        tf32=bool(training["tf32"]),
        seed=int(training["seed"]),
        data_seed=int(training["seed"]),
        report_to="none",
        completion_only_loss=bool(training["completion_only_loss"]),
        eos_token=training["eos_token"],
        packing=bool(training["packing"]),
        shuffle_dataset=bool(training["shuffle_dataset"]),
        include_num_input_tokens_seen="all",
    )
    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=sft_dataset(data_cfg["train_file"]),
        eval_dataset=sft_dataset(data_cfg["validation_file"]),
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
    parser.add_argument("--config", type=Path, default=Path("configs/continued_sft.yaml"))
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
