from __future__ import annotations

import argparse
from pathlib import Path

from src.training.callbacks import JsonlLogCallback
from src.training.common import load_base_model, load_tokenizer, require_cuda, sft_dataset
from src.training.run_utils import prepare_run, train_and_record
from src.utils.config import load_config, require_keys
from src.utils.reproducibility import set_seed


def run(config_path: Path) -> None:
    from peft import LoraConfig
    from trl import SFTConfig, SFTTrainer

    config = load_config(config_path)
    require_cuda()
    for section in ("model", "data", "lora", "training"):
        require_keys(config, [section])
    model_cfg, data_cfg, lora_cfg, training = (
        config["model"],
        config["data"],
        config["lora"],
        config["training"],
    )
    set_seed(int(training["seed"]))
    prepare_run(config_path, training["output_dir"], training["results_dir"])

    tokenizer = load_tokenizer(
        model_cfg["name"], model_cfg.get("trust_remote_code", False), model_cfg.get("revision")
    )
    model = load_base_model(model_cfg)
    train_data = sft_dataset(data_cfg["train_file"])
    validation_data = sft_dataset(data_cfg["validation_file"])
    peft_config = LoraConfig(
        r=int(lora_cfg["rank"]),
        lora_alpha=int(lora_cfg["alpha"]),
        lora_dropout=float(lora_cfg["dropout"]),
        target_modules=list(lora_cfg["target_modules"]),
        bias="none",
        task_type="CAUSAL_LM",
    )
    args = SFTConfig(
        output_dir=training["output_dir"],
        num_train_epochs=float(training["epochs"]),
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
    callback = JsonlLogCallback(Path(training["results_dir"]) / "train_log.jsonl")
    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=train_data,
        eval_dataset=validation_data,
        processing_class=tokenizer,
        peft_config=peft_config,
        callbacks=[callback],
    )
    train_and_record(
        trainer,
        output_dir=training["output_dir"],
        results_dir=training["results_dir"],
        config_path=config_path,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/sft.yaml"))
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
