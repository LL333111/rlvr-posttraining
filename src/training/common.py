from __future__ import annotations

import json
from pathlib import Path
from typing import Any

SYSTEM_PROMPT = (
    "Solve the math problem carefully. Show your reasoning, then put the final "
    "numeric answer on its own line in the form: #### answer"
)


def require_cuda() -> None:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is not installed; run scripts/setup.sh") from exc
    if not torch.cuda.is_available():
        raise RuntimeError("This training pipeline requires an NVIDIA CUDA GPU")


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(
            f"Missing prepared data: {source}. Run the data-preparation command first."
        )
    rows = []
    with source.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON at {source}:{line_number}") from exc
    return rows


def sft_dataset(path: str | Path):
    from datasets import Dataset

    rows = []
    for item in load_jsonl(path):
        rows.append(
            {
                "prompt": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": item["question"]},
                ],
                "completion": [
                    {"role": "assistant", "content": item["reasoning_solution"]}
                ],
            }
        )
    return Dataset.from_list(rows)


def grpo_dataset(path: str | Path):
    from datasets import Dataset

    rows = []
    for item in load_jsonl(path):
        rows.append(
            {
                "id": item["id"],
                "prompt": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": item["question"]},
                ],
                "ground_truth": item["ground_truth"],
            }
        )
    return Dataset.from_list(rows)


def torch_dtype(name: str):
    import torch

    mapping = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    if name not in mapping:
        raise ValueError(f"Unsupported dtype: {name}")
    return mapping[name]


def load_tokenizer(
    model_name: str, trust_remote_code: bool = False, revision: str | None = None
):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=trust_remote_code,
        revision=revision,
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_base_model(model_config: dict[str, Any]):
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        model_config["name"],
        dtype=torch_dtype(model_config.get("dtype", "bfloat16")),
        revision=model_config.get("revision"),
        trust_remote_code=bool(model_config.get("trust_remote_code", False)),
        attn_implementation=model_config.get("attn_implementation", "sdpa"),
    )
    model.config.use_cache = False
    return model


def load_trainable_adapter(model_config: dict[str, Any], adapter_path: str | Path):
    from peft import PeftModel

    source = Path(adapter_path)
    if not (source / "adapter_config.json").is_file():
        raise FileNotFoundError(f"No LoRA adapter found at {source}")
    base = load_base_model(model_config)
    model = PeftModel.from_pretrained(base, source, is_trainable=True)
    model.config.use_cache = False
    return model


def ensure_trainable_adapter(model: Any) -> int:
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    if trainable <= 0:
        raise RuntimeError("No trainable parameters; the adapter was loaded frozen")
    return trainable


def validate_prompt_lengths(dataset: Any, tokenizer: Any, max_tokens: int) -> None:
    too_long: list[tuple[int, int]] = []
    for index, prompt in enumerate(dataset["prompt"]):
        rendered = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
        length = len(tokenizer(rendered, add_special_tokens=False)["input_ids"])
        if length > max_tokens:
            too_long.append((index, length))
    if too_long:
        preview = ", ".join(f"{index}:{length}" for index, length in too_long[:5])
        raise ValueError(f"{len(too_long)} prompts exceed {max_tokens} tokens ({preview})")
