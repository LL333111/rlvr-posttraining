from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

from src.training.common import SYSTEM_PROMPT, load_base_model, load_tokenizer


def load_evaluation_model(model_config: dict[str, Any], adapter_path: str | Path | None):
    model = load_base_model(model_config)
    if adapter_path is not None:
        from peft import PeftModel

        path = Path(adapter_path)
        if not (path / "adapter_config.json").is_file():
            raise FileNotFoundError(f"No adapter found at {path}")
        model = PeftModel.from_pretrained(model, path, is_trainable=False)
    model.config.use_cache = True
    model.eval()
    return model


def _chunks(items: list[Any], size: int) -> Iterable[list[Any]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def generate_completions(
    rows: list[dict[str, Any]],
    *,
    model_config: dict[str, Any],
    adapter_path: str | Path | None,
    batch_size: int,
    max_new_tokens: int,
) -> list[dict[str, Any]]:
    import torch

    tokenizer = load_tokenizer(
        model_config["name"],
        model_config.get("trust_remote_code", False),
        model_config.get("revision"),
    )
    tokenizer.padding_side = "left"
    model = load_evaluation_model(model_config, adapter_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    outputs: list[dict[str, Any]] = []
    for batch in _chunks(rows, batch_size):
        messages = [
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": item["question"]},
            ]
            for item in batch
        ]
        prompts = [
            tokenizer.apply_chat_template(
                conversation, tokenize=False, add_generation_prompt=True
            )
            for conversation in messages
        ]
        encoded = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                do_sample=False,
                num_beams=1,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        input_width = encoded["input_ids"].shape[1]
        for item, token_ids in zip(batch, generated[:, input_width:], strict=True):
            completion = tokenizer.decode(token_ids, skip_special_tokens=True)
            outputs.append(
                {
                    **item,
                    "completion": completion,
                    "completion_tokens": int(token_ids.ne(tokenizer.pad_token_id).sum().item()),
                }
            )
    return outputs
