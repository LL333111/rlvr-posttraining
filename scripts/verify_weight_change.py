from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_adapter(path: Path):
    safetensors_path = path / "adapter_model.safetensors"
    binary_path = path / "adapter_model.bin"
    if safetensors_path.is_file():
        from safetensors.torch import load_file

        return load_file(str(safetensors_path), device="cpu")
    if binary_path.is_file():
        import torch

        return torch.load(binary_path, map_location="cpu", weights_only=True)
    raise FileNotFoundError(f"No adapter weights in {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("before", type=Path)
    parser.add_argument("after", type=Path)
    parser.add_argument("--metadata", type=Path)
    args = parser.parse_args()
    before = load_adapter(args.before)
    after = load_adapter(args.after)
    common = sorted(set(before) & set(after))
    if not common:
        raise SystemExit("No common adapter tensors")
    import torch

    changed = [key for key in common if not torch.equal(before[key], after[key])]
    if not changed:
        raise SystemExit("FAIL: GRPO did not change any adapter tensor")
    if args.metadata:
        metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
        if int(metadata.get("optimizer_steps", 0)) < 1:
            raise SystemExit("FAIL: no optimizer step was recorded")
    print(f"PASS: {len(changed)}/{len(common)} common adapter tensors changed")


if __name__ == "__main__":
    main()
