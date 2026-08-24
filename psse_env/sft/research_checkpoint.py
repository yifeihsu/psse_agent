"""Select the newest fully readable research Trainer checkpoint."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any


CHECKPOINT_PATTERN = re.compile(r"checkpoint-([1-9][0-9]*)")


class IncompleteCheckpointError(RuntimeError):
    """No checkpoint candidate is complete enough for a safe Trainer resume."""


def _require_nonempty_regular(path: Path) -> None:
    if path.is_symlink() or not path.is_file() or path.stat().st_size <= 0:
        raise ValueError(f"missing, empty, or non-regular {path.name}")


def _load_json_object(path: Path) -> dict[str, Any]:
    _require_nonempty_regular(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path.name} is not a JSON object")
    return payload


def _validate_adapter_weights(path: Path) -> None:
    safetensors_path = path / "adapter_model.safetensors"
    binary_path = path / "adapter_model.bin"
    errors: list[str] = []
    if safetensors_path.exists() or safetensors_path.is_symlink():
        try:
            _require_nonempty_regular(safetensors_path)
            from safetensors import safe_open

            tensor_count = 0
            element_count = 0
            with safe_open(
                str(safetensors_path), framework="pt", device="cpu"
            ) as handle:
                for key in handle.keys():
                    tensor = handle.get_tensor(key)
                    tensor_count += 1
                    element_count += int(tensor.numel())
            if tensor_count <= 0 or element_count <= 0:
                raise ValueError("adapter safetensors contains no parameters")
            return
        except Exception as exc:
            errors.append(f"adapter_model.safetensors: {type(exc).__name__}: {exc}")
    if binary_path.exists() or binary_path.is_symlink():
        try:
            _require_nonempty_regular(binary_path)
            payload = _load_torch_mapping(binary_path)
            if not payload:
                raise ValueError("adapter binary contains no parameters")
            return
        except Exception as exc:
            errors.append(f"adapter_model.bin: {type(exc).__name__}: {exc}")
    if not errors:
        errors.append("adapter weights are missing")
    raise ValueError("; ".join(errors))


def _load_torch_mapping(path: Path) -> Mapping[str, Any]:
    _require_nonempty_regular(path)
    import torch

    payload = torch.load(str(path), map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise ValueError(f"{path.name} is not a serialized mapping")
    return payload


def validate_complete_checkpoint(
    path: Path, *, step: int, expected_base_model: str
) -> None:
    if path.is_symlink() or not path.is_dir():
        raise ValueError("not a regular checkpoint directory")
    state = _load_json_object(path / "trainer_state.json")
    global_step = state.get("global_step")
    if (
        isinstance(global_step, bool)
        or not isinstance(global_step, (int, float))
        or not float(global_step).is_integer()
        or int(global_step) != step
    ):
        raise ValueError(
            f"trainer global_step {global_step!r} does not match directory step {step}"
        )
    adapter = _load_json_object(path / "adapter_config.json")
    if str(adapter.get("peft_type") or "").upper() != "LORA":
        raise ValueError("adapter_config.json does not describe LoRA")
    if adapter.get("base_model_name_or_path") != expected_base_model:
        raise ValueError(
            "adapter base model mismatch: "
            f"{adapter.get('base_model_name_or_path')!r} != {expected_base_model!r}"
        )
    _validate_adapter_weights(path)
    for name in ("optimizer.pt", "scheduler.pt", "rng_state.pth"):
        _load_torch_mapping(path / name)


def select_newest_complete_checkpoint(
    output_dir: str | Path, *, expected_base_model: str
) -> Path | None:
    root = Path(output_dir).expanduser().resolve(strict=True)
    candidates: list[tuple[int, Path]] = []
    for path in root.iterdir():
        match = CHECKPOINT_PATTERN.fullmatch(path.name)
        if match:
            candidates.append((int(match.group(1)), path))
    failures: list[str] = []
    for step, path in sorted(candidates, reverse=True):
        try:
            validate_complete_checkpoint(
                path, step=step, expected_base_model=expected_base_model
            )
        except Exception as exc:
            failures.append(f"{path.name}: {type(exc).__name__}: {exc}")
            continue
        return path.resolve(strict=True)
    if candidates:
        raise IncompleteCheckpointError(
            "checkpoint directories exist but none is complete: " + " | ".join(failures)
        )
    return None


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Print the newest fully readable research Trainer checkpoint"
    )
    result.add_argument("--output-dir", required=True, type=Path)
    result.add_argument("--expected-base-model", required=True)
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        selected = select_newest_complete_checkpoint(
            args.output_dir, expected_base_model=args.expected_base_model
        )
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    if selected is not None:
        print(selected)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
