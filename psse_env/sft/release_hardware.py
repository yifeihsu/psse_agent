"""Fail-closed accelerator checks shared by the BC0 release launchers."""

from __future__ import annotations

import json
from typing import Any


RTX6000_MIN_MEMORY_MIB = 90_000
_MIB = 1024**2


def normalize_accelerator_class(
    name: str,
    total_memory_bytes: int,
) -> str | None:
    """Return the approved release accelerator class, if any."""

    normalized_name = str(name).upper()
    if "H200" in normalized_name:
        return "h200"
    if "H100" in normalized_name:
        return "h100"
    if (
        "RTX" in normalized_name
        and "6000" in normalized_name
        and int(total_memory_bytes) >= RTX6000_MIN_MEMORY_MIB * _MIB
    ):
        return "rtx6000"
    return None


def validate_torch_release_accelerator(torch_module: Any) -> dict[str, Any]:
    """Validate and describe the one GPU visible to a release process."""

    cuda = torch_module.cuda
    failures: list[str] = []
    if not cuda.is_available():
        failures.append("torch.cuda.is_available() is false")
        device_count = 0
    else:
        device_count = int(cuda.device_count())
        if device_count != 1:
            failures.append(
                f"exactly one visible GPU is required; found {device_count}"
            )

    bf16_supported = bool(cuda.is_bf16_supported()) if device_count else False
    if device_count and not bf16_supported:
        failures.append("allocated GPU does not support bf16")

    devices: list[dict[str, Any]] = []
    for index in range(device_count):
        properties = cuda.get_device_properties(index)
        name = str(properties.name)
        total_memory_bytes = int(properties.total_memory)
        accelerator_class = normalize_accelerator_class(name, total_memory_bytes)
        if accelerator_class is None:
            failures.append(
                "GPU "
                f"{index} is not an H100, H200, or RTX 6000 with at least "
                f"{RTX6000_MIN_MEMORY_MIB} MiB: "
                f"{name!r} ({total_memory_bytes / _MIB:.0f} MiB)"
            )
        capability = cuda.get_device_capability(index)
        devices.append(
            {
                "index": index,
                "name": name,
                "total_memory_bytes": total_memory_bytes,
                "compute_capability": [int(capability[0]), int(capability[1])],
                "accelerator_class": accelerator_class,
            }
        )

    if failures:
        raise RuntimeError(
            "release accelerator contract failed:\n- " + "\n- ".join(failures)
        )
    return {
        "device_count": device_count,
        "bf16_supported": bf16_supported,
        "torch_cuda_version": str(torch_module.version.cuda),
        "devices": devices,
    }


def main() -> int:
    import torch

    try:
        attestation = validate_torch_release_accelerator(torch)
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps(attestation, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
