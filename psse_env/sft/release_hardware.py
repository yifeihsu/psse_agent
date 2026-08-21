"""Fail-closed accelerator checks shared by the BC0 release launchers."""

from __future__ import annotations

import argparse
import json
import re
from typing import Any


# ``rtx6000`` is NYU Torch's Slurm feature/GRES spelling.  The approved
# hardware is specifically NVIDIA RTX Pro 6000 Blackwell, not the older
# 48-GiB RTX 6000 Ada Generation card.
RTX_PRO_6000_MIN_MEMORY_MIB = 90_000
RTX6000_MIN_MEMORY_MIB = RTX_PRO_6000_MIN_MEMORY_MIB
H100_MIN_MEMORY_MIB = 75_000
H200_MIN_MEMORY_MIB = 130_000
APPROVED_ACCELERATOR_CLASSES = ("h100", "h200", "rtx6000")
_MIB = 1024**2
_NVIDIA_NAME_PREFIX = "NVIDIA "


def _has_model_token(normalized_name: str, model: str) -> bool:
    """Match one complete NVIDIA model token while allowing driver qualifiers."""

    # CUDA device names begin ``NVIDIA <model>`` and may append qualifiers such
    # as ``80GB HBM3``, ``NVL``, or ``Blackwell Server Edition``.  Anchor that
    # prefix and the trailing model-token boundary rather than accepting
    # arbitrary intervening labels or supersets such as FAKE H200, H2000, or
    # RTX PRO 60000.
    return re.match(
        rf"\A{re.escape(_NVIDIA_NAME_PREFIX)}{re.escape(model)}(?![A-Z0-9])",
        normalized_name,
    ) is not None


def normalize_accelerator_class(
    name: str,
    total_memory_bytes: int,
) -> str | None:
    """Return the approved release accelerator class, if any."""

    normalized_name = " ".join(str(name).upper().split())
    if _has_model_token(normalized_name, "H200") and int(
        total_memory_bytes
    ) >= H200_MIN_MEMORY_MIB * _MIB:
        return "h200"
    if _has_model_token(normalized_name, "H100") and int(
        total_memory_bytes
    ) >= H100_MIN_MEMORY_MIB * _MIB:
        return "h100"
    if _has_model_token(normalized_name, "RTX PRO 6000") and int(
        total_memory_bytes
    ) >= RTX_PRO_6000_MIN_MEMORY_MIB * _MIB:
        return "rtx6000"
    return None


def validate_torch_release_accelerator(
    torch_module: Any,
    *,
    required_class: str | None = None,
) -> dict[str, Any]:
    """Validate and describe the one GPU visible to a release process."""

    normalized_required_class = (
        str(required_class).strip().lower() if required_class is not None else None
    )
    if normalized_required_class not in {None, *APPROVED_ACCELERATOR_CLASSES}:
        raise ValueError(
            "required_class must be one of "
            + ", ".join(APPROVED_ACCELERATOR_CLASSES)
        )

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
                f"{index} is not an H100, H200, or RTX Pro 6000 with at least "
                f"{RTX_PRO_6000_MIN_MEMORY_MIB} MiB: "
                f"{name!r} ({total_memory_bytes / _MIB:.0f} MiB)"
            )
        elif (
            normalized_required_class is not None
            and accelerator_class != normalized_required_class
        ):
            failures.append(
                "GPU "
                f"{index} accelerator class {accelerator_class!r} does not match "
                f"required class {normalized_required_class!r}: {name!r}"
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
        "required_accelerator_class": normalized_required_class,
        "required_accelerator_class_matched": bool(
            normalized_required_class is None
            or all(
                device.get("accelerator_class") == normalized_required_class
                for device in devices
            )
        ),
        "devices": devices,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Fail-closed release accelerator attestation."
    )
    parser.add_argument(
        "--require-class",
        choices=APPROVED_ACCELERATOR_CLASSES,
        help=(
            "Optionally require the allocated accelerator to be one exact "
            "approved class. Use this for same-class canaries; omit it for "
            "portable production jobs."
        ),
    )
    args = parser.parse_args(argv)

    import torch

    try:
        attestation = validate_torch_release_accelerator(
            torch,
            required_class=args.require_class,
        )
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps(attestation, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
