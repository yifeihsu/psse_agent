"""Preliminary-only accelerator gate with an exact NYU L40S route.

Release training remains limited by :mod:`psse_env.sft.release_hardware`.  This
module delegates H100/H200/RTX Pro 6000 decisions to that contract and adds one
strict L40S profile only for explicitly non-release E2B debugging.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

from .release_hardware import (
    APPROVED_ACCELERATOR_CLASSES,
    normalize_accelerator_class,
    validate_torch_release_accelerator,
)


PRELIMINARY_HARDWARE_CONTRACT = "preliminary_dagger_hardware_attestation_v1"
PRELIMINARY_ACCELERATOR_CLASSES = (*APPROVED_ACCELERATOR_CLASSES, "l40s")
L40S_MIN_MEMORY_MIB = 45_000
_MIB = 1024**2


class PreliminaryHardwareError(RuntimeError):
    """The preliminary accelerator failed its distinct fail-closed gate."""


def _normalized_name(name: Any) -> str:
    return " ".join(str(name).upper().split())


def _is_l40s(name: Any, total_memory_bytes: int) -> bool:
    return (
        _normalized_name(name) == "NVIDIA L40S"
        and int(total_memory_bytes) >= L40S_MIN_MEMORY_MIB * _MIB
    )


def _wrap_release_attestation(attestation: Mapping[str, Any]) -> dict[str, Any]:
    devices = attestation.get("devices")
    actual_class = (
        devices[0].get("accelerator_class")
        if isinstance(devices, list)
        and len(devices) == 1
        and isinstance(devices[0], Mapping)
        else None
    )
    return {
        "contract": PRELIMINARY_HARDWARE_CONTRACT,
        "artifact_type": "preliminary_dagger_nonrelease_hardware_attestation",
        "release_eligible": False,
        "accelerator_class": actual_class,
        **dict(attestation),
    }


def validate_preliminary_accelerator(
    torch_module: Any,
    *,
    required_class: str | None = None,
) -> dict[str, Any]:
    """Accept release hardware or one exact >=45,000-MiB NVIDIA L40S."""

    requested = str(required_class or "auto").strip().lower()
    if requested not in {"auto", *PRELIMINARY_ACCELERATOR_CLASSES}:
        raise ValueError(
            "required_class must be auto or one of "
            + ", ".join(PRELIMINARY_ACCELERATOR_CLASSES)
        )
    cuda = torch_module.cuda
    available = bool(cuda.is_available())
    device_count = int(cuda.device_count()) if available else 0
    if device_count == 1:
        properties = cuda.get_device_properties(0)
        name = str(properties.name)
        total_memory_bytes = int(properties.total_memory)
        release_class = normalize_accelerator_class(name, total_memory_bytes)
        if release_class is not None:
            if requested == "l40s":
                raise PreliminaryHardwareError(
                    f"allocated accelerator is {release_class}, required l40s"
                )
            release_required = None if requested == "auto" else requested
            try:
                release = validate_torch_release_accelerator(
                    torch_module, required_class=release_required
                )
            except RuntimeError as exc:
                raise PreliminaryHardwareError(str(exc)) from exc
            return _wrap_release_attestation(release)
        if _is_l40s(name, total_memory_bytes):
            failures: list[str] = []
            if requested not in {"auto", "l40s"}:
                failures.append(
                    f"allocated accelerator is l40s, required {requested}"
                )
            if not bool(cuda.is_bf16_supported()):
                failures.append("allocated L40S does not support bf16")
            cuda_version = str(torch_module.version.cuda or "").strip()
            if not cuda_version:
                failures.append("Torch does not report a CUDA runtime")
            capability = cuda.get_device_capability(0)
            if int(capability[0]) < 8:
                failures.append(
                    f"L40S compute capability is not CUDA/bf16 capable: {capability}"
                )
            if failures:
                raise PreliminaryHardwareError(
                    "preliminary L40S contract failed:\n- " + "\n- ".join(failures)
                )
            return {
                "contract": PRELIMINARY_HARDWARE_CONTRACT,
                "artifact_type": (
                    "preliminary_dagger_nonrelease_hardware_attestation"
                ),
                "release_eligible": False,
                "accelerator_class": "l40s",
                "device_count": 1,
                "bf16_supported": True,
                "torch_cuda_version": cuda_version,
                "required_accelerator_class": (
                    None if requested == "auto" else requested
                ),
                "required_accelerator_class_matched": True,
                "devices": [
                    {
                        "index": 0,
                        "name": name,
                        "total_memory_bytes": total_memory_bytes,
                        "compute_capability": [
                            int(capability[0]),
                            int(capability[1]),
                        ],
                        "accelerator_class": "l40s",
                    }
                ],
            }

    # Reuse the release gate's complete diagnostics for unavailable, multiple,
    # undersized, fake, or otherwise unsupported devices.
    release_required = requested if requested in APPROVED_ACCELERATOR_CLASSES else None
    try:
        validate_torch_release_accelerator(
            torch_module, required_class=release_required
        )
    except RuntimeError as exc:
        raise PreliminaryHardwareError(str(exc)) from exc
    raise PreliminaryHardwareError("preliminary accelerator classification failed")


def validate_preliminary_hardware_attestation(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the persisted, explicitly non-release hardware evidence."""

    if value.get("contract") != PRELIMINARY_HARDWARE_CONTRACT:
        raise PreliminaryHardwareError("invalid preliminary hardware contract")
    if value.get("release_eligible") is not False:
        raise PreliminaryHardwareError(
            "preliminary hardware attestation must be release_eligible=false"
        )
    accelerator_class = value.get("accelerator_class")
    if accelerator_class not in PRELIMINARY_ACCELERATOR_CLASSES:
        raise PreliminaryHardwareError("unsupported preliminary accelerator class")
    if value.get("device_count") != 1 or value.get("bf16_supported") is not True:
        raise PreliminaryHardwareError(
            "preliminary attestation must bind one bf16-capable GPU"
        )
    if not str(value.get("torch_cuda_version") or "").strip():
        raise PreliminaryHardwareError("preliminary attestation lacks CUDA runtime")
    devices = value.get("devices")
    if not isinstance(devices, list) or len(devices) != 1:
        raise PreliminaryHardwareError("preliminary attestation must contain one device")
    device = devices[0]
    if not isinstance(device, Mapping) or device.get(
        "accelerator_class"
    ) != accelerator_class:
        raise PreliminaryHardwareError(
            "preliminary device class does not match the attestation"
        )
    name = device.get("name")
    try:
        memory_bytes = int(device.get("total_memory_bytes") or 0)
    except (TypeError, ValueError) as exc:
        raise PreliminaryHardwareError(
            "preliminary device memory is not an integer"
        ) from exc
    capability = device.get("compute_capability")
    if (
        not isinstance(capability, list)
        or len(capability) != 2
        or any(isinstance(value, bool) or not isinstance(value, int) for value in capability)
        or capability[0] < 8
    ):
        raise PreliminaryHardwareError(
            "preliminary device lacks CUDA/bf16 compute capability"
        )
    if accelerator_class == "l40s":
        if not _is_l40s(name, memory_bytes):
            raise PreliminaryHardwareError(
                "persisted L40S is not exact-name or >=45,000 MiB"
            )
    elif normalize_accelerator_class(str(name), memory_bytes) != accelerator_class:
        raise PreliminaryHardwareError(
            "persisted release-class device name/memory does not match its class"
        )
    if value.get("required_accelerator_class_matched") is not True:
        raise PreliminaryHardwareError(
            "preliminary hardware required-class match is not true"
        )
    required = value.get("required_accelerator_class")
    if required is not None and required != accelerator_class:
        raise PreliminaryHardwareError(
            "preliminary required accelerator class differs from actual class"
        )
    return dict(value)


def load_preliminary_hardware_attestation(
    path: str | os.PathLike[str],
) -> dict[str, Any]:
    candidate = Path(path)
    if candidate.is_symlink() or not candidate.is_file():
        raise PreliminaryHardwareError(
            "hardware attestation must be a regular non-symlink file"
        )
    try:
        value = json.loads(candidate.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PreliminaryHardwareError(
            f"cannot read preliminary hardware attestation: {exc}"
        ) from exc
    if not isinstance(value, Mapping):
        raise PreliminaryHardwareError("hardware attestation must be one JSON object")
    return validate_preliminary_hardware_attestation(value)


def _write_current_attestation(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o400)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            if os.name == "nt":
                os.chmod(temporary, 0o600)
            temporary.unlink(missing_ok=True)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Attest non-release E2B preliminary accelerator hardware."
    )
    parser.add_argument(
        "--require-class",
        choices=("auto", *PRELIMINARY_ACCELERATOR_CLASSES),
        default="auto",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    import torch

    result = validate_preliminary_accelerator(
        torch, required_class=args.require_class
    )
    if args.output is not None:
        _write_current_attestation(args.output, result)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
