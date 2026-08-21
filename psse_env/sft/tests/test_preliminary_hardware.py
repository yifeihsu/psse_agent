"""Preliminary L40S acceptance stays isolated from release hardware."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from psse_env.sft.preliminary_hardware import (
    L40S_MIN_MEMORY_MIB,
    PreliminaryHardwareError,
    validate_preliminary_accelerator,
    validate_preliminary_hardware_attestation,
)


@dataclass
class _Properties:
    name: str
    total_memory: int


class _Cuda:
    def __init__(
        self,
        name: str,
        memory_mib: int,
        *,
        bf16: bool = True,
        count: int = 1,
    ) -> None:
        self.properties = _Properties(name, memory_mib * 1024**2)
        self.bf16 = bf16
        self.count = count

    def is_available(self) -> bool:
        return self.count > 0

    def device_count(self) -> int:
        return self.count

    def is_bf16_supported(self) -> bool:
        return self.bf16

    def get_device_properties(self, _index: int) -> _Properties:
        return self.properties

    def get_device_capability(self, _index: int) -> tuple[int, int]:
        return (8, 9)


def _torch(name: str, memory_mib: int, **kwargs: object) -> SimpleNamespace:
    return SimpleNamespace(
        cuda=_Cuda(name, memory_mib, **kwargs),
        version=SimpleNamespace(cuda="12.8"),
    )


def test_exact_high_memory_l40s_is_preliminary_only() -> None:
    result = validate_preliminary_accelerator(
        _torch("NVIDIA L40S", 46_068), required_class="l40s"
    )
    assert result["release_eligible"] is False
    assert result["accelerator_class"] == "l40s"
    assert validate_preliminary_hardware_attestation(result) == result


@pytest.mark.parametrize(
    ("name", "memory_mib"),
    (
        ("NVIDIA L40S", L40S_MIN_MEMORY_MIB - 1),
        ("NVIDIA L40S0", 48_000),
        ("NVIDIA L40S FAKE", 48_000),
        ("NVIDIA FAKE L40S", 48_000),
    ),
)
def test_l40s_supersets_fakes_and_low_memory_fail(
    name: str, memory_mib: int
) -> None:
    with pytest.raises(PreliminaryHardwareError):
        validate_preliminary_accelerator(
            _torch(name, memory_mib), required_class="l40s"
        )


def test_l40s_requires_bf16_and_requested_class() -> None:
    with pytest.raises(PreliminaryHardwareError, match="bf16"):
        validate_preliminary_accelerator(
            _torch("NVIDIA L40S", 46_068, bf16=False), required_class="l40s"
        )
    with pytest.raises(PreliminaryHardwareError, match="required h200"):
        validate_preliminary_accelerator(
            _torch("NVIDIA L40S", 46_068), required_class="h200"
        )


def test_release_hardware_still_delegates_without_l40s_expansion() -> None:
    result = validate_preliminary_accelerator(
        _torch("NVIDIA H200", 141_000), required_class="h200"
    )
    assert result["accelerator_class"] == "h200"
    assert result["devices"][0]["accelerator_class"] == "h200"


def test_persisted_release_class_cannot_be_forged() -> None:
    result = validate_preliminary_accelerator(
        _torch("NVIDIA H200", 141_000), required_class="h200"
    )
    result["devices"][0]["name"] = "NVIDIA FAKE H200"
    with pytest.raises(PreliminaryHardwareError, match="name/memory"):
        validate_preliminary_hardware_attestation(result)
