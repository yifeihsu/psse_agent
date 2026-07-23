from __future__ import annotations

from types import SimpleNamespace
import unittest

from psse_env.sft.release_hardware import (
    RTX6000_MIN_MEMORY_MIB,
    normalize_accelerator_class,
    validate_torch_release_accelerator,
)


class _FakeCuda:
    def __init__(
        self,
        *,
        name: str,
        memory_mib: int,
        available: bool = True,
        bf16: bool = True,
        count: int = 1,
    ) -> None:
        self._name = name
        self._memory_mib = memory_mib
        self._available = available
        self._bf16 = bf16
        self._count = count

    def is_available(self) -> bool:
        return self._available

    def is_bf16_supported(self) -> bool:
        return self._bf16

    def device_count(self) -> int:
        return self._count

    def get_device_properties(self, _index: int) -> SimpleNamespace:
        return SimpleNamespace(
            name=self._name,
            total_memory=self._memory_mib * 1024**2,
        )

    def get_device_capability(self, _index: int) -> tuple[int, int]:
        return (10, 0)


def _fake_torch(**kwargs: object) -> SimpleNamespace:
    return SimpleNamespace(
        cuda=_FakeCuda(**kwargs),
        version=SimpleNamespace(cuda="12.8"),
    )


class ReleaseHardwareTests(unittest.TestCase):
    def test_hopper_classes_are_approved(self) -> None:
        self.assertEqual(
            normalize_accelerator_class("NVIDIA H100 80GB HBM3", 80 * 1024**3),
            "h100",
        )
        self.assertEqual(
            normalize_accelerator_class("NVIDIA H200", 140 * 1024**3),
            "h200",
        )

    def test_high_memory_rtx6000_is_approved_and_attested(self) -> None:
        torch_module = _fake_torch(
            name="NVIDIA RTX PRO 6000 Blackwell Server Edition",
            memory_mib=96 * 1024,
        )
        result = validate_torch_release_accelerator(torch_module)
        self.assertEqual(result["device_count"], 1)
        self.assertTrue(result["bf16_supported"])
        self.assertEqual(
            result["devices"][0]["accelerator_class"],
            "rtx6000",
        )

    def test_48gb_rtx6000_is_rejected(self) -> None:
        torch_module = _fake_torch(
            name="NVIDIA RTX 6000 Ada Generation",
            memory_mib=48 * 1024,
        )
        with self.assertRaisesRegex(
            RuntimeError,
            str(RTX6000_MIN_MEMORY_MIB),
        ):
            validate_torch_release_accelerator(torch_module)

    def test_nonrelease_gpu_and_multiple_devices_are_rejected(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "not an H100"):
            validate_torch_release_accelerator(
                _fake_torch(name="NVIDIA A100", memory_mib=80 * 1024)
            )
        with self.assertRaisesRegex(RuntimeError, "exactly one visible GPU"):
            validate_torch_release_accelerator(
                _fake_torch(
                    name="NVIDIA H200",
                    memory_mib=140 * 1024,
                    count=2,
                )
            )

    def test_cuda_and_bf16_are_required(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "is false"):
            validate_torch_release_accelerator(
                _fake_torch(
                    name="NVIDIA H200",
                    memory_mib=140 * 1024,
                    available=False,
                )
            )
        with self.assertRaisesRegex(RuntimeError, "does not support bf16"):
            validate_torch_release_accelerator(
                _fake_torch(
                    name="NVIDIA H200",
                    memory_mib=140 * 1024,
                    bf16=False,
                )
            )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
