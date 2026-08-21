from __future__ import annotations

from types import SimpleNamespace
import unittest

from psse_env.sft.release_hardware import (
    APPROVED_ACCELERATOR_CLASSES,
    RTX_PRO_6000_MIN_MEMORY_MIB,
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
    def test_approved_classes_are_stable(self) -> None:
        self.assertEqual(
            APPROVED_ACCELERATOR_CLASSES,
            ("h100", "h200", "rtx6000"),
        )

    def test_hopper_classes_are_approved(self) -> None:
        for name in ("NVIDIA H100 80GB HBM3", "NVIDIA H100 NVL"):
            with self.subTest(name=name):
                self.assertEqual(
                    normalize_accelerator_class(name, 80 * 1024**3),
                    "h100",
                )
        for name in ("NVIDIA H200", "NVIDIA H200 NVL", "NVIDIA H200-SXM5"):
            with self.subTest(name=name):
                self.assertEqual(
                    normalize_accelerator_class(name, 140 * 1024**3),
                    "h200",
                )

    def test_model_names_require_nvidia_and_complete_tokens(self) -> None:
        rejected = (
            ("NVIDIA H1000", 140 * 1024**3),
            ("NVIDIA H2000", 140 * 1024**3),
            ("NVIDIA RTX PRO 60000", 96 * 1024**3),
            ("NVIDIA RTX PRO 6000X", 96 * 1024**3),
            ("NVIDIA FAKE H100", 140 * 1024**3),
            ("NVIDIA FAKE H200", 140 * 1024**3),
            ("NVIDIA GEFORCE RTX PRO 6000", 96 * 1024**3),
            ("ACME H200", 140 * 1024**3),
            ("ACME RTX PRO 6000", 96 * 1024**3),
        )
        for name, total_memory_bytes in rejected:
            with self.subTest(name=name):
                self.assertIsNone(
                    normalize_accelerator_class(name, total_memory_bytes)
                )

    def test_hopper_names_without_hopper_memory_are_rejected(self) -> None:
        self.assertIsNone(normalize_accelerator_class("NVIDIA H100", 1))
        self.assertIsNone(normalize_accelerator_class("NVIDIA H200", 1))

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
        self.assertIsNone(result["required_accelerator_class"])
        self.assertTrue(result["required_accelerator_class_matched"])

    def test_documented_rtx_pro_6000_variants_are_approved(self) -> None:
        for name in (
            "NVIDIA RTX PRO 6000 Blackwell Server Edition",
            "NVIDIA RTX PRO 6000 Blackwell Workstation Edition",
            "NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition",
        ):
            with self.subTest(name=name):
                self.assertEqual(
                    normalize_accelerator_class(name, 96 * 1024**3),
                    "rtx6000",
                )

    def test_required_class_accepts_only_an_exact_class_match(self) -> None:
        torch_module = _fake_torch(
            name="NVIDIA RTX PRO 6000 Blackwell Server Edition",
            memory_mib=96 * 1024,
        )
        result = validate_torch_release_accelerator(
            torch_module,
            required_class="RTX6000",
        )
        self.assertEqual(result["required_accelerator_class"], "rtx6000")
        self.assertTrue(result["required_accelerator_class_matched"])

        with self.assertRaisesRegex(RuntimeError, "does not match required class 'h200'"):
            validate_torch_release_accelerator(
                torch_module,
                required_class="h200",
            )

    def test_unknown_required_class_is_rejected_before_cuda_access(self) -> None:
        with self.assertRaisesRegex(ValueError, "required_class must be one of"):
            validate_torch_release_accelerator(
                object(),
                required_class="a100",
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

    def test_high_memory_non_pro_rtx_name_is_rejected(self) -> None:
        torch_module = _fake_torch(
            name="NVIDIA RTX 6000 Ada Generation",
            memory_mib=96 * 1024,
        )
        with self.assertRaisesRegex(RuntimeError, "RTX Pro 6000"):
            validate_torch_release_accelerator(torch_module)
        self.assertEqual(
            RTX_PRO_6000_MIN_MEMORY_MIB,
            RTX6000_MIN_MEMORY_MIB,
        )

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
