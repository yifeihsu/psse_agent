import json
import unittest

from preprocess import (
    combo_distribution,
    dataset_qa,
    exact_balanced_split,
    extract_label,
    multi_combo_exact_split,
    multi_label_qa,
)
from trace_protocol import BALANCED_SPLIT_COUNTS, ERROR_FAMILIES


def make_sample(label: str, idx: int) -> dict:
    payload = {
        "verdict": {
            "has_error": label != "no_error",
            "error_family": label,
            "confidence": 0.97,
        },
        "evidence": {
            "global_metrics": {
                "global_residual_sum": 80.0 if label == "no_error" else 140.0,
                "global_residual_threshold": 100.0,
                "global_residual_ratio": 0.8 if label == "no_error" else 1.4,
            },
            "top_residuals": [] if label == "no_error" else [{"index0": idx % 5, "value": 4.2}],
            "top_lagrange": [],
        },
        "suspect_location": {"domain": "none", "details": {}},
        "action": {"verification_summary": {"success": True} if label != "no_error" else None},
        "summary": f"{label}-{idx}",
    }
    return {
        "messages": [
            {"role": "system", "content": "system"},
            {
                "role": "user",
                "content": json.dumps({"case_path": f"case_{label}_{idx}", "z_obs": [1.0, 2.0, 3.0]}),
            },
            {"role": "assistant", "content": json.dumps(payload, ensure_ascii=False)},
        ]
    }


def make_multi_sample(families: list[str], idx: int) -> dict:
    payload = {
        "verdict": {
            "has_error": True,
            "error_family": families[0],
            "error_families": families,
            "confidence": 0.96,
        },
        "evidence": {
            "global_metrics": {
                "global_residual_sum": 150.0,
                "global_residual_threshold": 100.0,
                "global_residual_ratio": 1.5,
            },
            "top_residuals": [{"index0": idx % 5, "value": 4.2}],
            "top_lagrange": [{"lambda_index0": idx % 4, "value": 5.1}],
        },
        "suspect_location": {"domain": "measurement", "details": {}},
        "suspect_locations": [{"domain": family.split("_")[0], "details": {}} for family in families],
        "action": {
            "applied_tool": "correct_parameters_from_path",
            "applied_tools": [
                {
                    "measurement_error": "correct_measurements_from_path",
                    "parameter_error": "correct_parameters_from_path",
                    "topology_error": "correct_topology_from_path",
                    "harmonic_anomaly": "run_hse_from_path",
                }[family]
                for family in families
            ],
            "verification_summary": None,
        },
        "summary": f"{'+'.join(families)}-{idx}",
    }
    return {
        "messages": [
            {"role": "system", "content": "system"},
            {
                "role": "user",
                "content": json.dumps({"case_path": f"case_multi_{idx}", "z_obs": [1.0, 2.0, 3.0]}),
            },
            {"role": "assistant", "content": json.dumps(payload, ensure_ascii=False)},
        ]
    }


class PreprocessBalancingTests(unittest.TestCase):
    def test_exact_balanced_split_returns_400_50_50_for_all_classes(self) -> None:
        samples = []
        for label in ERROR_FAMILIES:
            for idx in range(sum(BALANCED_SPLIT_COUNTS.values())):
                samples.append(make_sample(label, idx))

        train, valid, test, meta = exact_balanced_split(samples, seed=42)

        self.assertEqual(meta["mode"], "exact_balanced")
        self.assertEqual(len(train), BALANCED_SPLIT_COUNTS["train"] * len(ERROR_FAMILIES))
        self.assertEqual(len(valid), BALANCED_SPLIT_COUNTS["valid"] * len(ERROR_FAMILIES))
        self.assertEqual(len(test), BALANCED_SPLIT_COUNTS["test"] * len(ERROR_FAMILIES))

        for split, expected_count in (
            (train, BALANCED_SPLIT_COUNTS["train"]),
            (valid, BALANCED_SPLIT_COUNTS["valid"]),
            (test, BALANCED_SPLIT_COUNTS["test"]),
        ):
            counts = {label: 0 for label in ERROR_FAMILIES}
            for sample in split:
                counts[extract_label(sample)] += 1
            self.assertEqual(set(counts.keys()), set(ERROR_FAMILIES))
            for label in ERROR_FAMILIES:
                self.assertEqual(counts[label], expected_count)

    def test_dataset_qa_flags_clean_trace_regressions(self) -> None:
        samples = [make_sample("no_error", 0), make_sample("measurement_error", 1)]
        no_error_payload = json.loads(samples[0]["messages"][-1]["content"])
        no_error_payload["evidence"]["global_metrics"]["global_residual_ratio"] = 0.95
        no_error_payload["evidence"]["top_residuals"] = [{"index0": 1, "value": 3.4}]
        samples[0]["messages"][-1]["content"] = json.dumps(no_error_payload, ensure_ascii=False)

        qa = dataset_qa(samples)

        self.assertEqual(qa["thresholds_null"], 0)
        self.assertEqual(qa["ratios_null"], 0)
        self.assertEqual(qa["no_error_ratio_violations"], 1)
        self.assertEqual(qa["no_error_nonempty_evidence"], 1)

    def test_exact_balanced_split_prioritizes_resolved_verification_cases(self) -> None:
        samples = []
        total = sum(BALANCED_SPLIT_COUNTS.values())
        for label in ERROR_FAMILIES:
            for idx in range(total):
                sample = make_sample(label, idx)
                if label == "measurement_error":
                    payload = json.loads(sample["messages"][-1]["content"])
                    payload["action"]["verification_summary"] = {
                        "post_action_resolved": idx < 10,
                        "post_action_improved": True,
                        "post_action_global_residual_ratio": 0.8 if idx < 10 else 1.3,
                    }
                    sample["messages"][-1]["content"] = json.dumps(payload, ensure_ascii=False)
                samples.append(sample)

        extra_unresolved = []
        for idx in range(30):
            sample = make_sample("measurement_error", 1000 + idx)
            payload = json.loads(sample["messages"][-1]["content"])
            payload["action"]["verification_summary"] = {
                "post_action_resolved": False,
                "post_action_improved": True,
                "post_action_global_residual_ratio": 1.6,
            }
            sample["messages"][-1]["content"] = json.dumps(payload, ensure_ascii=False)
            extra_unresolved.append(sample)

        train, valid, test, meta = exact_balanced_split(samples + extra_unresolved, seed=42)
        self.assertEqual(meta["selected_per_class"]["measurement_error"], total)
        resolved_selected = 0
        resolved_by_split = []
        for split in (train, valid, test):
            split_count = 0
            for sample in split:
                if extract_label(sample) != "measurement_error":
                    continue
                payload = json.loads(sample["messages"][-1]["content"])
                summary = payload.get("action", {}).get("verification_summary") or {}
                if summary.get("post_action_resolved") is True:
                    resolved_selected += 1
                    split_count += 1
            resolved_by_split.append(split_count)
        self.assertEqual(resolved_selected, 10)
        self.assertGreater(resolved_by_split[1], 0)
        self.assertGreater(resolved_by_split[2], 0)

    def test_multi_combo_exact_split_returns_400_50_50_for_all_combos(self) -> None:
        combos = [
            ["measurement_error", "parameter_error"],
            ["measurement_error", "topology_error"],
            ["measurement_error", "harmonic_anomaly"],
            ["parameter_error", "topology_error"],
            ["parameter_error", "harmonic_anomaly"],
            ["topology_error", "harmonic_anomaly"],
            ["measurement_error", "parameter_error", "topology_error"],
        ]
        samples = []
        total = sum(BALANCED_SPLIT_COUNTS.values())
        for combo in combos:
            for idx in range(total):
                samples.append(make_multi_sample(combo, idx))

        train, valid, test, meta = multi_combo_exact_split(samples, seed=42)

        self.assertEqual(meta["mode"], "multi_combo_exact")
        for split, expected_count in (
            (train, BALANCED_SPLIT_COUNTS["train"]),
            (valid, BALANCED_SPLIT_COUNTS["valid"]),
            (test, BALANCED_SPLIT_COUNTS["test"]),
        ):
            distribution = combo_distribution(split)
            self.assertEqual(len(distribution), len(combos))
            for count in distribution.values():
                self.assertEqual(count, expected_count)

        qa = multi_label_qa(train + valid + test)
        self.assertEqual(qa["invalid_family_lists"], 0)
        self.assertEqual(qa["missing_applied_tools"], 0)
        self.assertEqual(qa["missing_suspect_locations"], 0)
        self.assertEqual(qa["family_count_distribution"]["3"], sum(BALANCED_SPLIT_COUNTS.values()))


if __name__ == "__main__":
    unittest.main()
