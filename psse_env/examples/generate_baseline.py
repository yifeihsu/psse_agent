from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any

from psse_env import TransactionalPSSEEnv
from psse_env.dagger import DaggerRolloutCollector, examples_to_chat_sft, write_jsonl
from psse_env.oracle import ExpertPolicyOracle


SEED = 42


class DeterministicPolicy:
    def act(self, observation: dict[str, Any]) -> dict[str, Any]:
        return {
            "tool": "run_wls",
            "arguments": {"state_id": observation["active_state_id"]},
        }


def scenario() -> dict[str, Any]:
    return {
        "scenario_id": "deterministic_case14_seed42",
        "case": {"name": "case14", "branch": [{"x": 0.1}]},
        "measurements": [1.2, 2.0, 3.0],
        "clean_case": {"name": "case14", "branch": [{"x": 0.1}]},
        "clean_measurements": [1.0, 2.0, 3.0],
        "true_measurement_errors": [{"index": 0, "observed": 1.2, "clean": 1.0}],
        "true_parameter_errors": [],
        "true_topology_errors": [],
        "oracle_action_hints": [
            {
                "tool": "correct_measurements",
                "arguments": {"state_id": "legacy:s0", "measurement_updates": {0: 1.0}},
            }
        ],
    }


def generate(output_dir: Path) -> list[dict[str, Any]]:
    collector = DaggerRolloutCollector(
        env=TransactionalPSSEEnv(max_steps=6),
        policy=DeterministicPolicy(),
        expert_oracle=ExpertPolicyOracle(),
        rng=random.Random(SEED),
    )
    rows = collector.collect_iteration(
        scenarios=[scenario()],
        iteration=0,
        beta=1.0,
        max_steps=6,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "sample_rollout.jsonl", rows)
    write_jsonl(output_dir / "dagger_example.jsonl", rows[:1])
    write_jsonl(output_dir / "chat_sft_example.jsonl", examples_to_chat_sft(rows[:1]))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate deterministic DAgger baseline fixtures.")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    rows = generate(args.output_dir)
    print(f"wrote {len(rows)} deterministic rollout steps to {args.output_dir}")


if __name__ == "__main__":
    main()
