"""LLM leg of the triage benchmark: prompts and truth-derived first actions for a classification fine-tune.

    python -m research.classifier_triage.llm_dataset --output-dir output/classifier_triage_20261004/llm --variant prompt_top5

Each row is the agent's decision right after the opening WLS of an alarmed
state, rendered exactly as the DAgger pipeline renders it (the canonical
tool surface, the compacted model view), but with no screen report: the
environment runs under ``wls_gated_diagnostics``, where no balanced screen
runs, and the contract paragraph of the system prompt is replaced by the
triage one (``triage_system_prompt``; the scorer installs the same text, so
training and scoring read identical prompts).  The target is the
truth-derived first action (decision T1):
request phase-resolved measurements when the truth holds an HIF, an
unbalance or a harmonic source, else the context of the balanced family to
investigate first.

Two prompt variants, because what the prompt shows bounds what the LLM can
decide (docs/classifier_triage_plan_20261004.md, section 7):

* ``prompt_top5``: today's WLS summary, the five largest residual
  magnitudes and the five largest multipliers;
* ``prompt_top10_signed``: the ten largest residuals with their signs.

Files: ``train.jsonl`` and ``validation.jsonl`` (chat rows for
``python -m psse_env.sft research-train``; validation is the tenth of the
train parents the GNN also holds out), ``score.jsonl`` (calibration, test
and probe prompts without a target) and ``score_labels.json``.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research.classifier_triage import data  # noqa: E402

VARIANTS = {"prompt_top5": (5, False), "prompt_top10_signed": (10, True)}
#: The profile the observations are generated and scored under: balanced SCADA and WLS, no screen.
TRIAGE_PROFILE = "wls_gated_diagnostics"
TRIAGE_PROMPT_PARAGRAPH = (
    f" Evidence profile: {TRIAGE_PROFILE}, triage benchmark. Start with WLS on the configured "
    "balanced-network model, the observed SCADA voltage magnitudes and P/Q "
    "injections/flows and the declared sensor noise; no fault flags, screens or "
    "precomputed diagnoses are provided. After a WLS alarm, decide from the WLS "
    "result itself: request phase-resolved PMU phasors (get_three_phase_context) "
    "when the residual pattern points at a high-impedance fault, a load unbalance "
    "or harmonic distortion, which no balanced correction can repair; otherwise "
    "investigate the balanced error family the residuals and branch multipliers "
    "point at (a meter, a branch parameter, or a branch status). Every substation "
    "can supply phasors and spectra, and each request has a cost, so request them "
    "only when the balanced evidence calls for it. Harmonic spectra follow phasors "
    "that came back balanced. run_alternative_test and the multi-scan HIF estimator "
    "are unavailable. If supported recovery cannot resolve the discrepancy, request "
    "operator review without inventing a fault-family diagnosis."
)
def triage_system_prompt() -> str:
    """The system prompt of every triage row (canonical preamble plus the triage contract)."""
    from psse_env.dagger.dataset_builder import CANONICAL_DAGGER_SYSTEM_PROMPT

    return CANONICAL_DAGGER_SYSTEM_PROMPT + TRIAGE_PROMPT_PARAGRAPH


FIRST_ACTION = {
    "request": "get_three_phase_context", "measurement": "get_measurement_context",
    "parameter": "get_parameter_context", "topology": "get_topology_context",
}
MAX_STEPS = 40
_ENVIRONMENT = None


def target_class(triage: Mapping[str, Any]) -> str:
    """The truth-derived first decision: request phasors, else the first balanced family (meter by default)."""
    if triage["needs_aux"]:
        return "request"
    return triage["first"] or "measurement"


def _environment():
    global _ENVIRONMENT
    if _ENVIRONMENT is None:
        import scripts.run_dagger_research as research
        from psse_env.evidence_profile import WLS_GATED_PROFILE

        research.RESEARCH_ENVIRONMENT_OPTIONS["evidence_profile"] = WLS_GATED_PROFILE
        research.RESEARCH_ENVIRONMENT_OPTIONS["normalized_residual_threshold"] = 4.0
        _ENVIRONMENT = research.resolve_environment_factory("research", WLS_GATED_PROFILE)()
    return _ENVIRONMENT


def _summary_residuals(payload: Mapping[str, Any], case: str, top_k: int, signed: bool) -> list[dict[str, Any]]:
    """The summary's residual list in the provider's format, with ``top_k`` entries and optional signs."""
    from research.classifier_triage.features import BLOCKS, network

    net = network(case)
    nb, nl = int(net["nb"]), int(net["nl"])
    edges = np.cumsum([0, nb, nb, nb, nl, nl, nl, nl])
    residual = np.asarray(payload["signed_normalized_residual"], dtype=float)
    listed = []
    for index in np.argsort(-np.abs(residual))[:top_k]:
        if abs(residual[index]) < 3.0:  # the provider lists residuals at or above its evidence threshold
            break
        block = int(np.searchsorted(edges, index, side="right") - 1)
        value = float(residual[index]) if signed else float(abs(residual[index]))
        listed.append({"channel": BLOCKS[block], "channel_offset": int(index - edges[block]), "index0": int(index),
                       "value": round(value, 4)})
    return listed


def _patch_summary(node: Any, residuals: list[dict[str, Any]]) -> None:
    """Replace every ``wls_summary.top_residuals`` in an observation tree."""
    if isinstance(node, dict):
        summary = node.get("wls_summary")
        if isinstance(summary, dict) and "top_residuals" in summary:
            summary["top_residuals"] = copy.deepcopy(residuals)
        for value in node.values():
            _patch_summary(value, residuals)
    elif isinstance(node, list):
        for value in node:
            _patch_summary(value, residuals)


def render_row(item: Mapping[str, Any]) -> dict[str, Any] | None:
    """One chat row: reset on the state, run the opening WLS, render the next decision without a screen."""
    from psse_env.dagger import dataset_builder
    from psse_env.state_store import policy_safe_copy

    top_k, signed = VARIANTS[item["variant"]]
    env = _environment()
    scenario = {"scenario_id": item["scenario_id"], "case": item["case"], "measurements": item["measurements"],
                "metadata": copy.deepcopy(item["metadata"])}
    try:
        state = env.reset(scenario)
        active = state["active_state_id"]
        action = {"tool": "run_wls", "arguments": {"state_id": active}}
        _, output = env.step(copy.deepcopy(action))
        if output.get("execution_status") != "success":
            return None
        history = [{"state_id": active, "action": policy_safe_copy(action), "tool_output": policy_safe_copy(output)}]
        observation = replace(env.get_policy_observation(history), remaining_budget=MAX_STEPS - 1).as_dict()
    except Exception as exc:  # an unloadable state: dropped and counted by the caller
        return {"id": item["id"], "error": f"{type(exc).__name__}: {str(exc)[:200]}"}
    if (top_k, signed) != VARIANTS["prompt_top5"]:
        _patch_summary(observation, _summary_residuals(item["payload"], item["case"], top_k, signed))
    target = {"tool": FIRST_ACTION[item["target"]], "arguments": {"state_id": active}}
    rows = dataset_builder.examples_to_chat_sft([{
        "example_id": f"triage_{item['id']}", "policy_observation": observation, "preferred_action": target,
        "scenario_id": item["scenario_id"], "scenario_family": item["family"],
    }], protocol="canonical")
    if not rows:
        return {"id": item["id"], "error": "no chat row rendered"}
    row = rows[0]
    system = row["messages"][0]["content"]
    if dataset_builder.WLS_GATED_PROMPT_PARAGRAPH not in system:
        return {"id": item["id"], "error": "unexpected system prompt"}
    row["messages"][0]["content"] = system.replace(dataset_builder.WLS_GATED_PROMPT_PARAGRAPH, TRIAGE_PROMPT_PARAGRAPH)
    if row["messages"][0]["content"] != triage_system_prompt():
        return {"id": item["id"], "error": "system prompt is not the triage prompt"}
    return {"id": item["id"], "messages": row["messages"], "tools": row["tools"], "metadata": row.get("metadata") or {},
            "alarm": bool((observation.get("fresh_context_evidence") or {}).get("wls", {}).get("chi_square_alarm")
                          or (observation.get("fresh_context_evidence") or {}).get("wls", {}).get("normalized_residual_alarm"))}


def _items(rows: Sequence[Mapping[str, Any]], variant: str) -> list[dict[str, Any]]:
    return [{
        "id": str(row["id"]), "variant": variant, "scenario_id": f"triage{index:05d}", "case": str(row["record"]["case"]),
        "measurements": [float(v) for v in row["record"]["measurements"]], "metadata": dict(row["record"].get("metadata") or {}),
        "family": row["family"], "payload": {"signed_normalized_residual": row["payload"]["signed_normalized_residual"]},
        "target": target_class(row["triage"]),
    } for index, row in enumerate(rows)]


def main(argv: Sequence[str] | None = None) -> int:
    from research.classifier_triage.train_gnn import in_validation

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-dir", default=str(data.DEFAULT_DATASET))
    parser.add_argument("--benchmark-dir", default="output/classifier_triage_20261004/ieee14", help="holds the probe files")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--variant", choices=sorted(VARIANTS), required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, help="rows per group, for a smoke run")
    args = parser.parse_args(argv)
    out = Path(args.output_dir) / args.variant
    out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    rows = data.load_rows(Path(args.dataset_dir), workers=args.workers)
    split = data.split_rows(rows)
    bench = Path(args.benchmark_dir)
    probe: list[dict[str, Any]] = []
    for name in ("probe.jsonl", "probe_vm_large.jsonl", "probe_vm_small.jsonl"):
        if (bench / name).is_file():
            with (bench / name).open(encoding="utf-8") as stream:
                probe += data.probe_rows(json.loads(line) for line in stream if line.strip())
    probe = [row for row in data.attach_payloads(probe, bench / "probe_payloads.pkl", workers=args.workers) if row["split"] != "train"]
    groups = {
        "train": [r for r in split["train"] if not in_validation(str(r["parent"]))],
        "validation": [r for r in split["train"] if in_validation(str(r["parent"]))],
        "score": split["calibration"] + split["test"] + probe,
    }
    if args.limit:
        groups = {name: group[:args.limit] for name, group in groups.items()}
    summary: dict[str, Any] = {"variant": args.variant, "profile": TRIAGE_PROFILE, "groups": {}}
    for name, group in groups.items():
        items = _items(group, args.variant)
        if args.workers > 1 and len(items) > 32:
            with ProcessPoolExecutor(max_workers=args.workers) as pool:
                rendered = list(pool.map(render_row, items, chunksize=16))
        else:
            rendered = [render_row(item) for item in items]
        by_id = {str(row["id"]): row for row in group}
        kept, errors, labels = [], Counter(), []
        for item, result in zip(items, rendered):
            if result is None or result.get("error"):
                errors[(result or {}).get("error", "wls_failed")[:60]] += 1
                continue
            source = by_id[item["id"]]
            if name == "score":  # the prompt only: the target stays out of the scored file
                kept.append({"id": item["id"], "messages": result["messages"][:2], "tools": result["tools"]})
                labels.append({"id": item["id"], "split": source["split"], "kind": source["kind"], "family": source["family"],
                               "parent": str(source["parent"]), "needs_aux": int(source["triage"]["needs_aux"]),
                               "families": list(source["triage"]["families"]), "first": source["triage"]["first"],
                               "target": item["target"], "probe_kind": source.get("probe_kind"),
                               "alarm_in_environment": result["alarm"]})
            else:
                # The trainer needs the canonical-protocol metadata of the rendered row and a physical root
                # that is disjoint between train and validation: the study's physical parent.
                kept.append({"example_id": f"triage_{item['id']}", "messages": result["messages"], "tools": result["tools"],
                             "metadata": {**result["metadata"], "scenario_family": source["family"], "target": item["target"]},
                             "physical_root_fingerprint": f"triage_parent:{source['parent']}",
                             "scenario_family": source["family"], "evidence_profile": TRIAGE_PROFILE})
        with (out / f"{name}.jsonl").open("w", encoding="utf-8") as stream:
            for row in kept:
                stream.write(json.dumps(row, sort_keys=True) + "\n")
        if name == "score":
            (out / "score_labels.json").write_text(json.dumps(labels), encoding="utf-8")
        characters = [sum(len(json.dumps(m)) for m in row["messages"]) + len(json.dumps(row["tools"])) for row in kept]
        summary["groups"][name] = {
            "rows": len(kept), "dropped": dict(errors), "targets": dict(Counter(item["target"] for item in items)),
            "characters_median": int(np.median(characters)) if characters else 0,
            "characters_max": int(max(characters)) if characters else 0,
        }
        print(f"[llm] {args.variant} {name}: {len(kept)} rows, dropped {dict(errors)}, targets {summary['groups'][name]['targets']}", flush=True)
    if not args.limit:
        # The trainer's own input gates, run here so a cluster job cannot fail on them.
        from psse_env.sft.gates import load_jsonl
        from psse_env.sft.research_cli import validate_research_splits
        from psse_env.sft.research_rows import normalize_research_rows

        train_rows, validation_rows = load_jsonl(out / "train.jsonl"), load_jsonl(out / "validation.jsonl")
        summary["trainer_gates"] = {"splits": validate_research_splits(train_rows, validation_rows)}
        for name, loaded in (("train", train_rows), ("validation", validation_rows)):
            _, report = normalize_research_rows(loaded, source_label=name)
            summary["trainer_gates"][name] = {"rows": report["rows"]}
        print(f"[llm] trainer input gates passed: {summary['trainer_gates']['splits']}", flush=True)
    summary["seconds"] = time.perf_counter() - started
    (out / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(f"[llm] wrote {out} in {summary['seconds']:.0f} s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
