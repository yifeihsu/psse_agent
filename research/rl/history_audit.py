"""Fidelity and size of the complete-history observation mode (workplan of 2026-11-10, stage 1).

    python -m research.rl.history_audit run --suite output/compound_rl_20261107/ieee14_dev300.json \\
        --per-class 3 --output-dir output/rl_improvement_20261110/history_audit --workers 6 \\
        --env-option lifecycle_choice=1 --env-option detailed_evidence=1 --env-option pair_refit_adjacent=1 \\
        --env-option flow_at_noise_cue=1 --env-option route_comparison=0 \\
        --expert-option abstain_isolated_lines=1 --expert-option state_ambiguity=1
    python -m research.rl.history_audit trajectories --store out/ieee14_v1/iteration_000/trajectories \\
        --suite data/suites/ieee14_train.json --output-dir out/history_audit_s0 --tokenizer google/gemma-4-12B-it \\
        --revision 707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7

``run`` rolls the compound study's teacher out under ``complete_history_v1``
on a few roots per class (CPU; the teacher reads the bounded window in
either mode, so its actions are those of the bounded run).  ``trajectories``
reads the episodes a rollout store already holds (the learner's own states).
Both render every policy step in the two observation modes and report

* **fidelity**, per step and per earlier event: every transition present,
  each event's evidence equal to what the model was shown when that output
  was the latest one, case aliases bound to the current state and stable for
  the episode, every candidate's fate listed, no history-size marker;
* **size**, by compound class and action position: median, 95th percentile
  and maximum of the prompt in both modes, and the input total of an
  episode.

Sizes are characters of the system prompt, the tool schemas and the state
message, and tokens by the best counter the run is given:

* ``--tokenizer`` (the model's processor, on the cluster): the prompt as the
  policy renders and tokenizes it, the count the input limit applies to;
* ``--tokenizer-file`` (a ``tokenizer.json`` of the model family): the three
  texts tokenized as they are, without the chat template.  On the 8,866
  cloning rows of 2026-11-06 this count has a median of 5,894 and a maximum
  of 13,189 against 5,708 and 12,987 from the processor on the cluster, so
  it runs 2 to 3% above the real prompt;
* neither: an estimate from the characters (:data:`CHARS_PER_TOKEN`).

The report says which one it used.

The teacher's episodes stand in for the student's lengths only as far as the
student follows the teacher; the token gate of the workplan is the
``trajectories`` run on the learner's own rollouts with the real tokenizer.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from psse_env.actions import COMMIT_STATE, CORRECT_MEASUREMENTS, CORRECT_PARAMETERS, CORRECT_TOPOLOGY, ROLLBACK_STATE  # noqa: E402
from psse_env.dagger.dataset_builder import (  # noqa: E402
    CANONICAL_DAGGER_SYSTEM_PROMPT,
    NEWEST_EVENT_EVIDENCE,
    _compact_last_tool_output,
    alias_model_visible_state,
    expand_tables,
    prepare_model_policy_observation,
    system_prompt_for_observation,
    tool_schemas_for_observation,
    without_controller_fingerprints,
)
from psse_env.dagger.observation_modes import (  # noqa: E402
    BOUNDED_HISTORY,
    CANDIDATE_EXPERIMENTS_KEY,
    COMPLETE_HISTORY,
    HISTORY_RECORD_KEY,
    OBSERVABLE_HISTORY_KEY,
    observable_history_from_trace,
    with_observable_history,
)

AUDIT_CONTRACT = "complete_history_audit_v1"
#: Characters per prompt token, for the estimate a run without a tokenizer reports: the median over the 8,866
#: cloning rows of 2026-11-06 under a Gemma 4 tokenizer file (3.15), rounded down so the estimate errs high.
CHARS_PER_TOKEN = 3.1
TOKEN_BASES = {"processor": "the model's processor on the rendered prompt",
               "tokenizer_file": "a tokenizer file on the prompt's texts, without the chat template (2 to 3% above the processor's count)",
               "estimate": f"estimated from characters at {CHARS_PER_TOKEN} per token"}
CORRECTIONS = frozenset({CORRECT_MEASUREMENTS, CORRECT_PARAMETERS, CORRECT_TOPOLOGY})
SIZE_MARKERS = ("events_omitted_before", "history omitted for size")
WLS_TABLE_KEYS = ("bus_table", "branch_table")


# --------------------------------------------------------------------------- measuring a prompt


class PromptMeasurer:
    """Characters of a model-visible view's prompt, and its tokens when a processor is given."""

    def __init__(self, tokenizer: str | None = None, *, revision: str | None = None, local_files_only: bool = True,
                 architecture: str = "gemma4_unified", tokenizer_file: str | None = None) -> None:
        self.tokenizer = tokenizer
        self.policy = None
        self.raw = None
        self._tools = None
        self._raw_cache: dict[str, int] = {}
        self.basis = "processor" if tokenizer else "tokenizer_file" if tokenizer_file else "estimate"
        if tokenizer_file and not tokenizer:
            from tokenizers import Tokenizer

            self.raw = Tokenizer.from_file(str(tokenizer_file))
        if tokenizer:
            from transformers import AutoProcessor

            from psse_env.dagger.release_factories import _ModelBundle
            from psse_env.dagger.research_policy_factory import _CanonicalResearchNativePolicy

            processor = AutoProcessor.from_pretrained(tokenizer, revision=revision, local_files_only=local_files_only)
            bundle = _ModelBundle(model=None, processor=processor, model_id=str(tokenizer), model_revision=str(revision or "local"),
                                  adapter_snapshot_path="")
            self.policy = _CanonicalResearchNativePolicy(bundle, architecture=architecture)

    def _schemas(self) -> list[dict[str, Any]]:
        if self._tools is None:
            from psse_env.dagger.preliminary_e2b_eval import canonical_prompt_tool_schemas

            self._tools = canonical_prompt_tool_schemas()
        return self._tools

    def measure(self, view: Mapping[str, Any]) -> dict[str, Any]:
        state = json.dumps({"state": view}, sort_keys=True, allow_nan=False)
        system = system_prompt_for_observation(CANONICAL_DAGGER_SYSTEM_PROMPT, view)
        tools = json.dumps(tool_schemas_for_observation(self._schemas(), view), sort_keys=True)
        chars = len(state) + len(system) + len(tools)
        if self.policy is not None:
            rendered, _tools, _parameters = self.policy.render_prompt(view)
            tokens = int(self.policy.prompt_token_ids(rendered)[1])  # the length before any cut
        elif self.raw is not None:
            tokens = self._raw_tokens(state, cache=False) + self._raw_tokens(system) + self._raw_tokens(tools)
        else:
            tokens = int(math.ceil(chars / CHARS_PER_TOKEN))
        return {"chars": chars, "state_chars": len(state), "tokens": tokens}

    def _raw_tokens(self, text: str, *, cache: bool = True) -> int:
        if cache and text in self._raw_cache:
            return self._raw_cache[text]
        count = len(self.raw.encode(text, add_special_tokens=False).ids)
        if cache:
            self._raw_cache[text] = count
        return count


# --------------------------------------------------------------------------- the audit of one episode


def _policy_positions(trace: Sequence[Mapping[str, Any]]) -> list[int]:
    return [index for index, row in enumerate(trace)
            if isinstance(row, Mapping) and not row.get("intervention") and isinstance(row.get("policy_observation"), Mapping)]


def _bounded_observation(row: Mapping[str, Any]) -> dict[str, Any]:
    observation = copy.deepcopy(dict(row["policy_observation"]))
    observation.pop(OBSERVABLE_HISTORY_KEY, None)
    return observation


def _target_state(event: Mapping[str, Any]) -> Any:
    """The case a transition's tool call was made on: the one its arguments name, else the active case at the time."""

    arguments = (event.get("action") or {}).get("arguments") if isinstance(event.get("action"), Mapping) else {}
    arguments = arguments if isinstance(arguments, Mapping) else {}
    return arguments.get("state_id") or arguments.get("candidate_state_id") or event.get("state_id")


def audit_episode(trace: Sequence[Mapping[str, Any]], measurer: PromptMeasurer | None = None) -> dict[str, Any]:
    """Render every policy step of an episode in both modes; check the complete view and measure both."""

    measurer = measurer or PromptMeasurer()
    events = observable_history_from_trace(trace)
    raw_views = [_compact_last_tool_output(without_controller_fingerprints(event.get("tool_output") or {})) for event in events]
    steps: list[dict[str, Any]] = []
    problems: list[dict[str, Any]] = []
    historical_alias: dict[str, set[str]] = defaultdict(set)

    def problem(kind: str, position: int, **detail: Any) -> None:
        problems.append({"kind": kind, "trace_position": position, **detail})

    for position in _policy_positions(trace):
        bounded = _bounded_observation(trace[position])
        complete = with_observable_history(bounded, events[:position])
        view_b, _control_b = prepare_model_policy_observation(bounded)
        view_c, control_c = prepare_model_policy_observation(complete)
        id_of = dict(control_c["state_aliases"])
        alias_of = {identifier: alias for alias, identifier in id_of.items()}
        history = view_c.get("history_window") or []
        record = view_c.get(HISTORY_RECORD_KEY) or {}

        # Every earlier transition is there, in order, and the marker says so.
        if len(history) != position or record.get("events") != position or record.get("contract") != COMPLETE_HISTORY:
            problem("events_missing", position, shown=len(history), expected=position)
        if [event.get("index") for event in history] != list(range(len(history))):
            problem("events_out_of_order", position)
        text = json.dumps(history, sort_keys=True)
        if any(marker in text for marker in SIZE_MARKERS):
            problem("size_marker_in_history", position)

        # Each event carries what the model was shown when its output was the latest one: written as tables,
        # which expand back to the same records, and for the newest event by reference to the last tool output.
        for index, event in enumerate(history[:position]):
            expected = alias_model_visible_state(raw_views[index].get("observable_metrics") or {}, id_of,
                                                 episode_aliases=control_c["episode_aliases"],
                                                 hash_aliases=control_c["hash_aliases"])
            shown = event.get("observable_metrics") or {}
            if index == position - 1 and expected:
                if shown != NEWEST_EVENT_EVIDENCE or ((view_c.get("last_tool_output") or {}).get("observable_metrics") or {}) != expected:
                    problem("newest_event_evidence_differs", position, event=index)
            elif expand_tables(shown) != expected:
                problem("event_evidence_differs", position, event=index)

        # The newest event is the last tool output the bounded view shows at this step (same compaction, own aliases).
        if position:
            shown = (view_b.get("last_tool_output") or {}).get("observable_metrics") or {}
            kept = (view_c.get("last_tool_output") or {}).get("observable_metrics") or {}
            if json.dumps(shown, sort_keys=True) != json.dumps(kept, sort_keys=True):
                # The two views alias a state by their own maps; compare through the raw ids.
                if _dealias(shown, _control_b["state_aliases"], _control_b["hash_aliases"]) != _dealias(
                        kept, control_c["state_aliases"], control_c["hash_aliases"]):
                    problem("last_tool_output_differs_between_modes", position)

        # Case aliases: evidence on the active case reads "active", on the open candidate "candidate";
        # an earlier case keeps one sN for the whole episode.
        active, candidate = bounded.get("active_state_id"), bounded.get("candidate_state_id")
        for index, raw in enumerate(events[:position]):
            target = _target_state(raw)
            if target is None:
                continue
            alias = alias_of.get(str(target))
            wanted = "active" if target == active else "candidate" if target == candidate else None
            if wanted is not None and alias != wanted:
                problem("current_case_alias_wrong", position, event=index, alias=alias, wanted=wanted)
            if wanted is None:
                if alias is None or not alias.startswith("s"):
                    problem("earlier_case_without_alias", position, event=index, alias=alias)
                else:
                    historical_alias[str(target)].add(alias)

        # Every successful correction is a listed experiment whose fate matches the trace so far.
        experiments = view_c.get(CANDIDATE_EXPERIMENTS_KEY) or []
        expected_fates = _candidate_fates(events[:position])
        if [item.get("disposition") for item in experiments] != [fate for _index, fate in expected_fates]:
            problem("candidate_experiments_differ", position, shown=[item.get("disposition") for item in experiments],
                    expected=[fate for _index, fate in expected_fates])

        size_b, size_c = measurer.measure(view_b), measurer.measure(view_c)
        tables = sum(1 for event in history
                     if any(key in (expand_tables(event.get("observable_metrics") or {}).get("wls_summary") or {})
                            for key in WLS_TABLE_KEYS))
        steps.append({
            "trace_position": position,
            "tool": str((trace[position].get("action") or {}).get("tool")),
            "events": len(history),
            "events_bounded": len(view_b.get("history_window") or []),
            "events_with_wls_tables": tables,
            "bounded": size_b,
            "complete": size_c,
        })
    unstable = {identifier: sorted(aliases) for identifier, aliases in historical_alias.items() if len(aliases) > 1}
    for identifier, aliases in unstable.items():
        problems.append({"kind": "earlier_case_alias_changed", "aliases": aliases})
    return {"steps": steps, "problems": problems, "events": len(events)}


def _dealias(value: Any, state_aliases: Mapping[str, str], hash_aliases: Mapping[str, str]) -> Any:
    """``value`` with every local alias string replaced by what it stands for (for comparing two views of one output)."""

    table = {**{str(alias): str(identifier) for alias, identifier in hash_aliases.items()},
             **{str(alias): str(identifier) for alias, identifier in state_aliases.items()}}

    def visit(item: Any) -> Any:
        if isinstance(item, Mapping):
            return {str(key): visit(child) for key, child in item.items()}
        if isinstance(item, (list, tuple)):
            return [visit(child) for child in item]
        if isinstance(item, str):
            return table.get(item, item)
        return item

    return visit(value)


def _candidate_fates(events: Sequence[Mapping[str, Any]]) -> list[tuple[int, str]]:
    """``(event index, fate)`` of every candidate a successful correction opened, read from the raw transitions."""

    fates: list[list[Any]] = []
    open_candidate: Any = None
    for index, event in enumerate(events):
        action = event.get("action") if isinstance(event.get("action"), Mapping) else {}
        tool = str(action.get("tool"))
        output = event.get("tool_output") if isinstance(event.get("tool_output"), Mapping) else {}
        if output.get("execution_status") != "success":
            continue
        if tool in CORRECTIONS and output.get("candidate_state_id") is not None:
            open_candidate = output.get("candidate_state_id")
            fates.append([index, "open"])
        elif tool in (COMMIT_STATE, ROLLBACK_STATE) and open_candidate is not None and fates:
            fates[-1][1] = "committed" if tool == COMMIT_STATE else "rolled_back"
            open_candidate = None
    return [(int(index), str(fate)) for index, fate in fates]


# --------------------------------------------------------------------------- aggregation


def _quantiles(values: Sequence[float]) -> dict[str, Any]:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return {"n": 0, "median": None, "p95": None, "max": None}

    def at(q: float) -> float:
        return ordered[min(len(ordered) - 1, int(math.ceil(q * len(ordered))) - 1)]

    return {"n": len(ordered), "median": at(0.5), "p95": at(0.95), "max": ordered[-1]}


def _tokens(size: Mapping[str, Any]) -> int:
    return int(size["tokens"])


def summarize(records: Sequence[Mapping[str, Any]], *, limit: int, basis: str) -> dict[str, Any]:
    """Sizes by class and action position, the fidelity findings, and what exceeds the usable prompt limit."""

    by_class: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    by_position: dict[int, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    overall: dict[str, list[float]] = defaultdict(list)
    episode_totals: dict[str, list[float]] = defaultdict(list)
    problems: dict[str, int] = defaultdict(int)
    over_limit = {"steps": 0, "episodes": 0, "roots": []}
    steps = 0
    for record in records:
        name = str(record.get("class"))
        totals: dict[str, float] = defaultdict(float)
        exceeded = False
        for number, step in enumerate(record["steps"]):
            steps += 1
            for mode in ("bounded", "complete"):
                tokens, chars = _tokens(step[mode]), step[mode]["chars"]
                for bucket in (by_class[name], by_position[number], overall):
                    bucket[f"{mode}_tokens"].append(tokens)
                    bucket[f"{mode}_chars"].append(chars)
                totals[f"{mode}_tokens"] += tokens
            if _tokens(step["complete"]) > limit:
                over_limit["steps"] += 1
                exceeded = True
        for key, value in totals.items():
            episode_totals[key].append(value)
        if exceeded:
            over_limit["episodes"] += 1
            over_limit["roots"].append(record.get("scenario_id"))
        for item in record["problems"]:
            problems[str(item["kind"])] += 1

    def table(buckets: Mapping[Any, Mapping[str, Sequence[float]]]) -> dict[str, Any]:
        return {str(key): {name: _quantiles(values) for name, values in sorted(bucket.items())}
                for key, bucket in sorted(buckets.items(), key=lambda item: (isinstance(item[0], str), item[0]))}

    return {
        "contract": AUDIT_CONTRACT,
        "episodes": len(records),
        "policy_steps": steps,
        "token_counts": TOKEN_BASES[basis],
        "token_basis": basis,
        "usable_prompt_limit": int(limit),
        "over_limit": over_limit,
        "fidelity": {"problems": dict(sorted(problems.items())), "episodes_with_problems":
                     sum(1 for record in records if record["problems"]), "passed": not problems},
        "overall": {name: _quantiles(values) for name, values in sorted(overall.items())},
        "episode_input_tokens": {name: _quantiles(values) for name, values in sorted(episode_totals.items())},
        "by_class": table(by_class),
        "by_action_position": table(by_position),
    }


def markdown(summary: Mapping[str, Any], *, title: str) -> str:
    def row(name: str, entry: Mapping[str, Any]) -> str:
        b, c = entry.get("bounded_tokens") or {}, entry.get("complete_tokens") or {}
        return (f"| {name} | {b.get('n')} | {_fmt(b.get('median'))} | {_fmt(b.get('p95'))} | {_fmt(b.get('max'))} | "
                f"{_fmt(c.get('median'))} | {_fmt(c.get('p95'))} | {_fmt(c.get('max'))} |")

    head = ["| | steps | bounded median | bounded p95 | bounded max | complete median | complete p95 | complete max |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    lines = [f"# {title}", "",
             f"{summary['episodes']} episodes, {summary['policy_steps']} policy steps. Prompt tokens: {summary['token_counts']}. "
             f"Usable prompt limit {summary['usable_prompt_limit']} tokens: {summary['over_limit']['steps']} steps in "
             f"{summary['over_limit']['episodes']} episodes are above it in the complete mode.", "",
             f"Fidelity: {'passed' if summary['fidelity']['passed'] else 'FAILED'} "
             f"({summary['fidelity']['episodes_with_problems']} episodes with a finding; {summary['fidelity']['problems'] or 'none'}).", "",
             "## Prompt tokens", "", *head, row("all steps", summary["overall"]), "",
             "## By compound class", "", *head, *[row(name, entry) for name, entry in summary["by_class"].items()], "",
             "## By action position (0 is the episode's first action)", "", *head,
             *[row(name, entry) for name, entry in summary["by_action_position"].items()], "",
             "## Input tokens of a whole episode", "",
             "| mode | episodes | median | p95 | max |", "| --- | ---: | ---: | ---: | ---: |"]
    for mode in ("bounded", "complete"):
        entry = summary["episode_input_tokens"].get(f"{mode}_tokens") or {}
        lines.append(f"| {mode} | {entry.get('n')} | {_fmt(entry.get('median'))} | {_fmt(entry.get('p95'))} | {_fmt(entry.get('max'))} |")
    return "\n".join(lines) + "\n"


def _fmt(value: Any) -> str:
    return "" if value is None else f"{float(value):,.0f}"


# --------------------------------------------------------------------------- running


_WORKER: dict[str, Any] = {}


def _init_worker(settings: Mapping[str, Any], measure: Mapping[str, Any]) -> None:
    from research.rl import decision_audit as audit

    audit._init_worker(settings)
    audit._HARNESS.contract["observation_mode"] = COMPLETE_HISTORY
    _WORKER.update(harness=audit._HARNESS, measurer=PromptMeasurer(**measure))


def _run_root(root: Mapping[str, Any]) -> dict[str, Any]:
    harness = _WORKER["harness"]
    scenario_id = str(root["execution"]["scenario_id"])
    name = ((root.get("audit") or {}).get("compound") or {}).get("class")
    started = time.perf_counter()
    try:
        episode = harness.run(root, harness.expert_policy())
        record = audit_episode(episode.get("trace") or [], _WORKER["measurer"])
        record.update(success=bool(episode.get("truth_audited_task_success")), actions=int(episode.get("steps") or 0))
    except Exception as exc:  # keep the sweep going; the record names the failure
        record = {"steps": [], "problems": [{"kind": "episode_error", "detail": f"{type(exc).__name__}: {exc}"}], "events": 0}
    record.update(scenario_id=scenario_id, seconds=round(time.perf_counter() - started, 1))
    record["class"] = name
    return record


def _sample(roots: Sequence[Mapping[str, Any]], per_class: int | None, wanted: Sequence[str] | None) -> list[Mapping[str, Any]]:
    if wanted:
        names = set(wanted)
        roots = [root for root in roots if str(root["execution"]["scenario_id"]) in names]
    if not per_class:
        return list(roots)
    taken: dict[Any, int] = defaultdict(int)
    kept = []
    for root in roots:
        name = ((root.get("audit") or {}).get("compound") or {}).get("class")
        if taken[name] < per_class:
            taken[name] += 1
            kept.append(root)
    return kept


def _usable_limit(explicit: int | None) -> int:
    if explicit:
        return int(explicit)
    from psse_env.dagger.research_policy_factory import usable_prompt_tokens

    return usable_prompt_tokens()


def _write(out: Path, records: Sequence[Mapping[str, Any]], summary: Mapping[str, Any], title: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    with (out / "episodes.jsonl").open("w", encoding="utf-8", newline="\n") as stream:
        for record in records:
            stream.write(json.dumps(record, sort_keys=True, default=str) + "\n")
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    (out / "report.md").write_text(markdown(summary, title=title), encoding="utf-8", newline="\n")


def run(args: argparse.Namespace) -> int:
    from psse_env.oracle.expert_variants import compound_study_expert_variant
    from research.compound_errors import rollout
    from research.rl.decision_audit import parse_run_options

    roots = _sample(rollout.load_suite(args.suite), args.per_class, args.roots)
    system = rollout.suite_system(roots, args.system)
    settings = dict(profile=args.profile, expert_variant=args.expert or compound_study_expert_variant(system), triage_model=None,
                    progress_basis=rollout.system_progress_basis(system, args.progress_basis), max_steps=args.max_steps,
                    seed=args.seed, reward=None, env_options=parse_run_options(args.env_option),
                    expert_options=parse_run_options(args.expert_option))
    measure = dict(tokenizer=args.tokenizer, revision=args.revision, local_files_only=not args.allow_download,
                   tokenizer_file=str(args.tokenizer_file) if args.tokenizer_file else None)
    basis = "processor" if args.tokenizer else "tokenizer_file" if args.tokenizer_file else "estimate"
    started = time.perf_counter()
    records: list[dict[str, Any]] = []
    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker, initargs=(settings, measure)) as pool:
            for record in pool.map(_run_root, roots, chunksize=1):
                records.append(record)
                if len(records) % 10 == 0 or len(records) == len(roots):
                    print(json.dumps({"done": len(records), "of": len(roots), "elapsed": round(time.perf_counter() - started, 1)}), flush=True)
    else:
        _init_worker(settings, measure)
        for root in roots:
            records.append(_run_root(root))
            print(json.dumps({"root": records[-1]["scenario_id"], "class": records[-1]["class"], "steps": len(records[-1]["steps"]),
                              "problems": len(records[-1]["problems"])}), flush=True)
    summary = summarize(records, limit=_usable_limit(args.limit), basis=basis)
    summary["source"] = {"kind": "teacher rollouts under complete_history_v1", "suite": str(args.suite), "roots": len(roots),
                         "settings": {key: value for key, value in settings.items() if key != "reward"},
                         "seconds": round(time.perf_counter() - started, 1)}
    _write(args.output_dir, records, summary, "Complete-history observation: fidelity and size (teacher rollouts)")
    print(markdown(summary, title="Complete-history observation: fidelity and size (teacher rollouts)"), flush=True)
    return 0 if summary["fidelity"]["passed"] else 1


def trajectories(args: argparse.Namespace) -> int:
    from psse_env.rl.rollout import TrajectoryStore
    from research.compound_errors import rollout

    classes = {}
    if args.suite:
        classes = {str(root["execution"]["scenario_id"]): ((root.get("audit") or {}).get("compound") or {}).get("class")
                   for root in rollout.load_suite(args.suite)}
    measurer = PromptMeasurer(tokenizer=args.tokenizer, revision=args.revision, local_files_only=not args.allow_download,
                              tokenizer_file=str(args.tokenizer_file) if args.tokenizer_file else None)
    records: list[dict[str, Any]] = []
    for root, group in TrajectoryStore(args.store).iter_groups():
        for trajectory in group[: args.samples_per_root] if args.samples_per_root else group:
            record = audit_episode(trajectory.evaluation.get("trace") or [], measurer)
            record.update(scenario_id=root, sample_index=trajectory.sample_index,
                          success=bool(trajectory.evaluation.get("truth_audited_task_success")),
                          actions=int(trajectory.evaluation.get("steps") or 0))
            record["class"] = classes.get(root)
            records.append(record)
    summary = summarize(records, limit=_usable_limit(args.limit), basis=measurer.basis)
    summary["source"] = {"kind": "stored learner trajectories", "store": str(args.store), "suite": str(args.suite) if args.suite else None}
    _write(args.output_dir, records, summary, "Complete-history observation: fidelity and size (stored trajectories)")
    print(markdown(summary, title="Complete-history observation: fidelity and size (stored trajectories)"), flush=True)
    return 0 if summary["fidelity"]["passed"] else 1


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)

    def common(sub: argparse.ArgumentParser) -> None:
        sub.add_argument("--output-dir", required=True, type=Path)
        sub.add_argument("--tokenizer", default=None, help="model id or path of the processor that counts tokens")
        sub.add_argument("--tokenizer-file", default=None, type=Path,
                         help="a tokenizer.json of the model family, for counts without the chat template (omit both: an estimate)")
        sub.add_argument("--revision", default=None)
        sub.add_argument("--allow-download", action="store_true", help="let the processor be fetched (default: local files only)")
        sub.add_argument("--limit", type=int, default=None, help="usable prompt tokens (default: the policy's input limit less its longest completion)")

    runner = commands.add_parser("run", help="roll the teacher out under complete_history_v1 and audit its episodes")
    common(runner)
    runner.add_argument("--suite", required=True, type=Path)
    runner.add_argument("--per-class", type=int, default=None)
    runner.add_argument("--roots", nargs="*", default=None)
    runner.add_argument("--profile", default="scada_only")
    runner.add_argument("--expert", default=None, help="unnamed: the compound study's teacher for the suite's system")
    runner.add_argument("--system", default=None)
    runner.add_argument("--progress-basis", default=None)
    runner.add_argument("--max-steps", type=int, default=40)
    runner.add_argument("--seed", type=int, default=20261006)
    runner.add_argument("--workers", type=int, default=1)
    runner.add_argument("--env-option", action="append", default=None, metavar="KEY=VALUE")
    runner.add_argument("--expert-option", action="append", default=None, metavar="KEY=VALUE")
    stored = commands.add_parser("trajectories", help="audit the episodes of a rollout store (the learner's own states)")
    common(stored)
    stored.add_argument("--store", required=True, type=Path, help="an iteration's trajectories directory")
    stored.add_argument("--suite", default=None, type=Path, help="the suite the roots come from, for their compound class")
    stored.add_argument("--samples-per-root", type=int, default=None)
    args = parser.parse_args(argv)
    return run(args) if args.command == "run" else trajectories(args)


if __name__ == "__main__":
    raise SystemExit(main())
