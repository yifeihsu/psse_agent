"""Learned ranker on the policy-visible balanced evidence (hypothesis-ranking plan, steps 4 and 5).

Step 4 (docs/hypothesis_ranking_step4_20261001.md) trained classifiers on
the step-1 states and found the balanced evidence separates the families
almost perfectly, that a learned HIF score removes the same-sign flow-meter
mimic the physics screen flags, and that as an acquisition gate the learned
score trades recall for false acquisitions instead of dominating the rule.
The decision was to keep the physics rule as the admission gate and to use
the learned scores as an ordering signal for the ledger expert.

This module is the deployable half of that decision:

* ``policy_visible_features`` reads one state's evidence exactly as the
  policy observation carries it, the WLS ledger, the unresolved signatures
  and the compact screen report the student also sees, never the offline
  study fields;
* ``LearnedRanker`` applies a model exported by
  ``research.hypothesis_ranking.ranker`` as JSON (no learning library at
  runtime): the study's gradient-boosted trees, or a standardized logistic
  model, Platt-scaled on the study's calibration split, with the
  operating-point thresholds the study measured;
* ``acquisition_deferral`` is the one decision the expert takes from it:
  when the physics rule admits phasors on this state and the model finds an
  auxiliary stream unlikely to be needed, the leading balanced hypothesis of
  the ledger is tried first, one verified candidate, then the acquisition.

The ranker recommends; it never bypasses the controller.  The physics rule
stays the admission gate for every auxiliary stream, the process oracle
judges every action, a voltage-meter edit still waits for the phasors (D3),
and the deferral is spent once per state.  A model trained on one network
is refused on another (bus count), which is what the IEEE 57 transfer
result asked for.
"""
from __future__ import annotations

import json
import math
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

from psse_env.actions import current_screen_report, current_suspicion, hif_suspicion_refuted, phasor_ledger
from psse_env.evidence_profile import is_suspicion_gated
from psse_env.oracle.hypothesis_ledger import FAMILY_TOOLS, hypothesis_ledger

RANKER_CONTRACT = "learned_ranker_v1"
MODEL_TYPES = ("logistic", "hgb")
FEATURE_SET = "policy_visible"
#: The study's export for IEEE 14 (research/hypothesis_ranking/ranker.py --export-model).
DEFAULT_RANKER_MODEL = Path(__file__).resolve().with_name("models") / "learned_ranker_ieee14_20261001.json"
#: Evidence code the expert attaches to the balanced proposals it tries first.
DEFERRAL_EVIDENCE = "learned_ranker_deferred_acquisition"
#: Operating point of the deferral: the score below which the study's
#: calibration split keeps the physics rule's recall (no auxiliary root the
#: rule reaches is deferred there, up to the measured 3%).
DEFAULT_DEFERRAL_THRESHOLD = "at_rule_recall"
DEFERRAL_TARGET = "needs_aux"

CLASSES = ("meter", "parameter", "topology", "hif")
#: Measurement blocks as the WLS ledger names them (trace_protocol.MEASUREMENT_ORDER).
BLOCKS = ("Vm", "Pinj", "Qinj", "Pf", "Qf", "Pt", "Qt")
_RESIDUAL_SIGNATURE = re.compile(
    r"^wls_residual_outlier(?P<dominant>_dominant)?\s+index=(?P<index>\d+)\s+channel=(?P<channel>[A-Za-z]+)\s*$"
)
_BRANCH_SIGNATURE = re.compile(r"^wls_branch_multiplier(?P<dominant>_dominant)?\s")
_MISSING_SCORE = -50.0


def _as_mapping(state: Any) -> Mapping[str, Any]:
    if isinstance(state, Mapping):
        nested = state.get("policy_observation")
        return nested if isinstance(nested, Mapping) else state
    as_dict = getattr(state, "as_dict", None)
    return as_dict() if callable(as_dict) else {}


def _log(value: Any, floor: float = 1e-6) -> float:
    try:
        return float(math.log(max(float(value), floor)))
    except (TypeError, ValueError):
        return float(math.log(floor))


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _wls_ledger(policy: Mapping[str, Any]) -> Mapping[str, Any]:
    contexts = policy.get("fresh_context_evidence")
    wls = contexts.get("wls") if isinstance(contexts, Mapping) else None
    return wls if isinstance(wls, Mapping) else {}


def _flow_offsets(bus_count: int, branch_count: int) -> dict[str, int]:
    nb, nl = int(bus_count), int(branch_count)
    return {"Pf": 3 * nb, "Qf": 3 * nb + nl, "Pt": 3 * nb + 2 * nl, "Qt": 3 * nb + 3 * nl}


def policy_visible_features(observation: Any, *, branch_count: int | None = None) -> dict[str, float]:
    """Fixed-length numeric view of one state's evidence as the policy sees it.

    Every value comes from the policy observation: the WLS ledger of the
    active state (alarm flags, the largest normalized residual, the residual
    breadth and its dominant block), the ``wls_`` residual and multiplier
    signatures (which channels stand out, which dominance tag the solve
    gave), the remaining anomaly score, and the compact balanced screen
    report (first and last compared round's class scores and winner, the
    accepted sequence, each class's best refit objective and its ranked
    alternatives, the channels set aside, the three suspicions).  The flow
    pair feature needs the branch count of the operator model, which the
    ledger does not carry; without it the feature is zero.
    """
    policy = _as_mapping(observation)
    wls = _wls_ledger(policy)
    f: dict[str, float] = {}
    f["wls_current"] = float(wls.get("successful") is True)
    max_rn = _number(wls.get("max_normalized_residual")) or 0.0
    f["max_rn_log1p"] = math.log1p(max(max_rn, 0.0))
    f["rn_alarm"] = float(bool(wls.get("normalized_residual_alarm")))
    f["chi_alarm"] = float(bool(wls.get("chi_square_alarm")))
    f["anomaly_breadth"] = _number(wls.get("anomaly_breadth")) or 0.0
    # The ledger's dominant_residual_block is not read: the provider's breadth
    # metric never fills it (it indexes slice objects), so the first residual
    # signature's block (sig1_*) carries that information instead.
    score = _number(policy.get("remaining_anomaly_score"))
    f["anomaly_score_log"] = _log(score) if score is not None else 0.0

    residual: list[tuple[int, str, bool]] = []
    branch_any = False
    branch_dominant = False
    for item in policy.get("unresolved_signatures") or []:
        text = str(item)
        match = _RESIDUAL_SIGNATURE.match(text)
        if match:
            residual.append((int(match["index"]), str(match["channel"]), bool(match["dominant"])))
            continue
        branch = _BRANCH_SIGNATURE.match(text)
        if branch:
            branch_any = True
            branch_dominant = branch_dominant or bool(branch["dominant"])
    f["n_residual_sigs"] = float(len(residual))
    f["meas_dominant"] = float(any(dominant_flag for _, _, dominant_flag in residual))
    f["branch_sig"] = float(branch_any)
    f["branch_dominant"] = float(branch_dominant)
    for block in BLOCKS:
        f[f"sig_{block}"] = float(sum(1 for _, channel, _ in residual if channel == block))
        f[f"sig1_{block}"] = float(bool(residual) and residual[0][1] == block)
    f["vm_frac_sig3"] = sum(1 for _, channel, _ in residual[:3] if channel == "Vm") / 3.0
    flow_pair = 0.0
    bus_count = wls.get("bus_count")
    if branch_count is not None and isinstance(bus_count, int) and not isinstance(bus_count, bool) and bus_count > 0:
        offsets = _flow_offsets(bus_count, branch_count)
        ends: dict[tuple[str, int], set[str]] = {}
        for index, channel, _ in residual:
            if channel in offsets:
                ends.setdefault((channel[0], index - offsets[channel]), set()).add(channel[1])
        flow_pair = float(any(seen >= {"f", "t"} for seen in ends.values()))
    f["flow_pair_sig"] = flow_pair

    report = current_screen_report(policy, "hif")
    valid = report.get("status") == "valid"
    f["screen_valid"] = float(valid)
    rounds = [r for r in (report.get("rounds") or []) if isinstance(r, Mapping) and r.get("scores")]
    first = rounds[0] if rounds else {}
    last = rounds[-1] if rounds else {}
    scores = first.get("scores") if isinstance(first.get("scores"), Mapping) else {}
    for name in CLASSES:
        f[f"s1_{name}"] = _number(scores.get(name)) if _number(scores.get(name)) is not None else _MISSING_SCORE
    ordered = sorted(value for value in (_number(v) for v in scores.values()) if value is not None)
    f["s1_margin"] = (ordered[-1] - ordered[-2]) if len(ordered) > 1 else 0.0
    f["s1_hif_minus_meter"] = f["s1_hif"] - f["s1_meter"]
    winner = str(first.get("winner") or "")
    for name in CLASSES:
        f[f"w1_{name}"] = float(winner == name)
    last_scores = last.get("scores") if isinstance(last.get("scores"), Mapping) else {}
    for name in CLASSES:
        value = _number(last_scores.get(name))
        f[f"sl_{name}"] = value if value is not None else _MISSING_SCORE
    accepted = [str(item.get("class")) for item in (report.get("accepted_hypotheses") or []) if isinstance(item, Mapping)]
    f["n_accepted"] = float(len(accepted))
    for position in (0, 1):
        for name in CLASSES:
            f[f"acc{position}_{name}"] = float(len(accepted) > position and accepted[position] == name)
    f["explained"] = float(bool(report.get("explained")))
    f["unexplained"] = float(bool(report.get("unexplained")))
    f["suspected"] = float(bool(report.get("suspected")))
    f["vm_channels_n"] = float(len(report.get("voltage_meter_channels") or []))
    f["n_rounds"] = float(len(rounds))
    best = first.get("best") if isinstance(first.get("best"), Mapping) else {}
    objectives: dict[str, float | None] = {}
    for name in CLASSES:
        item = best.get(name) if isinstance(best.get(name), Mapping) else {}
        objective = _number(item.get("J"))
        objectives[name] = objective
        f[f"j1_{name}_log1p"] = math.log1p(max(objective, 0.0)) if objective is not None else 0.0
        f[f"j1_{name}_known"] = float(objective is not None)
    meter_objective = objectives.get("meter")
    for name in ("parameter", "topology", "hif"):
        objective = objectives.get(name)
        f[f"j1_{name}_vs_meter_log"] = (
            _log((max(objective, 0.0) + 1.0) / (max(meter_objective, 0.0) + 1.0))
            if objective is not None and meter_objective is not None else 0.0
        )
    hif_best = best.get("hif") if isinstance(best.get("hif"), Mapping) else {}
    f["hif_alpha"] = _number(hif_best.get("alpha_from_from_bus")) or 0.0
    ranked = first.get("ranked") if isinstance(first.get("ranked"), Mapping) else {}
    for name in CLASSES:
        items = [item for item in (ranked.get(name) or []) if isinstance(item, Mapping)]
        gap = 0.0
        if len(items) > 1:
            top, second = _number(items[0].get("J")), _number(items[1].get("J"))
            if top is not None and second is not None:
                gap = _log((max(second, 0.0) + 1.0) / (max(top, 0.0) + 1.0))
        f[f"{name}_rank_gap_log"] = gap
    f["set_aside_n"] = float(len(last.get("set_aside_channels") or [])) if isinstance(last, Mapping) else 0.0
    f["clean_after_removal"] = float(any(
        bool(r.get("clean_after_removal")) for r in (report.get("rounds") or []) if isinstance(r, Mapping)
    ))
    return f


def _sigmoid(value: float) -> float:
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exp = math.exp(value)
    return exp / (1.0 + exp)


def _logit(probability: float) -> float:
    clipped = min(max(probability, 1e-6), 1.0 - 1e-6)
    return math.log(clipped / (1.0 - clipped))


def _hgb_raw_prediction(x: list[float], spec: Mapping[str, Any]) -> float:
    """Raw (pre-sigmoid) prediction of sklearn's HistGradientBoostingClassifier from its exported trees.

    Each tree is a node list; a non-leaf sends ``x[feature] <= threshold`` to
    the left child (a missing value follows ``missing_left``), and the raw
    prediction is the baseline plus the leaf values of every tree.
    """
    total = float(spec.get("baseline") or 0.0)
    for tree in spec.get("trees") or []:
        node = tree[0]
        while not node["leaf"]:
            value = x[int(node["feature"])]
            if value != value:  # NaN
                node = tree[int(node["left"] if node["missing_left"] else node["right"])]
            elif value <= float(node["threshold"]):
                node = tree[int(node["left"])]
            else:
                node = tree[int(node["right"])]
        total += float(node["value"])
    return total


class LearnedRanker:
    """A JSON-exported model per target (gradient-boosted trees or standardized logistic), Platt-scaled."""

    def __init__(self, payload: Mapping[str, Any], *, source: str | None = None) -> None:
        if payload.get("contract") != RANKER_CONTRACT:
            raise ValueError(f"unsupported ranker contract: {payload.get('contract')!r}")
        if payload.get("feature_set") != FEATURE_SET:
            raise ValueError(f"ranker feature set must be {FEATURE_SET!r}, got {payload.get('feature_set')!r}")
        self.payload = dict(payload)
        self.source = source
        self.features = [str(name) for name in payload.get("features") or []]
        if not self.features:
            raise ValueError("ranker declares no features")
        self.system = dict(payload.get("system") or {})
        self.targets: dict[str, dict[str, Any]] = {}
        for name, spec in (payload.get("targets") or {}).items():
            if not isinstance(spec, Mapping):
                continue
            model_type = str(spec.get("type") or "logistic")
            if model_type not in MODEL_TYPES:
                raise ValueError(f"ranker target {name!r}: unsupported model type {model_type!r}")
            if model_type == "logistic":
                for key in ("mean", "scale", "coef"):
                    if len(spec.get(key) or []) != len(self.features):
                        raise ValueError(f"ranker target {name!r}: {key} length differs from the feature list")
            elif not spec.get("trees"):
                raise ValueError(f"ranker target {name!r}: gradient-boosted model without trees")
            self.targets[str(name)] = dict(spec)

    @classmethod
    def from_json(cls, path: Any) -> "LearnedRanker":
        return _load_ranker(str(Path(path).resolve()))

    @property
    def bus_count(self) -> int | None:
        value = self.system.get("bus_count")
        return int(value) if isinstance(value, int) and not isinstance(value, bool) else None

    @property
    def branch_count(self) -> int | None:
        value = self.system.get("branch_count")
        return int(value) if isinstance(value, int) and not isinstance(value, bool) else None

    def applies_to(self, observation: Any) -> bool:
        """A current, valid screen on the active state of the network the model was trained on."""
        policy = _as_mapping(observation)
        wls = _wls_ledger(policy)
        bus_count = wls.get("bus_count")
        if not isinstance(bus_count, int) or isinstance(bus_count, bool):
            return False
        if self.bus_count is not None and int(bus_count) != self.bus_count:
            return False
        return current_screen_report(policy, "hif").get("status") == "valid"

    def feature_vector(self, observation: Any) -> list[float]:
        values = policy_visible_features(observation, branch_count=self.branch_count)
        return [float(values.get(name, 0.0)) for name in self.features]

    def probability(self, observation: Any, target: str) -> float | None:
        """Calibrated probability of ``target`` on this state, or None when the model does not apply."""
        spec = self.targets.get(target)
        if spec is None or not self.applies_to(observation):
            return None
        x = self.feature_vector(observation)
        if str(spec.get("type") or "logistic") == "hgb":
            logit = _hgb_raw_prediction(x, spec)
        else:
            logit = float(spec.get("intercept") or 0.0)
            for value, mean, scale, coefficient in zip(x, spec["mean"], spec["scale"], spec["coef"]):
                standardized = (value - float(mean)) / float(scale) if float(scale) else 0.0
                logit += float(coefficient) * standardized
        probability = _sigmoid(logit)
        platt = spec.get("platt")
        if isinstance(platt, Mapping) and platt.get("coef") is not None:
            probability = _sigmoid(float(platt["coef"]) * _logit(probability) + float(platt.get("intercept") or 0.0))
        return float(probability)

    def threshold(self, target: str, name: str = DEFAULT_DEFERRAL_THRESHOLD) -> float | None:
        spec = self.targets.get(target) or {}
        thresholds = spec.get("thresholds") if isinstance(spec.get("thresholds"), Mapping) else {}
        value = _number(thresholds.get(name))
        return value


@lru_cache(maxsize=8)
def _load_ranker(path: str) -> LearnedRanker:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return LearnedRanker(payload, source=path)


def resolve_ranker(value: Any) -> LearnedRanker | None:
    """A ranker from an instance, a JSON path, or None."""
    if value is None:
        return None
    if isinstance(value, LearnedRanker):
        return value
    return LearnedRanker.from_json(value)


def acquisition_deferral(
    observation: Any, ranker: LearnedRanker | None, *,
    threshold_name: str = DEFAULT_DEFERRAL_THRESHOLD,
) -> dict[str, Any] | None:
    """Whether the ledger's leading balanced hypothesis is tried before the admitted acquisition.

    Returns None (acquire as the physics rule says) unless every condition
    holds: suspicion-gated profile; a valid current screen on a network the
    model knows; the rule admits phasors on this state now (an HIF won, or a
    voltage channel was set aside) and none were requested here yet; no
    candidate is open and no verified candidate was rejected on this state
    (the deferral is spent once); the ledger's leading family is a balanced
    family with an untested target (a failed execution ends the deferral
    too), and that target is not a voltage
    channel (D3 holds the voltage-meter edit for the phasors); and the
    model's probability that an auxiliary stream is needed lies below the
    chosen operating point.  The returned record names the probability,
    the threshold and the hypothesis tried first, for the proposal's
    evidence codes and the audit.
    """
    if ranker is None:
        return None
    policy = _as_mapping(observation)
    if not is_suspicion_gated(policy):
        return None
    if policy.get("has_open_candidate") or phasor_ledger(policy):
        return None
    if not ranker.applies_to(policy):
        return None
    hif = bool(current_suspicion(policy, "hif") and not hif_suspicion_refuted(policy))
    voltage = bool(current_suspicion(policy, "voltage_meter"))
    if not (hif or voltage):
        return None
    ledger = hypothesis_ledger(policy)
    if not ledger["available"] or int(ledger.get("rejected_total") or 0) > 0:
        return None
    leading = next(
        (item for item in ledger["hypotheses"]
         if item["family"] in FAMILY_TOOLS and item["status"] in {"untested", "execution_failed"}),
        None,
    )
    if leading is None or leading["status"] != "untested":
        # A hypothesis whose execution failed on this state is the ledger's
        # to retry through the recovery route, not a reason to hold the
        # acquisition any longer.
        return None
    target = leading.get("target")
    if leading["family"] == "measurement":
        bus_count = _wls_ledger(policy).get("bus_count")
        if target is None or not isinstance(bus_count, int) or int(target) < int(bus_count):
            return None
    probability = ranker.probability(policy, DEFERRAL_TARGET)
    threshold = ranker.threshold(DEFERRAL_TARGET, threshold_name)
    if probability is None or threshold is None or probability >= threshold:
        return None
    return {
        "probability": float(probability),
        "threshold": float(threshold),
        "threshold_name": str(threshold_name),
        "suspicion": "hif" if hif else "voltage_meter",
        "family": str(leading["family"]),
        "target": target,
    }


def deferral_evidence_codes(deferral: Mapping[str, Any]) -> list[str]:
    return [
        f"{DEFERRAL_EVIDENCE} p_needs_aux={deferral['probability']:.3f} threshold={deferral['threshold']:.3f}",
        f"ledger_balanced_hypothesis_first family={deferral['family']} target={deferral.get('target')} "
        f"suspicion={deferral['suspicion']}",
    ]


__all__ = [
    "BLOCKS", "CLASSES", "DEFAULT_DEFERRAL_THRESHOLD", "DEFAULT_RANKER_MODEL", "DEFERRAL_EVIDENCE", "DEFERRAL_TARGET",
    "FEATURE_SET", "LearnedRanker", "MODEL_TYPES", "RANKER_CONTRACT", "acquisition_deferral", "deferral_evidence_codes",
    "policy_visible_features", "resolve_ranker",
]
