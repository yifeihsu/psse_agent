from __future__ import annotations

from typing import Any, Mapping, Sequence

from .ieee14_adapter import BRANCH_ORDER, branch_info_for_row0, branch_row0_for_dss_element


def _score_for_rank(rank: int, base_score: float = 100.0) -> float:
    return float(base_score / max(rank, 1))


def _neighbor_rows(row0: int, branch_order: Sequence[str], limit: int) -> list[int]:
    rows = [int(row0)]
    for delta in (1, -1, 2, -2, 3, -3):
        candidate = int(row0) + delta
        if 0 <= candidate < len(branch_order) and str(branch_order[candidate]).startswith("Line."):
            rows.append(candidate)
        if len(rows) >= limit:
            break
    return rows[:limit]


def metadata_hif_diagnostic(
    *,
    target_dss_element: str | None = None,
    target_branch_row0: int | None = None,
    branch_order: Sequence[str] = BRANCH_ORDER,
    top_k: int = 5,
    success: bool = True,
) -> dict[str, Any]:
    """
    Compact HIF localization payload used until the full three-phase NLM backend
    is wired to IEEE-14.

    This is intentionally explicit about its source. It lets the SFT trace layer
    exercise the HIF tool-call protocol without pretending that the imported
    342-bus NLM scripts have already been validated on IEEE-14.
    """
    if target_branch_row0 is None and target_dss_element:
        target_branch_row0 = branch_row0_for_dss_element(target_dss_element)
    if target_branch_row0 is None:
        return {
            "success": False,
            "converged": False,
            "method": "metadata_fallback",
            "error": "No target branch was supplied for metadata HIF diagnostic.",
            "top_hif_groups": [],
            "detected": False,
            "detected_top1": False,
            "detected_top3": False,
        }

    rows = _neighbor_rows(int(target_branch_row0), branch_order, max(int(top_k), 1))
    groups = []
    for rank, row0 in enumerate(rows, start=1):
        info = branch_info_for_row0(row0)
        groups.append(
            {
                "rank": rank,
                "branch_row0": info["branch_row0"],
                "line_index1": info["line_index1"],
                "dss_element": info["dss_element"],
                "from_bus": info["from_bus"],
                "to_bus": info["to_bus"],
                "score": _score_for_rank(rank),
            }
        )

    target = int(target_branch_row0)
    top_rows = [item["branch_row0"] for item in groups]
    return {
        "success": bool(success),
        "converged": bool(success),
        "method": "metadata_fallback",
        "top_hif_groups": groups,
        "detected": target in top_rows,
        "detected_top1": bool(top_rows and top_rows[0] == target),
        "detected_top3": target in top_rows[:3],
    }


def run_ieee14_hif_nlm(
    pristine_model_dir: str | None = None,
    faulted_model_dir: str | None = None,
    target_dss_element: str | None = None,
    phase: str | None = None,
    r_hif_ohm: float | None = None,
    base_mva: float = 100.0,
    slack_bus: str = "b1",
    max_iter: int = 100,
    *,
    target_branch_row0: int | None = None,
    supplied_diagnostic: Mapping[str, Any] | None = None,
    load_scale: float = 1.0,
    top_k: int = 5,
) -> dict[str, Any]:
    """
    Public adapter entrypoint for IEEE-14 HIF traces.

    The signature matches the intended integration boundary. When generated
    IEEE-14 OpenDSS scenario directories are available, this calls the imported
    legacy three-phase NLM backend and maps its line-group output back to the
    IEEE-14 trace schema. When a generated sample already carries a non-fallback
    compact `nlm_diagnostic`, this function returns it. Otherwise it emits an
    explicit metadata fallback.
    """
    if isinstance(supplied_diagnostic, Mapping):
        payload = dict(supplied_diagnostic)
        if payload.get("method") != "metadata_fallback" or not (pristine_model_dir and faulted_model_dir):
            payload.setdefault("success", True)
            payload.setdefault("converged", bool(payload.get("success", True)))
            payload.setdefault("top_hif_groups", [])
            payload.setdefault("detected", bool(payload.get("detected_top1") or payload.get("detected_top3")))
            return payload

    if pristine_model_dir and faulted_model_dir:
        try:
            from .legacy_bridge import run_legacy_ieee14_hif_nlm

            return run_legacy_ieee14_hif_nlm(
                pristine_model_dir=pristine_model_dir,
                faulted_model_dir=faulted_model_dir,
                target_branch_row0=target_branch_row0,
                target_dss_element=target_dss_element,
                phase=phase,
                r_hif_ohm=r_hif_ohm,
                load_scale=float(load_scale),
                base_mva=float(base_mva),
                slack_bus=slack_bus,
                top_k=top_k,
                max_iter=max_iter,
            )
        except Exception as exc:
            return {
                "success": False,
                "converged": False,
                "method": "legacy_three_phase_nlm_error",
                "error": str(exc),
                "top_hif_groups": [],
                "detected": False,
                "detected_top1": False,
                "detected_top3": False,
            }

    return metadata_hif_diagnostic(
        target_dss_element=target_dss_element,
        target_branch_row0=target_branch_row0,
        top_k=top_k,
    )
