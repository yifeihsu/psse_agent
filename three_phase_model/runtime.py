"""Isolated DSS contexts; compilation never edits a user's existing circuit."""
from __future__ import annotations

from pathlib import Path

import opendssdirect


def solve(dss) -> None:
    dss.Solution.Solve()
    if not dss.Solution.Converged():
        raise RuntimeError("OpenDSS snapshot did not converge")


def compile_model(master: str | Path):
    path = Path(master).resolve(strict=True)
    dss = opendssdirect.NewContext()
    dss.Basic.AllowChangeDir(False)
    dss.Text.Command(f'Compile "{path}"')
    if not dss.Solution.Converged():
        raise RuntimeError(f"OpenDSS did not converge: {path}")
    return dss


def redistribute_load(dss, registry, *, bus: int, delta: float) -> dict:
    """Change phase demand while preserving the bus's total complex power.

    delta=0 restores the generated reference. No phasors are edited; the
    modified circuit is solved to obtain the new measurements.
    """
    import math

    if not math.isfinite(delta) or not -1.0 < delta < 1.0:
        raise ValueError("delta must be finite and strictly between -1 and 1")
    loads = [row for row in registry["loads"] if row["bus"] == bus]
    if len(loads) != 3 or {row["phase"] for row in loads} != {1, 2, 3}:
        raise ValueError("Selected bus must have three generated single-phase loads")
    factors = {1: 1 + delta, 2: 1 - delta, 3: 1.0}
    before = [sum(row[key] for row in loads) for key in ("kw", "kvar")]
    after = [sum(row[key] * factors[row["phase"]] for row in loads) for key in ("kw", "kvar")]
    if not all(math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-9) for a, b in zip(before, after)):
        raise ValueError("Load redistribution requires a balanced reference phase split")
    commands = []
    for row in loads:
        factor = factors[row["phase"]]
        command = f"Edit {row['element']} kW={row['kw']*factor:.16g} kvar={row['kvar']*factor:.16g}"
        dss.Text.Command(command)
        commands.append(command)
    solve(dss)
    return {"bus": bus, "delta": delta, "phase_factors": factors,
            "before_kw_kvar": before, "after_kw_kvar": after, "commands": commands,
            "method": "physical_phase_load_redistribution_then_three_phase_solve"}
