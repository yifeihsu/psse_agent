"""The unbalance localizer's noise floor follows the bus current (2026-10-11).

Phasor noise is absolute, so the per-phase shunt power at a bus carrying ``|I|`` pu is noisy by
``|I| * sigma`` from the voltage as well as ``|V| * sigma * sqrt(n)`` from the summed currents.
The old floor had only the second term: at IEEE 118's bus 10 (4 pu of generation on one line) the
noise spread of a balanced injection cleared it and the three-phase estimator reported a
"significant" unbalance on harmonic roots, which the expert then finalized as explained.
"""
from __future__ import annotations

import cmath
import math

import numpy as np

from three_phase_nlm.branch_current_analysis import bus_shunt_power_unbalance, unbalance_source_localization

SIGMA = 1e-4
ANGLES = (0.0, -120.0, 120.0)


def _rows(current_pu: float, unbalance: float = 0.0, *, rng: np.random.Generator, bus: int = 10, other: int = 9):
    """One line ``other -> bus`` feeding a shunt that draws ``current_pu`` per phase, phase A scaled by (1 + unbalance)."""
    volts, currents = [], []
    for b in (other, bus):
        v = [cmath.rect(1.0, math.radians(a)) + rng.normal(0.0, SIGMA) + 1j * rng.normal(0.0, SIGMA) for a in ANGLES]
        volts.append({"bus": b, "kvbase_ln": 79.67, "vln_pu": [abs(x) for x in v], "ang_deg": [math.degrees(cmath.phase(x)) for x in v]})
    scale = [1.0 + unbalance, 1.0, 1.0]
    i_to = [-cmath.rect(current_pu * scale[k], math.radians(ANGLES[k])) for k in range(3)]       # leaving the line at ``bus`` into the shunt
    i_from = [-x for x in i_to]
    noisy = lambda xs: [x + rng.normal(0.0, SIGMA) + 1j * rng.normal(0.0, SIGMA) for x in xs]
    i_from, i_to = noisy(i_from), noisy(i_to)
    currents.append({"branch": "Line.9-10", "branch_row0": 8, "from_bus": other, "to_bus": bus,
                     "i_from_pu": [abs(x) for x in i_from], "ang_from_deg": [math.degrees(cmath.phase(x)) for x in i_from],
                     "i_to_pu": [abs(x) for x in i_to], "ang_to_deg": [math.degrees(cmath.phase(x)) for x in i_to],
                     "ibase_from_a": 418.4, "ibase_to_a": 418.4})
    return volts, currents


def test_a_balanced_four_pu_bus_is_not_a_significant_unbalance_source():
    insignificant = 0
    for seed in range(40):
        volts, currents = _rows(4.0, rng=np.random.default_rng(seed))
        record = next(r for r in bus_shunt_power_unbalance(volts, currents, sigma_pu=SIGMA) if r["bus"] == 10)
        assert record["spread_noise_floor_pu"] > 6 * SIGMA * 4.0  # the current term is in the floor
        insignificant += not record["significant"]
    assert insignificant == 40


def test_a_five_percent_unbalance_at_a_loaded_bus_stays_significant():
    # The fixture's single line makes both terminals carry the unbalanced current, so the
    # localization's top bus is either terminal; the shunt at bus 10 must be significant.
    for seed in range(10):
        volts, currents = _rows(1.0, 0.05, rng=np.random.default_rng(seed))
        record = next(r for r in bus_shunt_power_unbalance(volts, currents, sigma_pu=SIGMA) if r["bus"] == 10)
        assert record["significant"] and record["phase_power_spread_rel"] > 0.02
        assert record["phase_power_spread_pu"] > 20 * record["spread_noise_floor_pu"]
        top = unbalance_source_localization(volts, currents, sigma_pu=SIGMA)
        assert top is not None and top["significant"] and top["bus_1based"] in (9, 10)
