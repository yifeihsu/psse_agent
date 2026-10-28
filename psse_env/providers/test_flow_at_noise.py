"""The open-terminal cue the provider reports beside the branch ranking (compound study, stage 2)."""
from __future__ import annotations

import unittest

from psse_env.providers.matpower import MatpowerDeploymentProviders, measurement_index_map

NB, NL = 14, 20
ROW0 = 4
SIGMA = 0.01


def _solved(*, p_meas: float, q_meas: tuple[float, float] = (5.0, 1.0), p_resid: float, sigma: float | None = SIGMA) -> dict:
    """A solve bundle whose line ``ROW0`` reads ``p_meas`` sigmas of P at both ends, ``q_meas`` sigmas of Q and ``p_resid`` of P residual."""
    index_map = measurement_index_map(NB, NL)
    nz = max(block.stop for block in index_map.values())
    z = [0.0] * nz
    residual = [0.0] * nz
    z[index_map["Pf"].start + ROW0] = p_meas * SIGMA
    z[index_map["Pt"].start + ROW0] = -p_meas * SIGMA
    z[index_map["Qf"].start + ROW0] = q_meas[0] * SIGMA
    z[index_map["Qt"].start + ROW0] = q_meas[1] * SIGMA
    residual[index_map["Pf"].start + ROW0] = p_resid
    residual[index_map["Pt"].start + ROW0] = -0.5 * p_resid
    options = {} if sigma is None else {"measurement_sigma": [sigma] * nz}
    return {"payload": {"r": residual}, "z": z, "index_map": index_map, "nb": NB, "noise_options": options}


class FlowAtNoiseTests(unittest.TestCase):
    def setUp(self) -> None:
        self.providers = MatpowerDeploymentProviders(chi2_alpha=0.01, evidence_profile="scada_only")

    def test_defaults(self) -> None:
        self.assertEqual(self.providers.flow_noise_sigma, 3.0)
        self.assertEqual(self.providers.flow_noise_residual, 8.0)

    def test_open_terminal_fires_and_loaded_or_quiet_lines_do_not(self) -> None:
        flag = self.providers._flow_at_noise_lines
        self.assertEqual(flag(_solved(p_meas=1.0, p_resid=12.0), [ROW0]), [ROW0 + 1])
        # A line that carries flow is not an open terminal, however large its residual.
        self.assertEqual(flag(_solved(p_meas=10.0, p_resid=12.0), [ROW0]), [])
        # A lightly loaded line with a small residual (an impedance error at
        # low flow) stays below the residual floor.
        self.assertEqual(flag(_solved(p_meas=1.0, p_resid=5.0), [ROW0]), [])
        # Reactive flow at both ends means the line is energised end to end.
        self.assertEqual(flag(_solved(p_meas=1.0, q_meas=(5.0, 4.0), p_resid=12.0), [ROW0]), [])
        # Only the requested rows are examined.
        self.assertEqual(flag(_solved(p_meas=1.0, p_resid=12.0), [ROW0 + 1]), [])

    def test_thresholds_are_provider_options(self) -> None:
        lenient = MatpowerDeploymentProviders(chi2_alpha=0.01, evidence_profile="scada_only", flow_noise_residual=4.0)
        self.assertEqual(lenient._flow_at_noise_lines(_solved(p_meas=1.0, p_resid=5.0), [ROW0]), [ROW0 + 1])
        strict = MatpowerDeploymentProviders(chi2_alpha=0.01, evidence_profile="scada_only", flow_noise_sigma=0.5)
        self.assertEqual(strict._flow_at_noise_lines(_solved(p_meas=1.0, p_resid=12.0), [ROW0]), [])

    def test_sigma_falls_back_to_the_nominal_one_and_bad_rows_are_skipped(self) -> None:
        flag = self.providers._flow_at_noise_lines
        self.assertEqual(flag(_solved(p_meas=1.0, p_resid=12.0, sigma=None), [ROW0]), [ROW0 + 1])
        self.assertEqual(flag(_solved(p_meas=1.0, p_resid=12.0), [NL + 5, -1]), [])
        degenerate = _solved(p_meas=1.0, p_resid=12.0)
        degenerate["noise_options"]["measurement_sigma"][degenerate["index_map"]["Qt"].start + ROW0] = 0.0
        self.assertEqual(flag(degenerate, [ROW0]), [])


if __name__ == "__main__":
    unittest.main()
