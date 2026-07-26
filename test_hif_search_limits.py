from __future__ import annotations

import unittest

from hif_search_limits import (
    HIF_ALPHA_GRID_SIZE_MAX,
    HIF_MAX_SCANS_MAX,
    HIF_R_GRID_SIZE_MAX,
    validate_hif_search_limits,
)
from three_phase_nlm.hif_multiscan_estimator import (
    estimate_hif_location_magnitude_multiscan,
)
from three_phase_nlm.hif_parameter_estimator import estimate_hif_location_magnitude


class HIFSearchLimitTests(unittest.TestCase):
    def test_limits_accept_closed_boundaries_and_reject_non_integer_values(self) -> None:
        self.assertEqual(
            validate_hif_search_limits(
                alpha_grid_size=2,
                r_grid_size=HIF_R_GRID_SIZE_MAX,
                max_scans=HIF_MAX_SCANS_MAX,
            ),
            (2, HIF_R_GRID_SIZE_MAX, HIF_MAX_SCANS_MAX),
        )
        for value in (True, 3.5, "3"):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "alpha_grid_size must be an integer"):
                    validate_hif_search_limits(
                        alpha_grid_size=value,
                        r_grid_size=2,
                    )

    def test_single_scan_estimator_rejects_oversized_grid_before_runtime_work(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            rf"alpha_grid_size must be in \[2, {HIF_ALPHA_GRID_SIZE_MAX}\]",
        ):
            estimate_hif_location_magnitude(
                candidate_branch_row0=0,
                z_obs=[],
                alpha_grid_size=HIF_ALPHA_GRID_SIZE_MAX + 1,
                r_grid_size=2,
            )

    def test_multiscan_estimator_rejects_oversized_scan_count_before_loading_window(
        self,
    ) -> None:
        with self.assertRaisesRegex(
            ValueError,
            rf"max_scans must be in \[1, {HIF_MAX_SCANS_MAX}\]",
        ):
            estimate_hif_location_magnitude_multiscan(
                candidate_branch_row0=0,
                scan_window_path="/path/that/must/not/be-read.json",
                alpha_grid_size=2,
                r_grid_size=2,
                max_scans=HIF_MAX_SCANS_MAX + 1,
            )


if __name__ == "__main__":
    unittest.main()
