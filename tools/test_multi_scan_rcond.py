"""The multi-scan corrector's conditioning guard: the sparse symmetric computation keeps the dense SVD's decision (2026-10-28)."""
from __future__ import annotations

import os
import unittest
from unittest import mock

import numpy as np
import scipy.sparse as sp

from tools import correct_parameter_group_multi_scan_port as port


def _arrowhead_gain(scans: int = 4, states: int = 30, scale: float = 1.0, seed: int = 0) -> sp.csc_matrix:
    """A gain shaped like the corrector's: one H'WH block per scan, coupled through two parameter columns."""
    rng = np.random.default_rng(seed)
    n = scans * states + 2
    G = np.zeros((n, n))
    for k in range(scans):
        H_x = rng.normal(size=(3 * states, states))
        H_p = rng.normal(size=(3 * states, 2)) * scale
        rows = slice(k * states, (k + 1) * states)
        G[rows, rows] += H_x.T @ H_x
        G[rows, -2:] += H_x.T @ H_p
        G[-2:, rows] += H_p.T @ H_x
        G[-2:, -2:] += H_p.T @ H_p
    return sp.csc_matrix(G)


class GainConditionGuardTests(unittest.TestCase):
    def test_the_default_is_the_dense_svd(self) -> None:
        G = _arrowhead_gain()
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop(port.RCOND_METHOD_ENV, None)
            self.assertEqual(port._rcond_gain(G), port._rcond_dense(G.toarray()))

    def test_sparse_symmetric_agrees_with_the_svd_on_a_well_conditioned_gain(self) -> None:
        G = _arrowhead_gain(seed=3)
        dense = port._rcond_dense(G.toarray())
        with mock.patch.dict(os.environ, {port.RCOND_METHOD_ENV: "sparse_symmetric"}):
            sparse = port._rcond_gain(G)
        self.assertGreater(dense, port._SPARSE_RCOND_FALLBACK)
        self.assertAlmostEqual(sparse / dense, 1.0, places=8)

    def test_near_the_floor_the_dense_svd_decides(self) -> None:
        G = _arrowhead_gain(seed=5).toarray()
        G[-1, :] = G[-2, :]  # the two parameter columns coincide: singular
        G[:, -1] = G[:, -2]
        G[-1, -1] += 1e-13 * np.abs(G).max()
        singular = sp.csc_matrix(G)
        with mock.patch.dict(os.environ, {port.RCOND_METHOD_ENV: "sparse_symmetric"}):
            value = port._rcond_gain(singular)
        self.assertEqual(value, port._rcond_dense(G))
        self.assertLess(value, port._SPARSE_RCOND_FALLBACK)

    def test_an_unknown_method_is_refused(self) -> None:
        with mock.patch.dict(os.environ, {port.RCOND_METHOD_ENV: "eig"}):
            with self.assertRaises(ValueError):
                port._rcond_gain(_arrowhead_gain())


if __name__ == "__main__":
    unittest.main()
