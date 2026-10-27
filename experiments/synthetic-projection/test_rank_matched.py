import unittest
import numpy as np
from rank_matched import rank, random_row_projector, validate_projector

class TestControl(unittest.TestCase):
    def test_rejects_approximate_projector_outside_absolute_tolerance(self):
        p = np.eye(4); p[0, 0] += 1e-9
        with self.assertRaises(AssertionError):
            validate_projector(p, 0)

    def test_rejects_wrong_deletion_rank(self):
        with self.assertRaises(AssertionError):
            validate_projector(np.eye(4), 1)

    def test_deletes_exact_row_space_dimensions(self):
        X = np.zeros((9, 12)); X[:8, :8] = np.diag(np.arange(1, 9))
        V = np.eye(12)[:, :8]
        P = random_row_projector(V, 4, 10)
        np.testing.assert_allclose(P, P.T, atol=1e-12, rtol=0)
        np.testing.assert_allclose(P @ P, P, atol=1e-12, rtol=0)
        self.assertEqual(rank(X @ P), 4)
        np.testing.assert_allclose(P, random_row_projector(V, 4, 10))

if __name__ == '__main__': unittest.main()
