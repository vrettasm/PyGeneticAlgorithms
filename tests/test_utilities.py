import unittest
import numpy as np

from utils.utilities import (np_pareto_front,
                             np_pareto_front_index)
from pygenalgo.utils.utilities import two_indices_fast


class TestUtilities(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        print(">> TestUtilities - START -")

        # Create a random generator with fixed seed.
        cls.rng = np.random.default_rng(42)
    # _end_def_

    @classmethod
    def tearDownClass(cls) -> None:
        print(">> TestUtilities - FINISH -", end='\n\n')
    # _end_def_

    def test_two_indices_fast(self) -> None:
        """
        Test the functionality of two_indices_fast.
        """

        # Num should be more than 1.
        with self.assertRaises(ValueError):
            two_indices_fast(rng=self.rng, num=1)

        # Call the function 1000 times.
        for _ in range(1000):
            # Generate a random size >= 2.
            n_size = self.rng.integers(low=2, high=100)

            # Get two random numbers.
            i, j = two_indices_fast(rng=self.rng, num=n_size)

            # They must be different.
            self.assertNotEqual(i, j)
    # _end_def_

    def test_two_indices_fast_in_order(self) -> None:
        """
        Test the functionality in_order functionality
        of two_indices_fast function.
        """

        # Call the function 1000 times.
        for _ in range(1000):
            # Generate a random size >= 2.
            n_size = self.rng.integers(low=2, high=100)

            # Get two random numbers.
            i, j = two_indices_fast(rng=self.rng, num=n_size, in_order=True)

            # Output must be ordered.
            self.assertLess(i, j)
    # _end_def_

    def test_np_pareto_front(self) -> None:
        """
        Test the functionality of np_pareto_from_index.

        :return:
        """
        # Select randomly n_points.
        n_points = self.rng.integers(low=10, high=100, dtype=int)

        # Select randomly n_dim.
        n_dim = self.rng.integers(low=2, high=10, dtype=int)

        # Generate random points.
        points = np.random.randn(n_points, n_dim)

        # Remove duplicates.
        x_points = np.unique(points, axis=0)

        # Extract the pareto points.
        p_front = np_pareto_front(x_points)

        # Extract the indices of the pareto points.
        i_front = np_pareto_front_index(x_points)

        # The indexes should point to the same pareto points.
        for pp, k in zip(p_front, i_front):
            self.assertAlmostEqual(0.0,
                                   np.sum(pp-x_points[k]))
    # _end_def_

    def assertFrontEqual(self, actual, expected_set) -> None:
        """
        Helper function for asserting front equality.

        :param actual: points.
        :param expected_set: points
        :return: None.
        """
        actual_set = set(np.asarray(actual).tolist())
        self.assertEqual(actual_set, set(expected_set))
    # _end_def_

    def test_ndim_validation_raises(self) -> None:
        """
        Test the functionality of np_pareto_from_index
        when raising ndim argument error.

        :return: None.
        """
        # Test 1D sample.
        pts = np.array([1, 2, 3])

        with self.assertRaises(RuntimeError):
            # It should be 2D.
            np_pareto_front_index(pts)
    # _end_def_

    def test_mode_validation_raises(self) -> None:
        """
        Test the functionality of np_pareto_from_index
        when we pass unknown mode argument.

        :return: None.
        """
        # Sample points are 2D.
        pts = np.array([
            [1.0, 2.0],
            [3.0, 4.0]
        ], dtype=float)

        with self.assertRaises(ValueError):
            # Mode must be either 'max' or 'min'.
            np_pareto_front_index(pts, mode="median")
    # _end_def_

    def test_simple_max_2d_front(self):
        """
        Test the 'max' functionality.

        :return: None.
        """
        # Sample 2D points.
        pts = np.array([
            [1, 1],  # 0 dominated
            [2, 5],  # 1 front
            [5, 2],  # 2 front
            [3, 3],  # 3 front (not dominated)
            [1.5, 4],  # 4 dominated by [2, 5]
        ], dtype=float)

        idx = np_pareto_front_index(pts, mode="max", rel_eps=0.0,
                                    exclude_duplicates=True)
        self.assertFrontEqual(idx, expected_set={1, 2, 3})
    # _end_def_

    def test_simple_min_2d_front_equivalence(self) -> None:
        """
        Test the 'min' functionality.

        :return: None.
        """
        # Sample 2D points.
        pts = np.array([
            [1.0, 1.0],  # 0 front (min)
            [1.0, 2.0],  # 1 dominated by 0
            [2.0, 1.0],  # 2 dominated by 0
            [3.0, 3.0],  # 3 dominated by 0
            [1.0, 0.5],  # 4 front
        ], dtype=float)

        idx = np_pareto_front_index(pts, mode="min", rel_eps=0.0,
                                    exclude_duplicates=True)
        self.assertFrontEqual(idx, expected_set={4})
    # _end_def_

    def test_mode_max_vs_min_negative_transform(self) -> None:
        """
        Test the 'min' and 'max' functionality.

        :return: None.
        """
        # Generate random sample points.
        pts = self.rng.normal(size=(30, 3))

        # Solve as min problem.
        idx_min = np_pareto_front_index(+pts, mode="min", rel_eps=0.0,
                                        exclude_duplicates=True)
        # Solve as max problem.
        idx_max = np_pareto_front_index(-pts, mode="max", rel_eps=0.0,
                                        exclude_duplicates=True)
        # The indices should be identical.
        self.assertEqual(set(idx_min.tolist()),
                         set(idx_max.tolist()))
    # _end_def_

    def test_exclude_duplicates_true(self) -> None:
        """
        Test the 'exclude_duplicates' is True functionality.

        :return: None.
        """
        # Sample points.
        pts = np.array([
            [1.0, 2.0],  # 0
            [1.0, 2.0],  # 1 duplicate
            [0.0, 1.0],  # 2 dominated
        ], dtype=float)

        # Exclude duplicate points.
        idx_dedup = np_pareto_front_index(pts, mode="max", rel_eps=0.0,
                                          exclude_duplicates=True)

        # With exclude_duplicates=True, only one of {0, 1}
        # should appear.
        self.assertFrontEqual(idx_dedup, expected_set={0})
    # _end_def_

    def test_exclude_duplicates_false(self) -> None:
        """
        Test the 'exclude_duplicates' is False functionality.

        :return: None.
        """
        # Sample points.
        pts = np.array([
            [1.0, 2.0],  # 0
            [1.0, 2.0],  # 1
            [0.0, 1.0],  # 2 dominated
        ], dtype=float)

        # Keep duplicate points.
        idx_keep = np_pareto_front_index(pts, mode="max", rel_eps=0.0,
                                         exclude_duplicates=False)
        # We should have both identical points.
        self.assertFrontEqual(idx_keep, expected_set={0, 1})
    # _end_def_

    def test_branching_large_input_returns_expected_set(self) -> None:
        """
        Force the loop-branch by making memory_bytes exceed 500MB.

        :return: None.
        """
        # memory_bytes = n_points^2 * n_dims * dtype.itemsize
        # float64 itemsize=8 -> choose values so it's > 500_000_000
        n_points, n_dims = 4000, 4

        # Print info.
        print(f"memory_bytes = {n_points**2 * n_dims * 8}")

        # Large sample points.
        pts = np.full((n_points, n_dims), -1.0, dtype=np.float64)

        # Two dominating points
        pts[123] = 0.0
        pts[456] = 0.0

        # One dominated point
        pts[789] = -0.5

        # Should use the loop-branch.
        idx = np_pareto_front_index(pts, mode="max", rel_eps=0.0,
                                    exclude_duplicates=False)
        # Expected set.
        self.assertFrontEqual(idx, expected_set={123, 456})
    # _end_def_

# _end_class_


if __name__ == '__main__':
    unittest.main()
