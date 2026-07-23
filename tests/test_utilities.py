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

        # Extract the pareto points.
        p_front = np_pareto_front(points)

        # Generate a new set of the same points.
        new_points = np.zeros((n_points, n_dim+1))
        new_points[:, :-1] = points.copy()

        # In the last column add an artificial index.
        new_points[:, -1] = np.arange(n_points)

        # Extract the indices of the pareto points.
        i_front = np_pareto_front_index(new_points)

        # The indexes should point to the same pareto points.
        for pp, k in zip(p_front, i_front):
            self.assertAlmostEqual(0.0,
                                   np.sum(pp-new_points[k,:-1]))
    # _end_def_

# _end_class_


if __name__ == '__main__':
    unittest.main()
