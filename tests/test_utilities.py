import unittest
import numpy as np

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

# _end_class_


if __name__ == '__main__':
    unittest.main()
