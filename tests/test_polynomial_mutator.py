import unittest

from pygenalgo.operators.mutation.polynomial_mutator import PolynomialMutator


class TestPolynomialMutatorInitialization(unittest.TestCase):
    """
    Unit tests for the PolynomialMutator initialization and parameter checking.
    """

    def test_valid_default_initialization(self):
        """
        Test that the default constructor parameters are accepted smoothly.
        """
        # Simple dummy bounds.
        lower = [ 0.0,  0.0]
        upper = [10.0, 10.0]

        mutator = PolynomialMutator(mutate_probability=0.1,
                                    eta_pm=20.0,
                                    lower_lim=lower,
                                    upper_lim=upper)

        # Verify that eta_pm was assigned correctly.
        eta_assigned = mutator._items[0]
        self.assertEqual(eta_assigned, 20.0)
        self.assertIsInstance(eta_assigned, float)
    # _end_def_

    def test_eta_zero_raises_value_error(self):
        """
        Test that setting eta_pm exactly to 0 raises a ValueError.
        """
        with self.assertRaises(ValueError):
            PolynomialMutator(eta_pm=0.0)
    # _end_def_

    def test_eta_negative_raises_value_error(self):
        """
        Test that setting a negative eta_pm raises a ValueError.
        """
        with self.assertRaises(ValueError):
            PolynomialMutator(eta_pm=-5.5)
    # _end_def_

    def test_eta_string_conversion(self):
        """
        Test that a string numeric value is correctly converted to a float.
        """
        # Simple dummy bounds.
        lower = [0.0, 0.0]
        upper = [10.0, 10.0]

        mutator = PolynomialMutator(eta_pm="30",
                                    lower_lim=lower,
                                    upper_lim=upper)

        self.assertEqual(mutator._items[0], 30.0)
    # _end_def_

# _end_class_

if __name__ == "__main__":
    unittest.main()
