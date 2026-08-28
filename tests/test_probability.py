import unittest
from math import inf, nan

from pygenalgo.utils.auxiliary import Probability


class TestProbability(unittest.TestCase):

    # --- Initialization Tests ---
    def test_valid_float_initialization(self):
        """
        Test initialization with valid float values.
        """
        p = Probability(0.5)
        self.assertEqual(p.value, 0.5)
    # _end_def_

    def test_valid_int_initialization(self):
        """
        Test initialization with valid integer values
        (should convert to float).
        """
        p_zero = Probability(0)
        p_one = Probability(1)
        self.assertEqual(p_zero.value, 0.0)
        self.assertEqual(p_one.value, 1.0)
        self.assertIsInstance(p_zero.value, float)
        self.assertIsInstance(p_one.value, float)
    # _end_def_

    def test_boundary_values(self):
        """
        Test exact boundary conditions of 0.0 and 1.0.
        """
        self.assertEqual(Probability(0.0).value, 0.0)
        self.assertEqual(Probability(1.0).value, 1.0)
    # _end_def_

    # --- Type Validation Tests ---

    def test_invalid_type_string(self):
        """
        Test that passing a string raises a TypeError.
        """
        with self.assertRaises(TypeError):
            Probability("0.5")
    # _end_def_

    def test_invalid_type_boolean(self):
        """
        Test that passing a boolean raises a TypeError.
        """
        with self.assertRaises(TypeError):
            Probability(True)
    # _end_def_

    def test_invalid_type_none(self):
        """
        Test that passing None raises a TypeError.
        """
        with self.assertRaises(TypeError):
            Probability(None)
    # _end_def_

    # --- Value Range & Finiteness Tests ---

    def test_value_below_range(self):
        """
        Test that a value less than 0 raises a ValueError.
        """
        with self.assertRaises(ValueError):
            Probability(-0.01)
    # _end_def_

    def test_value_above_range(self):
        """
        Test that a value greater than 1 raises a ValueError.
        """
        with self.assertRaises(ValueError):
            Probability(1.01)
    # _end_def_

    def test_non_finite_infinity(self):
        """
        Test that positive and negative infinity raise a ValueError.
        """
        with self.assertRaises(ValueError):
            Probability(inf)

        with self.assertRaises(ValueError):
            Probability(-inf)
    # _end_def_

    def test_non_finite_nan(self):
        """
        Test that NaN (Not a Number) raises a ValueError.
        """
        with self.assertRaises(ValueError):
            Probability(nan)
    # _end_def_

    # --- String Representation Tests ---

    def test_string_representation(self):
        """
        Test the __str__ output matches the string version of the float.
        """
        p = Probability(0.25)
        self.assertEqual(str(p), "0.25")
    # _end_def_

    def test_repr_representation(self):
        """
        Test the __repr__ output format.
        """
        p = Probability(0.75)
        self.assertEqual(repr(p), "Probability(0.75)")
    # _end_def_

if __name__ == "__main__":
    unittest.main()
