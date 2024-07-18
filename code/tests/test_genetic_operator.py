import unittest
from src.operators.genetic_operator import GeneticOperator


class TestGeneticOperator(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        print(" >> TestGeneticOperator - START -")

    # _end_def_

    @classmethod
    def tearDownClass(cls) -> None:
        print(" >> TestGeneticOperator - FINISH -")
    # _end_def_

    def setUp(self) -> None:
        """
        Creates the test object with default settings.

        :return: None.
        """
        # Create an object with a valid probability.
        self.gen_op = GeneticOperator(_probability=0.5)
    # _end_def_

    def test_probability_range(self):
        """
        The probability should be range [0.0, 1.0].

        :return: None.
        """
        # Check: probability <= 1.0
        self.assertLessEqual(self.gen_op.probability, 1.0)

        # Check: probability >= 0.0
        self.assertGreaterEqual(self.gen_op.probability, 0.0)
    # _end_def_

    def test_probability(self):
        """
        The probability should be range [0.0, 1.0].

        :return: None.
        """

        # Check the input value.
        with self.assertRaises(ValueError):
            # Try to set the probability to an invalid value.
            self.gen_op.probability = -1.0
        # _end_with_

        # Check the input type.
        with self.assertRaises(TypeError):
            # Try to set the probability to an invalid type.
            self.gen_op.probability = int(0)
        # _end_with_
    # _end_def_

    def test_counter(self):
        """
        Test the operator counter functionality.

        :return: None.
        """
        # Number of repetitions.
        N = 10

        for i in range(N):
            self.gen_op.inc_counter()
        # _end_for_

        # The counter should have been increased 'N' times.
        self.assertEqual(N, self.gen_op.counter)

        # Reset the counter.
        self.gen_op.reset_counter()

        # The counter should have been reset to '0'.
        self.assertEqual(0, self.gen_op.counter)
    # _end_def_

# _end_class_


if __name__ == '__main__':
    unittest.main()
