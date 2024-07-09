import unittest
from src.genome.chromosome import Chromosome
from src.operators.crossover.crossover_operator import CrossoverOperator

class TestCrossoverOperator(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        print(" >> TestCrossoverOperator - START -")

    # _end_def_

    @classmethod
    def tearDownClass(cls) -> None:
        print(" >> TestCrossoverOperator - FINISH -")
    # _end_def_

    def setUp(self) -> None:
        """
        Creates the test object with default settings.

        :return: None.
        """
        # Create an object with a valid probability.
        self.cross_op = CrossoverOperator(crossover_probability=0.5)
    # _end_def_

    def test_crossover(self):
        """
        The crossover method should be implemented.

        :return: None.
        """

        # Create two empty test parents.
        parent1 = Chromosome()
        parent2 = Chromosome()

        # Check if the crossover method is implemented.
        with self.assertRaises(NotImplementedError):
            self.cross_op.crossover(parent1, parent2)
        # _end_with_

    # _end_def_

# _end_class_


if __name__ == '__main__':
    unittest.main()
