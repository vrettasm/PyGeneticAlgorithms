import unittest
from pygenalgo.genome.gene import Gene
from pygenalgo.genome.chromosome import Chromosome
from pygenalgo.operators.crossover.crossover_operator import CrossoverOperator


class TestCrossoverOperator(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        print(">> TestCrossoverOperator - START -")

    # _end_def_

    @classmethod
    def tearDownClass(cls) -> None:
        print(">> TestCrossoverOperator - FINISH -", end='\n\n')
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
        # Dummy genome with two genes.
        genome = [Gene('1', lambda x: int(x), True),
                  Gene('2', lambda x: int(x), True)]

        # Create two empty test parents.
        parent1 = Chromosome(genome)
        parent2 = Chromosome(genome)

        # Check if the crossover method is implemented.
        with self.assertRaises(NotImplementedError):
            self.cross_op.crossover(parent1, parent2)
    # _end_def_

# _end_class_


if __name__ == '__main__':
    unittest.main()
