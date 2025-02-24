import unittest
from numpy.random import default_rng
from pygenalgo.genome.gene import Gene
from pygenalgo.genome.chromosome import Chromosome
from pygenalgo.operators.crossover.partially_mapped_crossover import PartiallyMappedCrossover


class TestPartiallyMappedCrossover(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        print(">> TestPartiallyMappedCrossover - START -")

    # _end_def_

    @classmethod
    def tearDownClass(cls) -> None:
        print(">> TestPartiallyMappedCrossover - FINISH -", end='\n\n')
    # _end_def_

    def setUp(self) -> None:
        """
        Creates the test object with default settings.

        :return: None.
        """
        # Create an object with a crossover probability of 1.0.
        self.cross_op = PartiallyMappedCrossover(crossover_probability=1.0)

        # Create a random number generator.
        self.rng = default_rng()
    # _end_def_

    def test_crossover(self):
        """
        The crossover method should be implemented.

        :return: None.
        """

        # Create two dummy test parents.
        parent1 = Chromosome([Gene('1', lambda: str('x')),
                              Gene('2', lambda: str('x')),
                              Gene('3', lambda: str('x')),
                              Gene('4', lambda: str('x')),
                              Gene('5', lambda: str('x')),
                              Gene('6', lambda: str('x')),
                              Gene('7', lambda: str('x')),
                              Gene('8', lambda: str('x')),
                              Gene('9', lambda: str('x'))])

        # Get the number of 'unique' genes BEFORE crossover.
        N1 = len(set(parent1.genome))

        # Make a second of parent.
        parent2 = Chromosome([Gene('5', lambda: str('x')),
                              Gene('4', lambda: str('x')),
                              Gene('6', lambda: str('x')),
                              Gene('9', lambda: str('x')),
                              Gene('2', lambda: str('x')),
                              Gene('1', lambda: str('x')),
                              Gene('7', lambda: str('x')),
                              Gene('8', lambda: str('x')),
                              Gene('3', lambda: str('x'))])

        # Get the number of 'unique' genes BEFORE crossover.
        N2 = len(set(parent2.genome))

        # Print parents BEFORE crossover.
        print("Parent-1: ", " ".join([xi.value for xi in parent1]))
        print("Parent-2: ", " ".join([xi.value for xi in parent2]))

        # Perform the crossover.
        child1, child2 = self.cross_op(parent1, parent2)
        print("---------")

        # Print offsprings AFTER crossover.
        print("Child--1: ", " ".join([xi.value for xi in child1]))
        print("Child--2: ", " ".join([xi.value for xi in child2]))

        # Get the number of 'unique' genes AFTER crossover.
        M1 = len(set(child1.genome))

        # Get the number of 'unique' genes AFTER crossover.
        M2 = len(set(child2.genome))

        self.assertEqual(N1, N2, msg="Parents1/2 genome does not match.")
        self.assertEqual(N1, M1, msg="Parent1/Offspring1 genome does not match after crossover.")

        self.assertEqual(M1, M2, msg="Offsprings1/2 genome does not match.")
        self.assertEqual(N2, M2, msg="Parent2/Offspring2 genome does not match after crossover.")
    # _end_def_

# _end_class_


if __name__ == '__main__':
    unittest.main()
