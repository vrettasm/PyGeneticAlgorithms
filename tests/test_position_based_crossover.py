import unittest
from numpy.random import default_rng
from pygenalgo.genome.gene import Gene
from pygenalgo.genome.chromosome import Chromosome
from pygenalgo.operators.crossover.position_based_crossover import PositionBasedCrossover


class TestPositionBasedCrossover(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        print(">> TestPositionBasedCrossover - START -")

    # _end_def_

    @classmethod
    def tearDownClass(cls) -> None:
        print(">> TestPositionBasedCrossover - FINISH -", end='\n\n')
    # _end_def_

    def setUp(self) -> None:
        """
        Creates the test object with default settings.

        :return: None.
        """
        # Create an object with a crossover probability of 1.0.
        self.cross_op = PositionBasedCrossover(crossover_probability=1.0)

        # Create a random number generator.
        self.rng = default_rng()
    # _end_def_

    def test_crossover(self):
        """
        The crossover method should be implemented.

        :return: None.
        """

        # Create two dummy test parents.
        parent1 = Chromosome([Gene('A', lambda: str('x')),
                              Gene('B', lambda: str('x')),
                              Gene('C', lambda: str('x')),
                              Gene('D', lambda: str('x')),
                              Gene('E', lambda: str('x')),
                              Gene('F', lambda: str('x')),
                              Gene('G', lambda: str('x')),
                              Gene('H', lambda: str('x')),
                              Gene('I', lambda: str('x')),
                              Gene('J', lambda: str('x')),
                              Gene('K', lambda: str('x'))])

        # Get the number of 'unique' genes BEFORE crossover.
        n1 = len(set(parent1.genome))

        # Make a second of parent.
        parent2 = Chromosome([Gene('C', lambda: str('x')),
                              Gene('G', lambda: str('x')),
                              Gene('K', lambda: str('x')),
                              Gene('E', lambda: str('x')),
                              Gene('I', lambda: str('x')),
                              Gene('A', lambda: str('x')),
                              Gene('F', lambda: str('x')),
                              Gene('J', lambda: str('x')),
                              Gene('H', lambda: str('x')),
                              Gene('B', lambda: str('x')),
                              Gene('D', lambda: str('x'))])

        # Get the number of 'unique' genes BEFORE crossover.
        n2 = len(set(parent2.genome))

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
        m1 = len(set(child1.genome))

        # Get the number of 'unique' genes AFTER crossover.
        m2 = len(set(child2.genome))

        self.assertEqual(n1, n2, msg="Parents1/2 genome does not match.")
        self.assertEqual(n1, m1, msg="Parent1/Offspring1 genome does not match after crossover.")

        self.assertEqual(m1, m2, msg="Offsprings1/2 genome does not match.")
        self.assertEqual(n2, m2, msg="Parent2/Offspring2 genome does not match after crossover.")
    # _end_def_

# _end_class_


if __name__ == '__main__':
    unittest.main()
