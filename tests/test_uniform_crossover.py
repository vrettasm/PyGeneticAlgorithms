import unittest
from pygenalgo.genome.gene import Gene
from pygenalgo.genome.chromosome import Chromosome
from pygenalgo.operators.crossover.uniform_crossover import UniformCrossover


class TestUniformCrossover(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        print(">> TestUniformCrossover - START -")

    # _end_def_

    @classmethod
    def tearDownClass(cls) -> None:
        print(">> TestUniformCrossover - FINISH -", end='\n\n')
    # _end_def_

    def setUp(self) -> None:
        """
        Creates the test object with default settings.

        :return: None.
        """
        # Create an object with a crossover probability of 1.0.
        self.cross_op = UniformCrossover(crossover_probability=1.0)
    # _end_def_

    def test_crossover(self):
        """
        The crossover method should be implemented.

        :return: None.
        """

        # Create two dummy test parents.
        parent1 = Chromosome([Gene('a', lambda: str('!')),
                              Gene('b', lambda: str('!')),
                              Gene('c', lambda: str('!')),
                              Gene('d', lambda: str('!')),
                              Gene('e', lambda: str('!')),
                              Gene('f', lambda: str('!')),
                              Gene('g', lambda: str('!')),
                              Gene('h', lambda: str('!')),
                              Gene('i', lambda: str('!')),
                              Gene('k', lambda: str('!'))])

        parent2 = Chromosome([Gene('0', lambda: str('!')),
                              Gene('1', lambda: str('!')),
                              Gene('2', lambda: str('!')),
                              Gene('3', lambda: str('!')),
                              Gene('4', lambda: str('!')),
                              Gene('5', lambda: str('!')),
                              Gene('6', lambda: str('!')),
                              Gene('7', lambda: str('!')),
                              Gene('8', lambda: str('!')),
                              Gene('9', lambda: str('!'))])

        # Print parents BEFORE crossover.
        print("Parent-1: ", " ".join([xi.value for xi in parent1]))
        print("Parent-2: ", " ".join([xi.value for xi in parent2]))

        # Perform the crossover.
        child1, child2 = self.cross_op(parent1, parent2)
        print("---------")

        # Print offsprings AFTER crossover.
        print("Child-1: ", " ".join([xi.value for xi in child1]))
        print("Child-2: ", " ".join([xi.value for xi in child2]))
    # _end_def_

# _end_class_


if __name__ == '__main__':
    unittest.main()
