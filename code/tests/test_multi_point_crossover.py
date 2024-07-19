import unittest
from src.genome.gene import Gene
from src.genome.chromosome import Chromosome
from src.operators.crossover.mutli_point_crossover import MultiPointCrossover


class TestMultiPointCrossover(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        print(">> TestMultiPointCrossover - START -")

    # _end_def_

    @classmethod
    def tearDownClass(cls) -> None:
        print(">> TestMultiPointCrossover - FINISH -", end='\n\n')
    # _end_def_

    def setUp(self) -> None:
        """
        Creates the test object with default settings.

        :return: None.
        """
        # Create an object with a crossover probability of 1.0.
        self.cross_op = MultiPointCrossover(crossover_probability=1.0)
    # _end_def_

    def test_crossover(self):
        """
        The crossover method should be implemented.

        :return: None.
        """

        # Create two dummy test parents.
        parent1 = Chromosome([Gene('a', lambda: str('x')),
                              Gene('b', lambda: str('x')),
                              Gene('c', lambda: str('x')),
                              Gene('d', lambda: str('x')),
                              Gene('e', lambda: str('x')),
                              Gene('f', lambda: str('x')),
                              Gene('g', lambda: str('x')),
                              Gene('h', lambda: str('x')),
                              Gene('i', lambda: str('x'))])

        parent2 = Chromosome([Gene('1', lambda: str(0)),
                              Gene('2', lambda: str(0)),
                              Gene('3', lambda: str(0)),
                              Gene('4', lambda: str(0)),
                              Gene('5', lambda: str(0)),
                              Gene('6', lambda: str(0)),
                              Gene('7', lambda: str(0)),
                              Gene('8', lambda: str(0)),
                              Gene('9', lambda: str(0))])

        # Print parents BEFORE crossover.
        print("Parent-1: ", " ".join([xi.datum for xi in parent1]))
        print("Parent-2: ", " ".join([xi.datum for xi in parent2]))

        # Perform the crossover.
        child1, child2 = self.cross_op(parent1, parent2, num_loci=3)
        print("---------")

        # Print offsprings AFTER crossover.
        print("Child-1: ", " ".join([xi.datum for xi in child1]))
        print("Child-2: ", " ".join([xi.datum for xi in child2]))
    # _end_def_

# _end_class_


if __name__ == '__main__':
    unittest.main()
