import unittest
from pygenalgo.genome.gene import Gene
from pygenalgo.genome.chromosome import Chromosome
from pygenalgo.operators.mutation.swap_mutator import SwapMutator


class TestSwapMutator(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        print(">> TestSwapMutator - START -")

    # _end_def_

    @classmethod
    def tearDownClass(cls) -> None:
        print(">> TestSwapMutator - FINISH -", end='\n\n')
    # _end_def_

    def setUp(self) -> None:
        """
        Creates the test object with default settings.

        :return: None.
        """
        # Create an object with a mutation probability of 1.0.
        self.mut_op = SwapMutator(mutate_probability=1.0)
    # _end_def_

    def test_mutate(self):
        """
        The mutation method should be implemented.

        :return: None.
        """

        # Create a dummy test chromosome.
        chromo = Chromosome([Gene('a', lambda: str('_')),
                             Gene('b', lambda: str('_')),
                             Gene('c', lambda: str('_')),
                             Gene('d', lambda: str('_')),
                             Gene('e', lambda: str('_')),
                             Gene('f', lambda: str('_')),
                             Gene('g', lambda: str('_')),
                             Gene('h', lambda: str('_')),
                             Gene('i', lambda: str('_')),
                             Gene('j', lambda: str('_')),
                             Gene('k', lambda: str('_'))])

        # Print chromosome BEFORE mutation.
        print("Before: ", " ".join([xi.value for xi in chromo]))

        # Perform the mutation (in place).
        self.mut_op(chromo)
        print("-------")

        # Print chromosome AFTER mutation.
        print("After : ", " ".join([xi.value for xi in chromo]))
    # _end_def_

# _end_class_


if __name__ == '__main__':
    unittest.main()
