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

        # Get chromosome values before the mutation.
        c_before = [xi.value for xi in chromo]

        # Print chromosome BEFORE mutation.
        print("Before: ", " ".join(c_before))

        # Perform the mutation (in place).
        self.mut_op(chromo)
        print("-------")

        # Get chromosome values after the mutation.
        c_after = [xi.value for xi in chromo]

        # Print chromosome AFTER mutation.
        print("After : ", " ".join(c_after))

        # Get the lengths of the sets.
        len1 = len(set(c_before))
        len2 = len(set(c_after))

        # Assert that we don't have any losses.
        self.assertEqual(len1, len2)
    # _end_def_

# _end_class_


if __name__ == '__main__':
    unittest.main()
