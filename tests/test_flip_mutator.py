import unittest
from math import fsum
from pygenalgo.genome.gene import Gene
from pygenalgo.genome.chromosome import Chromosome
from pygenalgo.operators.mutation.flip_mutator import FlipMutator


class TestFlipMutator(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        print(">> TestFlipMutator - START -")
    # _end_def_

    @classmethod
    def tearDownClass(cls) -> None:
        print(">> TestFlipMutator - FINISH -", end='\n\n')
    # _end_def_

    def setUp(self) -> None:
        """
        Creates the test object with default settings.

        :return: None.
        """
        # Create an object with a mutation probability of 1.0.
        self.mut_op = FlipMutator(mutate_probability=1.0)
    # _end_def_

    def test_mutate(self):
        """
        The mutation method should be implemented.

        :return: None.
        """

        # Dummy 'random' function.
        def func():
            pass
        # _end_def_

        # Create a dummy test chromosome.
        chromo = Chromosome([Gene(1, func),
                             Gene(1, func),
                             Gene(0, func),
                             Gene(0, func),
                             Gene(0, func),
                             Gene(1, func),
                             Gene(0, func),
                             Gene(1, func),
                             Gene(0, func),
                             Gene(1, func),
                             Gene(1, func),
                             Gene(0, func),
                             Gene(0, func),
                             Gene(0, func),
                             Gene(1, func),
                             Gene(0, func),
                             Gene(1, func),
                             Gene(0, func)], _fitness=0.0)

        # Print chromosome BEFORE mutation.
        list_before = [xi.value for xi in chromo]
        print("Before: ", list_before)

        # Perform the mutation (in place).
        self.mut_op(chromo)
        print("-------")

        # Print chromosome AFTER mutation.
        list_after = [xi.value for xi in chromo]
        print("After : ", list_after)

        # Get the absolute difference from both lists.
        list_diff = [abs(i-j) for i, j in zip(list_before, list_after)]

        # Their difference should be exactly 1.
        self.assertEqual(1.0, fsum(list_diff))
    # _end_def_

# _end_class_


if __name__ == '__main__':
    unittest.main()
