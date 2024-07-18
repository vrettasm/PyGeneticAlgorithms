import unittest
from src.genome.chromosome import Chromosome
from src.operators.mutation.mutate_operator import MutationOperator


class TestMutationOperator(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        print(" >> TestMutationOperator - START -")
    # _end_def_

    @classmethod
    def tearDownClass(cls) -> None:
        print(" >> TestMutationOperator - FINISH -")
    # _end_def_

    def setUp(self) -> None:
        """
        Creates the test object with default settings.

        :return: None.
        """
        # Create an object with a valid probability.
        self.mut_op = MutationOperator(mutation_probability=0.5)
    # _end_def_

    def test_mutate(self):
        """
        The mutate method should be implemented.

        :return: None.
        """

        # Create an empty test parent.
        parent1 = Chromosome()

        # Check if the mutate method is implemented.
        with self.assertRaises(NotImplementedError):
            self.mut_op.mutate(parent1)
        # _end_with_

    # _end_def_

# _end_class_


if __name__ == '__main__':
    unittest.main()
