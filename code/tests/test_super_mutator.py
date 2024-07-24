import unittest
from src.genome.gene import Gene
from src.genome.chromosome import Chromosome
from src.operators.mutation.super_mutator import SuperMutator


class TestSuperMutator(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        print(">> TestSuperMutator - START -")

    # _end_def_

    @classmethod
    def tearDownClass(cls) -> None:
        print(">> TestSuperMutator - FINISH -", end='\n\n')
    # _end_def_

    def setUp(self) -> None:
        """
        Creates the test object with default settings.

        :return: None.
        """
        # Create an object with a mutation probability of 1.0.
        self.mut_op = SuperMutator(mutate_probability=1.0)
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
                             Gene('j', lambda: str('_'))])

        # Print chromosome BEFORE mutation.
        print("Before: ", " ".join([xi.datum for xi in chromo]))

        # Perform the mutation 10 times.
        for i in range(10):

            # Perform the mutation (in place).
            self.mut_op(chromo)
        # _end_for_

        print("-------")

        # Print chromosome AFTER mutation.
        print("After : ", " ".join([xi.datum for xi in chromo]))

        # Get the counter values.
        all_counters = 0
        for _, counter in self.mut_op.all_counters.items():
            all_counters += counter
        # _end_for_

        # Make sure the counters add-up.
        self.assertEqual(all_counters, self.mut_op.counter,
                         "Operator counters do not add up.")

        # Reset the counters.
        self.mut_op.reset_counter()

        # Make sure the self.counter is zero.
        self.assertEqual(0, self.mut_op.counter)

        # Get the counter values.
        all_counters = 0
        for _, counter in self.mut_op.all_counters.items():
            all_counters += counter
        # _end_for_

        # Make sure the counters are '0'.
        self.assertEqual(0, all_counters)

    # _end_def_

# _end_class_


if __name__ == '__main__':
    unittest.main()
