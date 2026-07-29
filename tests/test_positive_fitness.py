import unittest

import numpy as np
from pygenalgo.genome.gene import Gene
from pygenalgo.genome.chromosome import Chromosome
from pygenalgo.operators.selection.select_operator import ensure_positive_fitness


class TestPositiveFitness(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        print(">> TestPositiveFitness - START -")
    # _end_def_

    @classmethod
    def tearDownClass(cls) -> None:
        print(">> TestPositiveFitness - FINISH -", end='\n\n')
    # _end_def_

    def setUp(self) -> None:
        """
        Creates the test object with default settings.

        :return: None.
        """
        # Local dummy function.
        def func():
            pass

        # Create a demo population.
        self.population = [
            Chromosome(Gene(i, func), None, True)
            for i in range(10)
        ]
    # _end_def_

    def test_shift_up_all_positive(self):
        """
        Verifies that all-positive populations
        bypass transformation entirely.

        :return: None.
        """

        # Store the expected fit values.
        expected = []

        # Set the fitness values.
        for n, p in enumerate(self.population,
                              start=1):
            p.fitness = n
            expected.append(n)
        # _end_for_

        result = ensure_positive_fitness(self.population,
                                         mode="shift_up")

        self.assertTrue(np.all(result == expected))
    # _end_def_

    def test_shift_up_all_negative(self):
        """
        Verifies that all-negative
        populations shift up entirely.

        :return: None.
        """

        # Store the expected fit values.
        expected: list[float] = []

        # Population size.
        p_size: int = len(self.population)

        # Set the fitness values.
        for n, p in enumerate(self.population,
                              start=1):
            p.fitness = -n
            expected.append(p_size - n + 1)
        # _end_for_

        result = ensure_positive_fitness(self.population,
                                         mode="shift_up")

        self.assertTrue(np.all(result == expected))
    # _end_def_

    def test_shift_up_all_zero(self):
        """
        Verifies that all-zero populations
        are treated correctly.

        :return: None.
        """

        # Store the expected fit values.
        expected: list[float] = []

        # Set the fitness values.
        for n, p in enumerate(self.population,
                              start=0):
            p.fitness = 0
            expected.append(1)
        # _end_for_

        result = ensure_positive_fitness(self.population,
                                         mode="shift_up")

        self.assertTrue(np.all(result == expected))
    # _end_def_

# _end_class_


if __name__ == '__main__':
    unittest.main()
