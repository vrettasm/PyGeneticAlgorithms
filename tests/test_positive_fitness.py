import unittest

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

        self.assertEqual(result, expected)
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

        self.assertEqual(result, expected)
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
        for p in self.population:
            p.fitness = 0
            expected.append(1)
        # _end_for_

        result = ensure_positive_fitness(self.population,
                                         mode="shift_up")

        self.assertEqual(result, expected)
    # _end_def_

    def test_linear_scaled_normalizes_relative_to_worst(self) -> None:
        """
        Checks if linear scaling forces the worst performer to a baseline of 1.0.
        """
        # Minimum is 10.5.
        # Expected: [10.5-10.5+1, 100-10.5+1, 55.2-10.5+1]
        expected = [1.0, 90.5, 45.7]

        # Local dummy function.
        def func():
            pass

        population = [
            Chromosome(Gene(0, func), f_value, True)
            for f_value in expected
        ]

        result = ensure_positive_fitness(population, mode="linear_scaled")

        self.assertEqual(result, expected)
    # _end_def_

    def test_linear_scaled_handles_uniform_populations(self) -> None:
        """
        Ensures division by zero is safely avoided if everyone has equal scores.
        """
        expected = [1.0, 1.0]

        # Local dummy function.
        def func():
            pass

        population = [
            Chromosome(Gene(0, func), f_value, True)
            for f_value in expected
        ]

        result = ensure_positive_fitness(population, mode="linear_scaled")

        self.assertEqual(result, expected)
    # _end_def_

    def test_invalid_mode_raises_value_error(self) -> None:
        """
        Ensures an unrecognized processing mode forces a clear failure.
        """
        with self.assertRaises(ValueError):
            ensure_positive_fitness(self.population, mode="unsupported_mode")
    # _end_def_

# _end_class_


if __name__ == '__main__':
    unittest.main()
