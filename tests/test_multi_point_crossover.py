import unittest
from pygenalgo.genome.gene import Gene
from pygenalgo.genome.chromosome import Chromosome
from pygenalgo.operators.crossover.multi_point_crossover import MultiPointCrossover


class TestMultiPointCrossover(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        print(">> TestMultiPointCrossover - START -")

    # _end_def_

    @classmethod
    def tearDownClass(cls) -> None:
        print(">> TestMultiPointCrossover - FINISH -", end='\n\n')
    # _end_def_

    def setUp(self):
        """
        Set up mock parents and the crossover operator.
        """
        # Setup two unique permutation parents.
        p1_genome = [1, 2, 3, 4, 5]
        p2_genome = [6, 7, 8, 9, 0]

        # Dummy function.
        func = lambda x: _

        # Create two parents.
        self.parent1 = Chromosome([Gene(i, func) for i in p1_genome])
        self.parent2 = Chromosome([Gene(j, func) for j in p2_genome])

        # Basic instantiation (default 2 points)
        self.operator = MultiPointCrossover(crossover_probability=1.0, n_points=2)
    # _end_def_

    def test_constructor_enforces_minimum_points(self):
        """
        Ensure that n_points lower than 2 defaults to 2.
        """
        # Create an operator with low value.
        op_low = MultiPointCrossover(n_points=1)

        # Should default to 2.
        self.assertEqual(op_low._items, 2)
    # _end_def_

    def test_crossover_not_applicable(self):
        """
        Ensure original clones are returned if operator is not applicable.
        """
        # Set the probability to zero.
        self.operator.probability = 0.0

        # Crossover should not happen here.
        ch1, ch2 = self.operator.crossover(self.parent1, self.parent2)

        # Each child should be a clone of its parent.
        self.assertEqual(ch1, self.parent1)
        self.assertEqual(ch2, self.parent2)

        # Verifies that they are not the same object.s
        self.assertIsNot(ch1, self.parent1)
        self.assertIsNot(ch2, self.parent2)

        # Restore the probability to one.
        self.operator.probability = 1.0
    # _end_def_

    def test_crossover_identical_parents(self):
        """
        Ensure clones are returned if parents are identical.
        """
        # Perform the crossover one the same parent.
        ch1, ch2 = self.operator.crossover(self.parent1, self.parent1)

        # Each child should be a clone of its parent.
        self.assertEqual(ch1, self.parent1)
        self.assertEqual(ch2, self.parent1)

        # Verifies that they are not the same object.s
        self.assertIsNot(ch1, self.parent1)
        self.assertIsNot(ch2, self.parent1)
    # _end_def_

    def test_crossover_points_exceed_length_raises_error(self):
        """
        Ensure ValueError is raised if requested cuts exceed chromosome length.
        """
        # 5 cut points on chromosome of length 5 should fail.
        test_operator = MultiPointCrossover(crossover_probability=1.0, n_points=5)

        with self.assertRaises(ValueError):
            _ = test_operator.crossover(self.parent1, self.parent2)
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
        print("Parent-1: ", " ".join([xi.value for xi in parent1]))
        print("Parent-2: ", " ".join([xi.value for xi in parent2]))

        # Perform the crossover.
        child1, child2 = self.operator.crossover(parent1, parent2)
        print("---------")

        # Print offsprings AFTER crossover.
        print("Child-1: ", " ".join([xi.value for xi in child1]))
        print("Child-2: ", " ".join([xi.value for xi in child2]))
        print(" ")
    # _end_def_

    def test_uneven_chromosomes(self):
        """
        Test the crossover operator with
        chromosomes of different sizes.

        :return: None.
        """

        # Create two dummy test parents.
        parent1 = Chromosome([Gene('a', lambda: str('!')),
                              Gene('b', lambda: str('!')),
                              Gene('c', lambda: str('!')),
                              Gene('d', lambda: str('!')),
                              Gene('e', lambda: str('!')),
                              Gene('f', lambda: str('!'))])

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
        child1, child2 = self.operator.crossover(parent1, parent2)
        print("---------")

        # Print offsprings AFTER crossover.
        print("Child-1: ", " ".join([xi.value for xi in child1]))
        print("Child-2: ", " ".join([xi.value for xi in child2]))
        print(" ")
    # _end_def_

# _end_class_


if __name__ == '__main__':
    unittest.main()
