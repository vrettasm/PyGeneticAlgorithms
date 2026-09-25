import unittest

from pygenalgo.genome.gene import Gene
from pygenalgo.genome.chromosome import Chromosome
from pygenalgo.operators.crossover.cycle_crossover import CycleCrossover


class TestCycleCrossover(unittest.TestCase):

    def test_successful_crossover_cycles(self):
        """
        Verify standard Cycle Crossover correctly alternates parents per cycle.
        """
        # Setup two unique permutation parents.
        p1_genome = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        p2_genome = [4, 1, 2, 8, 9, 6, 7, 3, 5]

        # Dummy function.
        func = lambda x: _

        # Create two parents.
        parent1 = Chromosome([Gene(i, func) for i in p1_genome])
        parent2 = Chromosome([Gene(j, func) for j in p2_genome])

        # Force high probability so crossover always runs during testing.
        cx_operator = CycleCrossover(crossover_probability=1.0)

        # Execute crossover operation.
        child1, child2 = cx_operator.crossover(parent1, parent2)

        # Verify the structure matches the return type hint.
        self.assertIsInstance(child1, Chromosome)
        self.assertIsInstance(child2, Chromosome)

        # Expected outcome following alternating parents per cycle.
        expected_c1 = [1, 2, 3, 4, 9, 6, 7, 8, 5]
        expected_c2 = [4, 1, 2, 8, 5, 6, 7, 3, 9]

        # Verify the genomes matches the expected ones.
        self.assertEqual(child1.values(), expected_c1)
        self.assertEqual(child2.values(), expected_c2)
    # _end_def_

    def test_bypass_when_parents_are_identical(self):
        """
        Verify calculation is skipped if parents match exactly.
        """
        # Test genome.
        p_genome = [1, 2, 3, 4, 5]

        # Dummy function.
        func = lambda x: _

        # Create two identical parents.
        parent1 = Chromosome([Gene(i, func) for i in p_genome])
        parent2 = Chromosome([Gene(i, func) for i in p_genome])

        # Force high probability so crossover always runs during testing.
        cx_operator = CycleCrossover(crossover_probability=1.0)

        # Execute crossover operation.
        child1, child2 = cx_operator.crossover(parent1, parent2)

        # Verify the genomes matches the expected ones.
        self.assertEqual(child1.values(), p_genome)
        self.assertEqual(child2.values(), p_genome)
    # _end_def_

    def test_bypass_when_probability_fails(self):
        """
        Verify calculation is skipped if crossover probability fails.
        """

        # Test genomes.
        p1_genome = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        p2_genome = [9, 8, 7, 6, 5, 4, 3, 2, 1]

        # Dummy function.
        func = lambda x: _

        # Create two parents.
        parent1 = Chromosome([Gene(i, func) for i in p1_genome])
        parent2 = Chromosome([Gene(j, func) for j in p2_genome])

        # Force a 0% probability to skip the operation loop completely.
        cx_operator = CycleCrossover(crossover_probability=0.0)

        # (Not) execute crossover operation.
        child1, child2 = cx_operator.crossover(parent1, parent2)

        # Verify the genomes have been cloned.
        self.assertEqual(child1.values(), p1_genome)
        self.assertEqual(child2.values(), p2_genome)
    # _end_def_

# _end_class_

if __name__ == '__main__':
    unittest.main()

