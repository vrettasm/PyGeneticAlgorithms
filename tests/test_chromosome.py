import unittest
from numpy.random import randint
from pygenalgo.genome.gene import Gene
from pygenalgo.genome.chromosome import Chromosome


class TestChromosome(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        print(">> TestChromosome - START -")
    # _end_def_

    @classmethod
    def tearDownClass(cls) -> None:
        print(">> TestChromosome - FINISH -", end='\n\n')
    # _end_def_

    def test_fitness(self):
        """
        Check if the 'fitness' is float.

        :return: None.
        """
        # Dummy genome with two genes.
        genome = [Gene('1', lambda x: int(x), True),
                  Gene('2', lambda x: int(x), True)]

        # Create a 'dummy' chromosome.
        ch1 = Chromosome(genome)

        # Check if "fitness" is float/int.
        with self.assertRaises(TypeError):

            # We allow only float/int.
            ch1.fitness = '0.98'
        # _end_with_
    # _end_def_

    def test_genome_validity(self):
        """
        Check if the genome is valid.

        :return: None.
        """

        # Create a 'dummy' random function.
        def func():
            return randint(5)
        # _end_def_

        # Create a 'dummy' chromosome with 3 'genes'.
        ch_1 = Chromosome(genome=[Gene(0, func),
                                   Gene(1, func),
                                   Gene(2, func)],
                          fitness=0.0, valid=True)

        # This genome SHOULD be valid.
        self.assertTrue(ch_1.has_valid_genome())

        # Let's invalidate the genome by flipping one gene.
        ch_1[0].is_valid = False

        # This genome SHOULD NOT be valid.
        self.assertFalse(ch_1.has_valid_genome())
    # _end_def_

    def test_clone(self):
        """
        Make sure the clone method is working as intended.

        :return: None.
        """

        # Number of Genes.
        num_genes = 10

        # Create a "test" chromosome.
        chromo_1 = Chromosome(genome=[Gene(i, lambda: randint(num_genes))
                                      for i in range(num_genes)])

        # Make a "clone" of the first chromosome.
        chromo_2 = chromo_1.clone()

        # Check if the objects are equal.
        self.assertEqual(chromo_1, chromo_2)

        # Check if the objects are the same.
        self.assertTrue(chromo_1 is not chromo_2)
    # _end_def_

    def test_equal(self):
        """
        Make sure the equal method is working as intended.

        :return: None.
        """

        # Number of Genes.
        num_genes = 5

        # Create a "test" chromosome.
        chromo_1 = Chromosome([Gene(i, lambda: randint(num_genes))
                               for i in range(num_genes)])

        # Make a "clone" of the first chromosome.
        chromo_2 = chromo_1.clone()

        # Check if the objects are equal.
        self.assertEqual(chromo_1, chromo_2)

        # Make a change in the second chromosome.
        chromo_2[0] = Gene(100, lambda: randint(num_genes))

        # Check if the objects are equal.
        self.assertNotEqual(chromo_1, chromo_2)
    # _end_def_

    def test_humming_distance(self):
        """
        Check the Hamming distances of two identical
        and two completely different chromosomes.

        :return: None.
        """

        # Create a 'dummy' random function.
        def func():
            return 0
        # _end_def_

        # Create a 'dummy' chromosome with 3 'genes'.
        ch_1 = Chromosome(genome=[Gene(0, func),
                                  Gene(1, func),
                                  Gene(2, func)])

        # There are '0' dissimilarities when we compare
        # the same chromosome(s).
        self.assertEqual(0, ch_1.hamming_distance(ch_1))

        # Create a 'dummy' chromosome with 3 'genes'.
        ch_2 = Chromosome(genome=[Gene(3, func),
                                  Gene(4, func),
                                  Gene(5, func)])

        # All the genes are different here.
        self.assertEqual(3, ch_1.hamming_distance(ch_2))

        # Create a 'dummy' chromosome with 2 'genes'.
        ch_3 = Chromosome(genome=[Gene(3, func),
                                  Gene(4, func)])

        # Check if the chromosomes have the same length.
        with self.assertRaises(ValueError):
            ch_1.hamming_distance(ch_3)
    # _end_def_

# _end_class_


if __name__ == '__main__':
    unittest.main()
