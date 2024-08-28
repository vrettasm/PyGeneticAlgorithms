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

        # Create a 'dummy' chromosome.
        ch1 = Chromosome()

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
        ch_1 = Chromosome(_genome=[Gene(0, func),
                                   Gene(1, func),
                                   Gene(2, func)],
                          _fitness=0.0, _valid=True)

        # This genome SHOULD be valid.
        self.assertTrue(ch_1.is_genome_valid())

        # Let's invalidate the genome by flipping one gene.
        ch_1[0].is_valid = False

        # This genome SHOULD NOT be valid.
        self.assertFalse(ch_1.is_genome_valid())
    # _end_def_

    def test_humming_distance(self):
        """
        ...

        :return: None.
        """

        # Create a 'dummy' random function.
        def func():
            return randint(5)
        # _end_def_

        # Create a 'dummy' chromosome with 3 'genes'.
        ch_1 = Chromosome(_genome=[Gene(0, func),
                                   Gene(1, func),
                                   Gene(2, func)],
                          _fitness=0.0, _valid=True)

        # There are '0' dissimilarities when we compare
        # the same chromosome(s).
        self.assertEqual(0, ch_1.hamming_distance(ch_1))

        # Create a 'dummy' chromosome with 3 'genes'.
        ch_2 = Chromosome(_genome=[Gene(3, func),
                                   Gene(4, func),
                                   Gene(5, func)],
                          _fitness=0.0, _valid=True)

        # All the genes are different here.
        self.assertEqual(3, ch_1.hamming_distance(ch_2))
    # _end_def_

# _end_class_


if __name__ == '__main__':
    unittest.main()
