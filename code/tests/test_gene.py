import unittest
from numpy.random import randint
from src.genome.gene import Gene

class TestGene(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        print(" >> TestGene - START -")

    # _end_def_

    @classmethod
    def tearDownClass(cls) -> None:
        print(" >> TestGene - FINISH -")
    # _end_def_

    def test_init(self):
        """
        Check if the '_func' is callable.

        :return: None.
        """

        # Check the if "_func" is callable.
        with self.assertRaises(TypeError):
            # '0' is not a callable function.
            _ = Gene(_datum=[1, 0], _func=0, valid=True)
        # _end_with_
    # _end_def_

    def test_add(self):
        """
        Adding two Genes should return a list with two genes.

        :return: None.
        """

        # Create a 'dummy' gene.
        gene_1 = Gene(_datum=0, _func=lambda: randint(5), valid=True)

        # Create a 'dummy' gene.
        gene_2 = Gene(_datum=1, _func=lambda: randint(5), valid=True)

        # Add the two genes, should create a list.
        genome_list = gene_1 + gene_2

        # The result should be of type list.
        self.assertIsInstance(genome_list, list)

        # The size should be equal to 'two'.
        self.assertEqual(2, len(genome_list))

        # Adding a 'Gene' with another object should raise an error.
        with self.assertRaises(TypeError):
            _ = gene_1 + 0.0
        # _end_with_

        # An error will be raised if we try to add a gene to itself.
        with self.assertRaises(TypeError):
            _ = gene_1 + gene_1
        # _end_with_

    # _end_def_

# _end_class_


if __name__ == '__main__':
    unittest.main()
