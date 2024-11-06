import unittest
from numpy.random import randint
from pygenalgo.genome.gene import Gene


class TestGene(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        print(">> TestGene - START -")
    # _end_def_

    @classmethod
    def tearDownClass(cls) -> None:
        print(">> TestGene - FINISH -", end='\n\n')
    # _end_def_

    def test_init(self):
        """
        Check if the '_func' is callable.

        :return: None.
        """

        # Check if "_func" is callable.
        with self.assertRaises(TypeError):

            # '0' is not a callable function.
            _ = Gene(_datum=[1, 0], _func=0)
        # _end_with_
    # _end_def_

    def test_valid(self):
        """
        Test the valid flag.

        :return: None.
        """

        # Make a new test Gene, with _datum='None'.
        gene_1 = Gene(None, _func=lambda: randint(5))

        # Even though we have set the 'valid=True', because
        # of the 'datum=None' the Gene will be invalidated.
        self.assertFalse(gene_1.valid)
    # _end_def__

    def test_add(self):
        """
        Adding two Genes should return a list with two genes.

        :return: None.
        """

        # Create a 'dummy' gene.
        gene_1 = Gene(_datum=0, _func=lambda: randint(5))

        # Create a 'dummy' gene.
        gene_2 = Gene(_datum=1, _func=lambda: randint(5))

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

    def test_clone(self):
        """
        Make sure the clone method is working as intended.

        :return:
        """

        # Number of Genes.
        M = 5

        # Create a "test gene".
        gene_1 = Gene([i for i in range(M)], lambda: randint(M))

        # Make a "clone" of the first gene.
        gene_2 = gene_1.clone()

        # Check if the objects are equal.
        self.assertEqual(gene_1, gene_2)

        # Check if the objects are the same.
        self.assertTrue(gene_1 is not gene_2)
    # _end_def_

# _end_class_


if __name__ == '__main__':
    unittest.main()
