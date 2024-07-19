import unittest
from src.genome.gene import Gene
from src.genome.chromosome import Chromosome
from src.engines.auxiliary import apply_corrections


class TestAuxiliary(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        print(" >> TestAuxiliary - START -")

    # _end_def_

    @classmethod
    def tearDownClass(cls) -> None:
        print(" >> TestAuxiliary - FINISH -")

    # _end_def_

    def test_apply_corrections(self):
        """
        Tests the apply_corrections method, creating a 'dummy' population and invalidating manually
        some of the Genes.

        :return: None.
        """

        # Create a test population.
        test_population = [Chromosome([Gene('0', lambda: str('x')),
                                       Gene('1', lambda: str('x')),
                                       Gene('2', lambda: str('x'))]),
                           Chromosome([Gene('a', lambda: str('x')),
                                       Gene('b', lambda: str('x')),
                                       Gene('c', lambda: str('x'))]),
                           Chromosome([Gene('3', lambda: str('x')),
                                       Gene('4', lambda: str('x')),
                                       Gene('5', lambda: str('x'))])
                           ]

        # Run the corrections algorithms.
        t0_corrections = apply_corrections(test_population)

        # There should be exactly '0' corrected Genes.
        self.assertEqual(0, t0_corrections)

        # Now invalidate manually '2' Genes.
        test_population[0][0].valid = False
        test_population[1][1].valid = False

        # Here we set a new Chromosome in the population, where a Gene datum is not
        # initialized. When the datum field is 'None', the gene is assumed 'invalid'.
        test_population[2] = Chromosome([Gene('a0', lambda: str('x')),
                                         Gene('b0', lambda: str('x')),
                                         Gene(None, lambda: str('x'))])

        # Run the corrections algorithms.
        t3_corrections = apply_corrections(test_population)

        # There should be exactly '3' corrected Genes.
        self.assertEqual(3, t3_corrections)
    # _end_def_

# _end_class_


if __name__ == '__main__':
    unittest.main()