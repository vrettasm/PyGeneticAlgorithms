import unittest
from numpy import nan
from pygenalgo.genome.gene import Gene
from pygenalgo.genome.chromosome import Chromosome
from pygenalgo.engines.generic_ga import GenericGA
from pygenalgo.operators.mutation.mutate_operator import MutationOperator
from pygenalgo.operators.selection.select_operator import SelectionOperator
from pygenalgo.operators.crossover.crossover_operator import CrossoverOperator

class TestGenericGA(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        print(">> TestGenericGA - START -")

    # _end_def_

    @classmethod
    def tearDownClass(cls) -> None:
        print(">> TestGenericGA - FINISH -", end='\n\n')
    # _end_def_

    def setUp(self) -> None:
        """
        Creates the test object with default settings.

        :return: None.
        """
        pop = [Chromosome(_genome=[Gene('a', lambda: str('x')),
                                   Gene('b', lambda: str('x'))], _fitness=nan),
               Chromosome(_genome=[Gene('c', lambda: str('x')),
                                   Gene('d', lambda: str('x'))], _fitness=1.0),
               Chromosome(_genome=[Gene('e', lambda: str('x')),
                                   Gene('f', lambda: str('x'))], _fitness=2.0),
               Chromosome(_genome=[Gene('1', lambda: str('x')),
                                   Gene('2', lambda: str('x'))], _fitness=4.0),
               Chromosome(_genome=[Gene('3', lambda: str('x')),
                                   Gene('4', lambda: str('x'))], _fitness=5.0),
               Chromosome(_genome=[Gene('5', lambda: str('x')),
                                   Gene('6', lambda: str('x'))], _fitness=6.0),
               Chromosome(_genome=[Gene('7', lambda: str('x')),
                                   Gene('8', lambda: str('x'))], _fitness=7.0),
               Chromosome(_genome=[Gene('h', lambda: str('x')),
                                   Gene('i', lambda: str('x'))], _fitness=8.0),
               Chromosome(_genome=[Gene('j', lambda: str('x')),
                                   Gene('l', lambda: str('x'))], _fitness=9.0),
               Chromosome(_genome=[Gene('7', lambda: str('x')),
                                   Gene('8', lambda: str('x'))], _fitness=nan),
               Chromosome(_genome=[Gene('h', lambda: str('x')),
                                   Gene('i', lambda: str('x'))], _fitness=18.0),
               Chromosome(_genome=[Gene('j', lambda: str('x')),
                                   Gene('k', lambda: str('x'))], _fitness=19.0)
               ]
        # Create an object with a genetic probabilities of 1.0.
        self.ga = GenericGA(pop, lambda x: 0.0, SelectionOperator(1.0), MutationOperator(1.0),
                            CrossoverOperator(1.0))
    # _end_def_

    def test_best_chromosome(self):
        """
        Ensure the best_chromosome method behaves
        correctly in the presence of NaN entries.

        :return: None.
        """

        # Find the best chromosome.
        best_p = self.ga.best_chromosome()
        self.assertEqual(19.0, best_p.fitness)

        # Reverse the entries by subtracting the max value.
        for i in self.ga.population:
            i.fitness -= 19.0
        # _end_for_

        # Find the "new" best chromosome.
        best_k = self.ga.best_chromosome()
        self.assertEqual(0.0, best_k.fitness)
    # _end_def_

# _end_class_


if __name__ == '__main__':
    unittest.main()
