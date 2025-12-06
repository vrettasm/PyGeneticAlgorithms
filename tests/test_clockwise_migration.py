import unittest

from pygenalgo.genome.gene import Gene
from pygenalgo.genome.chromosome import Chromosome

from pygenalgo.utils.auxiliary import SubPopulation
from pygenalgo.operators.migration.clockwise_migration import ClockwiseMigration


class TestClockwiseMigration(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        print(">> TestClockwiseMigration - START -")

    # _end_def_

    @classmethod
    def tearDownClass(cls) -> None:
        print(">> TestClockwiseMigration - FINISH -", end='\n\n')
    # _end_def_

    def setUp(self) -> None:
        """
        Creates the test object with default settings.

        :return: None.
        """
        # Create an object with a migration probability of 1.0.
        self.mig_op = ClockwiseMigration(migration_probability=1.0)
    # _end_def_

    def test_migrate(self):
        """
        The migration method should be implemented.

        :return: None.
        """

        # Create a dummy list of three SubPopulation objects.
        pop = [SubPopulation(0, [Chromosome(genome=[Gene('a', lambda: str('x')),
                                                    Gene('b', lambda: str('x'))], fitness=0),
                                 Chromosome(genome=[Gene('c', lambda: str('x')),
                                                    Gene('d', lambda: str('x'))], fitness=1),
                                 Chromosome(genome=[Gene('e', lambda: str('x')),
                                                    Gene('f', lambda: str('x'))], fitness=2)]
                             ),
               SubPopulation(1, [Chromosome(genome=[Gene('1', lambda: str('x')),
                                                    Gene('2', lambda: str('x'))], fitness=4),
                                 Chromosome(genome=[Gene('3', lambda: str('x')),
                                                    Gene('4', lambda: str('x'))], fitness=5),
                                 Chromosome(genome=[Gene('5', lambda: str('x')),
                                                    Gene('6', lambda: str('x'))], fitness=6)]
                             ),
               SubPopulation(2, [Chromosome(genome=[Gene('7', lambda: str('x')),
                                                     Gene('8', lambda: str('x'))], fitness=7),
                                 Chromosome(genome=[Gene('h', lambda: str('x')),
                                                     Gene('i', lambda: str('x'))], fitness=8),
                                 Chromosome(genome=[Gene('j', lambda: str('x')),
                                                     Gene('l', lambda: str('x'))], fitness=9)]
                             )
               ]

        # Find the best chromosome of each subpopulation.
        best_chromosome = [(i, max(pop_i.population,
                                   key=lambda c: c.fitness)) for i, pop_i in enumerate(pop)]

        # Print best chromosome BEFORE the operation.
        print("Before Migration:")
        print("-----------------")
        for bc in best_chromosome:
            print(f"SubPop-{bc[0]}, with max.fit = {bc[1].fitness}")
        # _end_for_

        # Perform the migration.
        self.mig_op(pop)

        print(" ")

        # Find the best chromosome of each subpopulation.
        best_chromosome = [(i, max(pop_i.population,
                                   key=lambda c: c.fitness)) for i, pop_i in enumerate(pop)]

        # Print best chromosome AFTER the operation.
        print("After Migration:")
        print("----------------")
        for bc in best_chromosome:
            print(f"SubPop-{bc[0]}, with max.fit = {bc[1].fitness}")
        # _end_for_

    # _end_def_

# _end_class_


if __name__ == '__main__':
    unittest.main()
