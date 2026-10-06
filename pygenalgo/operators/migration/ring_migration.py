""" Ring migration module. """
from typing import Callable
from operator import attrgetter

# Custom code imports.
from pygenalgo.genome.chromosome import Chromosome
from pygenalgo.utils.auxiliary import SubPopulation
from pygenalgo.operators.migration.migration_operator import MigrationOperator


class RingMigration(MigrationOperator):
    """
    Description:

        Ring Migration implements a "very basic" migration policy in which
        each island migrates its best chromosome to the population on its
        right, following a "clockwise" (ring) rotation movement.
    """

    def __init__(self, migration_probability: float = 0.95) -> None:
        """
        Construct a 'RingMigration' object with a given probability value.

        :param migration_probability: (float) in [0, 1].
        """
        # Call the super constructor with the provided initial value.
        super().__init__(migration_probability=migration_probability)
    # _end_def_

    def migrate(self, islands: list[SubPopulation]) -> None:
        """
        Perform the migration operation on the list of SubPopulations.

        :param islands: list[SubPopulation].

        :return: None.
        """
        # Get the size of active islands.
        n_active: int = len(islands)

        # Perform the migration with a specified probability
        # and only if we have more than 1 active populations.
        if self.is_operator_applicable() and n_active > 1:
            # Define the key.
            key_sort: Callable = attrgetter("fitness")

            # First find the best individual chromosome
            # FROM EACH island.
            best_chromosomes: list[Chromosome] = [
                max(island.population, key=key_sort).clone()
                for island in islands
            ]

            # Go through all the destination islands.
            for i, island in enumerate(islands):

                # Select the individual with the lowest (worst)
                # fitness to be replaced.
                idx: int = self.find_worst_index(island.population)

                # Replace the worst chromosome with the best one
                # from its left.
                island.population[idx] = best_chromosomes[i - 1]
            # _end_for_

            # Increase the migration counter.
            self.inc_counter()
    # _end_def_

# _end_class_
