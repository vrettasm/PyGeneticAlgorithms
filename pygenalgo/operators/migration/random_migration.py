""" Random migration module. """
from typing import Callable
from operator import attrgetter

# Custom code imports.
from pygenalgo.genome.chromosome import Chromosome
from pygenalgo.utils.auxiliary import SubPopulation
from pygenalgo.operators.migration.migration_operator import MigrationOperator


class RandomMigration(MigrationOperator):
    """
    Description:

        Random Migration implements a "very basic" migration policy in which
        each island migrates its best chromosome to a randomly selected population.
    """

    def __init__(self, migration_probability: float = 0.95) -> None:
        """
        Construct a 'RandomMigration' object with a given probability value.

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
            # FROM EACH island (and clone it).
            best_chromosomes: list[Chromosome] = [
                max(island.population, key=key_sort).clone()
                for island in islands
            ]

            # Compute all available indices.
            all_indices: list[int] = list(range(n_active))

            # Go through all the indices.
            for source_idx in all_indices:
                # Omit the current index without rebuilding a full list.
                valid_destinations: list[int] = (all_indices[:source_idx] +
                                                 all_indices[source_idx + 1:])

                # Pick a random destination index.
                dest_idx: int = self.rng.choice(valid_destinations)

                # Get the island it points to.
                dest_island = islands[dest_idx]

                # Select the individual with the lowest fitness.
                idx: int = self._find_worst_index(dest_island.population)

                # Overwrite the worst target chromosome with the best.
                dest_island.population[idx] = best_chromosomes[source_idx]
            # _end_for_

            # Increase the migration counter.
            self.inc_counter()
    # _end_def_

# _end_class_
