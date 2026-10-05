""" Fully connected migration module. """
from typing import Callable
from operator import attrgetter

# Custom code imports.
from pygenalgo.genome.chromosome import Chromosome
from pygenalgo.utils.auxiliary import SubPopulation
from pygenalgo.operators.migration.migration_operator import MigrationOperator


class FullyConnectedMigration(MigrationOperator):
    """
    Description:

        Fully connected migration implements a migration policy in which
        every island has a direct migration link to every other island.

        The main advantage is rapid sharing of good genetic material, while
        the disadvantage is that it can cause all islands to become similar
        quickly, reducing diversity.
    """

    def __init__(self, migration_probability: float = 0.95) -> None:
        """
        Construct a 'FullyConnectedMigration' object with a given
        probability value.

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
            best_chromosomes: list[tuple[int, Chromosome]] = [
                (island.id, max(island.population, key=key_sort))
                for island in islands
            ]

            # Go through all the best chromosomes.
            for c_id, best_c in best_chromosomes:

                # Check all destination islands.
                for island in islands:

                    # Prevents self migration.
                    if island.id == c_id:
                        continue

                    # Get the population size of the destination island.
                    pop_size: int = len(island.population)

                    # Select randomly one individual chromosome.
                    idx: int = self.rng.integers(pop_size, dtype=int)

                    # Replace the randomly selected chromosome with
                    # the pre-selected best one from the list above.
                    island.population[idx] = best_c.clone()
            # _end_for_

            # Increase the migration counter.
            self.inc_counter()
    # _end_def_

# _end_class_
