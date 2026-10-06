""" Star connected migration module. """
from operator import attrgetter

# Custom code imports.
from pygenalgo.genome.chromosome import Chromosome
from pygenalgo.utils.auxiliary import SubPopulation
from pygenalgo.operators.migration.migration_operator import MigrationOperator


class StarConnectedMigration(MigrationOperator):
    """
    Description:

        Star connected migration implements a migration policy in which
        a randomly selected island has a direct migration link to every
        other island.
    """

    def __init__(self, migration_probability: float = 0.95) -> None:
        """
        Construct a 'StarConnectedMigration' object with a given
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
            # Select randomly one island as a hub.
            hub_k: int = self.rng.integers(n_active, dtype=int)

            # Then find its best individual chromosome.
            best_chromosome: Chromosome = max(islands[hub_k].population,
                                              key=attrgetter("fitness"))
            # Go through all the other destinations.
            for n, island in enumerate(islands):
                # Skip self migration.
                if n == hub_k:
                    continue

                # Select the individual with the lowest (worst)
                # fitness to be replaced.
                idx: int = self._find_worst_index(island.population)

                # Replace the randomly selected chromosome with
                # the pre-selected best one from the list above.
                island.population[idx] = best_chromosome.clone()
            # _end_for_

            # Increase the migration counter.
            self.inc_counter()
    # _end_def_

# _end_class_
