""" Random graph migration module. """
from typing import Callable
from operator import attrgetter

# Third party imports.
from numpy.typing import NDArray

# Custom code imports.
from pygenalgo.genome.chromosome import Chromosome
from pygenalgo.utils.auxiliary import SubPopulation
from pygenalgo.operators.migration.migration_operator import MigrationOperator


class RandomGraphMigration(MigrationOperator):
    """
    Description:

        Random graph migration implements a migration policy in which
        every island has a direct migration link to every other island
        but at runtime it randomly chooses which links will be active
        and which will not be.
    """

    def __init__(self, migration_probability: float = 0.95) -> None:
        """
        Construct a 'RandomGraphMigration' object with a given
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
                (n, max(island.population, key=key_sort))
                for n, island in enumerate(islands)
            ]

            # Generate a random matrix of True and False.
            neighbour: NDArray = self.rng.integers(0, 2,
                                                   size=(n_active, n_active),
                                                   dtype=bool)
            # Go through all the best chromosomes.
            for n, best_c in best_chromosomes:

                # Check all other destination neighbors.
                for k, is_available in enumerate(neighbour[n]):

                    # Check if the link is available.
                    # Also, avoid self migration.
                    if is_available and k != n:

                        # Get the population size of the destination island.
                        pop_k: int = len(islands[k].population)

                        # Select randomly one individual chromosome.
                        idx: int = self.rng.integers(pop_k, dtype=int)

                        # Replace the randomly selected chromosome with
                        # the pre-selected best one from the list above.
                        islands[k].population[idx] = best_c.clone()
            # _end_for_

            # Increase the migration counter.
            self.inc_counter()
    # _end_def_

# _end_class_
