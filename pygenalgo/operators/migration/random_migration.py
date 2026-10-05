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
        # Perform the migration with a specified probability
        # and only if we have more than 1 active populations.
        if self.is_operator_applicable() and len(islands) > 1:
            # Define the key.
            key_sort: Callable = attrgetter("fitness")

            # First find the best individual chromosome
            # FROM EACH island.
            best_chromosomes: list[tuple[int, Chromosome]] = [
                (island.id, max(island.population, key=key_sort))
                for island in islands
            ]

            # Shuffle the order of the best chromosomes
            # list to introduce some randomness.
            self.rng.shuffle(best_chromosomes)

            # Go through all the destination islands.
            for island, best_j in zip(islands, best_chromosomes):
                # Extract the values of the tuple.
                n, best_c = best_j

                # Prevents self migration.
                if island.id == n:
                    continue

                # Get the population size of the island.
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
