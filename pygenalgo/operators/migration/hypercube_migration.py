""" Hypercube migration module. """
from typing import Callable
from operator import attrgetter

# Custom code imports.
from pygenalgo.genome.chromosome import Chromosome
from pygenalgo.utils.auxiliary import SubPopulation
from pygenalgo.operators.migration.ring_migration import RingMigration
from pygenalgo.operators.migration.migration_operator import MigrationOperator


class HypercubeMigration(MigrationOperator):
    """
    Hypercube island migration.

    For N = pow(2, d) islands, each island is assigned an integer ID
    from 0 to N - 1. Two islands are neighbors when their IDs differ
    in exactly one binary bit.

    If the number of active islands is not a power of two, migration
    falls back to ring migration to preserve gene flow among islands.
    """

    def __init__(self, migration_probability: float = 0.95) -> None:
        """
        Construct a HypercubeMigration object.

        :param migration_probability: (float) in [0, 1].
        """
        super().__init__(migration_probability=migration_probability)

        # Create an auxiliary ring migration operator with 100%
        # probability. It is important because it will act as a
        # safety fallback operator.
        self._items: MigrationOperator = RingMigration(1.0)
    # _end_def_

    def migrate(self, islands: list[SubPopulation]) -> None:
        """
        Perform hypercube migration.

        Each island sends its best chromosome to every hypercube neighbor.
        The receiving island replaces one randomly selected chromosome for
        each incoming migrant.

        :param islands: list[SubPopulation].

        :return: None.
        """
        # Get the size of active islands.
        n_active: int = len(islands)

        # Perform the migration with a specified probability
        # and only if we have more than 1 active populations.
        if self.is_operator_applicable() and n_active > 1:

            # Check if n_active is not a power of 2.
            if (n_active & (n_active - 1)) != 0:
                # Local copy of the safety operator.
                ring_operator = self._items

                # Call its fallback migration policy.
                ring_operator.migrate(islands)

                # Increase the self migration counter.
                self.inc_counter()

                # Exit the call.
                return
            # _end_if_

            # Validate the number of islands and obtain
            # the hypercube dimension.
            dimension: int = n_active.bit_length() - 1

            # Define the key.
            key_sort: Callable = attrgetter("fitness")

            # Snapshot the best chromosome from every island before any
            # migration occurs. This prevents migrations in the current
            # round from affecting later source selections.
            best_chromosomes: list[tuple[int, Chromosome]] = [
                (n, max(island.population, key=key_sort).clone())
                for n, island in enumerate(islands)
            ]

            # Each dimension corresponds to one bit position.
            # Flipping that bit identifies one hypercube neighbor.
            #
            # For island i:
            #
            #     neighbor = i XOR (1 << bit)
            #
            # Example:
            #
            #     i = 2       -> binary 010
            #     bit = 0     -> 010 XOR 001 = 011 -> island 3
            #     bit = 1     -> 010 XOR 010 = 000 -> island 0
            #     bit = 2     -> 010 XOR 100 = 110 -> island 6
            for source_id, best_chromosome in best_chromosomes:
                # Find the destination islands for each source.
                for bit in range(dimension):
                    # Compute the destination index.
                    dest_k: int = source_id ^ (1 << bit)

                    # Local copy of the destination population.
                    dest_population = islands[dest_k].population

                    # Select a random individual in the destination island.
                    idx: int = self.rng.integers(len(dest_population),
                                                 dtype=int)

                    # Insert a clone so that islands do not share
                    # the same mutable chromosome object.
                    dest_population[idx] = best_chromosome.clone()
            # _end_for_

            # Increase the migration counter.
            self.inc_counter()
    # _end_def_

# _end_class_
