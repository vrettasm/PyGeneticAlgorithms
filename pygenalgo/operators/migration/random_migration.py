from pygenalgo.engines.auxiliary import SubPopulation
from pygenalgo.operators.migration.migration_operator import MigrationOperator


class RandomMigration(MigrationOperator):
    """
    Description:

        Random Migration implements a "very basic" migration policy in which
        each island migrates its best chromosome to a randomly selected population.
    """

    def __init__(self, migration_probability: float = 0.95):
        """
        Construct a 'RandomMigration' object with a given probability value.

        :param migration_probability: (float) in [0, 1].
        """

        # Call the super constructor with the provided probability value.
        super().__init__(migration_probability)
    # _end_def_

    def migrate(self, island: list[SubPopulation]) -> None:
        """
        Perform the migration operation on the list of SubPopulations.

        :param island: list[SubPopulation].

        :return: None.
        """

        # If we have only one active population exit without migration.
        if len(island) == 1:
            return None
        # _end_if_

        # First find the best individual chromosome FROM EACH island.
        best_chromosome = [max(pop_i.population,
                               key=lambda c: c.fitness) for pop_i in island]

        # Shuffle the order of the best chromosomes list.
        self.rng.shuffle(best_chromosome)

        # Go through all the islands.
        for i, (pop_i, best_j) in enumerate(zip(island, best_chromosome)):

            # Perform the migration with a predefined probability.
            if self.probability > self.rng.random():

                # Select randomly one individual chromosome.
                locus = self.rng.integers(0, len(pop_i.population))

                # Replace the chromosome with the randomly
                # selected best one from the list above.
                pop_i.population[locus] = best_j.clone()
            # _end_if_

        # _end_for_

        # Increase the migration counter.
        self.inc_counter()
    # _end_def_

# _end_class_
