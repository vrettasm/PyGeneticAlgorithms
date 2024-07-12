from src.engines.auxiliary import SubPopulation
from src.operators.migration.migration_operator import MigrationOperator


class ClockwiseMigration(MigrationOperator):
    """
    Description:

        Clockwise Migration implements a "very basic" migration policy in which
        each island migrates its best chromosome to the population on its right,
        following a "clockwise" rotation movement.

    """

    def __init__(self, migration_probability: float = 0.95):
        """
        Construct a 'SinglePointCrossover' object with a given probability value.

        :param migration_probability: (float).
        """

        # Call the super constructor with the provided
        # probability value.
        super().__init__(migration_probability)
    # _end_def_

    def migrate(self, island: list[SubPopulation]) -> None:
        """
        Perform the migration operation on the list of SubPopulations.

        :param island: list[SubPopulation].

        :return: None.
        """

        # First find the best individual chromosome FROM EACH island.
        best_chromosome = [max(pop_i.population,
                               key=lambda c: c.fitness) for pop_i in island]

        # Go through all the islands.
        for i, pop_i in enumerate(island):

            # Perform the migration with a predefined probability.
            if self.probability > self.rng.random():

                # Select randomly one individual chromosome.
                locus = self.rng.integers(0, len(pop_i.population))

                # Replace the chromosome with the best one from its left.
                pop_i.population[locus] = best_chromosome[i-1]
            # _end_if_

        # _end_for_

        # Increase the migration counter.
        self.inc_counter()
    # _end_def_

# _end_class_
