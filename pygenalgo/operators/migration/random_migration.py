from operator import attrgetter
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

    def migrate(self, islands: list[SubPopulation]) -> None:
        """
        Perform the migration operation on the list of SubPopulations.

        :param islands: list[SubPopulation].

        :return: None.
        """

        # If we have only one active population exit without migration.
        if len(islands) == 1:
            return None
        # _end_if_

        # First find the best individual chromosome FROM EACH island.
        best_chromosomes = [max(island_i.population, key=attrgetter("fitness"))
                            for island_i in islands]

        # Shuffle the order of the best chromosomes list.
        self.rng.shuffle(best_chromosomes)

        # Go through all the islands.
        for island_i, best_j in zip(islands, best_chromosomes):

            # Perform the migration with a predefined probability.
            if self.is_operator_applicable():

                # Select randomly one individual chromosome.
                idx = self.rng.integers(0, len(island_i.population))

                # Replace the randomly selected chromosome with
                # the selected best one from the list above.
                island_i.population[idx] = best_j.clone()
            # _end_if_

        # _end_for_

        # Increase the migration counter.
        self.inc_counter()
    # _end_def_

# _end_class_
