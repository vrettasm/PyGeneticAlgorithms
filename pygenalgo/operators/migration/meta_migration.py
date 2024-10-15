from pygenalgo.engines.auxiliary import SubPopulation
from pygenalgo.operators.migration.migration_operator import MigrationOperator
from pygenalgo.operators.migration.random_migration import RandomMigration
from pygenalgo.operators.migration.clockwise_migration import ClockwiseMigration


class MetaMigration(MigrationOperator):
    """
    Description:

        Meta-migrator, performs the migration betweem the sub-populations by applying
        randomly all other migrators (one at a time), with equal probability.

        NOTE: In the future the equal probabilities can be amended.
    """

    def __init__(self, migration_probability: float = 0.95):
        """
        Construct a 'MetaMigration' object with a predefined probability value.

        :param migration_probability: (float).
        """

        # Call the super constructor with the provided probability value.
        super().__init__(migration_probability)

        # NOTE: In here the migration probabilities for each policy are set to 1.0.
        self._items = (RandomMigration(1.0), ClockwiseMigration(1.0))
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

        # If the migration probability is higher than
        # a uniformly random value, make the transfer.
        if self.probability > self.rng.random():

            # Select randomly with equal probability
            # a method and call its migrate method.
            self.rng.choice(self.items).migrate(islands)

            # Increase the mutator counter.
            self.inc_counter()
        # _end_if_

    # _end_def_

    @property
    def all_counters(self) -> dict:
        """
        Accessor (getter) of the application counter from all the internal migrators.
        This is mostly to verify that everything is working as expected.

        :return: a dictionary with the counter calls for all migrator methods.
        """
        return {mig_op.__class__.__name__: mig_op.counter for mig_op in self.items}
    # _end_def_

    def reset_counter(self) -> None:
        """
        Sets ALL the counters to 'zero'. We have to override the super().reset_counter()
        method, because we have to call explicitly the reset_counter on all the internal
        operators.

        :return: None.
        """

        # First call the super() to reset the self internal counter.
        super().reset_counter()

        # Here call explicitly the reset on each of the internal mutators.
        for op in self.items:
            op.reset_counter()
        # _end_for_

    # _end_def_

# _end_class_
