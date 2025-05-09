from pygenalgo.engines.auxiliary import SubPopulation
from pygenalgo.operators.genetic_operator import GeneticOperator


class MigrationOperator(GeneticOperator):
    """
    Description:

        Provides the base class (interface) for a Migration Operator.
    """

    def __init__(self, migration_probability: float):
        """
        Construct a 'MigrationOperator' object with a given
        probability value.

        :param migration_probability: (float).
        """

        # Call the super constructor with the provided probability value.
        super().__init__(migration_probability)
    # _end_def_

    def migrate(self, islands: list[SubPopulation]) -> None:
        """
        Abstract method that "reminds" the user that if they want to
        create a Migration Class that inherits from here they should
        implement a migrate method.

        :param islands: list[SubPopulation].

        :return: Nothing but raising an error.
        """
        raise NotImplementedError(f"{self.__class__.__name__}: "
                                  f"You should implement this method!")
    # _end_def_

    def __call__(self, *args, **kwargs) -> None:
        """
        This is only a wrapper of the "migrate" method.
        """
        return self.migrate(*args, **kwargs)
    # _end_def_

# _end_class_
