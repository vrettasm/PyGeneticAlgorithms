from src.engines.auxiliary import SubPopulation
from src.operators.genetic_operator import GeneticOperator


class MigrationOperator(GeneticOperator):

    def __init__(self, migration_probability):
        """
        Construct a 'MigrationOperator' object with a given
        probability value.

        :param migration_probability: (float).
        """

        # Call the super constructor with the provided probability value.
        super().__init__(migration_probability)
    # _end_def_

    def migrate(self, islands: list[SubPopulation]):
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

    def __call__(self, *args, **kwargs):
        """
        This is only a wrapper of the "crossover" method.
        """
        return self.migrate(*args, **kwargs)
    # _end_def_

# _end_class_
