""" Migration operator module. """
# Custom code imports.
from pygenalgo.genome.chromosome import Chromosome
from pygenalgo.utils.auxiliary import SubPopulation
from pygenalgo.operators.genetic_operator import GeneticOperator


class MigrationOperator(GeneticOperator):
    """
    Description:

        Provides the base class (interface) for a Migration Operator.
    """

    def __init__(self, migration_probability: float) -> None:
        """
        Construct a 'MigrationOperator' object with a given
        probability value.

        :param migration_probability: (float).
        """
        # Call the super constructor with the provided initial value.
        super().__init__(probability=migration_probability)
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

    @staticmethod
    def find_best_index(population: list[Chromosome])-> int:
        """
        Finds the index of the chromosome with the highest
        fitness value within the given population.

        :param population: A list of Chromosome objects to
                           evaluate.
        :return: The index (int) of the chromosome with the
                 maximum fitness.
        """
        return max(enumerate(population),
                   key=lambda x: x[1].fitness
                   )[0]
    # _end_def_

    @staticmethod
    def find_worst_index(population: list[Chromosome]) -> int:
        """
        Finds the index of the chromosome with the lowest
        fitness value within the given population.

        :param population: A list of Chromosome objects to
                           evaluate.
        :return: The index (int) of the chromosome with the
                 lowest fitness.
        """
        return min(enumerate(population),
                   key=lambda x: x[1].fitness
                   )[0]
    # _end_def_

    def __call__(self, *args, **kwargs) -> None:
        """
        This is only a wrapper of the "migrate" method.
        """
        return self.migrate(*args, **kwargs)
    # _end_def_

# _end_class_
