from pygenalgo.genome.chromosome import Chromosome
from pygenalgo.operators.genetic_operator import GeneticOperator


class SelectionOperator(GeneticOperator):
    """
    Description:

        Provides the base class (interface) for a Selection Operator.  Note that even
        though the operator accepts a probability value, for the moment this operator
        is applied with 100% 'probability'.
    """

    def __init__(self, selection_probability: float) -> None:
        """
        Construct a 'SelectionOperator' object with a
        given probability value.

        :param selection_probability: (float).
        """
        # Call the super constructor with the provided
        # probability value.
        super().__init__(selection_probability)
    # _end_def_

    def select(self, population: list[Chromosome]) -> list[Chromosome]:
        """
        Abstract method that "reminds" the user that if they want to
        create a Selection Class that inherits from here they should
        implement a select method.

        :param population: is a list, with the chromosomes, to select
                           he parents for the next generation

        :return: Nothing but raising an error.
        """
        raise NotImplementedError(f"{self.__class__.__name__}: "
                                  f"You should implement this method!")
    # _end_def_

    def __call__(self, *args, **kwargs) -> list[Chromosome]:
        """
        This is only a wrapper of the "select" method.
        """
        return self.select(*args, **kwargs)
    # _end_def_

# _end_class_
