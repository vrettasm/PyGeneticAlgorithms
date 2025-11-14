from functools import cache
from pygenalgo.genome.chromosome import Chromosome
from pygenalgo.operators.genetic_operator import GeneticOperator


class SelectionOperator(GeneticOperator):
    """
    Description:

        Provides the base class (interface) for a Selection Operator.  Note that even
        though the operator accepts a probability value, for the moment this operator
        is applied with 100% 'probability'.
    """

    def __init__(self, selection_probability: float):
        """
        Construct a 'SelectionOperator' object with a
        given probability value.

        :param selection_probability: (float).
        """

        # Call the super constructor with the provided
        # probability value.
        super().__init__(selection_probability)
    # _end_def_

    def select(self, population: list[Chromosome]):
        """
        Abstract method that "reminds" the user that if they want to
        create a Selection Class that inherits from here they should
        implement a select method.

        :param population: is a list, with the chromosomes, to select
        the parents for the next generation

        :return: Nothing but raising an error.
        """
        raise NotImplementedError(f"{self.__class__.__name__}: "
                                  f"You should implement this method!")
    # _end_def_

    @staticmethod
    @cache
    def linear_rank_probabilities(p_size: int) -> list[float]:
        """
        Calculate the rank probability distribution over the population size.
        The function is cached so repeated calls with the same input should
        not recompute the same array since the population size of the swarm
        is not expected to change.

        NOTE: Probabilities are returned in ascending order.

        :param p_size: (int) population size.

        :return: (list) rank probability distribution in ascending order.
        """

        # Calculate the sum of '1 + 2 + 3 + ... + N'.
        # We know that this is equal to: N * (N+1)/2.
        sum_ranked_values = float(0.5 * p_size * (p_size + 1))

        # Return the probability values (ascending order).
        return [n / sum_ranked_values for n in range(1, p_size + 1)]
    # _end_def_

    def __call__(self, *args, **kwargs):
        """
        This is only a wrapper of the "select" method.
        """
        return self.select(*args, **kwargs)
    # _end_def_

# _end_class_
