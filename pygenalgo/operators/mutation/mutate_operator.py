from pygenalgo.genome.chromosome import Chromosome
from pygenalgo.operators.genetic_operator import GeneticOperator


class MutationOperator(GeneticOperator):
    """
    Description:

        Provides the base class (interface) for a Mutation Operator.
    """

    def __init__(self, mutation_probability: float):
        """
        Construct a 'MutationOperator' object with a
        given probability value.

        :param mutation_probability: (float).
        """

        # Call the super constructor with the provided
        # probability value.
        super().__init__(mutation_probability)
    # _end_def_

    def mutate(self, individual: Chromosome) -> None:
        """
        Abstract method that "reminds" the user that if they want to
        create a Mutation Class that inherits from here they should
        implement a mutate method.

        :param individual: the chromosome to be mutated.

        :return: Nothing but raising an error.
        """
        raise NotImplementedError(f"{self.__class__.__name__}: "
                                  f"You should implement this method!")
    # _end_def_

    def __call__(self, *args, **kwargs):
        """
        This is only a wrapper of the "mutate" method.
        """
        return self.mutate(*args, **kwargs)
    # _end_def_

# _end_class_
