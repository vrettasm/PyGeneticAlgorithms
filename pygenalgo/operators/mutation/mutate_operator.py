""" Mutation operator module. """
# Custom code imports.
from pygenalgo.genome.chromosome import Chromosome
from pygenalgo.operators.genetic_operator import GeneticOperator


class MutationOperator(GeneticOperator):
    """
    Description:

        Provides the base class (interface) for a Mutation Operator.
    """

    def __init__(self, mutation_probability: float) -> None:
        """
        Construct a 'MutationOperator' object with a
        given probability value.

        :param mutation_probability: (float).
        """
        # Call the super constructor with the provided initial value.
        super().__init__(probability=mutation_probability)
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

    def _finalize_mutation(self, individual: Chromosome) -> None:
        """
        Finalize the mutation operation by invalidating the
        fitness of individual chromosome and increasing the
        operations counter of the mutator.

        :param individual: the chromosome to be mutated.

        :return: None.
        """
        # Set the fitness to None.
        individual.invalidate_fitness()

        # Increase the mutator counter.
        self.inc_counter()
    # _end_def_

    def __call__(self, individual: Chromosome) -> None:
        """
        This is only a wrapper of the "mutate" method.
        """
        return self.mutate(individual)
    # _end_def_

# _end_class_
