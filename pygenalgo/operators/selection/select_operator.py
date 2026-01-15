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

    @staticmethod
    def ensure_positive_fitness(population: list[Chromosome]) -> list[float]:
        """
        Ensures that the fitness value of each chromosome is a positive number.

        This is useful because some of the selection methods require a positive
        fitness to operate. Also in minimization problems the fitness values are
        negated, therefore by using this transformation the methods that require
        positive values are guaranteed to work.

        :param population: (list) of chromosomes.

        :return: (list) of positive fitness values.
        """
        # Extract the fitness value of each chromosome.
        all_fitness = [p.fitness for p in population]

        # If there are negative values we perform a shift
        # transformation where all the values are shifted
        # so that the minimum fitness is going to be one.
        if any(fit_value < 0.0 for fit_value in all_fitness):
            # Compute the shift value.
            shift_value = fabs(min(all_fitness)) + 1.0

            # Shift all fitness values so that the minimum is '1'.
            all_fitness = [f + shift_value for f in all_fitness]
        # _end_if_

        return all_fitness
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
