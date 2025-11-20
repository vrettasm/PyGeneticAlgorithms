from pygenalgo.genome.chromosome import Chromosome
from pygenalgo.operators.genetic_operator import increase_counter
from pygenalgo.operators.selection.select_operator import SelectionOperator


class RandomSelector(SelectionOperator):
    """
    Description:

        Random Selection involves choosing individuals from a population without any regard
        to their fitness levels. Each member has an equal chance of being selected, which
        promotes genetic diversity by allowing exploration of various solutions. While this
        method is simple and easy to implement, it can lead to inefficient results, especially
        in populations with varying fitness values. Random selection may overlook high-quality
        solutions, making it less effective in later stages of evolution. However, it can serve
        as a useful mechanism in the initial stages or when diversity is paramount for avoiding
        premature convergence.

    """

    def __init__(self, select_probability: float = 1.0):
        """
        Construct a 'RandomSelector' object with a given probability value.

        :param select_probability: (float) in [0, 1].
        """
        # Call the super constructor with the provided probability value.
        super().__init__(select_probability)
    # _end_def_

    @increase_counter
    def select(self, population: list[Chromosome]) -> list[Chromosome]:
        """
        Select the individuals, from the input population, that will be
        passed on to the next genetic operations of crossover and mutation
        to form the new population of solutions.

        :param population: a list of chromosomes to select the parents from.

        :return: the selected parents population (as list of chromosomes).
        """
        # Get the population size.
        pop_size = len(population)

        # Select the new individuals indexes.
        index = self.rng.choice(pop_size, size=pop_size, replace=True, shuffle=False)

        # Return the new parents (individuals).
        return [population[i] for i in index]
    # _end_def_

# _end_class_
