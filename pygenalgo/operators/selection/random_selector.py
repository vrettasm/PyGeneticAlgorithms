from pygenalgo.genome.chromosome import Chromosome
from pygenalgo.operators.genetic_operator import increase_counter
from pygenalgo.operators.selection.select_operator import SelectionOperator


class RandomSelector(SelectionOperator):
    """
    Description:

        Random Selector implements selection assuming that all members of the population
        have the same probability to be selected as parents 1/N, effectively assuming a
        uniform probability.

        It does not favour the fit individuals therefore the mixing will be very slow.
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
    def select(self, population: list[Chromosome]):
        """
        Select the individuals, from the input population, that will be passed on
        to the next genetic operations of crossover and mutation to form the new
        population of solutions.

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
