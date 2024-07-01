from src.genome.chromosome import Chromosome
from src.operators.selection.select_operator import SelectionOperator

class RandomSelector(SelectionOperator):
    """
    Description:

        Random Selector implements selection assuming that all member of the population have the same probability
        to be selected as parents 1/N, (effectively assuming a uniform probability).

        It does not favour the fit individuals therefore the mixing will be very slow.
    """

    def __init__(self, select_probability: float = 1.0):
        """
        Construct a 'RandomSelector' object with a given probability value.

        :param select_probability: (float).
        """

        # Call the super constructor with the provided
        # probability value.
        super().__init__(select_probability)
    # _end_def_

    def select(self, population: list[Chromosome]):
        """
        Select the new individuals from the population that will be passed on to the next genetic operations
        of crossover and mutation to form the new population of solutions.

        :param population:

        :return: a new population (list of chromosomes).
        """
        # Get the length of the population list.
        N = len(population)

        # Increase the selection counter.
        self.inc_counter()

        # Select 'N' new individuals (indexes).
        index = self.rng.choice(N, size=N, replace=True, shuffle=False)

        # Return the (new) selected individuals.
        return [population[i] for i in index]
# _end_class_
