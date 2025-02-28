from operator import attrgetter
from pygenalgo.genome.chromosome import Chromosome
from pygenalgo.operators.genetic_operator import increase_counter
from pygenalgo.operators.selection.select_operator import SelectionOperator


class TournamentSelector(SelectionOperator):
    """
    Description:

        Tournament Selector implements an object that performs selection by choosing
        an individual from a set of individuals. The winner of each tournament i.e.
        (the one with the highest fitness value) is selected as new parent to perform
        crossover and mutation.
    """

    def __init__(self, select_probability: float = 1.0, k: int = 2):
        """
        Construct a 'TournamentSelector' object with a given probability value.

        :param select_probability: (float) in [0, 1].

        :param k: the number of participants in the tournament (int).
        """

        # Call the super constructor with the provided probability value.
        super().__init__(select_probability)

        # Number of participants of the tournament should be more than 2.
        self._items = max(2, int(k))
    # _end_def_

    @increase_counter
    def select(self, population: list[Chromosome]):
        """
        Select the individuals, from the input population that will be passed on
        to the next genetic operations of crossover and mutation to form the new
        population of solutions.

        :param population: a list of chromosomes to select the parents from.

        :return: the selected parents population (as list of chromosomes).
        """

        # Get the population size.
        pop_size = len(population)

        # Create a list that will contain the new parents.
        new_parents = pop_size * [None]

        # Repeat the 'tournament' N times.
        for i in range(pop_size):

            # Select randomly 'k' individuals (indexes)
            # from the initial population.
            index = self.rng.choice(pop_size, size=self._items,
                                    replace=False, shuffle=False)

            # Find the individual with the highest fitness value.
            winner = max([population[j] for j in index],
                         key=attrgetter("fitness"))

            # Add the best individual in the new list.
            new_parents[i] = winner
        # _end_for_

        # Return the new parents (individuals).
        return new_parents
    # _end_def_

# _end_class_
