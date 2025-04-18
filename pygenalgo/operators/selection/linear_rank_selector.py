from operator import attrgetter
from pygenalgo.genome.chromosome import Chromosome
from pygenalgo.operators.genetic_operator import increase_counter
from pygenalgo.operators.selection.select_operator import SelectionOperator


class LinearRankSelector(SelectionOperator):
    """
    Description:

        Linear Rank Selector implements an object that performs selection using ranking.
        The individuals first are sorted according to their fitness values. The rank 'N'
        is assigned to the best individual and the rank 1 to the worst individual.

        After that the selection process is similar to the one of RouletteWheelSelector.
    """

    def __init__(self, select_probability: float = 1.0):
        """
        Construct a 'LinearRankSelector' object with a given probability value.

        :param select_probability: (float) in [0, 1].
        """

        # Call the super constructor with the provided probability value.
        super().__init__(select_probability)

        # Initialize auxiliary parameters. These parameters will help us
        # NOT to recompute the selection probabilities every time unless
        # the population size changes during evolution.
        self._items = {"pop_size": 0, "selection_probs": None}
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

        # Check if the population size has changed.
        # This should run only ONE time during the
        # first iteration.
        if pop_size != self.items["pop_size"]:

            # Store the new population size.
            self.items["pop_size"] = pop_size

            # Calculate the sum of all the ranked fitness values: "1+2+3+...+N".
            sum_ranked_values = float(0.5 * pop_size * (pop_size + 1))

            # Calculate the selection probabilities of each member
            # in the population, using their ranking position.
            self.items["selection_probs"] = [n / sum_ranked_values
                                             for n in range(1, pop_size + 1)]
        # _end_if_

        # Sort the population in ascending order using their fitness value.
        sorted_population = sorted(population, key=attrgetter("fitness"))

        # Select the new individuals (indexes).
        index = self.rng.choice(pop_size, size=pop_size,
                                p=self.items["selection_probs"],
                                replace=True, shuffle=False)

        # Return the new parents (individuals).
        return [sorted_population[i] for i in index]
    # _end_def_

# _end_class_
