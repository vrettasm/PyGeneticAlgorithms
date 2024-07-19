from src.genome.chromosome import Chromosome
from src.operators.selection.select_operator import SelectionOperator

class LinearRankSelector(SelectionOperator):
    """
    Description:

        Linear Rank Selector implements an object that performs selection using ranking. The individuals first are
        sorted according to their fitness values. The rank N is assigned to the best individual and the rank 1 to
        the worst individual.

        After that the selection process is similar to the one of Roulette Wheel Selector.
    """

    def __init__(self, select_probability: float = 1.0):
        """
        Construct a 'LinearRankSelector' object with a given probability value.

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

        # Sort the population in ascending order using their fitness value.
        sorted_population = sorted(population, key=lambda p: p.fitness)

        # Get the length of the population list.
        N = len(sorted_population)

        # Calculate sum of all the ranked fitness values: "1 + 2 + 3 + ... + N".
        sum_ranked_values = int(0.5 * N * (N+1))

        # Calculate the "selection probabilities", of each member in the population.
        selection_probs = [n / sum_ranked_values for n in range(1, N+1)]

        # Increase the selection counter.
        self.inc_counter()

        # Select 'N' new individuals (indexes).
        index = self.rng.choice(N, size=N, p=selection_probs, replace=True, shuffle=False)

        # Return the (new) selected individuals.
        return [sorted_population[i].clone() for i in index]
    # _end_def_

# _end_class_
