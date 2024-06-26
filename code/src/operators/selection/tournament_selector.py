from src.genome.chromosome import Chromosome
from src.operators.selection.select_operator import SelectionOperator

class TournamentSelector(SelectionOperator):
    """
    Description:

        Tournament Selector implements an object that performs selection by choosing the individual from the set of
        individuals. The winner of each tournament i.e. (the one with the highest fitness value) is selected to perform
        crossover and mutation.
    """

    def __init__(self, select_probability: float = 1.0):
        """
        Construct a 'TournamentSelector' object with a given probability value.

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

        # Create a list that will contain the new parents.
        new_parents = []

        # Repeat the 'tournament' N times.
        for i in range(len(population)):

            # Select randomly '5' individuals from the initial population.
            parents = self.rng.choice(population, size=5, replace=False, shuffle=False)

            # Sort them (in descending order) according to their fitness value.
            parents = sorted(parents, key=lambda p: p._fitness, reverse=True)

            # Copy the best individual in the new list.
            new_parents.append(parents[0])
        # _end_for_

        return new_parents
    # _end_def_

# _end_class_
