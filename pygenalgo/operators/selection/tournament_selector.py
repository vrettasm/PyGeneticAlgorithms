from pygenalgo.genome.chromosome import Chromosome
from pygenalgo.operators.selection.select_operator import SelectionOperator


class TournamentSelector(SelectionOperator):
    """
    Description:

        Tournament Selector implements an object that performs selection by choosing the individual
        from the set of individuals. The winner of each tournament i.e. (the one with the highest
        fitness value) is selected to perform crossover and mutation.
    """

    def __init__(self, select_probability: float = 1.0, k: int = 5):
        """
        Construct a 'TournamentSelector' object with a given probability value.

        :param select_probability: (float).

        :param k: the number of participants in the tournament (int).
        """

        # Call the super constructor with the provided probability value.
        super().__init__(select_probability)

        # Make sure 'k' is integer.
        k = int(k)

        # Number of 'participants' of the tournament should be more than 2.
        self._k = k if k > 2 else 5
    # _end_def_

    def select(self, population: list[Chromosome]):
        """
        Select the new individuals from the population that will be passed on to the next
        genetic operations of crossover and mutation to form the new population of solutions.

        :param population:

        :return: a new population (list of chromosomes).
        """
        # Get the length of the population list.
        N = len(population)

        # Create a list that will contain the new parents.
        new_parents = []

        # Get the list append method locally.
        new_parents_append = new_parents.append

        # Define locally the choose(N,k) function.
        def choose_k():
            return self.rng.choice(N, size=self._k, replace=False, shuffle=False)
        # _end_def_

        # Repeat the 'tournament' N times.
        for i in range(N):

            # Select randomly 'k' individuals (indexes)
            # from the initial population.
            index = choose_k()

            # Find the individual with the highest fitness value.
            winner = max([population[j] for j in index],
                         key=lambda p: p.fitness)

            # Copy the best individual in the new list.
            new_parents_append(winner)
        # _end_for_

        # Increase the selection counter.
        self.inc_counter()

        # Return the (new) selected individuals.
        return new_parents
    # _end_def_

# _end_class_