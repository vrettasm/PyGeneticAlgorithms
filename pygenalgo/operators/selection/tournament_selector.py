from operator import attrgetter

from numpy.typing import NDArray
from numpy import array as np_array

from pygenalgo.genome.chromosome import Chromosome
from pygenalgo.operators.genetic_operator import increase_counter
from pygenalgo.operators.selection.select_operator import SelectionOperator


class TournamentSelector(SelectionOperator):
    """
    Description:

        Tournament Selection operates by randomly selecting a subset of individuals
        from the population and competing them against each other based on their fitness.
        The individual with the highest fitness in the mini-group is chosen for reproduction.
        This method provides a balance between selection pressure and diversity, as larger
        tournament sizes increase the likelihood of selecting fitter individuals, while smaller
        sizes maintain population diversity. The size of the tournament can be adjusted depending
        on the desired selection pressure. This technique is computationally efficient and
        effective, especially in dynamic environments or populations with diverse fitness
        landscapes.

    """

    def __init__(self, select_probability: float = 1.0, k: int = 5) -> None:
        """
        Construct a 'TournamentSelector' object with a given probability value.

        :param select_probability: (float) in [0, 1].

        :param k: the number of participants in the tournament (int).
        """
        # Call the super constructor with the provided initial value.
        super().__init__(selection_probability=select_probability)

        # Number of participants of the tournament should be more than 2.
        self._items: int = max(2, int(k))
    # _end_def_

    @increase_counter
    def select(self, population: list[Chromosome]) -> list[Chromosome]:
        """
        Select the individuals, from the input population that will be
        passed on to the next genetic operations of crossover and mutation
        to form the new population of solutions.

        :param population: a list of chromosomes to select the parents from.

        :return: the selected parents population (as list of chromosomes).
        """
        # Get the population size.
        pop_size: int = len(population)

        # Local copy of random choice.
        random_choice = self.rng.choice

        # Number of contestants.
        n_contestants = self._items

        # Select the contestants for each tournament.
        contestants: NDArray = np_array([
            random_choice(pop_size, size=n_contestants,
                          replace=False, shuffle=False)
            for _ in range(pop_size)
        ])

        # Define the key.
        key_sort = attrgetter("fitness")

        # Return the new parents.
        return [
            max((population[k] for k in row), key=key_sort)
            for row in contestants
        ]
    # _end_def_

# _end_class_
