from math import fsum, exp, isclose
from pygenalgo.genome.chromosome import Chromosome
from pygenalgo.operators.genetic_operator import increase_counter
from pygenalgo.operators.selection.select_operator import (SelectionOperator,
                                                           ensure_positive_fitness)


class BoltzmannSelector(SelectionOperator):
    """
    Description:

        Boltzmann Selection integrates principles from physical systems into genetic
        algorithms, using a temperature parameter to influence selection probability.
        The basic idea is that as the "temperature" decreases, the likelihood of selecting
        higher-fitness individuals increases, mimicking the cooling process in simulated
        annealing. At high temperatures, the selection is more random, allowing exploration.
        As temperature falls, the process becomes more exploitative, favoring superior solutions.
        This duality makes Boltzmann Selection versatile, although it requires careful tuning of
        the temperature parameter to maintain a balance between exploration and exploitation over
        generations.

    """

    def __init__(self, select_probability: float = 1.0, k: float = 100.0) -> None:
        """
        Construct a 'BoltzmannSelector' object with a given probability value.

        :param select_probability: (float) in [0, 1].

        :param k: constant of the Boltzmann distribution.
        """
        # Call the super constructor with the provided probability value.
        super().__init__(select_probability)

        # Make sure 'k' is float and at least 50.
        self._items = max(float(k), 50.0)
    # _end_def_

    @increase_counter
    def select(self, population: list[Chromosome]) -> list[Chromosome]:
        """
        Select the individuals, from the input population, that will be
        passed on to the next genetic operations of crossover and mutation
        to form the new population of solutions.

        NOTE: the Boltzmann constant is held in the '_items' variable.

        :param population: a list of chromosomes to select the parents from.

        :return: the selected parents population (as list of chromosomes).
        """
        # Extract the (positive) fitness values from the chromosomes.
        all_fitness = ensure_positive_fitness(population)

        # Compute the Temperature value (using the current iteration).
        temperature = max(0.1, exp(-self.iteration / self._items))

        # Convert the fitness values with the exponential.
        exp_fitness = [exp(-f/temperature) for f in all_fitness]

        # Calculate sum of all fitness.
        sum_fitness = fsum(exp_fitness)

        # Get the population size.
        pop_size = len(population)

        # If total fitness is zero (or effectively zero),
        # fall back to uniform random selection so every
        # individual has equal chance.
        if isclose(sum_fitness, 0.0):
            # Select chromosomes randomly with equal probability.
            safe_index = self.rng.choice(pop_size, size=pop_size,
                                         replace=True, shuffle=False)

            # Return the new parents to a list.
            return [population[i] for i in safe_index]
        # _end_if_

        # Calculate the selection probabilities of each member
        # in the population (Boltzmann distribution).
        selection_probs = [f / sum_fitness for f in exp_fitness]

        # Select the new individuals indexes.
        index = self.rng.choice(pop_size, size=pop_size, p=selection_probs,
                                replace=True, shuffle=False)

        # Return the new parents (individuals).
        return [population[i] for i in index]
    # _end_def_

# _end_class_
