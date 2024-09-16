from os import cpu_count
from math import isnan
from typing import Callable
from numpy.random import default_rng

from pygenalgo.genome.chromosome import Chromosome
from pygenalgo.operators.mutation.mutate_operator import MutationOperator
from pygenalgo.operators.selection.select_operator import SelectionOperator
from pygenalgo.operators.crossover.crossover_operator import CrossoverOperator

# Public interface.
__all__ = ["GenericGA"]


class GenericGA(object):
    """
    Description:

        Generic GA class models the interface of a specific genetic algorithm model (or engine).
        It provides the common variables and functionality that all GA models should share.
    """

    # Make a random number generator.
    rng_GA = default_rng()

    # Get the maximum number of CPUs (at least one).
    MAX_CPUs = 1 if not cpu_count() else cpu_count()

    # Object variables.
    __slots__ = ("population", "fitness_func", "_select_op", "_crossx_op", "_mutate_op", "_stats")

    def __init__(self, initial_pop: list[Chromosome], fit_func: Callable, select_op: SelectionOperator = None,
                 mutate_op: MutationOperator = None, cross_op: CrossoverOperator = None):
        """
        Default constructor of GenericGA object.

        :param initial_pop: list of the initial population of (randomized) chromosomes.

        :param fit_func: callable fitness function.

        :param select_op: selection operator (must inherit from class SelectionOperator).

        :param mutate_op: mutation operator (must inherit from class MutationOperator).

        :param cross_op: crossover operator (must inherit from class CrossoverOperator).
        """

        # Copy the reference of the population.
        self.population = initial_pop.copy()

        # Make sure the fitness function is indeed callable.
        if not callable(fit_func):
            raise TypeError(f"{self.__class__.__name__}: Fitness function is not callable.")
        else:
            # Get the fitness function.
            self.fitness_func = fit_func
        # _end_if_

        # Get Selection Operator.
        if select_op is None:
            raise ValueError(f"{self.__class__.__name__}: Selection operator is missing.")
        else:
            self._select_op = select_op
        # _end_if_

        # Get Mutation Operator.
        if mutate_op is None:
            raise ValueError(f"{self.__class__.__name__}: Mutation operator is missing.")
        else:
            self._mutate_op = mutate_op
        # _end_if_

        # Get Crossover Operator.
        if cross_op is None:
            raise ValueError(f"{self.__class__.__name__}: Crossover operator is missing.")
        else:
            self._crossx_op = cross_op
        # _end_if_

        # Placeholder for the stats.
        self._stats = None
    # _end_def_

    @property
    def stats(self) -> dict:
        """
        Accessor method that returns the 'stats' dictionary.

        :return: the dictionary with the statistics from the run.
        """
        return self._stats
    # _end_def_

    @property
    def select_op(self) -> SelectionOperator:
        """
        Accessor method that returns the selection operator reference.

        :return: the SelectionOperator.
        """
        return self._select_op
    # _end_def_

    @property
    def crossover_op(self) -> CrossoverOperator:
        """
        Accessor method that returns the crossover operator reference.

        :return: the CrossoverOperator.
        """
        return self._crossx_op
    # _end_def_

    @property
    def mutate_op(self) -> MutationOperator:
        """
        Accessor method that returns the mutation operator reference.

        :return: the MutationOperator.
        """
        return self._mutate_op
    # _end_def_

    def best_chromosome(self) -> Chromosome:
        """
        Auxiliary method that returns the chromosome with the
        highest fitness value. Safeguard to ignore NaN values.

        :return: Return the chromosome with the highest fitness.
        """
        # Return the chromosome with the highest fitness.
        return max(self.population,
                   key=lambda c: c.fitness if not isnan(c.fitness) else 0.0)
    # _end_def_

    def crossover_mutate(self, input_population: list[Chromosome]) -> None:
        """
        This is an auxiliary method that combines the crossover and mutation
        operations in one call. Since these operations happen in place the
        'input_population' will be modified directly.

        This method should be called AFTER the selection of the parents that
        have been selected for breeding.

        :param input_population: this is the population that we will apply
        the two genetic operators.
        """

        # Get the size of the input population.
        N = len(input_population)

        # CROSSOVER and MUTATE to produce the new offsprings.
        for j in range(0, N - 1, 2):
            # Replace directly the OLD parents with the NEW offsprings.
            input_population[j], input_population[j+1] = self._crossx_op(input_population[j],
                                                                         input_population[j+1])
            # MUTATE in place the 1st offspring.
            self._mutate_op(input_population[j])

            # MUTATE in place the 2nd offspring.
            self._mutate_op(input_population[j+1])
        # _end_for_
    # _end_def_

    def population_fitness(self) -> list[float]:
        """
        Get the fitness values of all the population.

        :return: A list with all the fitness values.
        """
        return [p.fitness for p in self.population]
    # _end_def_

    def individual_fitness(self, index: int) -> float:
        """
        Get the fitness value of an individual member of the population.

        :param index: Position of the individual in the population.

        :return: The fitness value (float).
        """
        return self.population[index].fitness
    # _end_def_

    def evaluate_fitness(self, *args, **kwargs):
        """
        This method evaluates all the chromosomes' of an input population
        with a custom fitness function. After updating all the chromosomes
        with their fitness, the method should return the average statistics
        of mean and std of the population fitness.
        """
        raise NotImplementedError(f"{self.__class__.__name__}: "
                                  f"You should implement this method!")
    # _end_def_

    def run(self, *args, **kwargs):
        """
        Main method of the Generic GA class, that implements the evolutionary routine.
        """
        raise NotImplementedError(f"{self.__class__.__name__}: "
                                  f"You should implement this method!")
    # _end_def_

    def __call__(self, *args, **kwargs):
        """
        This method is only a wrapper of the "run" method.
        """
        return self.run(*args, **kwargs)
    # _end_def_

# _end_class_
