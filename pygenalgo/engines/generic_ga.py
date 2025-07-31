from math import isnan
from os import cpu_count
from operator import attrgetter
from typing import Callable, Tuple
from collections import defaultdict

from joblib import (Parallel, delayed)
from numpy.random import default_rng, Generator

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
    rng_GA: Generator = default_rng()

    # Set the maximum number of CPUs (at least one).
    MAX_CPUs: int = 1 if not cpu_count() else cpu_count()

    # Object variables.
    __slots__ = ("population", "fitness_func", "_select_op", "_crossx_op", "_mutate_op",
                 "_stats", "_n_cpus", "_f_eval")

    def __init__(self, initial_pop: list[Chromosome], fit_func: Callable, select_op: SelectionOperator = None,
                 mutate_op: MutationOperator = None, crossx_op: CrossoverOperator = None, n_cpus: int = None):
        """
        Default constructor of GenericGA object.

        :param initial_pop: list of the initial population of (randomized) chromosomes.

        :param fit_func: callable fitness function.

        :param select_op: selection operator (must inherit from class SelectionOperator).

        :param mutate_op: mutation operator (must inherit from class MutationOperator).

        :param crossx_op: crossover operator (must inherit from class CrossoverOperator).

        :param n_cpus: Number of requested CPUs for the evolution process (Default=Max_CPU).
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
        if crossx_op is None:
            raise ValueError(f"{self.__class__.__name__}: Crossover operator is missing.")
        else:
            self._crossx_op = crossx_op
        # _end_if_

        # Get the number of requested CPUs.
        if n_cpus is None:

            # This is the default option.
            self._n_cpus = max(1, GenericGA.MAX_CPUs-1)
        else:

            # Assign the  requested number, making sure we have
            # enough CPUs and the value entered has the correct
            # type.
            self._n_cpus = max(1, min(GenericGA.MAX_CPUs-1, int(n_cpus)))
        # _end_if_

        # Dictionary with statistics.
        self._stats = defaultdict(list)

        # Set the function evaluation to zero.
        self._f_eval = 0
    # _end_def_

    @classmethod
    def set_seed(cls, new_seed=None) -> None:
        """
        Sets a new seed for the random number generator.

        :param new_seed: New seed value (default=None).

        :return: None.
        """
        # Re-initialize the class variable.
        cls.rng_GA = default_rng(seed=new_seed)
    # _end_def_

    @property
    def f_eval(self) -> int:
        """
        Accessor method that returns the value of the f_eval.

        :return: (int) the counted number of function evaluations.
        """
        return self._f_eval
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

    @property
    def n_cpus(self) -> int:
        """
        Accessor method that returns the number of CPUs.

        :return: the n_cpus.
        """
        return self._n_cpus
    # _end_def_

    def f_eval_increase_by(self, new_counts: int) -> None:
        """
        Utility method to allow the '_f_eval' to be updated with
        new counts outside the main class. This can happen during
        gene correction.

        :param new_counts: the new value of 'counts' that we want
        to add on the current f_eval.

        :return: None.
        """
        self._f_eval += new_counts
    # _end_def_

    def clear_all(self) -> None:
        """
        Make sure all the genetic operator counters and the stats
        are cleared. This reset everything before each run().

        :return: None.
        """
        # Ensure the genetic operator counters are reset before.
        self._crossx_op.reset_counter()
        self._mutate_op.reset_counter()
        self._select_op.reset_counter()

        # Reset stats dictionary.
        self._stats.clear()

        # Reset f_eval counter.
        self._f_eval = 0
    # _end_def_

    def best_chromosome(self) -> Chromosome:
        """
        Auxiliary method that returns the chromosome with the
        highest fitness value. Safeguarded with ignoring NaNs.

        :return: Return the chromosome with the highest fitness.
        """
        # Return the chromosome with the highest fitness.
        return max((p for p in self.population if not isnan(p.fitness)),
                   key=attrgetter("fitness"), default=None)
    # _end_def_

    def best_n(self, n: int = 1) -> list[Chromosome]:
        """
        Auxiliary method that returns the best 'n' chromosomes
        with the highest fitness value.

        :param n: the number of the best chromosomes. Default = 1.

        :return: Return the 'n' chromosomes with the highest fitness.
        """

        # Make sure 'n' is positive integer.
        if not isinstance(n, int) or n <= 0:
            raise ValueError(f"{self.__class__.__name__}: "
                             f"Input must be a positive integer.")
        # _end_if_

        # Ensure the number of return chromosome
        # do not exceed the size of the population.
        if n > len(self.population):
            raise RuntimeError(f"{self.__class__.__name__}: "
                               f"Best {n} exceeds population size.")
        # _end_if_

        # Sort the population in descending order.
        sorted_population = sorted([p for p in self.population if not isnan(p.fitness)],
                                   key=attrgetter("fitness"), reverse=True)

        # Return the best 'n' chromosomes.
        return sorted_population[0:n]
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
        pop_size = len(input_population)

        # CROSSOVER and MUTATE to produce the new offsprings.
        for j in range(0, pop_size, 2):

            # In case of 'odd sized' populations.
            if j == pop_size-1:

                # MUTATE in place the last offspring.
                self._mutate_op(input_population[j])

                # Exit the loop.
                break
            # _end_if_

            # Get the index of the next chromosome.
            k = j+1

            # Replace directly the OLD parents with the NEW offsprings.
            input_population[j], input_population[k] = self._crossx_op(input_population[j],
                                                                       input_population[k])
            # MUTATE in place the 1st offspring.
            self._mutate_op(input_population[j])

            # MUTATE in place the 2nd offspring.
            self._mutate_op(input_population[k])
        # _end_for_
    # _end_def_

    def adapt_probabilities(self, threshold: float = None) -> None:
        """
        This method is used (optionally) to adjust simultaneously the crossover
        and mutation parameters of the GenericGA object.

        :param threshold: (float) This parameter is used to determine whether we
        are going to increase or decrease the crossover and mutation parameters.
        """

        # Check if the threshold value is missing.
        if threshold is None:
            raise RuntimeError(f"{self.__class__.__name__}: "
                               f"Threshold parameter is missing.")
        # _end_if_

        # Initialize the trial values with the current
        # probabilities to avoid going out of limits.
        trial_pc = self._crossx_op.probability
        trial_pm = self._mutate_op.probability

        # Initialize the flag with "False"
        # to avoid unnecessary assignments.
        have_changed = False

        # Use the threshold value to adjust
        # the probabilities accordingly.
        if threshold < 0.1:

            trial_pc *= 0.9
            trial_pm *= 1.1
            have_changed = True

        elif threshold > 0.8:

            trial_pc *= 1.1
            trial_pm *= 0.9
            have_changed = True
        # _end_if_

        # Check if the probabilities have changed.
        if have_changed:
            # Ensure the probabilities stay within the range [0, 1].
            self._crossx_op.probability = min(max(trial_pc, 0.0), 1.0)
            self._mutate_op.probability = min(max(trial_pm, 0.0), 1.0)
        # _end_if_
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

    def evaluate_fitness(self, input_population: list[Chromosome],
                         parallel_mode: bool = False,
                         backend: str = "threading") -> Tuple[list[float], bool]:
        """
        Evaluate all the chromosomes of the input list with the custom
        fitness function. The parallel_mode is optional. Moreover, the
        default backend is "threading", but in the IslandModelGA it is
        better to select "loky".

        :param input_population: (list) The population of Chromosomes
        that we want to evaluate their fitness.

        :param parallel_mode: (bool) Enables parallel computation of
        the fitness function.

        :param backend: Backend for the parallel Joblib framework.

        :return: a list with the fitness values and the found solution flag.
        """

        # Get a local copy of the fitness function.
        fit_func = self.fitness_func

        # Check the 'parallel_mode' flag.
        if parallel_mode:

            # Evaluate the chromosomes in parallel mode.
            fitness_i = Parallel(n_jobs=self._n_cpus, backend=backend)(
                delayed(fit_func)(p) for p in input_population
            )
        else:

            # Evaluate the chromosomes in serial mode.
            fitness_i = [fit_func(p) for p in input_population]
        # _end_if_

        # Get the size of the population.
        p_size = len(fitness_i)

        # Preallocate the fitness list.
        fitness_values = p_size * [None]

        # Flag to indicate if a solution has been found.
        found_solution = False

        # Update all chromosomes with their fitness and check if a solution
        # has been found.
        for n, (p, fit_tuple) in enumerate(zip(input_population, fitness_i)):
            # Attach the fitness to each chromosome.
            p.fitness = fit_tuple[0]

            # Collect the fitness in a separate list.
            fitness_values[n] = fit_tuple[0]

            # Update the "found solution".
            found_solution |= fit_tuple[1]
        # _end_for_

        # Update the counter of function evaluations.
        self._f_eval += p_size

        # Return the fitness values.
        return fitness_values, found_solution
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
