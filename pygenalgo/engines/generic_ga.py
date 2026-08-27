""" Generic GA module. """
from os import cpu_count
from operator import attrgetter
from dataclasses import dataclass
from collections import defaultdict
from typing import Callable, Optional

from joblib import (Parallel, delayed)

from numpy import all as np_all
from numpy.typing import NDArray
from numpy.random import (default_rng, Generator)
from numpy import (array, nanmean, nanstd, isfinite)

from pygenalgo.engines import logger
from pygenalgo.genome.chromosome import Chromosome
from pygenalgo.utils.auxiliary import correct_chromosomes

from pygenalgo.operators.genetic_operator import GeneticOperator
from pygenalgo.operators.mutation.mutate_operator import MutationOperator
from pygenalgo.operators.selection.select_operator import SelectionOperator
from pygenalgo.operators.crossover.crossover_operator import CrossoverOperator

from pygenalgo.operators.mutation.meta_mutator import MetaMutator
from pygenalgo.operators.crossover.meta_crossover import MetaCrossover

# Define a fitness type.
Fitness = float | tuple[float, ...]

@dataclass(frozen=True)
class RunConfig:
    """
    Auxiliary dataclass to set the configuration parameters
    for all the GA engines.
    """

    # Number of iterations.
    epochs: int = 100
    '''
    Maximum number of iterations in the evolution process.
    '''

    # Algorithmic settings.
    elitism: bool = True
    '''
    If enabled then the chromosome with the highest fitness
    will always be copied unaltered to the next generation.
    '''

    shuffle: bool = True
    '''
    If enabled it will shuffle the population after
    the selection process. Default is set to True.
    '''

    correction: bool = False
    '''
    If enabled it will check the validity of the population
    at the gene level and attempt to correct the genome by
    calling the random() method of the flawed gene.
    '''

    adapt_probs: bool = False
    '''
    If enabled it will allow the crossover and mutation probabilities
    to adapt according to the convergence of the population to a single
    solution. Default is set to False.
    '''

    # Runtime settings.
    parallel: bool = False
    '''
    Enables parallel computation of the fitness function.
    '''

    verbose: bool = False
    '''
    If 'True' it will display periodically information about
    the current average fitness and spread of the population.
    '''

    # Stop criteria.
    f_tol: Optional[float] = None
    '''
    Tolerance in the difference between the average values of two
    consecutive populations. It is used to determine the convergence
    of the population. If this value is None (default) the algorithm
    will terminate using the epochs value.
    '''

    f_max_eval: Optional[int] = None
    '''
    Sets an upper limit of function evaluations. If this number is
    exceeded the genetic algorithm will terminate.
    '''

    # Migration ONLY parameters.
    allow_migration: bool = False
    '''
    If enabled it will allow the migration of the best individuals
    among the different co-evolving island populations.
    '''

    n_periods: int = 10
    '''
    The number of times that we will break the main evolution to allow
    for chromosomes to migrate. This setting is used only when the option
    allow_migration == True. Otherwise, is ignored.
    '''

    @staticmethod
    def _check_bool(name: str, var: bool) -> None:
        """
        Helper method to check if a value is True or False.

        :param name: variable name.
        :param var: variable value.
        :return: None.
        """
        if not isinstance(var, bool):
            raise TypeError(f"{name} must be bool, "
                            f"got {type(var).__name__}.")
    # _end_def_

    @staticmethod
    def _check_int_positive(name: str, var: Optional[int]) -> None:
        """
        Helper method to check if a value is a positive integer.

        :param name: variable name.
        :param var: variable value.
        :return: None.
        """
        # Sanity check 1.
        if var is None:
            return

        # Sanity check 2.
        # NOTE: In Python bool is a subclass of int!
        if not isinstance(var, int) or isinstance(var, bool):
            raise TypeError(f"{name} must be int, "
                            f"got {type(var).__name__}.")
        # Sanity check 3.
        if var <= 0:
            raise ValueError(f"{name} must be positive, "
                             f"got {var}.")
    # _end_def_

    @staticmethod
    def _check_float_non_negative(name: str, var: Optional[float | int]) -> None:
        """
        Helper method to check if a value is a non-negative float.

        :param name: variable name.
        :param var: variable value.
        :return: None.
        """
        # Sanity check 1.
        if var is None:
            return

        # Sanity check 2.
        # NOTE: In Python bool is a subclass of int!
        if not isinstance(var, (float, int)) or isinstance(var, bool):
            raise TypeError(f"{name} must be float or int, "
                            f"got {type(var).__name__}.")
        # Sanity check 3.
        if var < 0.0:
            raise ValueError(f"{name} must be non-negative, "
                             f"got {var}.")
    # _end_def_

    def __post_init__(self) -> None:
        """
        Post initialization checks.

        :return: None.
        """

        # Check bool parameters.
        self._check_bool("elitism", self.elitism)
        self._check_bool("shuffle", self.shuffle)
        self._check_bool("verbose", self.verbose)
        self._check_bool("parallel", self.parallel)
        self._check_bool("correction", self.correction)
        self._check_bool("adapt_probs", self.adapt_probs)
        self._check_bool("allow_migration", self.allow_migration)

        # Check integer parameters.
        self._check_int_positive("epochs", self.epochs)
        self._check_int_positive("n_periods", self.n_periods)
        self._check_int_positive("f_max_eval", self.f_max_eval)

        # Check float parameters.
        self._check_float_non_negative("f_tol", self.f_tol)
    # _end_def_
# _end_class_

# Public interface.
__all__ = ["GenericGA", "RunConfig", "Fitness"]


class GenericGA:
    """
    Description:

        Generic GA class models the interface of a specific genetic algorithm model
        (or engine). It provides the common variables and functionality that all GA
        models should share.
    """

    rng_GA: Generator = default_rng()
    """
    Random Number Generator for the whole class.
    """

    MAX_CPUs: int = 1 if not cpu_count() else cpu_count()
    """
    Set the maximum number of CPUs (at least one).
    """

    # Object variables.
    __slots__ = ("population", "fitness_func", "_select_op", "_crossx_op",
                 "_mutate_op", "_stats", "_n_cpus", "_f_evals", "_iteration")

    def __init__(self, initial_pop: list[Chromosome], fit_func: Callable,
                 select_op: SelectionOperator, mutate_op: MutationOperator,
                 crossx_op: CrossoverOperator, n_cpus: Optional[int] = None) -> None:
        """
        Default constructor of GenericGA object.

        :param initial_pop: list of the initial population of (randomized) chromosomes.

        :param fit_func: callable fitness function.

        :param select_op: selection operator (must inherit from class SelectionOperator).

        :param mutate_op: mutation operator (must inherit from class MutationOperator).

        :param crossx_op: crossover operator (must inherit from class CrossoverOperator).

        :param n_cpus: Number of requested CPUs for the evolution process (Default=Max_CPU).
        """
        # Sanity check.
        if not callable(fit_func):
            raise TypeError(f"{self.__class__.__name__}: Fitness function is not callable.")
        # _end_if_

        # Copy the reference of the population.
        self.population: list[Chromosome] = initial_pop.copy()

        # Get the fitness function.
        self.fitness_func: Callable = fit_func

        # Get Selection Operator.
        self._select_op: SelectionOperator = select_op

        # Get Mutation Operator.
        self._mutate_op: MutationOperator = mutate_op

        # Get Crossover Operator.
        self._crossx_op: CrossoverOperator = crossx_op

        # Get the number of requested CPUs.
        if n_cpus is None:
            # This is the default option.
            self._n_cpus: int = max(1, GenericGA.MAX_CPUs-1)
        else:
            # Assign the  requested number, making sure we have
            # enough CPUs and the value entered has the correct
            # type.
            self._n_cpus: int = max(1, min(GenericGA.MAX_CPUs-1, int(n_cpus)))
        # _end_if_

        # Log the number of CPUs.
        logger.debug("%s uses %s CPUs.", self.__class__.__name__, self._n_cpus)

        # Dictionary with statistics.
        self._stats: dict = defaultdict(list)

        # Set the function evaluation to zero.
        self._f_evals: int = 0

        # Set the iterations counter to zero.
        self._iteration: int = 0

        # Log the object initialization.
        logger.debug("%s initialization complete.", self.__class__.__name__)
    # _end_def_

    @property
    def rng(self) -> Generator:
        """
        Get access of the Class variable (rng_GA).

        :return: the random number generator.
        """
        return self.rng_GA
    # _end_def_

    @property
    def iteration(self) -> int:
        """
        Accessor (getter) of the iteration parameter.

        :return: the iteration value.
        """
        return self._iteration
    # _end_def_

    @iteration.setter
    def iteration(self, value: int) -> None:
        """
        Accessor (setter) of the iteration value.

        :param value: (int).
        """
        # Check for correct type and allow only the positive values.
        if not isinstance(value, int) or value < 0:
            raise RuntimeError(f"{self.__class__.__name__}: Iteration value "
                               f"should be positive 'int': {value.__class__.__name__}.")
        # _end_if_

        # Update the iteration value.
        self._iteration: int = value

        # Update the iteration value in the GeneticOperator Class.
        GeneticOperator.set_iteration(value)
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

        # Log the new seed.
        logger.debug("%s has a new seed.", cls.__name__)
    # _end_def_

    @property
    def f_evals(self) -> int:
        """
        Accessor method that returns the value of the f_eval.

        :return: (int) the counted number of function evaluations.
        """
        return self._f_evals
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
    def crossx_op(self) -> CrossoverOperator:
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
        self._f_evals = 0

        # Log the cleanup.
        logger.debug("%s cleared.", self.__class__.__name__)
    # _end_def_

    def update_stats(self, fit_list: list[float],
                     other_stats: dict = None) -> tuple:
        """
        Update the input stats dictionary with the mean / std
        values of the population fitness values.

        :param fit_list: (list) fitness values of the population.

        :param other_stats: (dict) stats dictionary.

        :return: the mean and std of the fitness values.
        """
        # Convert the fitness list in a numpy array.
        arr: NDArray = array(fit_list, dtype=float)

        # Compute the mean value.
        avg_fitness: NDArray = nanmean(arr, axis=0, dtype=float)

        # Compute the standard deviation value.
        std_fitness: NDArray = nanstd(arr, axis=0, dtype=float)

        # Update the population mean / std.
        if np_all(isfinite([avg_fitness, std_fitness])):

            if other_stats:
                # Store them in the input dictionary.
                other_stats["avg"].append(avg_fitness)
                other_stats["std"].append(std_fitness)
            else:
                # Store them in the self dictionary.
                self._stats["avg"].append(avg_fitness)
                self._stats["std"].append(std_fitness)
        else:
            raise RuntimeError(f"{self.__class__.__name__}:"
                               f"Something went wrong at {self._iteration} "
                               f"iteration. Mean={avg_fitness:.5f}, "
                               f"Std={std_fitness:.5f}.")
        # _end_if_

        # Return the average statistics.
        return avg_fitness, std_fitness
    # _end_def_

    def best_chromosome(self) -> Optional[Chromosome]:
        """
        Auxiliary method that returns the chromosome with the
        highest fitness value. Safeguarded with ignoring None.

        :return: Return the chromosome with the highest fitness.
        """
        # Define the key.
        key_sort: Callable = attrgetter("fitness")

        # Return the chromosome with the highest fitness.
        return max(
            (p for p in self.population if p.fitness is not None),
            key=key_sort, default=None
        )
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

        # Define the key.
        key_sort: Callable = attrgetter("fitness")

        # Sort the population in descending order.
        sorted_population: list[Chromosome] = sorted(
            [p for p in self.population if p.fitness is not None],
            key=key_sort, reverse=True
        )

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
        :return: None.
        """
        # Get the size of the input population.
        pop_size: int = len(input_population)

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
            k: int = j+1

            # Replace directly the OLD parents with the NEW offsprings.
            input_population[j], input_population[k] = self._crossx_op(input_population[j],
                                                                       input_population[k])
            # MUTATE in place the 1st offspring.
            self._mutate_op(input_population[j])

            # MUTATE in place the 2nd offspring.
            self._mutate_op(input_population[k])
    # _end_def_

    def adapt_probabilities(self, threshold: Optional[float] = None) -> bool:
        """
        This method is used (optionally) to adjust simultaneously the crossover
        and mutation parameters of the GenericGA object.

        :param threshold: This float parameter is used to determine whether we are
                          going to increase or decrease the crossover and mutation
                          parameters.
        :return: True if the parameters have changed, else False.
        """
        # Check if the threshold value is missing.
        if threshold is None:
            raise RuntimeError(f"{self.__class__.__name__}: "
                               f"Threshold parameter is missing.")
        # _end_if_

        # Sanity check.
        if not (isinstance(threshold, float) and 0.0 <= threshold <= 1.0):
            raise ValueError(f"{self.__class__.__name__}: "
                             f"Threshold value must be float in [0.0, 1.0].")
        # _end_if_

        # Initialize the trial values with the current
        # probabilities to avoid going out of limits.
        trial_pc: float = self._crossx_op.probability
        trial_pm: float = self._mutate_op.probability

        # Initialize the flag with "False"
        # to avoid unnecessary assignments.
        have_changed: bool = False

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

            # Log the update of the parameters.
            logger.debug("%s probabilities have been updated.",
                         self.__class__.__name__)
        # _end_if_

        return have_changed
    # _end_def_

    def population_fitness(self) -> list[float]:
        """
        Get the fitness values of all the population.

        :return: A list with all the fitness values.
        """
        return [p.fitness for p in self.population]
    # _end_def_

    def evaluate_fitness(self, input_population: list[Chromosome],
                         parallel_mode: bool = False,
                         backend: str = "threading") -> tuple[list[Fitness], bool]:
        """
        Evaluate all the chromosomes of the input list with the custom
        fitness function. The parallel_mode is optional. Moreover, the
        default backend is "threading", but in the IslandModelGA it is
        better to select "loky".

        :param input_population: (list) The population of Chromosomes
                                 that we want to evaluate their fitness.

        :param parallel_mode: (bool) Enables parallel computation of
                              the fitness function.

        :param backend: (str) Backend for the parallel Joblib framework.

        :return: a list with the fitness values and the found solution flag.
        """
        # Get a local copy of the fitness function.
        fit_func: Callable[[Chromosome], Fitness] = self.fitness_func

        # Check the 'parallel_mode' flag.
        if parallel_mode:

            # Evaluate the chromosomes in parallel mode.
            fitness_i: list[Fitness] = Parallel(n_jobs=self._n_cpus, backend=backend)(
                delayed(fit_func)(p) for p in input_population
            )
        else:

            # Evaluate the chromosomes in serial mode.
            fitness_i: list[Fitness] = [fit_func(p) for p in input_population]
        # _end_if_

        # Get the size of the population.
        p_size: int = len(fitness_i)

        # Preallocate the fitness list.
        fitness_values: list[Fitness] = [None] * p_size

        # Flag to indicate if a solution has been found.
        found_solution: bool = False

        # Update all chromosomes with their fitness and
        # check if a solution has been found.
        for n, (p, fit_result) in enumerate(zip(input_population, fitness_i)):
            # Attach the fitness to each chromosome.
            p.fitness = fit_result["f_value"]

            # Collect the fitness in a separate list.
            fitness_values[n] = fit_result["f_value"]

            # Update the "found solution".
            found_solution |= fit_result["solution_is_found"]
        # _end_for_

        # Update the counter of function evaluations.
        self._f_evals += p_size

        # Return the fitness values.
        return fitness_values, found_solution
    # _end_def_

    def correct_genome(self, input_population: list[Chromosome]) -> bool:
        """
        Applies the correction mechanism to the input population
        (the changes in the population are passed by reference).

        :param input_population: list of Chromosomes.

        :return: True if changes were made, False otherwise.
        """
        # Return flag.
        has_been_corrected: bool = False

        # Apply the "correct_chromosomes" function.
        total_corrections, f_counts = correct_chromosomes(input_population,
                                                          self.fitness_func)
        # If corrections were made we will
        # need to update the f_evals value.
        if total_corrections > 0:
            # Update the function evaluation counter.
            self._f_evals += f_counts

            # Log the corrections.
            logger.debug(
                "> %d correction(s) took place at epoch: %d",
                total_corrections, self._iteration
            )

            # Enable the flag.
            has_been_corrected = True

        return has_been_corrected
    # _end_def_

    def print_operator_stats(self) -> None:
        """
        Print the genetic operators stats.

        :return: None.
        """
        # First print the selection operator.
        print(self._select_op)

        # Second print the crossover operator.
        print(self._crossx_op)

        # Check if we used the MetaCrossover.
        if isinstance(self._crossx_op, MetaCrossover):
            # Call internally all operators.
            for op in self._crossx_op.items:
                print(op)
        # _end_if_

        # Lastly print the mutation operator.
        print(self._mutate_op)

        # Check if we used the MetaMutator.
        if isinstance(self._mutate_op, MetaMutator):
            # Call internally all operators.
            for op in self._mutate_op.items:
                print(op)
            # _end_for_

    # _end_def_

    def run(self, config: Optional[RunConfig] = None) -> None:
        """
        Main method of the Generic GA class that implements
        the evolutionary routine.
        """
        raise NotImplementedError(f"{self.__class__.__name__}: "
                                  f"You should implement this method!")
    # _end_def_

    def __call__(self, config: Optional[RunConfig] = None) -> None:
        """
        This method is only a wrapper of the "run" method.
        """
        return self.run(config)
    # _end_def_

# _end_class_
