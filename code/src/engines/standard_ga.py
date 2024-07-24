import time
import numpy as np
from typing import Callable
from collections import defaultdict

from src.genome.chromosome import Chromosome
from joblib import (Parallel, delayed, cpu_count)
from src.engines.auxiliary import apply_corrections
from src.operators.mutation.mutate_operator import MutationOperator
from src.operators.selection.select_operator import SelectionOperator
from src.operators.crossover.crossover_operator import CrossoverOperator

from src.operators.mutation.super_mutator import SuperMutator
from src.operators.crossover.super_crossover import SuperCrossover


# Public interface.
__all__ = ["StandardGA"]


class StandardGA(object):
    """
    Standard GA model that at each iteration replaces the whole population using
    the genetic operators (crossover and mutation).
    """

    # Make a random number generator.
    rng_GA = np.random.default_rng()

    # Get the maximum number of CPUs.
    MAX_CPUs = cpu_count()

    # Object variables.
    __slots__ = ("population", "fitness_func", "_select_op", "_cross_op", "_mutate_op",
                 "_stats")

    def __init__(self, initial_pop: list[Chromosome], fit_func: Callable, select_op: SelectionOperator = None,
                 mutate_op: MutationOperator = None, cross_op: CrossoverOperator = None):
        """
        Default constructor of StandardGA object.

        :param initial_pop: list of the initial population of (randomized) chromosomes.

        :param fit_func: callable fitness function.

        :param select_op: selection operator (must inherit from class SelectionOperator).

        :param mutate_op: mutation operator (must inherit from class MutationOperator).

        :param cross_op: crossover operator (must inherit from class CrossoverOperator).

        :return: a new GA object.
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
            self._cross_op = cross_op
        # _end_if_

        # Dictionary with stats.
        self._stats = defaultdict(list)

    # _end_def_

    @property
    def stats(self):
        """
        Accessor method that returns the 'stats' dictionary.

        :return: the dictionary with the statistics from the run.
        """
        return self._stats

    # _end_def_

    def update_stats(self, avg_fitness: float, std_fitness: float):
        """
        Update the stats dictionary with the input mean/std values
        of the fitness.

        :param avg_fitness: (float) mean fitness value of the population.

        :param std_fitness: (float) std fitness value of the population.

        :return: None.
        """

        # Make sure the input values are finite.
        if all(np.isfinite([avg_fitness, std_fitness])):

            # Store them in the dictionary.
            self._stats["avg"].append(avg_fitness)
            self._stats["std"].append(std_fitness)
        else:
            raise RuntimeError(f"{self.__class__.__name__}: Something went wrong with current "
                               f"population. Mean={avg_fitness:.5f}, Std={std_fitness:.5f}.")
        # _end_if_

    # _end_def_

    def individual_fitness(self, index: int) -> float:
        """
        Get the fitness value of an individual member of the population.

        :param index: Position of the individual in the population.

        :return: The fitness value (float).
        """
        return self.population[index].fitness

    # _end_def_

    def population_fitness(self) -> list[float]:
        """
        Get the fitness of all the population.

        :return: A list with all the fitness values.
        """
        return [p.fitness for p in self.population]

    # _end_def_

    def evaluate_fitness(self, input_population: list[Chromosome], parallel: bool = False):
        """
        Evaluate all the chromosomes of the input list with the custom fitness function.

        :param input_population: (list) The population of Chromosomes that we want to evaluate
        their fitness.

        :param parallel: (bool) Flag that enables parallel computation of the fitness function.

        :return: the mean and std of the fitness values.
        """

        # Get a local copy of the fitness function.
        fit_func = self.fitness_func

        # Check the 'parallel' flag.
        if parallel:

            # Evaluate the chromosomes in parallel mode.
            fitness_i = Parallel(n_jobs=self.MAX_CPUs,
                                 backend="threading")(
                delayed(fit_func)(p) for p in input_population
            )
        else:

            # Evaluate the chromosomes in serial mode.
            fitness_i = [fit_func(p) for p in input_population]

        # _end_if_

        # Attach the fitness to each chromosome.
        for p, fit_eval in zip(input_population, fitness_i):
            p.fitness = fit_eval
        # _end_for_

        # Convert list to numpy array.
        arr = np.array(fitness_i, dtype=float)

        # Return the mean and std of the fitness values.
        return np.mean(arr, dtype=float), np.std(arr, dtype=float)

    # _end_def_

    def best_chromosome(self):
        """
        Auxiliary method.

        :return: Return the chromosome with the highest fitness.
        """
        # Return the chromosome with the highest fitness.
        return max(self.population, key=lambda c: c.fitness)

    # _end_def_

    def run(self, epochs: int = 100, elitism: bool = True, correction: bool = False,
            f_tol: float = 1.0e-8, parallel: bool = False, verbose: bool = False):
        """
        Main method of the StandardGA class, that implements the evolutionary routine.

        :param epochs: (int) maximum number of iterations in the evolution process.

        :param elitism: (bool) flag that defines elitism. If 'True' then the chromosome
        with the higher fitness will always be copied to the next generation (unaltered).

        :param correction: (bool) flag that if set to 'True' will check the validity of
        the population (at the gene level) and attempt to correct the genome by calling
        the random() method of the flawed gene.

        :param f_tol: (float) tolerance in the difference between the average values of
        two consecutive populations. It is used to determine the convergence of the population.

        :param parallel: (bool) Flag that enables parallel computation of the fitness function.

        :param verbose: (bool) if 'True' it will display periodically information about
        the current average fitness and spread of the population.

        :return: None.
        """

        # Make sure the genetic operator counters are reset before each run().
        self._cross_op.reset_counter()
        self._mutate_op.reset_counter()
        self._select_op.reset_counter()

        # Reset stats dictionary.
        self._stats["avg"].clear()
        self._stats["std"].clear()

        # Get the size of the population.
        N = len(self.population)

        # Get the average/std fitness before optimisation.
        avg_fitness_0, std_fitness_0 = self.evaluate_fitness(self.population, parallel)

        # Update the mean/std in the dictionary.
        self.update_stats(avg_fitness_0, std_fitness_0)

        # Display an information message.
        print(f"Initial Avg. Fitness = {avg_fitness_0:.4f}")

        # Initial time instant.
        time_t0 = time.perf_counter()

        # Repeat 'epoch' times.
        for i in range(epochs):

            # SELECT the parents.
            # This will create a NEW copy of the population.
            population_i = self._select_op(self.population)

            # CROSSOVER/MUTATE to produce offsprings.
            for j in range(0, N - 1, 2):
                # Replace directly the OLD parents with the NEW offsprings.
                population_i[j], population_i[j + 1] = self._cross_op(population_i[j],
                                                                      population_i[j + 1])
                # MUTATE in place the 1st offspring.
                self._mutate_op(population_i[j])

                # MUTATE in place the 2nd offspring.
                self._mutate_op(population_i[j + 1])
            # _end_for_

            # Check if 'corrections' are enabled.
            if correction:
                # Apply the function.
                total_corrections = apply_corrections(population_i)

                # Print only if there were corrections,
                # to avoid cluttering the screen.
                if total_corrections:
                    print(f"> {total_corrections} correction(s) took place at epoch: {i}.")
                # _end_if_

            # _end_if_

            # Check if 'elitism' is enabled.
            if elitism:
                # Find the individual chromosome with the highest fitness value
                # (from the old population).
                best_chromosome = max(self.population, key=lambda c: c.fitness)

                # Select randomly a position.
                locus = self.rng_GA.integers(0, N)

                # Replace the chromosome with the previous best.
                population_i[locus] = best_chromosome.clone()

            # _end_if_

            # Calculate the (new) average/std of the fitness.
            avg_fitness_i, std_fitness_i = self.evaluate_fitness(population_i, parallel)

            # Update the mean/std in the dictionary.
            self.update_stats(avg_fitness_i, std_fitness_i)

            # Check if we want to print output.
            if verbose and np.mod(i, 10) == 0:
                # Display an information message.
                print(f"Epoch: {i + 1:>5} -> Avg. Fitness = {avg_fitness_i:.4f}, "
                      f"Spread = {std_fitness_i:.4f}")
            # _end_if_

            # Check for convergence.
            # Here we don't only check the average performance, but we also check the
            # spread of the population. If all the chromosomes are similar the spread
            # should be small (very close to zero).
            if np.fabs(avg_fitness_i - avg_fitness_0) < f_tol and std_fitness_i < 1.0E-1:
                # Display a warning message.
                print(f"{self.__class__.__name__} finished in {i + 1} iterations.")

                # Exit from the loop.
                break
            # _end_if_

            # Update the average value for the next iteration.
            avg_fitness_0 = avg_fitness_i

            # Update the old population with the new chromosomes.
            self.population = population_i.copy()
        # _end_for_

        # Final time instant.
        time_tf = time.perf_counter()

        # Display the final average fitness value.
        print(f"Final   Avg. Fitness = {avg_fitness_0:.4f}")

        # Print final duration in seconds.
        print(f"Elapsed time: {(time_tf - time_t0):.3f} seconds.", end='\n')

    # _end_def_

    # Auxiliary.
    def __call__(self, *args, **kwargs):
        """
        This is only a wrapper of the "run" method.
        """
        return self.run(*args, **kwargs)

    # _end_def_

    def print_operator_stats(self):
        """
        Print the genetic operators stats.

        :return: None.
        """

        # First print the selection operator.
        print(f"{self._select_op}")

        # Second print the crossover operator.
        print(f"{self._cross_op}")

        # Check if we used the SuperCrossover.
        if isinstance(self._cross_op, SuperCrossover):
            # Call internally all operators.
            for op in self._cross_op.list_all:
                print(f"{op}")
            # _end_for_
        # _end_if_

        # Lastly print the mutation operator.
        print(f"{self._mutate_op}")

        # Check if we used the SuperMutator.
        if isinstance(self._mutate_op, SuperMutator):
            # Call internally all operators.
            for op in self._mutate_op.list_all:
                print(f"{op}")
            # _end_for_
    # _end_def_

# _end_class_
