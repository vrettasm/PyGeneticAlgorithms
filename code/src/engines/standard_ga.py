import time
import numpy as np
from typing import Callable
from collections import defaultdict
from src.genome.chromosome import Chromosome
from src.engines.auxiliary import apply_corrections, calculate_mean_std
from src.operators.mutation.mutate_operator import MutationOperator
from src.operators.selection.select_operator import SelectionOperator
from src.operators.crossover.crossover_operator import CrossoverOperator

# Public interface.
__all__ = ["StandardGA"]


class StandardGA(object):
    """
    Standard GA model that at each iteration replaces the whole population using
    the genetic operators (crossover and mutation).
    """

    # Make a random number generator.
    rng_GA = np.random.default_rng()

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
        self._select_op = select_op

        # Get Mutation Operator.
        self._mutate_op = mutate_op

        # Get Crossover Operator.
        self._cross_op = cross_op

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

        :param std_fitness:(float) std fitness value of the population.

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

    def evaluate_fitness(self, input_population: list[Chromosome]):
        """
        Evaluate all the chromosomes of the input list with the custom
        fitness function.

        :param input_population: (list)

        :return: a list with all the fitness values.
        """
        # Get a local copy of the fitness function.
        fit_func = self.fitness_func

        # List with the fitness of the population.
        fitness_i = []

        # Make a local copy of the append method.
        fitness_i_append = fitness_i.append

        # Evaluate all the individuals.
        for p in input_population:

            # Assign the fitness to the chromosome.
            p.fitness = fit_func(p)

            # Add it to the return list.
            fitness_i_append(p.fitness)
        # _end_for_

        return fitness_i
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
            f_tol: float = 1.0e-8, verbose: bool = False):
        """
        Main method of the StandardGA class, that implements the evolutionary routine.

        :param epochs: (int) maximum number of iterations in the evolution process.

        :param elitism: (bool) flag that defines elitism. If 'True' then the chromosome
        with the higher fitness will always be copied to the next generation (unaltered).

        :param correction: (bool) flat that if set to 'True' will check the validity of
        the population (at the gene level) and attempt to correct the genome by calling
        the random() method of the flawed gene.

        :param f_tol: (float) tolerance in the difference between the average values of
        two consecutive populations. It is used to determine the convergence of the population.

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

        # STEP 1: Evaluate the initial population.
        fitness_0 = self.evaluate_fitness(self.population)

        # Get the average/std fitness before optimisation.
        avg_fitness_0, std_fitness_0 = calculate_mean_std(fitness_0)

        # Update the mean/std in the dictionary.
        self.update_stats(avg_fitness_0, std_fitness_0)

        # Display an information message.
        print(f"Initial Avg. Fitness = {avg_fitness_0:.4f}")

        # Initial time instant.
        time_t0 = time.perf_counter()

        # Repeat 'epoch' times.
        for i in range(epochs):

            # STEP 2: SELECT the parents.
            # This will create a NEW copy of the population.
            population_i = self._select_op(self.population)

            # STEP 3: CROSSOVER to produce offsprings.
            for j in range(0, N - 1, 2):
                # Replace directly the OLD parents with the NEW offsprings.
                population_i[j], population_i[j + 1] = self._cross_op(population_i[j],
                                                                      population_i[j + 1])
            # _end_for_

            # STEP 4: MUTATE in place the offsprings.
            for p in population_i:
                self._mutate_op(p)
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
                population_i[locus] = best_chromosome

            # _end_if_

            # STEP 5: EVALUATE the current population.
            fitness_i = self.evaluate_fitness(population_i)

            # Calculate the (new) average/std of the fitness.
            avg_fitness_i, std_fitness_i = calculate_mean_std(fitness_i)

            # Update the mean/std in the dictionary.
            self.update_stats(avg_fitness_i, std_fitness_i)

            # Check if we want to print output.
            if verbose and np.mod(i, 10) == 0:

                # Display an information message.
                print(f"Epoch: {i + 1:>5} -> Avg. Fitness = {avg_fitness_i:.4f}, "
                      f"Spread = {std_fitness_i:.4f}")
            # _end_if_

            # STEP 6: Check for convergence.
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
        print(f"{self._cross_op}")
        print(f"{self._select_op}")
        print(f"{self._mutate_op}")
    # _end_def_

# _end_class_
