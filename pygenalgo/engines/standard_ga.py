import time
import numpy as np
from math import fabs

from joblib import (Parallel, delayed)

from pygenalgo.genome.chromosome import Chromosome
from pygenalgo.engines.generic_ga import GenericGA

from pygenalgo.operators.mutation.meta_mutator import MetaMutator
from pygenalgo.operators.crossover.meta_crossover import MetaCrossover

from pygenalgo.engines.auxiliary import apply_corrections, avg_hamming_dist

# Public interface.
__all__ = ["StandardGA"]


class StandardGA(GenericGA):
    """
    Description:

        StandardGA model provides a basic implementation of the "GenericGA",
        which at each iteration (epoch) replaces the whole population using
        the genetic operators (crossover and mutation).
    """

    def __init__(self, **kwargs):
        """
        Default constructor of StandardGA object.
        """
        # Call the super constructor with all the input parameters.
        super().__init__(**kwargs)
    # _end_def_

    def update_stats(self, fit_list: list[float]) -> (float, float):
        """
        Update the stats dictionary with the mean/std values of the
        population fitness values.

        :param fit_list: (float) mean fitness value of the population.

        :return: the mean and std of the fitness values.
        """

        # Convert the fitness list in a numpy array.
        arr = np.array(fit_list)

        # Get the mean and std values.
        avg_fitness = np.nanmean(arr, dtype=float)
        std_fitness = np.nanstd(arr, dtype=float)

        # Make sure the stat values are finite.
        if all(np.isfinite([avg_fitness, std_fitness])):

            # Store them in the dictionary.
            self._stats["avg"].append(avg_fitness)
            self._stats["std"].append(std_fitness)
        else:
            raise RuntimeError(f"{self.__class__.__name__}: Something went wrong with current "
                               f"population. Mean={avg_fitness:.5f}, Std={std_fitness:.5f}.")
        # _end_if_

        # Return the average statistics.
        return avg_fitness, std_fitness
    # _end_def_

    def evaluate_fitness(self, input_population: list[Chromosome],
                         parallel: bool = False) -> (list[float], bool):
        """
        Evaluate all the chromosomes of the input list with the custom
        fitness function.

        :param input_population: (list) The population of Chromosomes that
        we want to evaluate their fitness.

        :param parallel: (bool) Flag that enables parallel computation of
        the fitness function.

        :return: a list with the fitness values and the found solution flag.
        """

        # Get a local copy of the fitness function.
        fit_func = self.fitness_func

        # Check the 'parallel' flag.
        if parallel:

            # Evaluate the chromosomes in parallel mode.
            fitness_i = Parallel(n_jobs=self.MAX_CPUs, backend="threading")(
                delayed(fit_func)(p) for p in input_population
            )
        else:

            # Evaluate the chromosomes in serial mode.
            fitness_i = [fit_func(p) for p in input_population]
        # _end_if_

        # Preallocate the fitness list.
        fitness_values = len(fitness_i) * [None]

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

        # Return the fitness values.
        return fitness_values, found_solution
    # _end_def_

    def run(self, epochs: int = 100, elitism: bool = True, correction: bool = False,
            f_tol: float = None, parallel: bool = False, adapt_probs: bool = False,
            verbose: bool = False) -> None:
        """
        Main method of the StandardGA class, that implements the evolutionary routine.

        :param epochs: (int) maximum number of iterations in the evolution process.

        :param elitism: (bool) flag that defines elitism. If 'True' then the chromosome
        with the higher fitness will always be copied to the next generation (unaltered).

        :param correction: (bool) flag that if set to 'True' will check the validity of
        the population (at the gene level) and attempt to correct the genome by calling
        the random() method of the flawed gene.

        :param f_tol: (float) tolerance in the difference between the average values of two
        consecutive populations. It is used to determine the convergence of the population.
        If this value is None (default) the algorithm will terminate using the epochs value.

        :param parallel: (bool) Flag that enables parallel computation of the fitness function.

        :param adapt_probs: (bool) If enabled (set to True), it will allow the crossover and
        mutation probabilities to adapt according to the convergence of the population to a
        single solution. Default is set to False.

        :param verbose: (bool) if 'True' it will display periodically information about the
        current average fitness and spread of the population.

        :return: None.
        """

        # Make sure the genetic operator counters are reset before each run().
        self._crossx_op.reset_counter()
        self._mutate_op.reset_counter()
        self._select_op.reset_counter()

        # Reset stats dictionary.
        self._stats.clear()

        # Get the size of the population.
        N = len(self.population)

        # Get the fitness values before optimisation.
        fit_list_0, _ = self.evaluate_fitness(self.population, parallel)

        # Update the average stats (mean/std) in the dictionary.
        avg_fitness_0, std_fitness_0 = self.update_stats(fit_list_0)

        # Store the initial crossover and mutation probabilities.
        self._stats["prob_crossx"].append(self._crossx_op.probability)
        self._stats["prob_mutate"].append(self._mutate_op.probability)

        # Display an information message.
        print(f"Initial Avg. Fitness = {avg_fitness_0:.4f}.")

        # Initial time instant.
        time_t0 = time.perf_counter()

        # Repeat 'epoch' times.
        for i in range(epochs):

            # Update current iteration in the selection operator.
            # Currently, this is used only from Boltzmann Selector.
            self._select_op.iter = i

            # SELECT the parents.
            population_i = self._select_op(self.population)

            # Shuffle the selected parents.
            self.rng_GA.shuffle(population_i)

            # CROSSOVER/MUTATE to produce offsprings.
            self.crossover_mutate(population_i)

            # Check if 'corrections' are enabled.
            if correction:
                # Apply the function.
                total_corrections = apply_corrections(population_i, self.fitness_func)

                # Print only if there were corrections,
                # to avoid cluttering the screen.
                if verbose and total_corrections:
                    print(f"> {total_corrections} correction(s) took place at epoch: {i}.")
                # _end_if_
            # _end_if_

            # Check if 'elitism' is enabled.
            if elitism:
                # Select randomly a position.
                locus = self.rng_GA.integers(N)

                # Replace the chromosome with the previous best.
                population_i[locus] = self.best_chromosome()
            # _end_if_

            # Calculate the new fitness values.
            fit_list_i, found_solution = self.evaluate_fitness(population_i, parallel)

            # Update the mean/std in the dictionary.
            avg_fitness_i, std_fitness_i = self.update_stats(fit_list_i)

            # Check if we want to print output.
            if verbose and (i % 10) == 0:
                # Display an information message.
                print(f"Epoch: {i + 1:>5} -> Avg. Fitness = {avg_fitness_i:.4f}, "
                      f"Spread = {std_fitness_i:.4f}.")
            # _end_if_

            # Update the old population with the new chromosomes.
            self.population = population_i

            # Check for termination.
            if found_solution:
                # Display a warning message.
                print(f"{self.__class__.__name__} finished in {i + 1} iterations.")

                # Exit from the loop.
                break
            # _end_if_

            # Compute the current average Hamming distance.
            d_avg = avg_hamming_dist(population_i)

            # Check for convergence.
            if f_tol and fabs(avg_fitness_i - avg_fitness_0) < f_tol and d_avg < 0.025:

                # Display a warning message.
                print(f"{self.__class__.__name__} converged in {i + 1} iterations.")

                # Exit from the loop.
                break
            # _end_if_

            # Update the average value for the next iteration.
            avg_fitness_0 = avg_fitness_i

            # Adap probabilities for the next generation.
            if adapt_probs:

                # For threshold, we use the mean Hamming
                # distance of the current population.
                self.adapt_probabilities(threshold=d_avg)
            # _end_if_

        # _end_for_

        # Final time instant.
        time_tf = time.perf_counter()

        # Display the final average fitness value.
        print(f"Final   Avg. Fitness = {avg_fitness_0:.4f}.")

        # Print final duration in seconds.
        print(f"Elapsed time: {(time_tf - time_t0):.3f} seconds.")
    # _end_def_

    def print_operator_stats(self) -> None:
        """
        Print the genetic operators stats.

        :return: None.
        """

        # First print the selection operator.
        print(self.select_op)

        # Second print the crossover operator.
        print(self.crossover_op)

        # Check if we used the MetaCrossover.
        if isinstance(self.crossover_op, MetaCrossover):
            # Call internally all operators.
            for op in self.crossover_op.items:
                print(op)
            # _end_for_
        # _end_if_

        # Lastly print the mutation operator.
        print(self.mutate_op)

        # Check if we used the MetaMutator.
        if isinstance(self.mutate_op, MetaMutator):
            # Call internally all operators.
            for op in self.mutate_op.items:
                print(op)
            # _end_for_
        # _end_if_

    # _end_def_

# _end_class_
