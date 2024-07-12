import time
import numpy as np
from typing import Callable
from collections import defaultdict
from src.genome.chromosome import Chromosome
from joblib import (Parallel, delayed, cpu_count)
from src.engines.auxiliary import apply_corrections, SubPopulation
from src.operators.mutation.mutate_operator import MutationOperator
from src.operators.selection.select_operator import SelectionOperator
from src.operators.crossover.crossover_operator import CrossoverOperator
from src.operators.migration.clockwise_migration import ClockwiseMigration

# Public interface.
__all__ = ["IslandModelGA"]


class IslandModelGA(object):
    """
    In Island Model GA we run in parallel a number of "islands", each one evolving its own
    (sub)-population. Optionally we can allow "migration", among the best individuals from
    each island.
    """

    # Make a random number generator.
    rng_GA = np.random.default_rng()

    # Get the maximum number of CPUs.
    MAX_CPUs = cpu_count()

    # Object variables.
    __slots__ = ("population", "fitness_func", "num_islands", "_stats",
                 "_select_op", "_cross_op", "_mutate_op", "_migrate_op")

    def __init__(self, initial_pop: list[Chromosome], fit_func: Callable, num_islands: int,
                 select_op: SelectionOperator = None, mutate_op: MutationOperator = None,
                 cross_op: CrossoverOperator = None, migrate_op: ClockwiseMigration = None):
        """
        Default constructor of StandardGA object.

        :param initial_pop: (list) of the initial population of (randomized) chromosomes.

        :param fit_func: (callable) fitness function.

        :param num_islands: (int) number of parallel evolving islands.

        :param select_op: selection operator (must inherit from class SelectionOperator).

        :param mutate_op: mutation operator (must inherit from class MutationOperator).

        :param cross_op: crossover operator (must inherit from class CrossoverOperator).

        :param migrate_op: migration operator (must inherit from class MigrationOperator).

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

        # Assign the number of islands.
        self.num_islands = num_islands

        # Get Selection Operator.
        self._select_op = select_op

        # Get Mutation Operator.
        self._mutate_op = mutate_op

        # Get Crossover Operator.
        self._cross_op = cross_op

        # Get Migration Operator.
        self._migrate_op = migrate_op

        # Dictionary with stats.
        self._stats = defaultdict(dict)
    # _end_def_

    @property
    def stats(self):
        """
        Accessor method that returns the 'stats' dictionary.

        :return: the dictionary with the statistics from the run.
        """
        return self._stats
    # _end_def_

    def best_chromosome(self):
        """
        Auxiliary method.

        :return: Return the chromosome with the highest fitness.
        """
        # Return the chromosome with the highest fitness.
        return max(self.population, key=lambda c: c.fitness)
    # _end_def_

    def evaluate_fitness(self, in_population: list[Chromosome]):
        """
        Evaluate all the chromosomes of the input population list with the
        custom fitness  function. After updating all  the chromosomes with
        their fitness, the method returns the average statistics mean/std.

        :param in_population: (list) The population of Chromosomes that we
        want to evaluate their fitness.

        :return: mean(fitness), std(fitness).
        """

        # Get a local copy of the fitness function.
        fit_func = self.fitness_func

        # Evaluate the chromosomes.
        for p in in_population:

            # Assign the chromosome its fitness value.
            p.fitness = fit_func(p)

        # _end_for_

        # Get all the fitness values in a numpy array.
        arr = np.array([p.fitness for p in in_population])

        # Return the mean and std values of the fitness.
        return np.mean(arr, dtype=float), np.std(arr, dtype=float)
    # _end_def_

    @staticmethod
    def evolve_population(island: SubPopulation, eval_fitness: Callable, epochs: int, crs_op: CrossoverOperator,
                          mut_op: MutationOperator, sel_op: SelectionOperator, rnd_gen, f_tol: float = 1.0e-6,
                          correction: bool = False, elitism: bool = True):

        # Keeps track of the convergence of the population,
        # along with the iteration that terminated.
        has_converged = (False, epochs)

        # Get the size of the population.
        N = len(island.population)

        # Define local dictionary to hold the statistics.
        local_stats = {"avg": [], "std": []}

        # Initial evaluation of the population.
        avg_fitness_0, std_fitness_0 = eval_fitness(island.population)

        # Update the local population mean/std.
        if all(np.isfinite([avg_fitness_0, std_fitness_0])):

            # Store them in the dictionary.
            local_stats["avg"].append(avg_fitness_0)
            local_stats["std"].append(std_fitness_0)
        else:
            raise RuntimeError(f"0: Mean={avg_fitness_0:.5f}, Std={std_fitness_0:.5f}.")
        # _end_if_

        # Start timing the loop.
        time_t0 = time.perf_counter()

        # Repeat 'epoch' times.
        for i in range(epochs):

            # SELECT the parents.
            population_i = sel_op(island.population)

            # CROSSOVER/MUTATE to produce offsprings.
            for j in range(0, N - 1, 2):
                # Replace directly the OLD parents with the NEW offsprings.
                population_i[j], population_i[j+1] = crs_op(population_i[j],
                                                            population_i[j+1])
                # MUTATE in place the 1st offspring.
                mut_op(population_i[j])

                # MUTATE in place the 2nd offspring.
                mut_op(population_i[j+1])
            # _end_for_

            # Check if 'corrections' are enabled.
            if correction:
                _ = apply_corrections(population_i)
            # _end_if_

            # Check if 'elitism' is enabled.
            if elitism:

                # Find the individual chromosome with the highest fitness value
                # (from the old population).
                best_chromosome = max(island.population, key=lambda c: c.fitness)

                # Select randomly a position.
                locus = rnd_gen.integers(0, N)

                # Replace the chromosome with the previous best.
                population_i[locus] = best_chromosome

            # _end_if_

            # EVALUATE the i-th population.
            avg_fitness_i, std_fitness_i = eval_fitness(population_i)

            # Update the i-th population mean/std.
            if all(np.isfinite([avg_fitness_i, std_fitness_i])):

                # Store them in the dictionary.
                local_stats["avg"].append(avg_fitness_i)
                local_stats["std"].append(std_fitness_i)
            else:
                raise RuntimeError(f"{i+1}: Mean={avg_fitness_i:.5f}, Std={std_fitness_i:.5f}.")
            # _end_if_

            # Check for convergence.
            if np.fabs(avg_fitness_i - avg_fitness_0) < f_tol and std_fitness_i < 1.0E-1:

                # Switch the convergence flag and track the current iteration.
                has_converged = (True, i)

                # Exit from the loop.
                break
            # _end_if_

            # Update the average value for the next iteration.
            avg_fitness_0 = avg_fitness_i

            # Update the old population with the new chromosomes.
            island.population = population_i.copy()
        # _end_for_

        # Compute the elapsed time (in seconds).
        elapsed_time = time.perf_counter() - time_t0

        return island, has_converged, local_stats, elapsed_time
    # _end_def_

    def run(self, epochs: int = 1000, correction: bool = False, elitism: bool = True,
            f_tol: float = 1.0e-6, allow_migration: bool = False, n_periods: int = 10,
            verbose: bool = False):
        """
        Main method of the StandardGA class, that implements the evolutionary routine.

        :param epochs: (int) maximum number of iterations in the evolution process.

        :param correction: (bool) flag that if set to 'True' will check the validity of
        the population (at the gene level) and attempt to correct the genome by calling
        the random() method of the flawed gene.

        :param elitism: (bool) flag that defines elitism. If 'True' then the chromosome
        with the higher fitness will always be copied to the next generation (unaltered).

        :param f_tol: (float) tolerance in the difference between the average values of two
        consecutive populations. It is used to determine the convergence of the population.

        :param allow_migration: (bool) flag that if set to 'True' will allow the migration
        of the best individuals among the different islands.

        :param n_periods: (int) the number of times that we will break the main evolution
        to allow for chromosomes to migrate. NB: This setting is active only when the option
        allow_migration == True. Otherwise is ignored.

        :param verbose: (bool) if 'True' it will display periodically information about the
        current stats of the subpopulations. NB: This setting is active only when the option
        allow_migration == True. Otherwise is ignored.

        :return: None.
        """

        # Reset stats dictionary.
        self._stats.clear()

        # Initial time instant.
        time_t0 = time.perf_counter()

        # Randomly shuffle (in place) the original population.
        self.rng_GA.shuffle(self.population)

        # Initial random split of the total population in (active) subpopulations.
        # Active here means 'still evolving'.
        active_population = [SubPopulation(i, self.population[i::self.num_islands])
                             for i in range(self.num_islands)]

        # Final population.
        final_population = []

        # Check if we allow migration among the populations.
        if allow_migration:

            # Make sure 'n_periods' is integer.
            n_periods = int(n_periods)

            # Compute the in-between evolving epochs.
            n_epochs = int(float(epochs)/n_periods)

            # Compute the remainder epochs (if any).
            rem_epochs = int(epochs % n_periods)

            # Break the total 'epochs' in 'n_periods'.
            for i in range(n_periods):

                # Check if we want information on the screen.
                if verbose:

                    # Print a message to the screen.
                    print(f"\nCurrent period {i+1} / {n_periods}:\n")
                # _end_if_

                # If the remainder epochs is not zero, add them in the
                # last iteration to complete the total number of epochs.
                if rem_epochs and i == n_periods-1:

                    # Update the n_epochs ONLY in the last period.
                    n_epochs += rem_epochs
                # _end_if_

                # Evolve the subpopulations in parallel for 'n_epochs'.
                results_i = Parallel(n_jobs=self.MAX_CPUs, backend="threading")(
                    delayed(self.evolve_population)(p, self.evaluate_fitness, n_epochs, self._cross_op,
                                                    self._mutate_op, self._select_op, self.rng_GA,
                                                    f_tol, correction, elitism) for p in active_population
                )

                # Empty the list of active populations.
                active_population = []

                # Process the results if the i-th period.
                for res in results_i:

                    # Extract the results.
                    island, has_converged, _, elapsed_time = res

                    # Check if we want information on the screen.
                    if verbose:

                        # Find the current highest fitness.
                        best_fitness = max([p.fitness for p in island.population])

                        # Print a message to the screen.
                        print(f"Best Fitness in island {island.id} is:= {best_fitness:.5f}")
                    # _end_if_

                    # First check if the island has converged.
                    if has_converged[0]:

                        # Copy the population in the final list.
                        final_population.extend(island.population)

                        # Print a message to the screen.
                        if verbose:
                            print(f"Island population {island.id}, finished in {elapsed_time} (sec).")
                        # _end_if_

                    else:

                        # Add the island population to the new active list.
                        active_population.append(island)
                    # _end_if_

                # _end_for_

                # Check for early termination.
                if len(active_population) == 0:
                    break
                # _end_if_

                # Here we call the migration policy.
                self._migrate_op(active_population)
            # _end_for_

            # Get the rest of the populations that have not yet converged.
            for pop_j in active_population:
                final_population.extend(pop_j.population)
            # _end_for_

        else:

            # Evolve the subpopulations in parallel for 'epoch' iterations.
            results = Parallel(n_jobs=self.MAX_CPUs, backend="threading")(
                delayed(self.evolve_population)(p, self.evaluate_fitness, epochs, self._cross_op,
                                                self._mutate_op, self._select_op, self.rng_GA,
                                                f_tol, correction, elitism) for p in active_population
            )

            # Process the final results.
            for res in results:

                # Copy only the population.
                final_population.extend(res[0].population)
            # _end_for_

        # _end_if_

        # Final time instant.
        time_tf = time.perf_counter()

        # Update the population in the class.
        self.population = final_population

        # Final fitness evaluation.
        avg_fitness_final, std_fitness_final = self.evaluate_fitness(self.population)

        # Print message.
        print(f"Final Avg. Fitness = {avg_fitness_final:.4f}, "
              f"Spread = {std_fitness_final:.4f}")

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

# _end_class_
