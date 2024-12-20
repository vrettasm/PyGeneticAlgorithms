import time
import numpy as np
from typing import Callable
from math import isnan, fabs
from collections import defaultdict
from joblib import (Parallel, delayed)

from pygenalgo.genome.chromosome import Chromosome
from pygenalgo.engines.generic_ga import GenericGA
from pygenalgo.engines.auxiliary import (apply_corrections,
                                         SubPopulation,
                                         avg_hamming_dist)
from pygenalgo.operators.mutation.mutate_operator import MutationOperator
from pygenalgo.operators.selection.select_operator import SelectionOperator
from pygenalgo.operators.crossover.crossover_operator import CrossoverOperator

from pygenalgo.operators.migration.meta_migration import MetaMigration
from pygenalgo.operators.migration.migration_operator import MigrationOperator
from pygenalgo.operators.migration.clockwise_migration import ClockwiseMigration

# Public interface.
__all__ = ["IslandModelGA"]


class IslandModelGA(GenericGA):
    """
    Description:

        In Island Model GA we run in parallel a number of "islands", each one evolving its own
        (sub)-population. Optionally we can allow "migration", among the best individuals from
        each island.
    """

    # Object variables (specific for the IslandModel).
    __slots__ = ("num_islands", "_migrate_op")

    def __init__(self, num_islands: int, migrate_op: ClockwiseMigration = None, **kwargs):
        """
        Default constructor of IslandModelGA object.

        :param num_islands: (int) number of parallel evolving islands.

        :param migrate_op: migration operator (must inherit from class MigrationOperator).

        :return: a new GA object.
        """

        # Call the super constructor with all the input parameters.
        super().__init__(**kwargs)

        # Sanity check.
        if num_islands < len(self.population):
            # Assign the number of islands.
            self.num_islands = num_islands
        else:
            # Raise an error if number of islands is too high.
            raise ValueError(f"{self.__class__.__name__}: "
                             f"Number of requested islands ({num_islands}) exceeds the size of the population.")
        # _end_if_

        # Get Migration Operator.
        if migrate_op is None:
            raise ValueError(f"{self.__class__.__name__}: Migration operator is missing.")
        else:
            self._migrate_op = migrate_op
        # _end_if_

        # Dictionary with statistics.
        self._stats = defaultdict(dict)
    # _end_def_

    @property
    def migrate_op(self) -> MigrationOperator:
        """
        Accessor method that returns the migration operator reference.

        :return: the MigrationOperator.
        """
        return self._migrate_op
    # _end_def_

    def evaluate_fitness(self, in_population: list[Chromosome]) -> (float, float):
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

        # Evaluate the fitness of all chromosomes.
        fit_list = [fit_func(p) for p in in_population]

        # Assign each chromosome its fitness value.
        for p, fit_value in zip(in_population, fit_list):
            p.fitness = fit_value
        # _end_for_

        # Convert the fitness values in a numpy array.
        fit_arr = np.array(fit_list)

        # Return the mean and std values of the fitness.
        return np.nanmean(fit_arr, dtype=float), np.nanstd(fit_arr, dtype=float)
    # _end_def_

    @classmethod
    def evolve_population(cls, island: SubPopulation, eval_fitness: Callable, epochs: int,
                          crs_op: CrossoverOperator, mut_op: MutationOperator, sel_op: SelectionOperator,
                          rnd_gen, f_tol: float = None, correction: bool = False, elitism: bool = True):
        """
        This method is called to evolve each subpopulation independently.  It is defined as 'classmethod'
        because we need access to the fitness function of the object. The input parameters have identical
        meaning with the ones from run().
        """

        # Keeps track of the convergence of the population,
        # along with the iteration that terminated.
        has_converged = (False, epochs)

        # Get the size of the population.
        N = len(island.population)

        # Define local dictionary to hold the statistics.
        local_stats = {"avg": [], "std": []}

        # Initial evaluation of the population.
        avg_fitness_0, std_fitness_0 = eval_fitness(island.population)

        # Check for initial errors.
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

            # Update current iteration in the selection operator.
            # Currently, this is used only from Boltzmann Selector.
            sel_op.iter = i

            # SELECT the parents.
            population_i = sel_op(island.population)

            # Shuffle the selected parents.
            rnd_gen.shuffle(population_i)

            # CROSSOVER/MUTATE to produce offsprings.
            for j in range(0, N - 1, 2):
                # Replace directly the OLD parents with the NEW offsprings.
                population_i[j], population_i[j + 1] = crs_op(population_i[j],
                                                              population_i[j + 1])
                # MUTATE in place the 1st offspring.
                mut_op(population_i[j])

                # MUTATE in place the 2nd offspring.
                mut_op(population_i[j + 1])
            # _end_for_

            # Check if 'corrections' are enabled.
            if correction:
                _ = apply_corrections(population_i, cls.fitness_func)
            # _end_if_

            # Check if 'elitism' is enabled.
            if elitism:

                # Find the individual chromosome with the highest fitness
                # value (from the old subpopulation of the current island).
                best_chromosome = max((p for p in island.population if not isnan(p.fitness)),
                                      key=lambda c: c.fitness, default=None)

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

            # Update the old population with the new chromosomes.
            island.population = population_i

            # Check for convergence.
            if f_tol and fabs(avg_fitness_i - avg_fitness_0) < f_tol and\
                    avg_hamming_dist(population_i) < 0.025:

                # Switch the convergence flag and track the current iteration.
                has_converged = (True, i+1)

                # Exit from the loop.
                break
            # _end_if_

            # Update the average value for the next iteration.
            avg_fitness_0 = avg_fitness_i
        # _end_for_

        # Compute the elapsed time (in seconds).
        elapsed_time = time.perf_counter() - time_t0

        return island, has_converged, local_stats, elapsed_time
    # _end_def_

    def run(self, epochs: int = 1000, correction: bool = False, elitism: bool = True,
            f_tol: float = None, allow_migration: bool = False, n_periods: int = 10,
            verbose: bool = False) -> None:
        """
        Main method of the IslandModelGA class, that implements the evolutionary routine.

        :param epochs: (int) maximum number of iterations in the evolution process.

        :param correction: (bool) flag that if set to 'True' will check the validity of
        the population (at the gene level) and attempt to correct the genome by calling
        the random() method of the flawed gene.

        :param elitism: (bool) flag that defines elitism. If 'True' then the chromosome
        with the higher fitness will always be copied to the next generation (unaltered).

        :param f_tol: (float) tolerance in the difference between the average values of two
        consecutive populations. It is used to determine the convergence of the population.
        If this value is None (default) the algorithm will terminate using the epochs value.

        :param allow_migration: (bool) flag that if set to 'True' will allow the migration
        of the best individuals among the different islands.

        :param n_periods: (int) the number of times that we will break the main evolution
        to allow for chromosomes to migrate. NB: This setting is active only when the option
        allow_migration == True. Otherwise, is ignored.

        :param verbose: (bool) if 'True' it will display periodically information about the
        current stats of the subpopulations. NB: This setting is active only when the option
        allow_migration == True. Otherwise, is ignored.

        :return: None.
        """

        # Reset stats dictionary.
        self._stats.clear()

        # Randomly shuffle (in place) the original population.
        self.rng_GA.shuffle(self.population)

        # Initial random split of the total population in (active) subpopulations.
        # Active here means 'still evolving'.
        active_population = [SubPopulation(i, self.population[i::self.num_islands])
                             for i in range(self.num_islands)]
        # Final population.
        final_population = []

        # Initial time instant.
        time_t0 = time.perf_counter()

        # Check if we allow migration among the populations.
        if allow_migration:

            # Initialize the statistics dictionary.
            for pop_n in active_population:
                self._stats[pop_n.id] = {"avg": [], "std": []}
            # _end_def_

            # Make sure 'n_periods' is integer.
            n_periods = int(n_periods)

            # Compute the in-between evolving epochs.
            n_epochs = int(float(epochs)/n_periods)

            # Compute the remainder epochs (if any).
            rem_epochs = int(epochs % n_periods)

            # Break the total 'epochs' in n_periods.
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

                # Evolve the subpopulations in parallel for n_epochs.
                results_i = Parallel(n_jobs=self.MAX_CPUs, backend="loky")(
                    delayed(self.evolve_population)(p, self.evaluate_fitness, n_epochs, self._crossx_op,
                                                    self._mutate_op, self._select_op, self.rng_GA,
                                                    f_tol, correction, elitism) for p in active_population
                )

                # Empty the list of active populations.
                active_population = []

                # Process the results if the i-th period.
                for res in results_i:

                    # Extract the results.
                    island, has_converged, local_stats, elapsed_time = res

                    # Check if we want information on the screen.
                    if verbose:

                        # Find the current highest fitness.
                        best_fitness = max((p.fitness for p in island.population
                                            if not isnan(p.fitness)))

                        # Print a message to the screen.
                        print(f"Best Fitness in island {island.id} is:= {best_fitness:.5f}.")
                    # _end_if_

                    # First check if the island has converged.
                    if has_converged[0]:

                        # Copy the population in the final list.
                        final_population.extend(island.population)

                        # Check for verbosity.
                        if verbose:
                            # Compute the total number of iterations.
                            itr = int(i*n_epochs + has_converged[1])

                            # Print a message to the screen.
                            print(f"Island population {island.id}, finished in {itr} iterations.")
                        # _end_if_

                    else:

                        # Add the island population to the new active list.
                        active_population.append(island)
                    # _end_if_

                    # Update statistics.
                    self._stats[island.id]["avg"].extend(local_stats["avg"])
                    self._stats[island.id]["std"].extend(local_stats["std"])
                # _end_for_

                # Check for early termination.
                if len(active_population) == 0:
                    break
                # _end_if_

                # Here we call the migration policy.
                self._migrate_op(active_population)
            # _end_for_

            # Get the rest of the populations that have not yet converged.
            for pop_n in active_population:
                final_population.extend(pop_n.population)
            # _end_for_

        else:

            # Evolve the subpopulations in parallel for 'epoch' iterations.
            results = Parallel(n_jobs=self.MAX_CPUs, backend="loky")(
                delayed(self.evolve_population)(p, self.evaluate_fitness, epochs, self._crossx_op,
                                                self._mutate_op, self._select_op, self.rng_GA,
                                                f_tol, correction, elitism) for p in active_population
            )

            # Process the final results.
            for res in results:

                # Extract the results.
                island, has_converged, local_stats, _ = res

                # Check if the island has converged early.
                if verbose and has_converged[0]:
                    print(f"Island population {island.id}, "
                          f"finished in {has_converged[1]} iterations.")
                # _end_if_

                # Copy only the population.
                final_population.extend(island.population)

                # Update the statistics.
                self._stats[island.id] = local_stats
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
              f"Spread = {std_fitness_final:.4f}.")

        # Print final duration in seconds.
        print(f"Elapsed time: {(time_tf - time_t0):.3f} seconds.")
    # _end_def_

    def print_migration_stats(self) -> None:
        """
        Print the migration operators stats.

        :return: None.
        """

        # First print the migration operator.
        print(self.migrate_op)

        # Check if we used the MetaMigration.
        if isinstance(self.migrate_op, MetaMigration):
            # Call internally all operators.
            for op in self.migrate_op.items:
                print(op)
            # _end_for_
        # _end_if_

    # _end_def_

# _end_class_
