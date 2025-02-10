import time
from collections import defaultdict
from math import isnan, fabs
from typing import Callable

import numpy as np
from numpy.random import Generator

from joblib import (Parallel, delayed)

from pygenalgo.engines.auxiliary import (SubPopulation,
                                         apply_corrections,
                                         average_hamming_distance)

from pygenalgo.engines.generic_ga import GenericGA
from pygenalgo.genome.chromosome import Chromosome
from pygenalgo.operators.crossover.crossover_operator import CrossoverOperator
from pygenalgo.operators.migration.meta_migration import MetaMigration
from pygenalgo.operators.migration.migration_operator import MigrationOperator
from pygenalgo.operators.mutation.mutate_operator import MutationOperator
from pygenalgo.operators.selection.select_operator import SelectionOperator

# Public interface.
__all__ = ["IslandModelGA"]


class HelperEvoParamGroup(object):
    """
    Description:

        Implements a helper class for the grouping the evolution parameters.
        This class is used to  create a single object with all the required
        parameters  to evolve a single  population in  the Parallel process.

    """

    # Object variables.
    __slots__ = ("fitness_func", "eval_fitness", "sel_op", "crs_op", "mut_op",
                 "rng_ga", "epochs", "f_tol", "correction", "shuffle", "elitism")

    def __init__(self, fitness_func: Callable, eval_fitness: Callable, sel_op: SelectionOperator,
                 crs_op: CrossoverOperator, mut_op: MutationOperator, rng_ga: Generator, epochs: int,
                 f_tol: float, correction: bool = False, shuffle: bool = True, elitism: bool = False):
        """
            The definition of all parameters is identical to the one in IslandModelGA.run().
        """

        # Functions.
        self.fitness_func = fitness_func
        self.eval_fitness = eval_fitness

        # Genetic operators.
        self.sel_op = sel_op
        self.crs_op = crs_op
        self.mut_op = mut_op
        self.rng_ga = rng_ga

        # Boolean flags.
        self.f_tol = f_tol
        self.epochs = epochs
        self.shuffle = shuffle
        self.elitism = elitism
        self.correction = correction
    # _end_def_

    @property
    def num_epochs(self) -> int:
        """
        Accessor (getter) of the epochs' parameter.

        :return: the epochs value.
        """
        return self.epochs
    # _end_def_

    @num_epochs.setter
    def num_epochs(self, new_value: int):
        """
        Update the value of epochs in the helper class.

        :param new_value: (int) new value of epochs.

        :return: None.
        """
        # Check for the correct type.
        if isinstance(new_value, int):

            # Ensure the correct range of values.
            if new_value > 1:

                # Update the epochs value.
                self.epochs = new_value
            else:
                raise ValueError(f"{self.__class__.__name__}: "
                                 f"Number of epochs should be > 1.")
                # _end_if_
        else:
            raise TypeError(f"{self.__class__.__name__}: "
                            f"Number of epochs should be int: {type(new_value)}.")
        # _end_if_
    # _end_def_

    def _adapt_local_probabilities(self, threshold: float) -> None:
        """
        This method is used to adjust simultaneously the crossover
        and mutation parameters of the HelperEvoParamGroup object.

        :param threshold: (float) This parameter is used to determine
        whether we are going to increase / decrease the crossover and
        mutation parameters.

        :return: None.
        """

        # Initialize the trial values with the current
        # probabilities to avoid going out of limits.
        trial_pc = self.crs_op.probability
        trial_pm = self.mut_op.probability

        # Use the threshold value to adjust
        # the probabilities accordingly.
        if threshold < 0.1:

            trial_pc *= 0.9
            trial_pm *= 1.1

        elif threshold > 0.8:

            trial_pc *= 1.1
            trial_pm *= 0.9
        # _end_if_

        # Ensure the probabilities stay within the range [0, 1].
        self.crs_op.probability = min(max(trial_pc, 0.0), 1.0)
        self.mut_op.probability = min(max(trial_pm, 0.0), 1.0)
    # _end_def_

    def __call__(self, island: SubPopulation, adapt_probs: bool = False,
                 initial_probs: dict = None):
        """
        This method is called to evolve each subpopulation independently
        inside the Parallel loop.

        :param island: (SubPopulation) contains a single evolving population.

        :param adapt_probs: (bool) If enabled (set to True), it will allow
        the crossover and mutation probabilities to adapt according to the
        convergence of the fitness values. Default is set to False.

        :param initial_probs: (dict) This optional dictionary contains the
        initial probability values for the crossover / mutation operators.
        It is used in the migration period to ensure the continuity of the
        values. If this is not set then the algorithm will use the original
        (initial) values every time is called.
        """

        # Get the BitGenerator used by default_rng.
        bit_Gen = type(self.rng_ga.bit_generator)

        # Use the state from a fresh bit generator to re-seed rng_ga.
        # This uses the current system time (in nanoseconds) to avoid
        # using the same seed value among different Parallel workers.
        self.rng_ga.bit_generator.state = bit_Gen(seed=time.time_ns()).state

        # Keeps track of the convergence /termination of the
        # population, along with the iteration that happened.
        has_converged = (False, self.epochs)

        # Get the size of the population.
        N = len(island.population)

        # Define local dictionary to hold the statistics.
        local_stats = {"avg": [], "std": [], "prob_crossx": [], "prob_mutate": []}

        # Initialize this auxiliary parameter to a large number.
        avg_fitness_0 = 1.0e+100

        # Check if initial probabilities have been given.
        if adapt_probs and initial_probs:
            self.crs_op.probability = initial_probs["crossx"]
            self.mut_op.probability = initial_probs["mutate"]
        # _end_if_

        # Start timing the loop.
        time_t0 = time.perf_counter()

        # Repeat 'epoch' times.
        for i in range(self.epochs):

            # Update current iteration in the selection operator.
            # Currently, this is used only from Boltzmann Selector.
            self.sel_op.iter = i

            # SELECT the parents.
            population_i = self.sel_op(island.population)

            # Shuffle the selected parents.
            if self.shuffle:
                self.rng_ga.shuffle(population_i)
            # _end_if_

            # CROSSOVER/MUTATE to produce offsprings.
            for j in range(0, N - 1, 2):
                # Replace directly the OLD parents with the NEW offsprings.
                population_i[j], population_i[j + 1] = self.crs_op(population_i[j],
                                                                   population_i[j + 1])
                # MUTATE in place the 1st offspring.
                self.mut_op(population_i[j])

                # MUTATE in place the 2nd offspring.
                self.mut_op(population_i[j + 1])
            # _end_for_

            # Check if 'corrections' are enabled.
            if self.correction:
                apply_corrections(population_i, self.fitness_func)
            # _end_if_

            # Check if 'elitism' is enabled.
            if self.elitism:

                # Find the individual chromosome with the highest fitness
                # value (from the old subpopulation of the current island).
                best_chromosome = max((p for p in island.population if not isnan(p.fitness)),
                                      key=lambda c: c.fitness, default=None)

                # Select randomly a position.
                locus = self.rng_ga.integers(0, N)

                # Replace the chromosome with the previous best.
                population_i[locus] = best_chromosome
            # _end_if_

            # EVALUATE the i-th population.
            avg_fitness_i, std_fitness_i, found_solution = self.eval_fitness(population_i)

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

            # Check for termination.
            if found_solution:
                # Switch the convergence flag and track the current iteration.
                has_converged = (True, i + 1)

                # Exit from the loop.
                break
            # _end_if_

            # Compute the current average Hamming distance.
            d_avg = average_hamming_distance(population_i)

            # Check for convergence.
            if self.f_tol and fabs(avg_fitness_i - avg_fitness_0) < self.f_tol and d_avg < 0.025:

                # Switch the convergence flag and track the current iteration.
                has_converged = (True, i+1)

                # Exit from the loop.
                break
            # _end_if_

            # Check the adaptive flag.
            if adapt_probs:

                # Update the genetic probabilities.
                self._adapt_local_probabilities(threshold=d_avg)

                # Store the updated crossover and mutation probabilities.
                local_stats["prob_crossx"].append(self.crs_op.probability)
                local_stats["prob_mutate"].append(self.mut_op.probability)
            # _end_if_

            # Update the average value for the next iteration.
            avg_fitness_0 = avg_fitness_i
        # _end_for_

        # Compute the elapsed time (in seconds).
        elapsed_time = time.perf_counter() - time_t0

        return island, has_converged, local_stats, elapsed_time
    # _end_def_

    def __getstate__(self) -> dict:
        """
        This method is used when pickling the
        object during the parallel execution.
        """
        return {
            attr: getattr(self, attr) for attr in self.__slots__
        }
    # _end_def_

    def __setstate__(self, state: dict) -> None:
        """
        This method works in tandem with the __getstate__()
        and used to unpickle the object.
        """
        for attr, value in state.items():
            setattr(self, attr, value)
    # _end_def_

# _end_class_

class IslandModelGA(GenericGA):
    """
    Description:

        In Island Model GA we run in parallel a number of "islands", each one evolving its own
        (sub)-population. Optionally we can allow "migration", among the best individuals from
        each island.
    """

    # Object variables (specific for the IslandModel).
    __slots__ = ("num_islands", "_migrate_op")

    def __init__(self, num_islands: int, migrate_op: MigrationOperator = None, **kwargs):
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
    # _end_def_

    @property
    def migrate_op(self) -> MigrationOperator:
        """
        Accessor method that returns the migration operator reference.

        :return: the MigrationOperator.
        """
        return self._migrate_op
    # _end_def_

    def evaluate_fitness(self, in_population: list[Chromosome],
                         parallel: bool = False) -> (float, float, bool):
        """
        Evaluate all the chromosomes of the input population list with the
        custom fitness  function. After updating all  the chromosomes with
        their fitness, the method returns the average statistics mean/std.

        :param in_population: (list) The population of Chromosomes that we
        want to evaluate their fitness.

        :param parallel: (bool) Flag that enables parallel computation of
        the fitness function.

        :return: mean(fitness), std(fitness), found_solution flag.
        """

        # Get a local copy of the fitness function.
        fit_func = self.fitness_func

        # Check the 'parallel' flag.
        if parallel:

            # Evaluate the chromosomes in parallel mode.
            fit_list = Parallel(n_jobs=self.n_cpus, backend="loky")(
                delayed(fit_func)(p) for p in in_population
            )
        else:

            # Evaluate the fitness of all chromosomes.
            fit_list = [fit_func(p) for p in in_population]
        # _end_if_

        # Preallocate the fitness list.
        fitness_values = len(fit_list) * [None]

        # Flag to indicate if a solution has been found.
        found_solution = False

        # Update all chromosomes with their fitness and check
        # if a solution has been found.
        for n, (p, fit_tuple) in enumerate(zip(in_population, fit_list)):
            # Attach the fitness to each chromosome.
            p.fitness = fit_tuple[0]

            # Collect the fitness in a separate list.
            fitness_values[n] = fit_tuple[0]

            # Update the "found solution".
            found_solution |= fit_tuple[1]
        # _end_for_

        # Convert the fitness values in a numpy array.
        fit_arr = np.array(fitness_values, dtype=float)

        # Return the mean, the std values of the fitness and
        # the flag to indicate if a solution has been found.
        return (np.nanmean(fit_arr, dtype=float),
                np.nanstd(fit_arr, dtype=float),
                found_solution)
    # _end_def_

    def run(self, epochs: int = 1000, correction: bool = False, elitism: bool = True,
            f_tol: float = None, allow_migration: bool = False, n_periods: int = 10,
            adapt_probs: bool = False, shuffle: bool = True, verbose: bool = False) -> None:
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

        :param adapt_probs: (bool) If enabled (set to True), it will allow the crossover and
        mutation probabilities to adapt according to the convergence of the population to a
        single solution. Default is set to False.

        :param shuffle: (bool) If enabled (set to True), it will shuffle the population before
        the application of the crossover and mutation operations. Default is set to True.

        :param verbose: (bool) if 'True' it will display periodically information about the
        current stats of the subpopulations. NB: This setting is active only when the option
        allow_migration == True. Otherwise, is ignored.

        :return: None.
        """

        # Reset stats dictionary.
        self._stats.clear()

        # Randomly shuffle (in place) the original population.
        if shuffle:
            self.rng_GA.shuffle(self.population)
        # _end_if_

        # Initial random split of the total population in (active) subpopulations.
        # Active here means 'still evolving'.
        active_population = [SubPopulation(i, self.population[i::self.num_islands])
                             for i in range(self.num_islands)]

        # Initial evaluation of the subpopulations.
        for pop_n in active_population:

            # Initialize the statistics dictionary.
            self._stats[pop_n.id] = {"avg": [], "std": [], "prob_crossx": [], "prob_mutate": []}

            # Initial evaluation of the population. This can run also in parallel.
            avg_fitness_0, std_fitness_0, _ = self.evaluate_fitness(pop_n.population,
                                                                    parallel=True)
            # Check for initial errors.
            if all(np.isfinite([avg_fitness_0, std_fitness_0])):

                # Store them in the dictionary.
                self._stats[pop_n.id]["avg"].append(avg_fitness_0)
                self._stats[pop_n.id]["std"].append(std_fitness_0)
            else:

                raise RuntimeError(f"{pop_n.id}: Mean={avg_fitness_0:.5f}, Std={std_fitness_0:.5f}.")
            # _end_if_

            # Store the initial crossover and mutation probabilities.
            self._stats[pop_n.id]["prob_crossx"].append(self._crossx_op.probability)
            self._stats[pop_n.id]["prob_mutate"].append(self._mutate_op.probability)
        # _end_for_

        # Set up the evolution helper object. Here we group all the parameters,
        # and subsequently we pass only this object to the delayed function.
        evolve_population = HelperEvoParamGroup(fitness_func=self.fitness_func,
                                                eval_fitness=self.evaluate_fitness,
                                                sel_op=self._select_op,
                                                crs_op=self._crossx_op,
                                                mut_op=self._mutate_op,
                                                rng_ga=self.rng_GA,
                                                epochs=epochs,
                                                f_tol=f_tol,
                                                correction=correction,
                                                shuffle=shuffle,
                                                elitism=elitism)
        # Display an information message.
        print(f"Parallel evolution in progress with {self.num_islands} islands ...")

        # Final population.
        final_population = []

        # Initial time instant.
        time_t0 = time.perf_counter()

        # Check if we allow migration among the populations.
        if allow_migration:

            # Initial values for the crossover and mutation operators will be used
            # to ensure continuity in the case of adaptable probabilities.
            genetic_probs = defaultdict(dict)

            # Initial assignment of the genetic probabilities.
            for pop_n in active_population:

                # Use the values of the object operators itself.
                genetic_probs[pop_n.id]["crossx"] = self._crossx_op.probability
                genetic_probs[pop_n.id]["mutate"] = self._mutate_op.probability
            # _end_for_

            # Make sure 'n_periods' is integer.
            n_periods = int(n_periods)

            # Compute the in-between evolving epochs.
            n_epochs = int(float(epochs)/n_periods)

            # Compute the remainder epochs (if any).
            rem_epochs = int(epochs % n_periods)

            # Reuse the pool of workers.
            with Parallel(n_jobs=self.n_cpus, backend="loky") as parallel:

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

                    # Update the n_epochs in the helper object.
                    evolve_population.num_epochs = n_epochs

                    # Evolve the subpopulations in parallel for 'n_epochs'.
                    results_i = parallel(
                        delayed(evolve_population)(island=pop_i, adapt_probs=adapt_probs,
                                                   initial_probs=genetic_probs[pop_i.id])
                        for pop_i in active_population
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

                        # Check if we were adapting the probabilities.
                        if adapt_probs:

                            # Make sure there is at least one entry
                            # to avoid "index out of bound" errors.
                            if len(local_stats["prob_crossx"]) > 0:

                                # Update the values for the next interval.
                                genetic_probs[island.id]["crossx"] = local_stats["prob_crossx"][-1]
                                genetic_probs[island.id]["mutate"] = local_stats["prob_mutate"][-1]
                            # _end_if_

                            # Store the updated crossover and mutation values.
                            self._stats[island.id]["prob_crossx"].extend(local_stats["prob_crossx"])
                            self._stats[island.id]["prob_mutate"].extend(local_stats["prob_mutate"])
                        # _end_if_

                    # _end_for_

                    # Check for early termination.
                    if len(active_population) == 0:
                        break
                    # _end_if_

                    # Here we call the migration policy.
                    self._migrate_op(active_population)
                # _end_for_

            # _end_parallel_with_

            # Get the rest of the populations that have not yet converged.
            for pop_n in active_population:
                final_population.extend(pop_n.population)
            # _end_for_

        else:

            # Evolve the subpopulations in parallel for 'epoch' iterations.
            results = Parallel(n_jobs=self.n_cpus, backend="loky")(
                delayed(evolve_population)(island=pop_n, adapt_probs=adapt_probs)
                for pop_n in active_population
            )

            # Process the final results.
            for res_n in results:

                # Extract the results.
                island, has_converged, local_stats, _ = res_n

                # Check if the island has converged early.
                if verbose and has_converged[0]:
                    print(f"Island population {island.id}, "
                          f"finished in {has_converged[1]} iterations.")
                # _end_if_

                # Copy only the population.
                final_population.extend(island.population)

                # Update the statistics.
                self._stats[island.id]["avg"].extend(local_stats["avg"])
                self._stats[island.id]["std"].extend(local_stats["std"])

                # Check if we were adapting the probabilities.
                if adapt_probs:

                    # Store the updated crossover and mutation values.
                    self._stats[island.id]["prob_crossx"].extend(local_stats["prob_crossx"])
                    self._stats[island.id]["prob_mutate"].extend(local_stats["prob_mutate"])
                # _end_if_

            # _end_for_

        # _end_if_

        # Final time instant.
        time_tf = time.perf_counter()

        # Update the population in the class.
        self.population = final_population

        # Make a final fitness evaluation (to ensure consistency).
        # This can run also in parallel.
        avg_fitness_final, std_fitness_final, _ = self.evaluate_fitness(self.population,
                                                                        parallel=True)
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
