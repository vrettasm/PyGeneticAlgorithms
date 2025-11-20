import time
from math import isnan, isclose
from operator import attrgetter
from collections import defaultdict

from joblib import (Parallel, delayed)
from numpy import (nanmean, nanstd, isfinite)

from pygenalgo.utils.auxiliary import (SubPopulation,
                                       apply_corrections,
                                       average_hamming_distance)

from pygenalgo.engines.generic_ga import GenericGA
from pygenalgo.operators.migration.meta_migration import MetaMigration
from pygenalgo.operators.migration.migration_operator import MigrationOperator

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
    # _end_def_

    @property
    def migrate_op(self) -> MigrationOperator:
        """
        Accessor method that returns the migration operator reference.

        :return: the MigrationOperator.
        """
        return self._migrate_op
    # _end_def_

    def _evolve_population(self, island: SubPopulation, epochs: int, shuffle: bool,
                           correction: bool, elitism: bool, f_tol: float, adapt_probs: bool = None,
                           prob_crossx: float = None, prob_mutate: float = None) -> tuple:
        """
        This is a helper method to be used inside the Parallel delayed method.
        It is responsible for running the evolution of a single population (island).

        :return: a tuple (island, has_converged, local_stats, elapsed_time)
        """
        # Get the BitGenerator used by default_rng.
        bit_gen = type(self.rng_GA.bit_generator)

        # Use the state from a fresh bit generator to re-seed rng_GA.
        # This uses the current system time (in nanoseconds) to avoid
        # using the same seed value among different Parallel workers.
        self.rng_GA.bit_generator.state = bit_gen(seed=time.time_ns()).state

        # Keeps track of the convergence /termination of the
        # population, along with the iteration that happened.
        has_converged = (False, epochs)

        # Get the size of the population.
        pop_size = len(island.population)

        # Define local dictionary to hold the statistics.
        local_stats = {"avg": [], "std": [], "prob_crossx": [], "prob_mutate": []}

        # Initialize this auxiliary parameter to a large number.
        avg_fitness_0 = 1.0e+100

        # Check if initial probabilities have been given.
        if prob_crossx and prob_mutate:
            self._crossx_op.probability = prob_crossx
            self._mutate_op.probability = prob_mutate
        # _end_if_

        # Start timing the loop.
        time_t0 = time.perf_counter()

        # Repeat 'epoch' times.
        for i in range(epochs):

            # Update current iteration.
            self.iteration = i

            # SELECT the parents.
            population_i = self.select_op(island.population)

            # Shuffle the selected parents.
            if shuffle:
                self.rng_GA.shuffle(population_i)
            # _end_if_

            # CROSSOVER/MUTATE to produce offsprings.
            self.crossover_mutate(population_i)

            # EVALUATE the i-th population.
            fit_list_i, found_solution = self.evaluate_fitness(population_i)

            # Check if 'corrections' are enabled.
            if correction:
                # Apply the function.
                total_corrections, _ = apply_corrections(population_i, self.fitness_func)

                # If corrections were made we will
                # need to update the fitness list.
                if total_corrections > 0:

                    # Update the fitness list to ensure consistency.
                    fit_list_i = [p.fitness for p in population_i]
            # _end_if_

            # Check if 'elitism' is enabled.
            if elitism:
                # Find the individual chromosome with the highest fitness
                # value (from the old subpopulation of the current island).
                best_chromosome = max([p for p in island.population if not isnan(p.fitness)],
                                      key=attrgetter("fitness"), default=None)

                # Check if the chromosome already exists.
                if best_chromosome not in population_i:
                    # Select a position at random.
                    locus = self.rng_GA.integers(pop_size)

                    # Replace it with the previous best.
                    population_i[locus] = best_chromosome

                    # Update the list of fitness values to reflect the update.
                    fit_list_i[locus] = population_i[locus].fitness
                # _end_if_
            # _end_if_

            # Compute the mean value.
            avg_fitness_i = nanmean(fit_list_i, dtype=float)

            # Compute the standard deviation value.
            std_fitness_i = nanstd(fit_list_i, dtype=float)

            # Update the i-th population mean/std.
            if all(isfinite([avg_fitness_i, std_fitness_i])):

                # Store them in the dictionary.
                local_stats["avg"].append(avg_fitness_i)
                local_stats["std"].append(std_fitness_i)
            else:
                raise RuntimeError(f"{i + 1}: Mean={avg_fitness_i:.5f}, Std={std_fitness_i:.5f}.")
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

            # Check for convergence.
            if f_tol and isclose(avg_fitness_i, avg_fitness_0, abs_tol=f_tol):
                # Switch the convergence flag and track the current iteration.
                has_converged = (True, i + 1)

                # Exit from the loop.
                break
            # _end_if_

            # Check the adaptive flag.
            if adapt_probs:

                # Compute the current average Hamming distance.
                avg_distance = average_hamming_distance(population_i)

                # Update the genetic probabilities.
                if self.adapt_probabilities(threshold=avg_distance):
                    # Store the updated crossover and mutation probabilities.
                    local_stats["prob_crossx"].append(self._crossx_op.probability)
                    local_stats["prob_mutate"].append(self._mutate_op.probability)
            # _end_if_

            # Update the average value for the next iteration.
            avg_fitness_0 = avg_fitness_i
        # _end_for_

        # Compute the elapsed time (in seconds).
        elapsed_time = time.perf_counter() - time_t0

        return island, has_converged, local_stats, elapsed_time
    # _end_def_

    def run(self, epochs: int = 500, correction: bool = False, elitism: bool = True,
            f_tol: float = None, allow_migration: bool = False, n_periods: int = 10,
            adapt_probs: bool = False, shuffle: bool = True, f_max_eval: int = None,
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

        :param adapt_probs: (bool) If enabled (set to True), it will allow the crossover and
        mutation probabilities to adapt according to the convergence of the population to a
        single solution. Default is set to False.

        :param shuffle: (bool) If enabled (set to True), it will shuffle the population before
        the application of the crossover and mutation operations. Default is set to True.

        :param f_max_eval: (int) it sets an upper limit of function evaluations. If the number
        is exceeded the genetic algorithm stops. If this value is set, the epochs will be
        ignored and re-adjusted to meet the new requirement.

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
            fit_list_0, _ = self.evaluate_fitness(pop_n.population, parallel_mode=True,
                                                  backend="loky")
            # Compute the mean value.
            avg_fitness_0 = nanmean(fit_list_0, dtype=float)

            # Compute the standard deviation value.
            std_fitness_0 = nanstd(fit_list_0, dtype=float)

            # Check for initial errors.
            if all(isfinite([avg_fitness_0, std_fitness_0])):

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

        # Check if we have set a maximum number on function
        # evaluations and re-adjust the epochs.
        if f_max_eval:

            # First remove the counts from the initial
            # evaluation of the population.
            total_f_counts = int(f_max_eval) - self.f_eval

            # Assuming each epoch performs N function evaluations.
            epochs = int(total_f_counts / len(self.population))

            # Display an information message.
            print(f"INFO: The 'f_max_eval' parameter has been set to: {f_max_eval}. "
                  f"The 'epochs' value has been re-adjusted to: {epochs}\n")
        # _end_if_

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

                    # Evolve the subpopulations in parallel for 'n_epochs'.
                    results_i = parallel(
                        delayed(self._evolve_population)(island=pop_i,
                                                         epochs=n_epochs,
                                                         shuffle=shuffle,
                                                         elitism=elitism,
                                                         correction=correction,
                                                         f_tol=f_tol,
                                                         adapt_probs=adapt_probs,
                                                         prob_crossx=genetic_probs[pop_i.id]["crossx"],
                                                         prob_mutate=genetic_probs[pop_i.id]["mutate"])
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
                delayed(self._evolve_population)(island=pop_n,
                                                 epochs=epochs,
                                                 shuffle=shuffle,
                                                 elitism=elitism,
                                                 correction=correction,
                                                 f_tol=f_tol,
                                                 adapt_probs=adapt_probs)
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
            # _end_for_

        # _end_if_

        # Update the population in the class.
        self.population = final_population

        # Make a final fitness evaluation (to ensure consistency).
        fit_list_final, _ = self.evaluate_fitness(self.population,
                                                  parallel_mode=True, backend="loky")
        # Compute the mean value.
        avg_fitness_final = nanmean(fit_list_final, dtype=float)

        # Final time instant.
        time_tf = time.perf_counter()

        # Print message.
        print(f"Final Avg. Fitness = {avg_fitness_final:.4f}.")

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
    # _end_def_

# _end_class_
