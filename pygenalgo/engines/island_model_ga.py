""" Island model GA module. """
import time
from math import isclose
from operator import attrgetter
from collections import defaultdict
from typing import (Optional, Callable)

# Third party code.
from numpy import nanmean
from joblib import (Parallel, delayed)

# Custom PyGenaAlgo code.
from pygenalgo.engines import logger
from pygenalgo.genome.chromosome import Chromosome
from pygenalgo.utils.auxiliary import (SubPopulation,
                                       average_hamming_distance)
# Custom PyGenaAlgo code.
from pygenalgo.operators.island_manager import IslandManager
from pygenalgo.engines.generic_ga import GenericGA, RunConfig
from pygenalgo.operators.migration.meta_migration import MetaMigration
from pygenalgo.operators.migration.migration_operator import MigrationOperator

# Public interface.
__all__ = ["IslandModelGA", "RunConfig"]


class IslandModelGA(GenericGA):
    """
    Description:

        In Island Model GA we run in parallel a number of "islands", each one evolving its own
        (sub)-population. Optionally we can allow "migration", among the best individuals from
        each island.
    """

    # Object variables (specific for the IslandModel).
    __slots__ = ("_num_islands", "_migrate_op")

    def __init__(self, num_islands: int,
                 migrate_op: MigrationOperator, **kwargs) -> None:
        """
        Default constructor of IslandModelGA object.

        :param num_islands: (int) number of parallel evolving islands.

        :param migrate_op: migration operator (must inherit from class
                           MigrationOperator).

        :return: a new GA object.
        """
        # Call the super constructor with all the input parameters.
        super().__init__(**kwargs)

        # Ensure the number of islands is integer.
        num_islands = int(num_islands)

        # Sanity check.
        if num_islands > len(self.population):
            # Raise an error if number of islands is too high.
            raise ValueError(f"{self.__class__.__name__}: "
                             f"Number of requested islands ({num_islands}) "
                             f"exceeds the size of the population.")
        # _end_if_

        # Assign the number of islands to the object.
        self._num_islands: int = max(1, num_islands)

        # Get Migration Operator.
        self._migrate_op: MigrationOperator = migrate_op
    # _end_def_

    @property
    def num_islands(self) -> int:
        """
        Accessor method that returns the
        number of islands in the model.

        :return: the number of islands.
        """
        return self._num_islands
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
                           correction: bool, elitism: bool, f_tol: float, adapt_probs: bool,
                           prob_crossx: Optional[float] = None,
                           prob_mutate: Optional[float] = None) -> tuple:
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
        pop_size: int = len(island.population)

        # Define local dictionary to hold the statistics.
        local_stats: dict = {
            "avg": [], "std": [], "prob_crossx": [], "prob_mutate": []
        }

        # Initialize this auxiliary parameter to a large number.
        avg_fitness_0: float = 1.0e+100

        # Check if initial probabilities have been given.
        if prob_crossx is not None and prob_mutate is not None:
            self.crossx_op.probability = prob_crossx
            self.mutate_op.probability = prob_mutate
        # _end_if_

        # Update the pointers of the IslandManagers.
        for gen_op in (self.select_op, self.mutate_op, self.crossx_op):
            # Get the '_items' feature (or None).
            feature = getattr(gen_op, "_items", None)

            # If it is an IslandManager, then it
            # will have the idx field to set up.
            if isinstance(feature, IslandManager):
                feature.idx = island.id
        # _end_for_

        # Start timing the loop.
        time_t0: float = time.perf_counter()

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

            # Check for termination.
            if found_solution:
                # Switch the convergence flag and track the current iteration.
                has_converged = (True, i + 1)

                # Update the old population with the current.
                island.population = population_i

                # Update the average statistics in the local_stats.
                _, _ = self.update_stats(fit_list_i, local_stats)

                # Exit from the loop.
                break
            # _end_if_

            # Check if 'corrections' are enabled.
            if correction and self.correct_genome(population_i):
                # Update the fitness list to ensure consistency.
                fit_list_i = [p.fitness for p in population_i]
            # _end_if_

            # Check if 'elitism' is enabled.
            if elitism:
                # Find the individual chromosome with the highest fitness
                # value from the old subpopulation of the current island.
                previous_best: Chromosome = max(
                    (p for p in island.population if p.fitness is not None),
                    key=attrgetter("fitness"), default=None
                )

                # Check if the chromosome already exists.
                if (previous_best is not None and
                        previous_best not in population_i):
                    # Select a position at random.
                    locus: int = self.rng_GA.integers(pop_size, dtype=int)

                    # Replace it with the previous best.
                    population_i[locus] = previous_best

                    # Update the list of fitness values to reflect the update.
                    fit_list_i[locus] = population_i[locus].fitness
            # _end_if_

            # Update the i-th population mean / std.
            avg_fitness_i, _ = self.update_stats(fit_list_i,
                                                 local_stats)
            # Update the old population with the current.
            island.population = population_i

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
                    local_stats["prob_crossx"].append(self.crossx_op.probability)
                    local_stats["prob_mutate"].append(self.mutate_op.probability)
            # _end_if_

            # Update the average value for the next iteration.
            avg_fitness_0 = avg_fitness_i
        # _end_for_

        # Compute the elapsed time (in seconds).
        elapsed_time: float = time.perf_counter() - time_t0

        return island, has_converged, local_stats, elapsed_time
    # _end_def_

    def run(self, config: Optional[RunConfig] = None) -> None:
        """
        Main method of the IslandModelGA class that implements
        the evolutionary routine.

        :param config: (RunConfig) the configuration params.

        :return: None.
        """
        # Initialize the configuration parameters.
        config = config or RunConfig()

        # Reset stats dictionary.
        self.stats.clear()

        # Initial random split of the total population in active
        # subpopulations. In this context 'active' means that is
        # still evolving.
        active_population: list[SubPopulation] = [
            SubPopulation(i, self.population[i::self._num_islands])
            for i in range(self._num_islands)
        ]

        # Initial evaluation of the subpopulations.
        for pop_n in active_population:
            # Extract the population id.
            p_id: int = pop_n.id

            # Initialize the statistics dictionary.
            self.stats[p_id]: dict = {
                "avg": [], "std": [], "prob_crossx": [], "prob_mutate": []
            }

            # Initial evaluation of the population.
            fit_list_0, _ = self.evaluate_fitness(pop_n.population,
                                                  parallel_mode=True,
                                                  backend="loky")

            # Compute the initial mean/std values and update the stats[pop_n.id].
            _, _ = self.update_stats(fit_list_0, self.stats[p_id])

            # Store the initial crossover and mutation probabilities.
            self.stats[p_id]["prob_crossx"].append(self.crossx_op.probability)
            self.stats[p_id]["prob_mutate"].append(self.mutate_op.probability)
        # _end_for_

        # Set the predefined value.
        new_epochs: int = config.epochs

        # Check if we have set a maximum number on function
        # evaluations and re-adjust the number of epochs.
        if config.f_max_eval is not None:

            # First remove the counts from the initial evaluation
            # of the population.
            total_f_counts = int(config.f_max_eval) - self.f_evals

            # Assuming each epoch performs N function evaluations.
            new_epochs = int(total_f_counts / len(self.population))

            # Display a warning message.
            logger.warning(
                "The 'f_max_eval' parameter has been set to: %s. "
                "The 'epochs' value has been re-adjusted to: %s\n",
                config.f_max_eval, new_epochs)
        # _end_if_

        # Display an information message.
        logger.info("Parallel evolution in progress with %s islands ...",
                    self._num_islands)

        # Final population.
        final_population = []

        # Local copy of evolve population.
        fn_evolve: Callable = self._evolve_population

        # Local copy of all the common parameters.
        # NOTE: 'epochs' value might have changed!
        common_parameters: dict = {
            "f_tol": config.f_tol,
            "epochs": new_epochs,
            "shuffle": config.shuffle,
            "elitism": config.elitism,
            "correction": config.correction,
            "adapt_probs": config.adapt_probs
        }

        # Initial time instant.
        time_t0: float = time.perf_counter()

        # Check if we allow migration among the populations.
        if config.allow_migration:

            # Initial values for the crossover and mutation operators will be used
            # to ensure continuity in the case of adaptable probabilities.
            genetic_probs = defaultdict(dict)

            # Initial assignment of the genetic probabilities.
            for pop_n in active_population:
                # Extract the population (island) id.
                p_id: int = pop_n.id

                # Use the values of the object operators itself.
                genetic_probs[p_id]["crossx"] = self.crossx_op.probability
                genetic_probs[p_id]["mutate"] = self.mutate_op.probability
            # _end_for_

            # Make sure 'n_periods' is integer.
            n_periods = int(config.n_periods)

            # Compute the in-between evolving epochs.
            n_epochs = int(new_epochs / n_periods)

            # Compute the remainder epochs (if any).
            rem_epochs = int(new_epochs % n_periods)

            # Type hint the work_parallel to avoid warnings.
            work_parallel: Parallel

            # Reuse the pool of workers.
            with Parallel(n_jobs=self.n_cpus, backend="loky") as work_parallel:

                # Break the total 'epochs' in n_periods.
                for i in range(n_periods):

                    # Check if we want information on to be logged.
                    if config.verbose:
                        logger.info("Current period %s / %s:", i + 1, n_periods)
                    # _end_if_

                    # If the remainder epochs is not zero, add them in the
                    # last iteration to complete the total number of epochs.
                    if rem_epochs and i == n_periods-1:

                        # Update the n_epochs ONLY in the last period.
                        n_epochs += rem_epochs
                    # _end_if_

                    # Update epochs to 'n_epochs'.
                    common_parameters["epochs"] = n_epochs

                    # Evolve the subpopulations in parallel for 'n_epochs'.
                    results_i = work_parallel(
                        delayed(fn_evolve)(island=pop_i,
                                           prob_crossx=genetic_probs[pop_i.id]["crossx"],
                                           prob_mutate=genetic_probs[pop_i.id]["mutate"],
                                           **common_parameters)
                        for pop_i in active_population
                    )

                    # Empty the list of active populations.
                    active_population = []

                    # Process the results if the i-th period.
                    for res in results_i:

                        # Extract the results.
                        island, has_converged, local_stats, _ = res

                        # Check if we want information on the screen.
                        if config.verbose:

                            # Find the current highest fitness.
                            best_fitness = max(
                                (p.fitness for p in island.population
                                 if p.fitness is not None)
                            )

                            # Log an update of the progress.
                            logger.info(
                                "Best Fitness in island %s is:= %.5f",
                                island.id, best_fitness
                            )
                        # _end_if_

                        # First check if the island has converged.
                        if has_converged[0]:
                            # Copy the population in the final list.
                            final_population.extend(island.population)

                            # Check for verbosity.
                            if config.verbose:
                                # Compute the total number of iterations.
                                itr = int(i*n_epochs + has_converged[1])

                                # Log a warning message to the screen.
                                logger.warning(
                                    "Island population %s finished in %s iterations.",
                                    island.id, itr
                                )
                            # _end_if_
                        else:
                            # Add the island population to the new active list.
                            active_population.append(island)
                        # _end_if_

                        # Update statistics.
                        self.stats[island.id]["avg"].extend(local_stats["avg"])
                        self.stats[island.id]["std"].extend(local_stats["std"])

                        # Check if we were adapting the probabilities.
                        if config.adapt_probs:

                            # Make sure there is at least one entry
                            # to avoid "index out of bound" errors.
                            if len(local_stats["prob_crossx"]) > 0:

                                # Update the values for the next interval.
                                genetic_probs[island.id]["crossx"] = local_stats["prob_crossx"][-1]
                                genetic_probs[island.id]["mutate"] = local_stats["prob_mutate"][-1]
                            # _end_if_

                            # Store the updated crossover and mutation values.
                            self.stats[island.id]["prob_crossx"].extend(local_stats["prob_crossx"])
                            self.stats[island.id]["prob_mutate"].extend(local_stats["prob_mutate"])
                        # _end_if_

                    # _end_for_

                    # Check for early termination.
                    if len(active_population) == 0:
                        logger.warning("No active islands found.")
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
                delayed(fn_evolve)(island=pop_n, **common_parameters)
                for pop_n in active_population
            )

            # Process the final results.
            for res_n in results:

                # Extract the results.
                island, has_converged, local_stats, _ = res_n

                # Check if we want to log output.
                if has_converged[0]:
                    logger.info(
                        "Island population %s, finished in %s iterations.",
                        island.id, has_converged[1]
                    )

                # Copy only the population.
                final_population.extend(island.population)

                # Update the statistics.
                self.stats[island.id]["avg"].extend(local_stats["avg"])
                self.stats[island.id]["std"].extend(local_stats["std"])

                # Check if we were adapting the probabilities.
                if config.adapt_probs:
                    # Store the updated crossover and mutation values.
                    self.stats[island.id]["prob_crossx"].extend(local_stats["prob_crossx"])
                    self.stats[island.id]["prob_mutate"].extend(local_stats["prob_mutate"])
            # _end_for_

        # _end_if_

        # Update the population in the class.
        self.population = final_population

        # Make a final fitness evaluation (to ensure consistency).
        fit_list_final, _ = self.evaluate_fitness(self.population,
                                                  parallel_mode=True,
                                                  backend="loky")
        # Compute the mean value.
        avg_fitness_final = nanmean(fit_list_final, dtype=float)

        # Final time instant.
        time_tf: float = time.perf_counter()

        # Print message.
        logger.info("Final Avg. Fitness = %.4f.", avg_fitness_final)

        # Print final duration in seconds.
        print(f"Elapsed time: {(time_tf - time_t0):.3f} seconds.")
    # _end_def_

    def print_migration_stats(self) -> None:
        """
        Print the migration operators stats.

        :return: None.
        """
        # First print the migration operator.
        print(self._migrate_op)

        # Check if we used the MetaMigration.
        if isinstance(self._migrate_op, MetaMigration):
            # Call internally all operators.
            for op in self._migrate_op.items:
                print(op)
    # _end_def_

# _end_class_
