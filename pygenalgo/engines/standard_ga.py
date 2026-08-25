""" Standard GA model module. """
import time
from math import isclose
from typing import Optional

# Custom PyGenaAlgo code.
from pygenalgo.engines import logger
from pygenalgo.engines.generic_ga import GenericGA, RunConfig
from pygenalgo.utils.auxiliary import average_hamming_distance

# Public interface.
__all__ = ["StandardGA", "RunConfig"]


class StandardGA(GenericGA):
    """
    Description:

        StandardGA model provides a basic implementation of the "GenericGA",
        which at each iteration (epoch) replaces the whole population using
        the genetic operators (crossover and mutation).
    """

    def __init__(self, **kwargs) -> None:
        """
        Default constructor of StandardGA object.
        """
        # Call the super constructor with all the input parameters.
        super().__init__(**kwargs)
    # _end_def_

    def run(self, config: Optional[RunConfig] = None) -> None:
        """
        Main method of the StandardGA class that implements
        the evolutionary routine.

        :param config: (RunConfig) the configuration params.

        :return: None.
        """
        # Initialize the configuration parameters.
        config = config or RunConfig()

        # Make sure everything is cleared.
        self.clear_all()

        # Get the size of the population.
        pop_size: int = len(self.population)

        # Get the fitness values before optimization.
        fit_list_0, found_solution = self.evaluate_fitness(self.population,
                                                           config.parallel)
        # Initial termination check.
        if found_solution:
            # Display the message for the user.
            logger.info("Optimization Finished!")
            return

        # Update the average statistics in the dictionary.
        avg_fitness_0, _ = self.update_stats(fit_list_0)

        # Store the initial crossover and mutation probabilities.
        self.stats["prob_crossx"].append(self.crossx_op.probability)
        self.stats["prob_mutate"].append(self.mutate_op.probability)

        # Local variable to display information on the screen.
        # To avoid cluttering the screen we print info only 10
        # times regardless of the total number of epochs.
        print_interval: int = config.epochs // 10 if config.epochs > 10 else 2

        # Display an information message.
        logger.info("Initial Avg. Fitness = %.4f", avg_fitness_0)

        # Initial time instant.
        time_t0: float = time.perf_counter()

        # Repeat 'epoch' times.
        for i in range(config.epochs):

            # Update current iteration.
            self.iteration = i

            # SELECT the parents.
            population_i = self.select_op(self.population)

            # Shuffle the selected parents.
            if config.shuffle:
                self.rng_GA.shuffle(population_i)
            # _end_def_

            # CROSSOVER/MUTATE to produce offsprings.
            self.crossover_mutate(population_i)

            # Calculate the new fitness values.
            fit_list_i, found_solution = self.evaluate_fitness(population_i,
                                                               config.parallel)
            # Check for termination.
            if found_solution:
                # Log a warning message.
                logger.warning("%s finished in %d iterations.",
                               self.__class__.__name__, i + 1)

                # Update the old population with the current.
                self.population = population_i

                # Final update the mean/std in the dictionary.
                avg_fitness_0, _ = self.update_stats(fit_list_i)

                # Exit.
                break
            # _end_if_

            # Check if 'corrections' are enabled.
            if config.correction and self.correct_genome(population_i):

                # Update the fitness list to ensure consistency.
                fit_list_i = [p.fitness for p in population_i]
            # _end_if_

            # Check if 'elitism' is enabled.
            if config.elitism:

                # Get the reference of the best chromosome
                # from the previous generation.
                previous_best = self.best_chromosome()

                # Check if the chromosome already exists in
                # the current generation to avoid flooding
                # the new pool with the same chromosome.
                if (previous_best is not None and
                        previous_best not in population_i):

                    # Select a position at random.
                    locus: int = self.rng_GA.integers(pop_size, dtype=int)

                    # Replace it with the previous best.
                    population_i[locus] = previous_best

                    # Update the list of fitness values to reflect the update.
                    fit_list_i[locus] = population_i[locus].fitness
            # _end_if_

            # Update the mean/std in the dictionary.
            avg_fitness_i, std_fitness_i = self.update_stats(fit_list_i)

            # Log the information message.
            if config.verbose and (i % print_interval) == 0:
                logger.info(
                    "Epoch: %5d -> Avg. Fitness = %.4f, Spread = %.4f",
                    i + 1, avg_fitness_i, std_fitness_i
                )
            # _end_if_

            # Update the old population with current.
            self.population = population_i

            # Check for the maximum function evaluations.
            if config.f_max_eval is not None and\
                    self.f_evals >= config.f_max_eval:
                # Log a warning message.
                logger.warning(
                    "%s reached the maximum number of function evaluations: %d",
                    self.__class__.__name__, config.f_max_eval
                )

                # Final update the mean value.
                avg_fitness_0 = avg_fitness_i

                # Exit.
                break
            # _end_if_

            # Check for convergence.
            if config.f_tol is not None and isclose(avg_fitness_i,
                                                    avg_fitness_0,
                                                    abs_tol=config.f_tol):
                # Display a warning message.
                logger.warning("%s converged in %d iterations.",
                               self.__class__.__name__, i + 1)

                # Final update the mean value.
                avg_fitness_0 = avg_fitness_i

                # Exit.
                break
            # _end_if_

            # Check the adaptive flag.
            if config.adapt_probs:
                # Compute the current average Hamming distance.
                avg_distance = average_hamming_distance(population_i)

                # Update the genetic probabilities.
                if self.adapt_probabilities(threshold=avg_distance):
                    # Store the updated crossover and mutation probabilities.
                    self.stats["prob_crossx"].append(self.crossx_op.probability)
                    self.stats["prob_mutate"].append(self.mutate_op.probability)
            # _end_if_

            # Update the average value for the next iteration.
            avg_fitness_0 = avg_fitness_i
        # _end_for_

        # Final time instant.
        time_tf: float = time.perf_counter()

        # Display the final average fitness value.
        logger.info("Final: Avg. Fitness = %.4f", avg_fitness_0)

        # Print final duration in seconds.
        print(f"Elapsed time: {(time_tf - time_t0):.3f} seconds.")
    # _end_def_

# _end_class_
