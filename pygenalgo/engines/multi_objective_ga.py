import time
from typing import Optional

# Third party numpy.
from numpy.typing import NDArray
from numpy import (array, nanmean, nanstd,
                   isfinite, isclose, array2string)
# Custom PyGenaAlgo code.
from pygenalgo.engines import logger
from pygenalgo.engines.generic_ga import GenericGA
from pygenalgo.utils.auxiliary import average_hamming_distance

# Supported selection operators.
from pygenalgo.operators.selection.pareto_front_selector import ParetoFrontSelector
from pygenalgo.operators.selection.pareto_tournament_selector import ParetoTournamentSelector

# Public interface.
__all__ = ["MultiObjectiveGA"]

def _to_str(x: NDArray, precision: int = 4) -> str:
    """
    Auxiliary function that converts an NDArray
    into a string, with fixed precision.

    :param x: (ndarray) the array to be converted.
    :param precision: (int) the precision of the
                       output string

    :return: (str) the converted string.
    """
    return array2string(x, precision=precision)
# _end_def_


class MultiObjectiveGA(GenericGA):
    """
    Description:

        MultiObjectiveGA model provides a basic implementation of the GenericGA
        which specializes in multi-objective optimization problems. The fitness
        in these problems is not a float but a tuple of floats (one for each of
        the objective values).
    """

    def __init__(self, **kwargs) -> None:
        """
        Default constructor of MultiObjectiveGA object.
        """
        # Call the super constructor with the input parameters.
        super().__init__(**kwargs)

        # Here we check if the select operator is supported.
        if not isinstance(self._select_op, (ParetoFrontSelector,
                                            ParetoTournamentSelector)):
            raise ValueError(f"The select_op: {self._select_op.__class__.__name__} "
                             f"is not supported in MultiObjectiveGA.")
    # _end_def_

    def update_stats(self, fit_list: list[tuple[float,...]]) -> tuple:
        """
        Update the stats dictionary with the mean and std values
        of the population fitness values.

        :param fit_list: (list) fitness values of the population.

        :return: the mean and std of the fitness values.
        """
        # Convert the fitness list in a numpy array.
        arr: NDArray = array(fit_list, dtype=float)

        # Get the mean and std values (row-wise).
        avg_fitness: NDArray = nanmean(arr, axis=0, dtype=float)
        std_fitness: NDArray = nanstd(arr, axis=0, dtype=float)

        # Make sure the stat values are finite.
        if all(isfinite(avg_fitness, std_fitness)):

            # Store them in the dictionary.
            self.stats["avg"].append(avg_fitness)
            self.stats["std"].append(std_fitness)
        else:
            raise RuntimeError(f"{self.__class__.__name__}:"
                               f"Something went wrong with current "
                               f"population. Mean={avg_fitness},"
                               f"Std={std_fitness}.")
        # _end_if_

        # Return the average statistics.
        return avg_fitness, std_fitness
    # _end_def_

    def run(self, epochs: int = 100, elitism: bool = True, correction: bool = False,
            parallel: bool = False, adapt_probs: bool = False, shuffle: bool = True,
            f_tol: Optional[float] = None, f_max_eval: Optional[int] = None,
            verbose: bool = False) -> None:
        """
        Main method of the MultiObjectiveGA class, that implements the evolutionary routine.

        :param epochs: (int) maximum number of iterations in the evolution process.

        :param elitism: (bool) flag that enables elitism. If True the chromosome with
                        the highest fitness will always be copied to the next generation
                        (unaltered).

        :param correction: (bool) flag that if set to 'True' will check the validity of
                           the population (at the gene level) and attempt to correct the
                           genome by calling the random() method of the flawed gene.

        :param parallel: (bool) Flag that enables parallel computation of the fitness function.

        :param adapt_probs: (bool) If enabled (set to True), it will allow the crossover and
                            mutation probabilities to adapt according to the convergence of
                            the population to a single solution. Default is set to False.

        :param shuffle: (bool) If enabled (set to True), it will shuffle the population before
                        the application of the crossover and mutation operations. The default
                        is set to True.

        :param f_tol: (float) tolerance in the difference between the average values of two
                              consecutive populations. It is used to determine the convergence
                              of the population. If this value is None (default) the algorithm
                              will terminate using the epochs value.

        :param f_max_eval: (int) it sets an upper limit of function evaluations. If the number
                           is exceeded the genetic algorithm stops.

        :param verbose: (bool) if 'True' it will display periodically information about the
                        current average fitness and spread of the population.
        :return: None.
        """
        # Make sure everything is cleared.
        self.clear_all()

        # Get the size of the population.
        pop_size: int = len(self.population)

        # Get the fitness values before optimization.
        fit_list_0, found_solution = self.evaluate_fitness(self.population,
                                                           parallel_mode=parallel)
        # Initial termination check.
        if found_solution:
            # Display the message for the user.
            logger.info("Optimization Finished!")
            return

        # Update the average statistics in the dictionary.
        avg_fitness_0, _ = self.update_stats(fit_list_0)

        # Local variable to display information on the screen.
        # To avoid cluttering the screen we print info only 10
        # times regardless of the total number of epochs.
        its_time_to_print = epochs // 10 if epochs > 10 else 2

        # Display an information message.
        logger.info("Initial Avg. Fitness = %s",
                    _to_str(avg_fitness_0))

        # Initial time instant.
        time_t0: float = time.perf_counter()

        # Repeat 'epoch' times.
        for i in range(epochs):

            # Update current iteration.
            self.iteration = i

            # SELECT the parents.
            population_i = self.select_op(self.population)

            # Shuffle the selected parents.
            if shuffle:
                self.rng_GA.shuffle(population_i)
            # _end_def_

            # CROSSOVER/MUTATE to produce offsprings.
            self.crossover_mutate(population_i)

            # Calculate the new fitness values.
            fit_list_i, found_solution = self.evaluate_fitness(population_i, parallel)

            # Check if 'corrections' are enabled.
            if correction and self.correct_genome(population_i):
                # Update the fitness list to ensure consistency.
                fit_list_i = [p.fitness for p in population_i]
            # _end_if_

            # Check if 'elitism' is enabled.
            if elitism:
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
            # _end_if_

            # Update the mean / std in the dictionary.
            avg_fitness_i, std_fitness_i = self.update_stats(fit_list_i)

            # Log the information message.
            if verbose and (i % its_time_to_print) == 0:
                logger.info(
                    "Epoch: %5d -> Avg. Fitness = %s, Spread = %s",
                    i+1, _to_str(avg_fitness_i), _to_str(std_fitness_i)
                )
            # _end_if_

            # Update the old population with the new chromosomes.
            self.population = population_i

            # Check for termination.
            if found_solution:
                # Log a warning message.
                logger.warning("%s finished in %d iterations.",
                               self.__class__.__name__, i + 1)
                # Exit.
                break
            # _end_if_

            # Check for the maximum function evaluations.
            if f_max_eval is not None and self.f_evals >= f_max_eval:
                # Log a warning message.
                logger.warning("%s reached the maximum number of function evaluations.",
                               self.__class__.__name__)
                # Exit.
                break
            # _end_if_

            # Check for convergence (in all the objectives).
            if f_tol is not None and all(isclose(avg_fitness_i,
                                                 avg_fitness_0,
                                                 atol=f_tol)):
                # Display a warning message.
                logger.warning("%s converged in %d iterations.",
                               self.__class__.__name__, i + 1)

                # Final update the mean value.
                avg_fitness_0 = avg_fitness_i

                # Exit.
                break
            # _end_if_

            # Update the average value for the next iteration.
            avg_fitness_0 = avg_fitness_i

            # Check the adaptive flag.
            if adapt_probs:
                # Compute the current average Hamming distance.
                avg_distance = average_hamming_distance(population_i)

                # Update the genetic probabilities.
                if self.adapt_probabilities(threshold=avg_distance):
                    # Store the updated crossover and mutation probabilities.
                    self.stats["prob_crossx"].append(self._crossx_op.probability)
                    self.stats["prob_mutate"].append(self._mutate_op.probability)
        # _end_for_

        # Final time instant.
        time_tf: float = time.perf_counter()

        # Display the final average fitness value.
        logger.info(
            "Final Avg. Fitness = %s", _to_str(avg_fitness_0)
        )

        # Print final duration in seconds.
        print(f"Elapsed time: {(time_tf - time_t0):.3f} seconds.")
    # _end_def_

# _end_class_
