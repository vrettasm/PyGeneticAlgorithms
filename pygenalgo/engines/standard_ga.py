import time
from math import isclose

from numpy import (array, nanmean, nanstd, isfinite)

from pygenalgo.engines import logger
from pygenalgo.engines.generic_ga import GenericGA
from pygenalgo.utils.auxiliary import (apply_corrections,
                                       average_hamming_distance)

from pygenalgo.operators.mutation.meta_mutator import MetaMutator
from pygenalgo.operators.crossover.meta_crossover import MetaCrossover


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

    def update_stats(self, fit_list: list[float]) -> tuple[float, float]:
        """
        Update the stats dictionary with the mean/std values of the
        population fitness values.

        :param fit_list: (list) fitness values of the population.

        :return: the mean and std of the fitness values.
        """
        # Convert the fitness list in a numpy array.
        arr = array(fit_list, dtype=float)

        # Get the mean and std values.
        avg_fitness = nanmean(arr, dtype=float)
        std_fitness = nanstd(arr, dtype=float)

        # Make sure the stat values are finite.
        if all(isfinite([avg_fitness, std_fitness])):

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

    def run(self, epochs: int = 100, elitism: bool = True, correction: bool = False,
            f_tol: float = None, parallel: bool = False, adapt_probs: bool = False,
            shuffle: bool = True, f_max_eval: int = None, verbose: bool = False) -> None:
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

        :param shuffle: (bool) If enabled (set to True), it will shuffle the population before
        the application of the crossover and mutation operations. Default is set to True.

        :param f_max_eval: (int) it sets an upper limit of function evaluations. If the number
        is exceeded the genetic algorithm stops.

        :param verbose: (bool) if 'True' it will display periodically information about the
        current average fitness and spread of the population.

        :return: None.
        """
        # Make sure everything is cleared.
        self.clear_all()

        # Get the size of the population.
        pop_size = len(self.population)

        # Get the fitness values before optimisation.
        fit_list_0, _ = self.evaluate_fitness(self.population, parallel)

        # Update the average statistics in the dictionary.
        avg_fitness_0, _ = self.update_stats(fit_list_0)

        # Store the initial crossover and mutation probabilities.
        self._stats["prob_crossx"].append(self._crossx_op.probability)
        self._stats["prob_mutate"].append(self._mutate_op.probability)

        # Local variable to display information on the screen.
        # To avoid cluttering the screen we print info only 10
        # times regardless of the total number of epochs.
        its_time_to_print = (epochs//10)

        # Display an information message.
        logger.info(f"Initial Avg. Fitness = {avg_fitness_0:.4f}")

        # Initial time instant.
        time_t0 = time.perf_counter()

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
            if correction:
                # Apply the function.
                total_corrections, f_counts = apply_corrections(population_i, self.fitness_func)

                # If corrections were made, we will need to make some updates.
                if total_corrections > 0:

                    # Update the function evaluation counter.
                    self.f_eval_increase_by(f_counts)

                    # Update the fitness list to ensure consistency.
                    fit_list_i = [p.fitness for p in population_i]

                    # Log the corrections.
                    logger.debug(f"> {total_corrections} "
                                 f"correction(s) took place at epoch: {i}")
            # _end_if_

            # Check if 'elitism' is enabled.
            if elitism:
                # Get the reference of the best chromosome
                # from the previous generation.
                previous_best = self.best_chromosome()

                # Check if the chromosome already exists in
                # the current generation to avoid flooding
                # the new pool with the same chromosome.
                if previous_best not in population_i:

                    # Select a position at random.
                    locus = self.rng_GA.integers(pop_size)

                    # Replace it with the previous best.
                    population_i[locus] = previous_best

                    # Update the list of fitness values to reflect the update.
                    fit_list_i[locus] = population_i[locus].fitness
                # _end_if_
            # _end_if_

            # Update the mean/std in the dictionary.
            avg_fitness_i, std_fitness_i = self.update_stats(fit_list_i)

            # Log the information message.
            if verbose and (i % its_time_to_print) == 0:
                logger.info(f"Epoch: {i + 1:>5} -> "
                            f"Avg. Fitness = {avg_fitness_i:.4f}, "
                            f"Spread = {std_fitness_i:.4f}")
            # _end_if_

            # Update the old population with the new chromosomes.
            self.population = population_i

            # Check for termination.
            if found_solution:
                # Log a warning message.
                logger.warning(f"{self.__class__.__name__} "
                               f"finished in {i + 1} iterations.")
                # Exit.
                break
            # _end_if_

            # Check for the maximum function evaluations.
            if f_max_eval and self.f_eval >= f_max_eval:
                # Log a warning message.
                logger.warning(f"{self.__class__.__name__} "
                               "reached the maximum number of function evaluations.")
                # Exit.
                break
            # _end_if_

            # Check for convergence.
            if f_tol and isclose(avg_fitness_i, avg_fitness_0, abs_tol=f_tol):
                # Display a warning message.
                logger.warning(f"{self.__class__.__name__} "
                               f"converged in {i + 1} iterations.")
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
                    self._stats["prob_crossx"].append(self._crossx_op.probability)
                    self._stats["prob_mutate"].append(self._mutate_op.probability)
        # _end_for_

        # Final time instant.
        time_tf = time.perf_counter()

        # Display the final average fitness value.
        logger.info(f"Final   Avg. Fitness = {avg_fitness_0:.4f}")

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
        print(self.crossx_op)

        # Check if we used the MetaCrossover.
        if isinstance(self.crossx_op, MetaCrossover):
            # Call internally all operators.
            for op in self.crossx_op.items:
                print(op)
        # _end_if_

        # Lastly print the mutation operator.
        print(self.mutate_op)

        # Check if we used the MetaMutator.
        if isinstance(self.mutate_op, MetaMutator):
            # Call internally all operators.
            for op in self.mutate_op.items:
                print(op)
            # _end_for_
    # _end_def_

# _end_class_
