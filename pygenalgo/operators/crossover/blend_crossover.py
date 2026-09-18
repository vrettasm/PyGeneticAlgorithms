""" Blend-a crossover (BLX-a) operator module. """
# Third party imports.
from numpy.typing import ArrayLike, NDArray

# Custom code imports.
from pygenalgo.genome.gene import Gene
from pygenalgo.utils.utilities import clamp
from pygenalgo.genome.chromosome import Chromosome
from pygenalgo.operators.crossover.crossover_operator import (CrossoverOperator, Offsprings)


class BlendCrossover(CrossoverOperator):
    """
    Description:

        Blend-a crossover (BLX-a) creates two children chromosomes (offsprings) by
        uniformly picking values that lie  between two points that contain the two
        parents but may extend equally on either side determined by a user specified
        parameter 'a'.

        NB: Used only for real coded genomes.
    """

    def __init__(self, crossover_probability: float = 0.9, p_alpha: float = 0.5,
                 lower_lim: ArrayLike = None, upper_lim: ArrayLike = None) -> None:
        """
        Construct a 'BlendCrossover' object with a given probability value.

        :param crossover_probability: (float).

        :param p_alpha: (float).

        :param lower_lim: (ArrayLike) lower limit values for the genes.

        :param upper_lim: (ArrayLike) upper limit values for the genes.
        """

        # Call the super constructor with the provided initial value.
        super().__init__(crossover_probability=crossover_probability)

        # Validate the bounds.
        lower_lim, upper_lim = self.validate_bounds(lower_lim,
                                                    upper_lim)

        # Ensure p_alpha parameter is float.
        p_alpha = clamp(float(p_alpha), 0.0, 1.0)

        # Assign variables to the _items placeholder.
        self._items: tuple[float, NDArray, NDArray] = (
            p_alpha, lower_lim, upper_lim
        )
    # _end_def_

    def crossover(self, parent1: Chromosome, parent2: Chromosome) -> Offsprings:
        """
        Perform the crossover operation on the two input parent chromosomes.

        :param parent1: (Chromosome).

        :param parent2: (Chromosome).

        :return: child1 and child2 (as Chromosomes).
        """
        # If the crossover probability is higher than a uniformly
        # random value and the parents aren't identical apply the
        # changes.
        if self.is_operator_applicable() and (parent1 != parent2):

            # Extract the values from the placeholder.
            p_alpha, x_lower, x_upper = self._items

            # Create the 1st offspring genome list.
            child_1: list[Gene] = parent1.clone_genome()

            # Create the 2nd offspring genome list.
            child_2: list[Gene] = parent2.clone_genome()

            # Find the minimum length of the two chromosomes.
            min_length: int = min(len(child_1), len(child_2))

            # Generate uniform random numbers in the [0.0, 1.0).
            random_uniform: NDArray = self.rng.random(size=(min_length, 2))

            # Set the new gene values iteratively.
            for i in range(min_length):
                # Extract gene values from both parents.
                g1: float = child_1[i].value
                g2: float = child_2[i].value

                # Skip if genes are (almost) identical.
                if isclose(g1, g2, rel_tol=1.0e-9, abs_tol=1.0e-15):
                    continue
                # _end_if_

                # Get the min / max values.
                min_value: float = min(g1, g2)
                max_value: float = max(g1, g2)

                # Get the offset by scaling the distance
                # between the two gene values with alpha.
                offset_distance: float = p_alpha * (max_value - min_value)

                # Compute the lower and upper limits by
                # removing / adding the offset distance.
                min_value -= offset_distance
                max_value += offset_distance

                # Extract the two random values.
                rv_1, rv_2 = random_uniform[i]

                # Compute the difference.
                diff: float = max_value - min_value

                # Create two new gene values.
                new_value_1: float = min_value + (diff * rv_1)
                new_value_2: float = min_value + (diff * rv_2)

                # Local bounds lookups.
                xl: float = x_lower[i]
                xu: float = x_upper[i]

                # Ensure the new values are within limits.
                child_1[i].value = min(max(new_value_1, xl), xu)
                child_2[i].value = min(max(new_value_2, xl), xu)
            # _end_for_

            # Increase the crossover counter.
            self.inc_counter()

            # Return two new offsprings.
            return Chromosome(child_1), Chromosome(child_2)
        # _end_if_

        # Return two cloned offsprings.
        return parent1.clone(), parent2.clone()
    # _end_def_

# _end_class_
