""" Arithmetic crossover (whole/linear) operator module. """
from math import isclose
from typing import Optional

# Third party imports.
from numpy.typing import ArrayLike, NDArray

# Custom code imports.
from pygenalgo.genome.gene import Gene
from pygenalgo.utils.utilities import clamp
from pygenalgo.genome.chromosome import Chromosome
from pygenalgo.operators.crossover.crossover_operator import (CrossoverOperator, Offsprings)


class ArithmeticCrossover(CrossoverOperator):
    """
    Description:

        The Whole ArithmeticCrossover operator with reflection is structurally quite similar
        to Blend Crossover (BLX-α), but they differ fundamentally in where they look for new
        solutions. They both use linear interpolation, but 'BLX-α' is explicitly designed to
        explore outside the space bounded by the parents, whereas standard ArithmeticCrossover
        stays inside (unless configured with an extraordinary α value).

        NB: Used only for real coded genomes.
    """

    def __init__(self, crossover_probability: float = 0.9,
                 p_alpha: Optional[float] = None,
                 lower_lim: ArrayLike = None,
                 upper_lim: ArrayLike = None) -> None:
        """
        Construct a 'ArithmeticCrossover' object with a given probability value.

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

        # Ensure that if p_alpha is given then it is float.
        if p_alpha is not None:
            p_alpha = clamp(float(p_alpha), 0.0, 1.0)
        # _end_if_

        # Assign variables to the _items placeholder.
        self._items: tuple[Optional[float], NDArray, NDArray] = (
            p_alpha, lower_lim, upper_lim
        )
    # _end_def_

    @staticmethod
    def _reflect_boundary(value: float, low: float, high: float) -> float:
        """
        Applies Mirroring (Reflection) strategy for out-of-bounds values.
        If reflection still falls outside the range, it clamps to safety.
        """
        # Handles extreme cases where value is out of bounds recursively
        # via reflection.
        while value > high or value < low:
            # Check upper limit.
            if value > high:
                overshoot = value - high
                value = high - overshoot
            # Check lower limit.
            elif value < low:
                undershoot = low - value
                value = low + undershoot
        # _end_while_

        # Return the value.
        return value
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

            # Extract values from the placeholder.
            p_alpha, x_lower, x_upper = self._items

            # If alpha is not given choose one at random.
            _alpha: float = self.rng.random() if p_alpha is None else p_alpha

            # Create the 1st offspring genome list.
            child_1: list[Gene] = parent1.clone_genome()

            # Create the 2nd offspring genome list.
            child_2: list[Gene] = parent2.clone_genome()

            # Find the minimum length of the two chromosomes.
            min_length: int = min(len(child_1), len(child_2))

            # Set the new gene values linearly.
            for i in range(min_length):
                # Extract gene values from both parents.
                v1: float = child_1[i].value
                v2: float = child_2[i].value

                # Skip if parents are (almost) identical.
                if isclose(v1, v2, rel_tol=1.0e-9, abs_tol=1.0e-15):
                    continue
                # _end_if_

                # Compute the new gene values.
                c1: float = _alpha * v1 + (1.0 - _alpha) * v2
                c2: float = (1.0 - _alpha) * v1 + _alpha * v2

                # Get the limits.
                xl: float = x_lower[i]
                xu: float = x_upper[i]

                # Update the genomes by applying mirroring strategy for out of bounds.
                child_1[i].value = self._reflect_boundary(c1, xl, xu)
                child_2[i].value = self._reflect_boundary(c2, xl, xu)
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
