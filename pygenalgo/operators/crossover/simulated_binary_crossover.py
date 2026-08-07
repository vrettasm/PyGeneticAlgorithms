from math import fabs, isclose

from numpy import asarray
from numpy import any as np_any
from numpy.typing import ArrayLike, NDArray

from pygenalgo.genome.gene import Gene
from pygenalgo.genome.chromosome import Chromosome
from pygenalgo.operators.crossover.crossover_operator import (CrossoverOperator, Offsprings)


class SimulatedBinaryCrossover(CrossoverOperator):
    """
    Description:

        Simulate binary crossover (SBX) creates two children chromosomes (offsprings)
        by replicating the search behavior of a single-point crossover operator used
        in binary-coded Genetic Algorithms onto real-coded variables.

        It uses a probability distribution centered around the parents' locations to
        calculate a spread factor, ensuring that the average distance of the offspring
        from the parents is proportional to the distance between the parents themselves.

        NB: Used only for real coded genomes.
    """

    def __init__(self, crossover_probability: float = 0.9, eta: float = 20.0,
                 lower_lim: ArrayLike = None, upper_lim: ArrayLike = None) -> None:
        """
        Construct a 'SimulatedBinaryCrossover' object with a given probability value.

        :param crossover_probability: (float).

        :param eta: (float) distribution index. Typically, between 5 - 50.
                    Higher values produce children closer to their parents.

        :param lower_lim: (ArrayLike) lower limit values for the genes.

        :param upper_lim: (ArrayLike) upper limit values for the genes.
        """

        # Call the super constructor with the provided initial value.
        super().__init__(crossover_probability=crossover_probability)

        # Check if the lower and upper bounds are set.
        if (lower_lim is None) or (upper_lim is None):
            raise ValueError(f"{self.__class__.__name__}: "
                             f"Lower or Upper limits are missing.")

        # Make sure the limits are numpy arrays.
        lower_lim = asarray(lower_lim, dtype=float)
        upper_lim = asarray(upper_lim, dtype=float)

        # Check if there is a size mismatch.
        if lower_lim.size != upper_lim.size:
            raise ValueError(f"{self.__class__.__name__}: "
                             f"Lower and Upper limits sizes do not match.")

        # Check if the boundaries are set correctly.
        if np_any(upper_lim <= lower_lim):
            raise ValueError(f"{self.__class__.__name__}: "
                             f"Lower and Upper limits are set incorrectly.")

        # Assign variables to the _items placeholder.
        self._items: tuple[float, NDArray, NDArray] = (
            max(5, min(float(eta), 50)), lower_lim, upper_lim
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
        if (parent1 != parent2) and self.is_operator_applicable():

            # Extract the values from the placeholder.
            eta, x_lower, x_upper = self._items

            # Create the 1st offspring genome list.
            child_1: list[Gene] = [
                gene.clone() for gene in parent1.genome
            ]

            # Create the 2nd offspring genome list.
            child_2: list[Gene] = [
                gene.clone() for gene in parent2.genome
            ]

            # Find the minimum length of the two chromosomes.
            min_length: int = min(len(child_1), len(child_2))

            for i in range(min_length):

                # Get the i-th position gene values
                # from both parents.
                x1: float = child_1[i].value
                x2: float = child_2[i].value

                # Skip if parents are (almost) identical.
                if isclose(x1, x2, rel_tol=1.0e-9, abs_tol=1.0e-15):
                    continue
                # _end_if_

                # Initialize boolean flag.
                swapped: bool = False

                # Ensure x1 <= x2 for consistency.
                if x1 > x2:
                    # Swap in place.
                    x1, x2 = x2, x1

                    # Swap the flag.
                    swapped = True
                # _end_if_

                # Compute the difference between the
                # two gene values.
                denominator: float = (x2 - x1)

                # Make the condition scale aware.
                scale = max(1.0, abs(x1), abs(x2))
                if fabs(denominator) <= 1.0e-15 * scale:
                    continue

                # Local bounds lookups.
                xl: float = x_lower[i]
                xu: float = x_upper[i]

                # Compute both distance factors to lower and upper bounds.
                beta1 = 1.0 + (2.0 * (x1 - xl) / denominator)
                beta2 = 1.0 + (2.0 * (xu - x2) / denominator)

                # Precompute repeated variables once.
                eta_1: float = eta + 1.0
                inv_eta_1: float = 1.0 / eta

                # Compute separate alpha values for balancing distributions.
                alpha1 = 2.0 - (beta1 ** -eta_1)
                alpha2 = 2.0 - (beta2 ** -eta_1)

                # Draw one random number to generate BOTH children.
                rand = self.rng.random()

                # Sample beta_q1 based on lower bound proximity.
                if rand <= (1.0 / alpha1):
                    beta_q1 = (rand * alpha1) ** inv_eta_1
                else:
                    beta_q1 = (1.0 / (2.0 - rand * alpha1)) ** inv_eta_1

                # Sample beta_q2 based on upper bound proximity.
                if rand <= (1.0 / alpha2):
                    beta_q2 = (rand * alpha2) ** inv_eta_1
                else:
                    beta_q2 = (1.0 / (2.0 - rand * alpha2)) ** inv_eta_1

                # Precompute repeated variable once.
                x1_plus_x2: float = x1 + x2

                # Apply the correct distinct scaling factors to each child.
                c1: float = 0.5 * (x1_plus_x2 - beta_q1 * denominator)
                c2: float = 0.5 * (x1_plus_x2 + beta_q2 * denominator)

                # Safe clipping.
                c1 = max(xl, min(c1, xu))
                c2 = max(xl, min(c2, xu))

                # Extract the gene function.
                gene_function = child_1[i].func

                if not swapped:
                    child_1[i] = Gene(datum=c1, func=gene_function)
                    child_2[i] = Gene(datum=c2, func=gene_function)
                else:
                    child_1[i] = Gene(datum=c2, func=gene_function)
                    child_2[i] = Gene(datum=c1, func=gene_function)
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
