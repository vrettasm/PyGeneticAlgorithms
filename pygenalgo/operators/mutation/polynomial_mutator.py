""" Polynomial mutator (PM-eta) module. """
# Third party imports.
from numpy.typing import ArrayLike

# Custom code imports.
from pygenalgo.utils.utilities import clamp
from pygenalgo.genome.chromosome import Chromosome
from pygenalgo.operators.mutation.mutate_operator import MutationOperator


class PolynomialMutator(MutationOperator):
    """
    Description:

        Polynomial mutator (PM-eta), mutates the chromosome by adjusting the
        values of genes according to a polynomial distribution. This results
        in a more controlled and smoother alteration of values.
    """

    def __init__(self, mutate_probability: float = 0.1,
                 eta_pm: float = 20.0,
                 lower_lim: ArrayLike = None,
                 upper_lim: ArrayLike = None) -> None:
        """
        Construct a 'PolynomialMutator' object with a given probability value.

        :param mutate_probability: (float).

        :param eta_pm: (float) the distribution index for polynomial mutation.
                        Higher values mean smaller perturbations (more local
                        search).

        :param lower_lim: (ArrayLike) lower limit values for the genes.

        :param upper_lim: (ArrayLike) upper limit values for the genes.
        """
        # Call the super constructor with the provided initial value.
        super().__init__(mutation_probability=mutate_probability)

        # Validate the bounds.
        lower_lim, upper_lim = self.validate_bounds(lower_lim,
                                                    upper_lim)

        # Ensure eta_pm parameter is float.
        eta_pm = float(eta_pm)

        # Assign variables to the _items placeholder.
        self._items: tuple[float, ...] = (
            eta_pm, lower_lim, upper_lim
        )
    # _end_def_

    def mutate(self, individual: Chromosome) -> None:
        """
        Perform the mutation operation by adjusting the values
        of genes according to a polynomial distribution.

        :param individual: (Chromosome).

        :return: None.
        """
        # If the mutation probability is higher than
        # a uniformly random value, make the changes.
        if self.is_operator_applicable():

            # Get the size of the chromosome.
            n_genes: int = len(individual)

            # Extract the variables from the placeholder.
            eta_pm, x_lower, x_upper = self._items

            # Select a random position in the genome.
            idx: int = self.rng.integers(n_genes, dtype=int)

            # Copy the old value of the Gene.
            old_value = individual[idx].value

            # Local bounds lookups.
            xl: float = x_lower[idx]
            xu: float = x_upper[idx]

            # Compute the difference.
            bound_span: float = xu - xl

            # Normalize variable to [0, 1]
            # distance to bounds.
            delta1 = (old_value - xl) / bound_span
            delta2 = (xu - old_value) / bound_span

            # Generate a random number in [0, 1).
            rand_u: float = self.rng.random()

            # Mutation power value.
            m_power: float = 1.0 / (eta_pm + 1.0)

            # Calculate delta (the perturbation factor).
            if rand_u <= 0.5:

                base_ratio = 1.0 - delta1
                val = 2.0 * rand_u + (1.0 - 2.0 * rand_u) * (base_ratio ** (eta_pm + 1.0))
                delta_q = (val ** m_power) - 1.0
            else:

                base_ratio = 1.0 - delta2
                val = 2.0 * (1.0 - rand_u) + 2.0 * (rand_u - 0.5) * (base_ratio ** (eta_pm + 1.0))
                delta_q = 1.0 - (val ** m_power)
            # _end_if_

            # Update old gene value.
            new_value: float = old_value + delta_q * bound_span

            # Update the genome of the offspring with the
            # new value ensuring it stays within limits.
            individual[idx].value = clamp(new_value, xl, xu)

            # Set the fitness to None.
            individual.invalidate_fitness()

            # Increase the mutator counter.
            self.inc_counter()
    # _end_def_

# _end_class_
