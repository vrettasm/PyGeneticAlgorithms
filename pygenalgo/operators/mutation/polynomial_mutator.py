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

    def __init__(self, mutate_probability: float = 0.1, eta_pm: float = 20.0,
                 lower_val: float = None, upper_val: float = None) -> None:
        """
        Construct a 'PolynomialMutator' object with a given probability value.

        :param mutate_probability: (float).

        :param eta_pm: (float) the distribution index for polynomial mutation.
                        Higher values mean smaller perturbations (more local search).

        :param lower_val: (float) lower limit value for the gene mutation.

        :param upper_val: (float) upper limit value for the gene mutation.
        """
        # Call the super constructor with the provided
        # probability value.
        super().__init__(mutate_probability)

        # Ensure eta_pm parameter is float.
        eta_pm = float(eta_pm)

        # Ensure that both lower and upper limits are provided.
        if lower_val is None or upper_val is None:
            raise ValueError(f"{self.__class__.__name__}: "
                             f"Lower or Upper limits are missing.")
        # _end_if_

        # Ensure lower_val parameter is float.
        lower_val = float(lower_val)

        # Ensure upper_val parameter is float.
        upper_val = float(upper_val)

        # Ensure the order is correct.
        if upper_val <= lower_val:
            raise ValueError(f"{self.__class__.__name__}: "
                             f"The limit values are incorrect.")

        # Assign variables to the _items placeholder.
        self._items = [eta_pm, lower_val, upper_val]
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
            n_genes = len(individual)

            # Extract the variables from the placeholder _items.
            eta_pm, xl, xu = self._items

            # Generate a random number in [0, 1).
            rand_u = self.rng.random()

            # Calculate delta (the perturbation factor).
            if rand_u <= 0.5:
                delta = ((2.0 * rand_u) ** (1.0 / (eta_pm + 1.0))) - 1.0
            else:
                delta = 1.0 - ((2.0 * (1.0 - rand_u)) ** (1.0 / (eta_pm + 1.0)))
            # _end_if_

            # Select a random position in the genome.
            i = self.rng.integers(n_genes)

            # Get the old value of the Gene.
            old_value = individual[i].value

            # Update the genome of the offspring with the new value ensuring it
            # stays within limits.
            individual[i].value = clamp(old_value + delta * (xu - xl), xl, xu)

            # Set the fitness to NaN.
            individual.invalidate_fitness()

            # Increase the mutator counter.
            self.inc_counter()
    # _end_def_

# _end_class_
