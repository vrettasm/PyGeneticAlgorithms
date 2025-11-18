from pygenalgo.genome.gene import Gene
from pygenalgo.genome.chromosome import Chromosome
from pygenalgo.operators.mutation.mutate_operator import MutationOperator


class PolynomialMutator(MutationOperator):
    """
    Description:

        Polynomial mutator, mutates the chromosome by adjusting the values of
        traits according to a polynomial distribution. This results in a more
        controlled and smoother alteration of values.
    """

    def __init__(self, mutate_probability: float = 0.1, eta_pm: float = 50.0,
                 lower_val: float = None, upper_val: float = None):
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

        # Ensure lower_val parameter is float.
        lower_val = float(lower_val)

        # Ensure upper_val parameter is float.
        upper_val = float(upper_val)

        # Ensure the order is correct.
        if upper_val < lower_val:
            raise ValueError(f"{self.__class__.__name__}: "
                             f"The limit values are incorrect.")
        # _end_if_

        # Assign to the _items placeholder.
        self._items = [eta_pm, lower_val, upper_val]
    # _end_def_

    def mutate(self, individual: Chromosome) -> None:
        """
        Perform the mutation operation by ...

        :param individual: (Chromosome).

        :return: None.
        """

        # If the mutation probability is higher than
        # a uniformly random value, make the changes.
        if self.is_operator_applicable():

            # Get the size of the chromosome.
            n_genes = len(individual)

            # Extract the values from the placeholder variable.
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

            # Calculate the new value ensuring it stays within limits.
            new_value = min(max(individual[i].value + delta * (xu - xl), xl), xu)

            # Update the genome of the offspring with the new Gene.
            individual[i] = Gene(datum=new_value, func=individual[i].func)

            # Set the fitness to NaN.
            individual.invalidate_fitness()

            # Increase the mutator counter.
            self.inc_counter()
        # _end_if_

    # _end_def_

# _end_class_
