from pygenalgo.genome.chromosome import Chromosome
from pygenalgo.operators.genetic_operator import increase_counter
from pygenalgo.operators.selection.select_operator import SelectionOperator

from pygenalgo.operators.selection.random_selector import RandomSelector
from pygenalgo.operators.selection.tournament_selector import TournamentSelector
from pygenalgo.operators.selection.linear_rank_selector import LinearRankSelector
from pygenalgo.operators.selection.roulette_wheel_selector import RouletteWheelSelector


class MetaSelector(SelectionOperator):
    """
    Description:

        Meta-selector, selects the chromosomes for the new population by applying
        randomly the predefined selectors (one at a time), with equal probability.

        NOTE: In the future the equal probabilities can be amended.
    """

    def __init__(self, select_probability: float = 1.0) -> None:
        """
        Construct a 'MetaSelector' object with a predefined probability value.

        :param select_probability: (float).
        """
        # Call the super constructor with the provided probability value.
        super().__init__(select_probability)

        # NOTE: In here the selection probabilities for each operator are
        #       set to 1.0.
        self._items = (RandomSelector(1.0), LinearRankSelector(1.0),
                       RouletteWheelSelector(1.0), TournamentSelector(1.0))
    # _end_def_

    @increase_counter
    def select(self, population: list[Chromosome]) -> list[Chromosome]:
        """
        Select the individuals, from the input population, that will be
        passed on to the next genetic operations of crossover and mutation
        to form the new population of solutions.

        :param population: a list of chromosomes to select the parents from.

        :return: the selected parents population (as list of chromosomes).
        """
        # Get the number of available selectors.
        n_operators = len(self.items)

        # Select randomly a method, with equal probability
        # (but this can be changed).
        return self.items[self.rng.integers(n_operators,
                                            dtype=int)].select(population)
    # _end_def_

    @property
    def all_counters(self) -> dict:
        """
        Accessor (getter) of the application counter from all the internal selectors.
        This is mostly to verify that everything is working as expected.

        :return: a dictionary with the counter calls for all selector methods.
        """
        return {sel_op.__class__.__name__: sel_op.counter for sel_op in self.items}
    # _end_def_

    def reset_counter(self) -> None:
        """
        Sets ALL the counters to 'zero'. We have to override the super().reset_counter()
        method, because we have to call explicitly the reset_counter on all the internal
        operators.

        :return: None.
        """
        # First call the super() to reset the self internal counter.
        super().reset_counter()

        # Here call explicitly the reset on each of the internal mutators.
        for op in self.items:
            op.reset_counter()
    # _end_def_

# _end_class_
