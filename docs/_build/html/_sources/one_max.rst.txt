OneMax
======

Description:

    - Optimization (max)
    - Single-objective
    - Constraints (no)

The general problem statement is given by:

We have a state vector :math:`\mathbf{x} \in [0, 1]^M` of bits, such as: :math:`\mathbf{x} = (0, 0, 1, 1, 0, ..., 1)`.

- The optimal solution is the one where each variable has the value of '1'.

    .. math::
        f(\mathbf{x}) = \sum_{i=1}^{M} x_i, \text{ with } x_i \in \{0, 1\}

- Global maximum is found at:

    .. math::
        f(1, 1, ..., 1) = M


Step 1: Import python libraries and PyGenAlgo classes
-----------------------------------------------------

.. code-block:: python

    import numpy as np
    from math import fsum

    # Import main classes.
    from pygenalgo.genome.gene import Gene
    from pygenalgo.genome.chromosome import Chromosome
    from pygenalgo.engines.standard_ga import StandardGA
    from pygenalgo.utils.utilities import cost_function

    # Import Selection Operator(s).
    from pygenalgo.operators.selection.linear_rank_selector import LinearRankSelector

    # Import Crossover Operator(s).
    from pygenalgo.operators.crossover.meta_crossover import MetaCrossover

    # Import Mutation Operator(s).
    from pygenalgo.operators.mutation.flip_mutator import FlipMutator

Step 2: Define the objective function
-------------------------------------

.. code-block:: python

    # OneMax function.
    @cost_function
    def fun_OneMax(individual: Chromosome):

        # Compute the function value.
        f_value = fsum(individual.values())

        # Condition for termination.
        solution_found = f_value == len(individual)

        # Return the solution tuple.
        return f_value, solution_found

Step 3: Set the GA parameters
-----------------------------

.. code-block:: python

    # Set a seed for reproducible initial population.
    SEED = 1821

    # Random number generator.
    rng = np.random.default_rng(SEED)

    # Random function: It is used only for compatibility.
    boundary_x = lambda: rng.integers(2)

    # Define the number of genes.
    M = 50

    # Define the number of chromosomes.
    N = 100

    # Draw random samples for the initial points.
    x_init = rng.integers(low=0, high=2, size=(N, M))

    # Initial population.
    population = [Chromosome([Gene(x_init[i, j], boundary_x)
                              for j in range(M)], np.nan, True)
                  for i in range(N)]

    # Create the StandardGA object that will carry on the optimization.
    test_GA = StandardGA(initial_pop=population,
                         fit_func=fun_OneMax,
                         select_op=LinearRankSelector(),
                         mutate_op=FlipMutator(),
                         crossx_op=MetaCrossover())

Step 4: Run the optimization
----------------------------

.. code-block:: python

    test_GA(epochs=200, elitism=True, f_tol=1.0e-6, verbose=False)

Step 5: Final output
--------------------

.. code-block:: python

    # Extract the optimal solution from the GA.
    optimal_solution = test_GA.best_chromosome()

    # Extract the fitness value from the optimal solution.
    optimal_fit = optimal_solution.fitness

    # Display the (final) optimal value.
    print(f"Optimum Found: {optimal_fit:.6f}\n")

    # Display each gene value separately.
    for i, xi in enumerate(optimal_solution.values(), start=1):
        print(f"x{i} = {xi:>10.6f}")
