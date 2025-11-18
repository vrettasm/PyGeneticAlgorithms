# PyGenAlgo:  A simple and powerful toolkit for genetic algorithms.

![Logo](./logo/pga_logo.png)

"Genetic Algorithms [(GA)](https://en.wikipedia.org/wiki/Genetic_algorithm), are meta heuristic algorithms
inspired by the process of natural selection and belong to a larger class of evolutionary algorithms (EA)."

-- (From Wikipedia, the free encyclopedia)

This repository implements a genetic algorithm (GA) in Python3 programming language, using only **Numpy** and **Joblib**
as additional libraries. The basic approach offers a "StandardGA" class, where the whole population of chromosomes is
replaced by a new one at the end of each iteration (or epoch). More recently, a new computational model was added named
"IslandModelGA" class that offers a new genetic operator (MigrationOperator), that allows for periodic migration of the
best individuals, among the (co-evolving) different island populations.
  
**NOTE**:
For computationally expensive fitness functions the StandardGA class provides the option of parallel evaluation
(of the individual chromosomes), by setting in the method run(..., parallel=True). However, for fast fitness
functions this will actually cause the algorithm to execute slower (due to the time required to open and close the
parallel pool). So the default setting here is "parallel=False". Regarding the IslandModelGA, this is running in
parallel mode by definition.

  > **NEWS**:
  > Recently three new features were added. The "Neighborhood Selector" operator is used for problems with multi-modal
  > objective functions. It allows the population to focus on multiple areas of the search space and detect more than
  > one optimal values. This features works in both computational modes (StandardGA and IslandModelGA). Moreover, the
  > Blend-α (BLX-α) crossover and Polynomial (PM-η) mutation operators were implemented with focus on real coded genomes.
  > 

The current implementation offers a variety of genetic operators including:

- **Selection operators**:
  - [Linear Rank Selector](pygenalgo/operators/selection/linear_rank_selector.py)
  - [Neighborhood Selector](pygenalgo/operators/selection/neighborhood_selector.py)
  - [Random Selector](pygenalgo/operators/selection/random_selector.py)
  - [Roulette Wheel Selector](pygenalgo/operators/selection/roulette_wheel_selector.py)
  - [Stochastic Universal Selector](pygenalgo/operators/selection/stochastic_universal_selector.py)
  - [Tournament Selector](pygenalgo/operators/selection/tournament_selector.py)
  - [Truncation Selector](pygenalgo/operators/selection/truncation_selector.py)
  - [Boltzmann Selector](pygenalgo/operators/selection/boltzmann_selector.py)

- **Crossover operators**:
  - [Single-Point Crossover](pygenalgo/operators/crossover/single_point_crossover.py)
  - [Multi-Point Crossover](pygenalgo/operators/crossover/multi_point_crossover.py)
  - [Uniform Crossover](pygenalgo/operators/crossover/uniform_crossover.py)
  - [Order Crossover (OX1)](pygenalgo/operators/crossover/order_crossover.py)
  - [Partially Mapped Crossover (PMX)](pygenalgo/operators/crossover/partially_mapped_crossover.py)
  - [Position Based Crossover (POS)](pygenalgo/operators/crossover/position_based_crossover.py)
  - [Blend-α Crossover (BLX-α)](pygenalgo/operators/crossover/blend_crossover.py)

- **Mutation operators**:
  - [Random Mutator](pygenalgo/operators/mutation/random_mutator.py)
  - [Shuffle Mutator](pygenalgo/operators/mutation/shuffle_mutator.py)
  - [Inverse Mutator](pygenalgo/operators/mutation/inverse_mutator.py)
  - [Gaussian Mutator](pygenalgo/operators/mutation/gaussian_mutator.py)
  - [Swap Mutator](pygenalgo/operators/mutation/swap_mutator.py)
  - [Flip Mutator](pygenalgo/operators/mutation/flip_mutator.py)
  - [Polynomial Mutator](pygenalgo/operators/mutation/polynomial_mutator.py)

- **Migration operators**
  - [Clockwise Migrator](pygenalgo/operators/migration/clockwise_migration.py)
  - [Random Migrator](pygenalgo/operators/migration/random_migration.py)

- **Meta operators**
  - [Meta Crossover](pygenalgo/operators/crossover/meta_crossover.py)
  - [Meta Mutator](pygenalgo/operators/mutation/meta_mutator.py)
  - [Meta Migration](pygenalgo/operators/migration/meta_migration.py)

(**NOTE:** Meta operators call randomly the other operators (crossover/mutation/migration) from a predefined set,
with equal probability.)

Incorporating additional genetic operators is easily facilitated by inheriting from the base classes:
- [SelectionOperator](pygenalgo/operators/selection/select_operator.py)
- [CrossoverOperator](pygenalgo/operators/crossover/crossover_operator.py)
- [MutationOperator](pygenalgo/operators/mutation/mutate_operator.py)
- [MigrationOperator](pygenalgo/operators/migration/migration_operator.py)

and implementing the basic interface as described therein. In the examples that follow I show how one can use this code
to run a GA for optimization problems (maximization/minimization) with and without constraints. The project is ongoing
so new things might come along the way.

### Installation

There are two options to install the software.

The easiest way is to visit the GitHub web-page of the project and simply download the source code in
[zip](https://github.com/vrettasm/PyGeneticAlgorithms/archive/refs/heads/master.zip) format. This option
does not require a prior installation of git on the computer.

Alternatively one can clone the project directly using git as follows:

    git clone https://github.com/vrettasm/PyGeneticAlgorithms.git

After the download of the code (or the git clone), one can use the command:

    pip install pygenalgo

This will install the PyGenAlgo in the package management system.

### Required packages

The recommended version is Python 3.10 (and above). To simplify the required packages just use:

    pip install -r requirements.txt

### Fitness function

The most important thing the user has to do is to define the "fitness function". A template is provided here,
in addition to the examples below. The cost_function decorator is used to indicate whether the function will
be maximized (default), or minimized. The second output parameter ("solution_found") is optional; only in the
cases where we can evaluate if a termination condition is satisfied.

```python
from pygenalgo.genome.chromosome import Chromosome
from pygenalgo.utils.utilities import cost_function


# Fitness function <template>.
@cost_function(minimize=True)
def fitness_func(individual: Chromosome):
    """
    This is how a fitness function should look like. The whole
    evaluation should be implemented (or wrapped around) this
    function.
    
    :param individual: Individual chromosome to be evaluated.
    
    :return: the function value evaluated at the individual.
    """

    # Extract gene values from the chromosome.
    x = individual.values()
    
    # ... CODE TO IMPLEMENT ...

    # Compute the function value.
    f_value = ...

    # Condition for termination.
    # We set it to True / False.
    solution_found = ...

    # Return the solution.
    return f_value, solution_found
# _end_def_
```
Once the fitness function is defined correctly the next steps are straightforward as described in the examples.

### Examples

Some optimization examples on how to use these algorithms:

| **Problem**                                                | **Variables** | **Objectives** | **Constraints** | **Optima** |
|:-----------------------------------------------------------|:-------------:|:--------------:|:---------------:|:----------:|
| [Sphere](examples/sphere.ipynb)                            |    M (=5)     |       1        |       no        |   single   |
| [Rastrigin](examples/rastrigin.ipynb)                      |    M (=5)     |       1        |       no        |   single   |
| [Rosenbrock](examples/rosenbrock_on_a_disk.ipynb)          |    M (=2)     |       1        |        1        |   single   |
| [Binh & Korn](examples/binh_and_korn_multiobjective.ipynb) |    M (=2)     |       2        |        2        |   single   |
| [Sphere (parallel)](examples/sphere_in_parallel.ipynb)     |    M (=10)    |       1        |       no        |   single   |
| [Easom (parallel)](examples/easom_in_parallel.ipynb)       |    M (=2)     |       1        |       no        |   single   |
| [Traveling Salesman](examples/tsp.ipynb)                   |    M (=10)    |       1        |       yes       |   single   |
| [N-Queens](examples/queens_puzzle.ipynb)                   |    M (=8)     |       1        |       yes       |   single   |
| [OneMax](examples/one_max.ipynb)                           |    M (=50)    |       1        |       no        |   single   |
| [Tanaka](examples/tanaka_multiobjective.ipynb)             |    M (=2)     |       2        |        2        |   single   |
| [Zakharov](examples/zakharov.ipynb)                        |    M (=8)     |       1        |       no        |   single   |
| [Osyczka](examples/osyczka_kundu_multiobjective.ipynb)     |       6       |       2        |        6        |   single   |
| [Shubert](examples/shubert_2D.ipynb)                       |       2       |       1        |       no        |  multiple  |
| [Gaussian Mixture](examples/gaussian_mixture_2D.ipynb)     |       2       |       1        |       no        |  multiple  |


Constraint optimization problems can be easily addressed using the [Penalty Method](https://en.wikipedia.org/wiki/Penalty_method).
Moreover, multi-objective optimizations (with or without constraints) can also be solved, using the _weighted sum method_,
as shown in the examples above. For multimodal optimizations check examples **Shubert** and **Gaussian Mixture**.

## References and Documentation

This work is described in:

- [Michail D. Vrettas and Stefano Silvestri (2025)](https://www.sciencedirect.com/science/article/pii/S2352711025000949)
"PyGenAlgo: a simple and powerful toolkit for genetic algorithms". SoftwareX, vol. 30. DOI: 10.1016/j.softx.2025.102127.

You can find the latest documentation [here](https://pygeneticalgorithms.readthedocs.io/en/latest/).

### Contact

For any questions/comments (**regarding this code**) please contact me at: vrettasm@gmail.com
