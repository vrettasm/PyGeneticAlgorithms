# PyGenAlgo:  A simple and powerful toolkit for genetic algorithms.

"Genetic Algorithms [(GA)](https://en.wikipedia.org/wiki/Genetic_algorithm), are meta heuristic algorithms
inspired by the process of natural selection and belong to a larger class of evolutionary algorithms (EA)."

-- (From Wikipedia, the free encyclopedia)

This repository implements a genetic algorithm (GA) in Python3 programming language, using only **Numpy** and **Joblib**
as additional libraries. The basic approach offers a "StandardGA" class, where the whole population of chromosomes is
replaced by a new one at the end of each iteration (or epoch). More recently, the new "IslandModelGA" class was added
that offers a new genetic operator (MigrationOperator), that allows for periodic migration of the best individuals,
among the different island populations.
  
**NOTE**:
For computationally expensive fitness functions the StandardGA class provides the option of parallel evaluation
(of the individual chromosomes), by setting in the method run(..., parallel=True). However, for fast fitness
functions this will actually cause the algorithm to execute slower (due to the time required to open and close the
parallel pool). So the default setting here is "parallel=False". Regarding the IslandModelGA, this is running in
parallel mode by definition.

  > **NEWS**:
  > Several new genetic operators have been added, such as: **PositionBasedCrossover(POS)**,
  > **PartiallyMappedCrossover (PMX)** and **OrderCrossover (OX1)**. These operators were added to address combinatorial
  > problems where the genome can become invalid by the application of the other standard operators. Additionally, the
  > **Boltzmann Selector** has been implemented where the individuals, that will form the new population, are selected
  > using a temperature controlled Boltzmann distribution.
  > 

The current implementation offers a variety of genetic operators including:

- **Selection operators**:
  - [Linear Rank Selector](pygenalgo/operators/selection/linear_rank_selector.py)
  - [Random Selector](pygenalgo/operators/selection/random_selector.py)
  - [Roulette Wheel Selector](pygenalgo/operators/selection/roulette_wheel_selector.py)
  - [Stochastic Universal Selector](pygenalgo/operators/selection/stochastic_universal_selector.py)
  - [Tournament Selector](pygenalgo/operators/selection/tournament_selector.py)
  - [Truncation Selector](pygenalgo/operators/selection/truncation_selector.py)
  - [Boltzmann Selector](pygenalgo/operators/selection/boltzmann_selector.py)

- **Crossover operators**:
  - [Single-Point Crossover](pygenalgo/operators/crossover/single_point_crossover.py)
  - [Multi-Point Crossover](pygenalgo/operators/crossover/mutli_point_crossover.py)
  - [Uniform Crossover](pygenalgo/operators/crossover/uniform_crossover.py)
  - [Order Crossover (OX1)](pygenalgo/operators/crossover/order_crossover.py)
  - [Partially Mapped Crossover (PMX)](pygenalgo/operators/crossover/partially_mapped_crossover.py)
  - [Position Based Crossover (POS)](pygenalgo/operators/crossover/position_based_crossover.py)

- **Mutation operators**:
  - [Random Mutator](pygenalgo/operators/mutation/random_mutator.py)
  - [Shuffle Mutator](pygenalgo/operators/mutation/shuffle_mutator.py)
  - [Inverse Mutator](pygenalgo/operators/mutation/inverse_mutator.py)
  - [Gaussian Mutator](pygenalgo/operators/mutation/gaussian_mutator.py)
  - [Swap Mutator](pygenalgo/operators/mutation/swap_mutator.py)
  - [Flip Mutator](pygenalgo/operators/mutation/flip_mutator.py)

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
[zip](https://github.com/vrettasm/PyGeneticAlgorithms/archive/refs/heads/master.zip) format. This option does not
require a prior installation of git on the computer.

Alternatively one can clone the project directly using git as follows:

    git clone https://github.com/vrettasm/PyGeneticAlgorithms.git

### Required packages

The recommended version is Python 3.10 (and above). To simplify the required packages just use:

    pip install -r requirements.txt

### Fitness function

The most important thing the user has to do is to define the "fitness function". A template is provided here,
in addition to the examples below.

```python
from pygenalgo.genome.chromosome import Chromosome

# Fitness function <template>.
def fitness_func(individual: Chromosome, f_min: bool = False):
    """
    This is how a fitness function should look like. The whole
    evaluation should be implemented (or wrapped around) this
    function.
    
    :param individual: Individual chromosome to be evaluated.
    
    :param f_min: Bool flag indicating whether we are dealing
    with a minimization or maximization problem.
    """
    
    # CODE TO IMPLEMENT.
    
    # Assign the estimated value.
    f_val = ...
    
    # If we want minimization we return the negative.
    return -f_val if f_min else f_val
# _end_def_
```
Once the fitness function is defined correctly the next steps are straightforward as described in the examples.

### Examples

Some optimization examples on how to use these algorithms:

| **Problem**                                                 | **Variables** | **Objectives** | **Constraints** | **Description** |
|:------------------------------------------------------------|:-------------:|:--------------:|:---------------:|:---------------:|
| [Sphere](examples/sphere.ipynb)                             |   M (=5)      |     1          |       no        |     serial      |
| [Rastrigin](examples/rastrigin.ipynb)                       |    M (=5)     |       1        |       no        |     serial      |
| [Rosenbrock](examples/rosenbrock_on_a_disk.ipynb)           |    M (=2)     |       1        |        1        |     serial      |
| [Binh & Korn](examples/binh_and_korn_multiobjective.ipynb)  |    M (=2)     |       2        |        2        |     serial      |
| [Sphere](examples/sphere_in_parallel.ipynb)                 |    M (=10)    |       1        |       no        |    parallel     |
| [Easom](examples/easom_in_parallel.ipynb)                   |    M (=2)     |       1        |       no        |    parallel     |
| [Traveling Salesman Problem](examples/tsp.ipynb)            |    M (=10)    |       1        |       yes       |     serial      |
| [N-Queens puzzle](examples/queens_puzzle.ipynb)             |    M (=8)     |       1        |       yes       |    parallel     |
| [OneMax](examples/one_max.ipynb)                            |    M (=50)    |       1        |       no        |     serial      |
| [Tanaka](examples/tanaka_multiobjective.ipynb)              |    M (=2)     |       2        |        2        |     serial      |
| [Zakharov](examples/zakharov.ipynb)                         |    M (=8)     |       1        |       no        |     serial      |
| [Osyczka](examples/osyczka_kundu_multiobjective.ipynb)      |       6       |       2        |        6        |    parallel     |

Constraint optimization problems can be easily addressed using the
[Penalty Method](https://en.wikipedia.org/wiki/Penalty_method). Moreover, multi-objective optimizations (with or without
constraints) can also be solved, using the _weighted sum method_, as shown in the examples above.

### Contact

For any questions/comments (**regarding this code**) please contact me at: vrettasm@gmail.com
