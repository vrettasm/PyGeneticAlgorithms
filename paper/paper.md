---
title: 'PyGenAlgo: A simple toolkit for genetic algorithms.'
tags:
  - Python
  - genetic algorithms
  - optimization
  - evolutionary algorithms
  - island model
authors:
  - name: Michail D. Vrettas
    orcid: 0000-0002-5456-3226
    affiliation: "1"
affiliations:
 - name: Institute of High Performance Computing and Networking (ICAR-CNR), Italy
   index: 1
date: 30 July 2024
bibliography: paper.bib
---

# Summary
Genetic algorithms (GAs) [@Holland1975, @Mitchel1998], are metaheuristic algorithms that are used for solving constrained
and unconstrained optimization problems, mimicking the process of natural selection in biological evolution. Due to the
fact that GAs do not require the optimization function to be differentiable, they are suitable for application in cases
where the derivative of the objective function is either unavailable or impractical to obtain numerically.

Conceptually GAs belong to the bigger family of Evolutionary Computation (EC) algorithms. They maintain a population of
individual solutions (also called chromosomes), where each chromosome is characterized by its genome and its fitness
value. At each iteration in the evolution process the GA will evaluate the chromosomes' fitness, select the fitter
individuals and through a set of genetic operators, such crossover and mutation, will reproduce them and generate a new
population of solutions with end goal to maximize their fitness [@DeJong2006].

This paper proposes a general purpose genetic algorithm, implemented in Python3 programming language, having only
minimum dependencies in **NumPy** [@Harris2020] and **Joblib**, that handle some numerical and parallel execution
details. The code is freely available on GitHub at [@PyGenAlgo] and distributed under the GPL3.0 license.

# Statement of need

Since genetic algorithms are not a new concept in computer science a few implementations, in Python programming language,
have already been introduced such as: 1) DEAP [@DEAPFortin2012], 2) LEAP [@LEAPColetti2020], 3) EasyGA [@EasyGA],
4)PyGAD [@PyGAD2023] and 5) pymoo [@pymoo2020]. All these frameworks come with their merits and their disadvantages. Our
goal was not to propose a new framework that will outperform all these approaches, but rather provide a simpler alternative,
focused only on GAs, that for some users would be easier to understand and apply, without the steep learning curve that
some of the aforementioned implementations require, whilst offering the means to extend it and customize it to match the
requirements of a wide range of problem setups.

`PyGenAlgo` is a research toolkit for genetic algorithms (GA). Using a fully object-oriented paradigm it provides a set
of classes that can be used to solve optimization problems (constrained and unconstrained). Its clear and simple design,
makes it useful for someone that just want to use GAs as a black-box. At the same time, its object-oriented structure
allows for easy additions of numerous new genetic operators that some researcher would like to incorporate.

The current implementation provides a standard genetic algorithm model (class StandardGA), where the whole population of
individual solutions (i.e. chromosomes) is replaced by another one of offsprings, at the end of each epoch. Additionally,
the toolkit offers an IslandModelGA class, where a parallel island model [@Whitley1999, @Izzo2012] is implemented, with
or without the use of a migration operator. Some of the features that both models provide can be found in the table below:

|                    | **StandardGA**              | **IslandModelGA**           |
|:-------------------|:----------------------------|:----------------------------|
| No. of populations | 1                           | M                           | 
| Elitism            | yes                         | yes                         |
| Parallelism        | yes (in fitness evaluation) | yes (in parallel evolution) |
| Migration operator | no                          | yes                         |
| Self correction    | yes (at the gene level)     | yes (at the gene level)     |

_Elitism_ is an option that can be used in both computational models, and if set to True it will ensure that the best
individual in the population (in terms of higher fitness value), will always be present in the next generation, thus
avoid backdrops in the evolution. Despite the fact that `PyGenAlgo` is not made to be computationally fast, specific
care was taken to use the **StandardGA** in parallel mode, when the fitness function is computationally demanding.
That means the user has the option to evaluate the fitness of the chromosomes in parallel, to speed up the execution
process. In contrast, the **IslandModelGA** runs by definition only in parallel mode. Finally, the '_self correction_'
is a function that, if enabled, will check the genome of each chromosome right after the mutation operation and if some
gene is invalidated it will apply a correction mechanism to bring back the gene in a 'valid state'. This basically means
that if a gene gets an invalid value, this mechanism will replace it with valid one that is randomly chosen.

A variety of selection, crossover, mutation and migration operators are offered, as follows:

| **Selection**        | **Crossover** | **Mutation** | **Migration** |
|:---------------------|:--------------|:-------------|:--------------|
| Linear Rank          | Single-Point  | Random       | Clockwise     | 
| Random               | Multi-Point   | Shuffle      | -             |
| Roulette Wheel       | Order (OX1)   | Swap         | -             |
| Stochastic Universal | Uniform       | Super        | -             |
| Tournament           | Super         | -            | -             |
| Truncation           | -             | -            | -             |

Due to its flexible implementation it further allows the addition of new operators, after inheriting from the base
classes of each genetic operator, and implementing their interface:

- SelectionOperator
- CrossoverOperator
- MutationOperator
- MigrationOperator

This allows the user, that is interested in testing a new approach, to easily incorporate it in the current framework
and use it.

# Basic steps

A few examples on how to use both computational models (StandardGA and IslandModelGA), for optimization problems
(maximization or minimization, with and without constraints) are listed on the project's repository on GitHub. At
a higher level the steps that required to use `PyGenAlgo` can be listed as follows:

1. **Define the fitness function:** This is a "universal requirement" when using any optimization algorithm. In a GA
    framework the fitness function encodes the constraints about the environment, where the chromosomes are evolving,
    thus providing direct feedback about how fit they are.

2. **Define the Gene:** Unlike other software implementations of GAs that provide restricted types of gene encoding in
    our toolkit the Gene class, is designed to hold any data type. That can range from a simple boolean type (0, 1) to
    any complex data structure. The benefit of that is the additional flexibility to use any type of data when one is
    forming their optimization problem. In addition, since we provide this option of using any data type, the user also
    has to specify a function (that could be unique to each gene, if necessary), that will be responsible to provide
    random values to the gene. This method is called usually during the mutation operation and the purpose of that is to
    allow the gene to be mutated randomly, but enforcing its validity.

3. **Construct the initial population of chromosomes:** Once the genes have been formed, constructing the initial
    population is something easy and straightforward. A simple list comprehension is enough to generate all the
    chromosomes.

4. **Instantiate an 'engine' (StandardGA or IslandModelGA):** The selection of the 'engine' (or computational model)
    here is to determine whether we want a simple GA approach, such as the one provided by the StandardGA class, or
    a more advanced such as the one provided by the IslandModelGA. The advantage that provides the Island Model is that
    if our objective function has many local minima (or maxima), spanning over a large search space, we can easily break
    up the initial population in subpopulations where, each one will be initialized and evolve in isolation (with an
    optional migration policy), thus allowing for a better exploration of the space.

5. **Start the evolution process:** This is the last step and usually involves a simple call to the 'run()' method.
    Here we have to emphasize that even though all the genetic operators, as well as the different 'engines', have 
    default parameters we suggest the user to experiment with different values if required, since there is not going
    to be 'one optimal setting fits all'.

## Final notes

1. **On probabilities:**
    All the genetic operators have an internal probability, inherited by their base class, that is set upon instantiation
    and determines if the operator will be applied or not. Internally the genetic operators might apply additional
    probabilities according to the way they operate. For example, if the **RandomMutator** has probability of 0.1 and is
    applied on a chromosome with 100 genes, since each gene has equal probability to be selected, it will have a total
    probability of '0.001' to be mutated.

2. **On multi-objective optimization:**
    Addressing multi-objective optimization problems (with or without constraints), is feasible as long as the user
    forms the fitness function (including the constraints), in an appropriate way. `PyGenAlgo` is 'agnostic' of the
    type of problem is solving (single or multi-objective). All that is required is to form the genes correctly, and
    implement the fitness function. Once these steps are completed, the rest follow exactly the same evolution cycle
    as shown in the provided examples.

3. **On visualization:**
    In this toolkit, we decided not to provide explicit plot functions. The reason is that including plot functions would
    increase the dependencies of `PyGenAlgo` in other third-party libraries (e.g. matplotlib or seaborn). Therefore, we
    decided to provide easy access to the quantities of interest, such as statistics about the fitness evolution, the
    best chromosome of the population, etc.

# References
