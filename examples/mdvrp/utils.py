import json
from collections import namedtuple

import numpy as np
from scipy.spatial import distance
from numpy.random import default_rng

from pygenalgo.genome.gene import Gene
from pygenalgo.genome.chromosome import Chromosome

# Data structure to represent a customer on a grid.
Customer = namedtuple("Customer",
                      ["ID", "x", "y", "demand"])

# Data structure to represent a depot on a grid.
Depot = namedtuple("Depot",
                   ["ID", "x", "y", "vehicles", "cap_per_vehicle"])

# Data structure to represent a cluster of customers around a depot.
Cluster = namedtuple("Cluster",
                     ["ID", "depot", "customers"])

# Function that loads the data.
def load_data(filepath: str) -> tuple[list[Customer], list[Depot]]:
    """
    Auxiliary function to load data from JSON files.

    :param filepath: the JSON data file.

    :return: the lists with the customers and the depots.
    """
    # Ensure the file exits.
    try:
        # Load the data from JSON file.
        with open(filepath, 'r') as data_file:
            data = json.load(data_file)
        # _end_with_

        # Create the customers list.
        customers = [Customer(c["ID"], c["X"], c["Y"], c["DEMAND"])
                     for c in data["customers"]]

        # Extract the depots list.
        depots = [Depot(d["ID"], d["X"], d["Y"], d["VEHICLES"], d["MAX_CAPACITY"])
                  for d in data["depots"]]
    except FileNotFoundError:
        # Raise an error message.
        raise RuntimeError(f"Cannot find file: {filepath}")
    # _end_try_

    # Return both lists.
    return customers, depots
# _end_def_

def cluster_customers_to_depots(depots: list[Depot],
                                customers: list[Customer],
                                verbose: bool = False) -> list[Cluster]:
    """
    Clusters the customers into their closer depots. Each cluster
    contains the depot (center) along with a list of customers.

    :param depots: a list of depots.

    :param customers: a list of customers.

    :param verbose: if True it will display the info from the clustering.
                    Default is False.

    :return: a list with all the clusters.
    """
    # Extract the depot coordinates (as centroids).
    centroids = np.array([(d.x, d.y) for d in depots])

    # Extract the customers coordinates.
    customer_data = np.array([(c.x, c.y) for c in customers])

    # Step 1: Compute the Euclidean distances.
    distances = distance.cdist(customer_data, centroids, "euclidean")

    # Step 2: Find the index of the closest points.
    closest_indices = np.argmin(distances, axis=1)

    # Initialize the clusters.
    clusters = [Cluster(ID=i, depot=dpt,
                        customers=[customers[n] for n, k in enumerate(closest_indices) if k == i])
                for i, dpt in enumerate(depots)]

    if verbose:
        # Display the information.
        for c_item in clusters:
            print(f"Depot-ID: {c_item.depot.ID:>2} "
                  f"has {len(c_item.customers):>3} customers")
    # _end_if_

    return clusters
# _end_def_

def initialize_population(customers: list[Customer], n_vehicles: int,
                          n_pop: int = 100, seed: int = None) -> list[Chromosome]:
    """
    Creates and returns an initial (random) population.

    :param customers: the list of customers.
    :param n_vehicles: number of vehicles in the depot.
    :param n_pop: the number of required population.
    :param seed: seed for the random number generator.
    :return: a list of chromosomes.
    """
    # Get a random number generator.
    local_rng = default_rng(seed)

    # Extract all the customer IDs.
    customer_ids = [cx.ID for cx in customers]

    # Create a list with "invalid" values.
    invalid_genes = [-(px + 1)
                     for px in range(len(customer_ids) * (n_vehicles - 1))]

    # Create an initial valid chromosome.
    valid_chromosome = customer_ids + invalid_genes

    # Population list.
    population = []

    # Initial population.
    for _ in range(n_pop):
        # Shuffle the order of the genes.
        local_rng.shuffle(valid_chromosome)

        # Create a Chromosome and add it to the population.
        population.append(Chromosome([Gene(value, lambda: None)
                                      for value in valid_chromosome], np.nan, True))
    # _end_for_

    return population
# _end_def_

def evaluate_solution(individual: Chromosome, assignment: Cluster,
                      verbose: bool = False) -> tuple[float, float]:
    """
    Evaluates a single chromosome solution. It estimates the total distance
    that all the vehicles have traveled and computes the penalty when the
    demand of the routes exceeds the that of the depot capacity.

    :param individual: the chromosome to be validated.

    :param assignment: the cluster (depot, customers).

    :param verbose: if True it will display the info from the clustering.

    :return: total distance traveled (float) along with the penalty term.
    """
    # Extract the depot and the customers list.
    depot = assignment.depot
    customers = assignment.customers

    # Number of depot vehicles.
    n_vehicles = depot.vehicles

    # Number of customers.
    n_customers = len(customers)

    # Reshape the solution array.
    x_solution = np.reshape(individual.values(),
                            (n_customers, n_vehicles))
    # Stores the distance.
    total_distance = 0.0

    # Stores the penalty.
    total_penalty = 0

    # Start computing the total cost.
    for i in range(n_vehicles):
        # Exclude the negative entries.
        customer_ids = x_solution[x_solution[:, i] >= 0, i]

        # If the array is emtpy
        # continue to the next column.
        if customer_ids.size == 0:
            continue

        # Initialize a new route with the depot.
        route = [depot]

        # Generate the route for this vehicle.
        for c_id in customer_ids.tolist():
            for cx in customers:
                if cx.ID == c_id:
                    route.append(cx)
        # _end_for_

        # Length of the i-th route.
        n_stops = len(route)

        # The distance of a single route.
        route_distance = 0.0

        # The demand of a single route.
        route_demand = 0

        # Iterate through the route.
        for j, start_point in enumerate(route):
            # Get the coordinates of the first point.
            xj, yj = start_point.x, start_point.y

            # The next index ('k') should point
            # at the beginning of the list once
            # we reach at the end.
            if j < n_stops - 1:
                k = j + 1
                demand_k = route[k].demand
            else:
                k, demand_k = 0, 0

            # Get the coordinates of the next stop.
            xk, yk = route[k].x, route[k].y

            # Compute the Euclidean distance.
            route_distance += distance.euclidean((xk, yk), (xj, yj))

            # Add the customer's demand to the route demand.
            route_demand += demand_k
        # _end_for_

        # Update total distance.
        total_distance += route_distance

        # Display info.
        if verbose:
            print(f"Depot-ID {depot.ID}: -> Route {i}, "
                  f"Distance {route_distance:.2f}, Demand {route_demand}")
        # _end_if_

        # Remove the depot max capacity from the route demand.
        diff_capacity = route_demand - depot.cap_per_vehicle

        # Check if the difference is positive.
        if diff_capacity > 0.0:
            # Add it to the total penalty.
            total_penalty += diff_capacity
    # _end_for_

    # Return the total distance and penalty.
    return total_distance, total_penalty
# _end_def_