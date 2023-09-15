import numpy as np
import random
np.random.seed(5)

adjacency_mat = np.asarray(
    #Remember that we use the encoding above, i.e. 0 refers to Amsterdam and 10 to Paris!
    [
        [0, 3082, 649, 209, 904, 1180, 2300, 494, 1782, 515], # Distance Amsterdam to the other cities
        [3082, 0, 2552, 3021, 3414, 3768, 4578, 3099, 3940, 3140], # Distance Athens to the other cities
        [649, 2552, 0, 782, 743, 1727, 3165, 1059, 2527, 1094], # Distance Berlin to the other cities
        [209, 3021, 782, 0, 1035, 996, 2080, 328, 1562, 294], # Distance Brussels to the other cities
        [904, 3414, 743, 1035, 0, 1864, 3115, 1196, 2597, 1329], # Distance Copenhagen to the other cities
        [1180, 3768, 1727, 996, 1864, 0, 2879, 656, 2372, 1082], # Distance Edinburgh to the other cities
        [2300, 4578, 3165, 2080, 3115, 2879, 0, 2210, 638, 1786], # Distance Lisbon to the other cities
        [494, 3099, 1059, 328, 1196, 656, 2210, 0, 1704, 414], # Distance London to the other cities
        [1782, 3940, 2527, 1562, 2597, 2372, 638, 1704, 0, 1268], # Distance Madrid to the other cities
        [515, 3140, 1094, 294, 1329, 1082, 1786, 414, 1268, 0] # Distance Paris to the other cities
    ]

)

(adjacency_mat==adjacency_mat.T).all() # Check if the matrix is symmetric


def compute_distance(route: list, adjacency_mat: np.ndarray) -> int:
    '''
    Calculates the total number of kilometers for a given route.
    '''
    total_distance = 0

    for i in range(len(route) - 1):
        total_distance += adjacency_mat[route[i], route[i + 1]]

    return total_distance


def fittest_solution_TSP(fitness_function: callable, generation, adjancency_mat) -> tuple:
    '''
    This function calculates the fitness values of all individuals of a generation.
    It then returns the best fitness value and the corresponding individual.
    '''

    for individual in generation:
        fitness_values = []
        for i in range(len(generation)):
            fitness_values.append(fitness_function(individual, adjancency_mat))

    fittest = np.argmin(fitness_values)

    return fitness_values[fittest], generation[fittest]


def initialize_population(n_population: int, city_list: list, start_city: int = None,
                          fixed_start=True, round_trip=True) -> list:
    '''This returns a randomly initialized list of individual solutions of size n_population.'''

    population = []
    city_list_adj = city_list.copy()

    if fixed_start:
        city_list_adj.remove(start_city)
        for _ in range(n_population):
            individual = random.sample(city_list_adj, len(city_list_adj))
            # Add the start city to the beginning
            individual = [start_city] + individual

            if round_trip:
                # Given the round trip we need to add the start city to the end
                individual = individual + [start_city]

            population.append(individual)
    else:
        for _ in range(n_population):
            population.append(random.sample(city_list, len(city_list)))

    return population


def inversion_mutation(x: list, p_mutation: float, fixed_start=True, fixed_end=True) -> list:
    '''This applies the inverse mutation operator to a list and returns the mutated list.'''

    if np.random.uniform() > p_mutation:
        return x

    else:
        index_list = np.arange(0, len(x)).tolist()  # create a list of index to sample from

        if fixed_start:
            index_list = index_list[1:]  ##Remove the first index 0 from the list
            # index_list.remove(0) #Remove the first index 0 from the list

        if fixed_end:
            index_list = index_list[:-1]  # Remove the last index from the list
            # index_list.remove(len(x)-1) #Remove the last index from the list

        # Sample two integers from the index list

        a, b = random.sample(index_list, 2)

        # Sort them to make it clear which is the lower splitting point and which one is upper
        if a > b:
            lower = b
            upper = a
        else:
            lower = a
            upper = b

        # Pick the part of the list that will be inversed

        # Increase the upper pointer by 1 as python does not include the upper limit

        upper = upper + 1
        selected_slice = x[lower:upper]

        # Inverse the selected slice
        inversed_slice = [i for i in reversed(selected_slice)]

        # Create the mutated individual
        x_mutated = x[:lower] + inversed_slice + x[upper:]

        # Implement some assertion tests for checking if the mutation goes as expected
        assert (x_mutated[0] == x[0] & x_mutated[-1] == x[-1]), 'First start route and last route do not match up'
        assert (np.sum([i == 0 for i in x_mutated]) == 2), 'The start and end city does not match up!'
        assert len(x_mutated) == len(x), 'The length of the chromosomes differ'

    return x_mutated


def PMX_algorithm(parent_1, parent_2, lower, upper) -> list:
    '''
    This function applies the PMX algorith as discussed in the textbook by Eiben and Smith (2015).
    It returns a list which corresponds to a solution that has undergone the crossover operation.
    '''

    # Initialize child_1, -1 as marker for elements that have not been filled in yet
    child = np.repeat(-1, len(parent_1)).tolist()

    # Now implement the algorithm
    child[lower:upper] = parent_1[lower:upper]

    # print(f'this is the step 1 child {child}')
    for index, element in enumerate(parent_2[lower:upper]):
        if element not in parent_1[lower:upper]:
            element_to_be_replaced = parent_1[lower:upper][index]
            while element_to_be_replaced in parent_2[lower:upper]:
                new_index = parent_2.index(element_to_be_replaced)
                element_to_be_replaced = parent_1[new_index]
            index_to_fill_new_element = parent_2.index(element_to_be_replaced)
            child[index_to_fill_new_element] = element

    # Now fill the elements that have not been filled:
    for index, element in enumerate(child):
        if element == -1:
            child[index] = parent_2[index]
    return child


def partially_mapped_crossover(parent_1: list, parent_2: list, p_crossover: float,
                               fixed_start=True, fixed_end=True) -> tuple:
    '''
    This function applies the PMX operation on two parents, p1 and p2 respectively and returns two children.
    '''

    if np.random.uniform() > p_crossover:
        # Do not perform crossover
        return parent_1, parent_2

    else:
        lower = np.random.randint(1, len(parent_1) - 1)
        upper = np.random.randint(lower + 1, len(parent_1))

        child_1 = PMX_algorithm(parent_1, parent_2, lower, upper)
        child_2 = PMX_algorithm(parent_2, parent_1, lower, upper)

        return child_1, child_2


def tournament_selection_TSP(generation: list,
                             fitness_function: callable, adjacency_mat: np.ndarray, k: int) -> int:
    '''
    Implements the tournament selection algorithm.
    It draws randomly with replacement k individuals and returns the index of the fittest individual.
    '''

    # First step: Choose a random individual and score it
    current_winner = np.random.randint(0, len(generation))
    current_winner_score = fitness_function(generation[current_winner], adjacency_mat)

    # Get the score which is the one to beat!
    for candidate in range(k - 1):  # We already have one candidate, so we are left with k-1 to choose
        random_individual = np.random.randint(0, len(generation))
        random_individual_score = fitness_function(generation[random_individual], adjacency_mat)
        if random_individual_score < current_winner_score:
            current_winner = random_individual

    return current_winner


# Now we can re-run the experiment from above, this time using tournament selection:

# Define the hyperparameters,
# following the recommendations presented in the textbook
# Eiben, A.E., Smith, J.E., Introduction to Evolutionary Computing., Springer, 2015, 2nd edition, page 100

# Define population size
n_population = 10

# Define mutation rate
p_mutation = 0.10

# Crossover probability
p_crossover = 0.6

# Number of iterations
n_iter = 500

# Set the seed for reproducibility

np.random.seed(5)

# Tournament size
k = 3

# City list, see the index from above
# 0: Amsterdam, 1: Athens, 2: Berlin, 3: Brussels,
# 4: Copenhagen, 5: Edinburgh, 6: Lisbon, 7: London, 8: Madrid, 9: Paris

city_list = np.arange(0, 10).tolist()

# Adjacency mat
adjacency_mat = np.asarray(
    # Remember that we use the encoding above, i.e. 1 refers to Amsterdam and 10 to Paris!
    [
        [0, 3082, 649, 209, 904, 1180, 2300, 494, 1782, 515],  # Distance Amsterdam to the other cities
        [3082, 0, 2552, 3021, 3414, 3768, 4578, 3099, 3940, 3140],  # Distance Athens to the other cities
        [649, 2552, 0, 782, 743, 1727, 3165, 1059, 2527, 1094],  # Distance Berlin to the other cities
        [209, 3021, 782, 0, 1035, 996, 2080, 328, 1562, 294],  # Distance Brussels to the other cities
        [904, 3414, 743, 1035, 0, 1864, 3115, 1196, 2597, 1329],  # Distance Copenhagen to the other cities
        [1180, 3768, 1727, 996, 1864, 0, 2879, 656, 2372, 1082],  # Distance Edinburgh to the other cities
        [2300, 4578, 3165, 2080, 3115, 2879, 0, 2210, 638, 1786],  # Distance Lisbon to the other cities
        [494, 3099, 1059, 328, 1196, 656, 2210, 0, 1704, 414],  # Distance London to the other cities
        [1782, 3940, 2527, 1562, 2597, 2372, 638, 1704, 0, 1268],  # Distance Madrid to the other cities
        [515, 3140, 1094, 294, 1329, 1082, 1786, 414, 1268, 0]  # Distance Paris to the other cities
    ]

)

# Initialize the number of children
number_of_children = 2

# Initiliaze the generation
generation = initialize_population(n_population, city_list, start_city=0)

# Compute the current best fitness
best = fittest_solution_TSP(compute_distance, generation, adjacency_mat)
print('The current best solution in the initial generation is {0} km and the route is {1}'.format(best[0], best[1]))

for i in range(1, n_iter + 1):

    # Initialize the list of new generation
    new_generation = []

    # We loop over the number of parent pairs we need to get
    for j in range(int(n_population / number_of_children)):

        mating_pool = []
        for child in range(number_of_children):
            mate = tournament_selection_TSP(generation, compute_distance, adjacency_mat, k)
            mating_pool.append(mate)

        # Cross-over

        child_1, child_2 = partially_mapped_crossover(generation[mating_pool[0]], generation[mating_pool[1]],
                                                      p_crossover, fixed_start=True, fixed_end=True)

        # Mutation

        child_1 = inversion_mutation(child_1, p_mutation, fixed_start=True, fixed_end=True)
        child_2 = inversion_mutation(child_2, p_mutation, fixed_start=True, fixed_end=True)

        # Survival selection is here generational, hence all children replace their parents

        new_generation.append(child_1)
        new_generation.append(child_2)

    generation = new_generation
    # Calculate the best solution and replace the current_best

    best_generation = fittest_solution_TSP(compute_distance, generation, adjacency_mat)

    if best_generation[0] < best[0]:
        best = best_generation

    if i % 25 == 0:
        print(
            'The current best population in generation {0} is {1} km and the route is {2}'.format(i, best[0], best[1]))

print('\n-----Final tour:----\n')
# Print out the result:
Decoding = {0: 'Ams',
            1: 'Athens',
            2: 'Berlin',
            3: 'Brussels',
            4: 'Copenhagen',
            5: 'Edinburg',
            6: 'Lisbon',
            7: 'London',
            8: 'Madrid',
            9: 'Paris'}

for index, city in enumerate(best[1]):
    if city == 0:
        if index == 0:
            print(f'You should start in {Decoding[0]}')
        elif index == 10:
            print(f'You should end in {Decoding[0]}')

    else:
        print(f'Then you should go to {Decoding[city]}')
