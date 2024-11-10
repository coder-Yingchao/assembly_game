import random
import pandas as pd
from deap import base, creator, tools, algorithms
import networkx as nx

# Load muscle fatigue parameters from Excel (n tasks x m muscles)
# Assuming the Excel file contains a table with n tasks and m=20 muscle fatigue columns for each task
fatigue_data = pd.read_excel('../assets/random_numbers.xlsx')  # Load the entire n x m table

# Step 1: Define the task dependencies using a directed graph (DAG)
G = nx.DiGraph()
G.add_edges_from([
    (1, 2), (1, 3), (1, 4), (1, 5),
    (2, 6), (3, 6), (4, 6), (5, 6),
    (2, 7), (3, 7), (4, 7), (5, 7),
    (6, 8), (7, 8)
])  # Example task dependencies

tasks = list(G.nodes)  # Tasks are represented as a list of integers

# Step 2: Define the Genetic Algorithm minimizing sum of mean and max fatigue
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))  # Minimizing the objective
creator.create("Individual", list, fitness=creator.FitnessMin)

def generate_random_valid_sequence(graph):
    """Generate a random valid task sequence that satisfies the DAG dependencies."""
    in_degree = {u: graph.in_degree(u) for u in graph.nodes()}
    zero_in_degree = [u for u in graph.nodes() if in_degree[u] == 0]
    sequence = []
    while zero_in_degree:
        u = random.choice(zero_in_degree)
        sequence.append(u)
        zero_in_degree.remove(u)
        for v in graph.successors(u):
            in_degree[v] -= 1
            if in_degree[v] == 0:
                zero_in_degree.append(v)
    return sequence

def is_valid_sequence(sequence, graph):
    """Check if the task sequence respects the dependencies defined in the graph."""
    completed_tasks = set()
    for task in sequence:
        for predecessor in graph.predecessors(task):
            if predecessor not in completed_tasks:
                return False
        completed_tasks.add(task)
    return True

def init_individual():
    """Initialize an individual with a valid task sequence and random assignments."""
    sequence = generate_random_valid_sequence(G)
    assignments = [random.randint(0, 1) for _ in range(len(sequence))]
    individual = [(task, assigned_to) for task, assigned_to in zip(sequence, assignments)]
    return creator.Individual(individual)

def eval_fatigue(individual):
    """
    Evaluate an individual solution.
    - `individual` is a list of tuples (task, assigned_to).
    - We calculate the sum of mean and max human muscle fatigue based on task assignments.
    """
    sequence = [task for task, _ in individual]

    # Penalize invalid sequences
    if not is_valid_sequence(sequence, G):
        return float('inf'),  # Invalid sequence, return a large penalty

    muscle_fatigue_list = []  # Store the individual muscle fatigue for calculating max later

    for task, assigned_to in individual:
        if assigned_to == 1:  # Human does the task
            task_fatigue_vector = fatigue_data.iloc[task - 1].values  # Adjusted index (assuming tasks start from 1)
            muscle_fatigue_list.append(task_fatigue_vector)  # Store the vector for later use

    # Accumulate all fatigue vectors
    total_fatigue_vector = [0] * 20  # Initialize a vector of length 20

    for fatigue_vector in muscle_fatigue_list:
        total_fatigue_vector = [total_fatigue_vector[i] + fatigue_vector[i] for i in range(20)]

    # Calculate mean fatigue across all muscles and tasks assigned to the human
    if muscle_fatigue_list:
        mean_fatigue = sum(sum(fatigue_vector) for fatigue_vector in muscle_fatigue_list) / (len(muscle_fatigue_list))
        # Calculate the maximum fatigue experienced by any muscle across all tasks
        max_fatigue = max(total_fatigue_vector)
    else:
        mean_fatigue, max_fatigue = 0, 0  # No human tasks assigned

    return mean_fatigue + max_fatigue,  # Return the sum of mean and max fatigue

def mutate_sequence(individual):
    """Mutate the sequence by swapping two tasks if the swap is valid."""
    sequence = [task for task, _ in individual]
    idx1, idx2 = random.sample(range(len(sequence)), 2)
    new_sequence = sequence[:]
    new_sequence[idx1], new_sequence[idx2] = new_sequence[idx2], new_sequence[idx1]
    if is_valid_sequence(new_sequence, G):
        # Apply the swap
        assignments = [assigned_to for _, assigned_to in individual]
        individual[:] = [(task, assigned_to) for task, assigned_to in zip(new_sequence, assignments)]
    return individual,

def mutate_assignment(individual):
    """Mutate the assignments by flipping the assigned_to value for a random task."""
    idx = random.randrange(len(individual))
    task, assigned_to = individual[idx]
    individual[idx] = (task, 1 - assigned_to)
    return individual,

def crossover(ind1, ind2):
    """Perform crossover between two individuals."""
    # For simplicity, take the sequence from one parent and assignments from both
    seq1 = [task for task, _ in ind1]
    seq2 = [task for task, _ in ind2]
    # Randomly choose a parent's sequence
    if random.random() < 0.5:
        new_sequence = seq1[:]
    else:
        new_sequence = seq2[:]
    # Mix assignments from both parents
    assignments1 = [assigned_to for _, assigned_to in ind1]
    assignments2 = [assigned_to for _, assigned_to in ind2]
    new_assignments = []
    for a1, a2 in zip(assignments1, assignments2):
        if random.random() < 0.5:
            new_assignments.append(a1)
        else:
            new_assignments.append(a2)
    ind1[:] = [(task, assigned_to) for task, assigned_to in zip(new_sequence, new_assignments)]
    ind2[:] = [(task, assigned_to) for task, assigned_to in zip(new_sequence, new_assignments)]
    return ind1, ind2

# Step 3: Create GA toolbox
toolbox = base.Toolbox()
toolbox.register("individual", init_individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Register genetic operators
toolbox.register("mate", crossover)
toolbox.register("mutate_sequence", mutate_sequence)
toolbox.register("mutate_assignment", mutate_assignment)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", eval_fatigue)

# Step 4: Initialize the population
population = toolbox.population(n=300)

# Step 5: Run the genetic algorithm
NGEN = 500
CXPB = 0.7  # Crossover probability
MUTPB = 0.2  # Mutation probability

for gen in range(NGEN):
    # Select the next generation individuals
    offspring = toolbox.select(population, len(population))
    offspring = list(map(toolbox.clone, offspring))

    # Apply crossover
    for child1, child2 in zip(offspring[::2], offspring[1::2]):
        if random.random() < CXPB:
            toolbox.mate(child1, child2)
            del child1.fitness.values
            del child2.fitness.values

    # Apply mutation
    for mutant in offspring:
        if random.random() < MUTPB:
            # Randomly decide which mutation to apply
            if random.random() < 0.5:
                toolbox.mutate_sequence(mutant)
            else:
                toolbox.mutate_assignment(mutant)
            del mutant.fitness.values

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
    fitnesses = map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    # Replace the population with the offspring
    population[:] = offspring

    # Optional: print the best fitness in each generation
    # best_ind = tools.selBest(population, 1)[0]
    # print(f"Generation {gen}: Best Fitness = {best_ind.fitness.values[0]}")

# Step 6: Get the best solution
best_individual = tools.selBest(population, 1)[0]
print("Best individual (task assignments):", best_individual)
print("Sum of mean and max fatigue:", eval_fatigue(best_individual)[0])
