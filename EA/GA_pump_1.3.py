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
    (1, 2), (2, 5), (5, 7), (5, 8),
    (7, 3), (8, 3), (3, 13), (3, 15),
    (13, 14), (14, 11), (15, 16), (16, 12),
    (11, 25), (12, 25),(25,17),(25,18),(17,26),(18,26),
    (26,6),(6,9),(6,10),(9,4),(10,4),
    (4,19),(4,20),(4,21),(4,22),
    (19,23),(20,23),(21,23),(22,23),
    (19, 24), (20, 24), (21, 24), (22, 24)
])  # Example task dependencies

tasks = list(G.nodes)  # Tasks are represented as a list of integers

# Step 2: Define the Genetic Algorithm minimizing sum of mean and max fatigue
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))  # Minimizing the objective
creator.create("Individual", list, fitness=creator.FitnessMin)

def generate_alternating_valid_sequence(graph):
    """
    Generate a valid task sequence that satisfies the DAG dependencies
    and alternates between human and robot assignments.
    """
    in_degree = {u: graph.in_degree(u) for u in graph.nodes()}
    zero_in_degree = [u for u in graph.nodes() if in_degree[u] == 0]
    sequence = []
    assignments = []
    current_assignment = random.randint(0, 1)  # Start with either human or robot

    while zero_in_degree:
        # Filter tasks that can be assigned to the current entity
        available_tasks = zero_in_degree[:]
        task_found = False
        for u in available_tasks:
            # Attempt to assign task u to the current entity
            sequence.append(u)
            assignments.append(current_assignment)
            zero_in_degree.remove(u)
            for v in graph.successors(u):
                in_degree[v] -= 1
                if in_degree[v] == 0:
                    zero_in_degree.append(v)
            task_found = True
            break  # Move to the next assignment
        if not task_found:
            # No task can be assigned; this should not happen if the graph is well-formed
            raise Exception("Cannot find a task to assign while maintaining alternation.")
        # Alternate the assignment for the next task
        current_assignment = 1 - current_assignment

    individual = [(task, assigned_to) for task, assigned_to in zip(sequence, assignments)]
    return individual

def is_valid_sequence(sequence, graph):
    """Check if the task sequence respects the dependencies defined in the graph."""
    completed_tasks = set()
    for task in sequence:
        for predecessor in graph.predecessors(task):
            if predecessor not in completed_tasks:
                return False
        completed_tasks.add(task)
    return True

def alternating_turn_constraint(individual):
    """Check if the tasks alternate between human (1) and robot (0)."""
    for i in range(1, len(individual)):
        if individual[i][1] == individual[i - 1][1]:
            return False  # Consecutive tasks assigned to the same entity
    return True

def init_individual():
    """Initialize an individual with a valid task sequence and alternating assignments."""
    individual = generate_alternating_valid_sequence(G)
    return creator.Individual(individual)

def eval_fatigue(individual):
    """
    Evaluate an individual solution.
    - `individual` is a list of tuples (task, assigned_to).
    - We calculate the sum of mean and max human muscle fatigue based on task assignments.
    """
    sequence = [task for task, _ in individual]

    # Check for sequence validity
    if not is_valid_sequence(sequence, G):
        return float('inf'),  # Invalid sequence

    # Check for alternating assignments
    if not alternating_turn_constraint(individual):
        return float('inf'),  # Violates alternation constraint

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
    """Mutate the sequence by swapping two tasks if the swap is valid and maintains alternation."""
    idx1, idx2 = sorted(random.sample(range(len(individual)), 2))
    new_sequence = individual[:]
    # Swap the tasks
    new_sequence[idx1], new_sequence[idx2] = new_sequence[idx2], new_sequence[idx1]

    # Check if the new sequence is valid
    sequence_tasks = [task for task, _ in new_sequence]
    if not is_valid_sequence(sequence_tasks, G):
        return individual,  # Return the original individual

    # Check if the assignments still alternate
    if not alternating_turn_constraint(new_sequence):
        return individual,  # Return the original individual

    individual[:] = new_sequence  # Apply the mutation
    return individual,

def mutate_assignment(individual):
    """Mutate the assignments by flipping the assignments while maintaining alternation."""
    # Select two positions to swap assignments
    idx = random.randrange(1, len(individual) - 1)

    # Flip the assignment at idx if it maintains alternation
    original_assignment = individual[idx][1]
    new_assignment = 1 - original_assignment

    # Check if flipping maintains alternation
    if individual[idx - 1][1] != new_assignment and individual[idx + 1][1] != new_assignment:
        individual[idx] = (individual[idx][0], new_assignment)
    # Else, do nothing
    return individual,

def crossover(ind1, ind2):
    """Perform crossover between two individuals while maintaining alternation."""
    size = len(ind1)
    cxpoint = random.randint(1, size - 1)

    # Create offspring by combining parts from both parents
    new_ind1 = ind1[:cxpoint] + ind2[cxpoint:]
    new_ind2 = ind2[:cxpoint] + ind1[cxpoint:]

    # Check if sequences are valid and assignments alternate
    for ind in [new_ind1, new_ind2]:
        sequence_tasks = [task for task, _ in ind]
        if not is_valid_sequence(sequence_tasks, G) or not alternating_turn_constraint(ind):
            continue  # Skip invalid offspring
        else:
            ind1[:] = ind
            ind2[:] = ind
            break  # Valid offspring found

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
NGEN = 1000
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
