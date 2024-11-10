import random
import pandas as pd
import numpy as np
from deap import base, creator, tools, algorithms
import networkx as nx

# Load muscle fatigue parameters from Excel (n tasks x m muscles)
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
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

def generate_alternating_valid_sequence(graph):
    in_degree = {u: graph.in_degree(u) for u in graph.nodes()}
    zero_in_degree = [u for u in graph.nodes() if in_degree[u] == 0]
    sequence = []
    assignments = []
    current_assignment = random.randint(0, 1)

    while zero_in_degree:
        available_tasks = zero_in_degree[:]
        task_found = False
        for u in available_tasks:
            sequence.append(u)
            assignments.append(current_assignment)
            zero_in_degree.remove(u)
            for v in graph.successors(u):
                in_degree[v] -= 1
                if in_degree[v] == 0:
                    zero_in_degree.append(v)
            task_found = True
            break
        current_assignment = 1 - current_assignment

    individual = [(task, assigned_to) for task, assigned_to in zip(sequence, assignments)]
    return individual

def is_valid_sequence(sequence, graph):
    completed_tasks = set()
    for task in sequence:
        for predecessor in graph.predecessors(task):
            if predecessor not in completed_tasks:
                return False
        completed_tasks.add(task)
    return True

def alternating_turn_constraint(individual):
    for i in range(1, len(individual)):
        if individual[i][1] == individual[i - 1][1]:
            return False
    return True

def init_individual():
    individual = generate_alternating_valid_sequence(G)
    return creator.Individual(individual)

def eval_fatigue(individual, human_skip_probability=0.7):
    sequence = [task for task, _ in individual]

    if not is_valid_sequence(sequence, G):
        return float('inf'),

    if not alternating_turn_constraint(individual):
        return float('inf'),

    muscle_fatigue_list = []

    for task, assigned_to in individual:
        if assigned_to == 1:
            if random.random() > human_skip_probability:  # Human executes task with certain probability
                task_fatigue_vector = fatigue_data.iloc[task - 1].values
                muscle_fatigue_list.append(task_fatigue_vector)

    total_fatigue_vector = [0] * 20

    for fatigue_vector in muscle_fatigue_list:
        total_fatigue_vector = [total_fatigue_vector[i] + fatigue_vector[i] for i in range(20)]

    if muscle_fatigue_list:
        mean_fatigue = sum(sum(fatigue_vector) for fatigue_vector in muscle_fatigue_list) / (len(muscle_fatigue_list))
        max_fatigue = max(total_fatigue_vector)
    else:
        mean_fatigue, max_fatigue = 0, 0

    return mean_fatigue + max_fatigue,

def mutate_sequence(individual):
    idx1, idx2 = sorted(random.sample(range(len(individual)), 2))
    new_sequence = individual[:]
    new_sequence[idx1], new_sequence[idx2] = new_sequence[idx2], new_sequence[idx1]
    sequence_tasks = [task for task, _ in new_sequence]
    if not is_valid_sequence(sequence_tasks, G):
        return individual,
    if not alternating_turn_constraint(new_sequence):
        return individual,

    individual[:] = new_sequence
    return individual,

def mutate_assignment(individual):
    idx = random.randrange(1, len(individual) - 1)
    original_assignment = individual[idx][1]
    new_assignment = 1 - original_assignment
    if individual[idx - 1][1] != new_assignment and individual[idx + 1][1] != new_assignment:
        individual[idx] = (individual[idx][0], new_assignment)
    return individual,

def crossover(ind1, ind2):
    size = len(ind1)
    cxpoint = random.randint(1, size - 1)
    new_ind1 = ind1[:cxpoint] + ind2[cxpoint:]
    new_ind2 = ind2[:cxpoint] + ind1[cxpoint:]
    for ind in [new_ind1, new_ind2]:
        sequence_tasks = [task for task, _ in ind]
        if not is_valid_sequence(sequence_tasks, G) or not alternating_turn_constraint(ind):
            continue
        else:
            ind1[:] = ind
            ind2[:] = ind
            break
    return ind1, ind2

toolbox = base.Toolbox()
toolbox.register("individual", init_individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("mate", crossover)
toolbox.register("mutate_sequence", mutate_sequence)
toolbox.register("mutate_assignment", mutate_assignment)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", eval_fatigue)

# Parameters for GA
NGEN = 500
CXPB = 0.7
MUTPB = 0.2
NUM_RUNS = 10  # Run the GA 10 times

# Store the best fitness results from each run
best_fitnesses = []

for run in range(NUM_RUNS):
    print(f"Run {run+1}/{NUM_RUNS}")
    population = toolbox.population(n=300)

    for gen in range(NGEN):
        offspring = toolbox.select(population, len(population))
        offspring = list(map(toolbox.clone, offspring))

        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CXPB:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < MUTPB:
                if random.random() < 0.5:
                    toolbox.mutate_sequence(mutant)
                else:
                    toolbox.mutate_assignment(mutant)
                del mutant.fitness.values

        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        population[:] = offspring

    # Get the best individual from the population
    best_individual = tools.selBest(population, 1)[0]
    best_fitness = eval_fatigue(best_individual)[0]
    best_fitnesses.append(best_fitness)
    print(f"Best fitness of this run: {best_fitness}")

# Calculate mean and standard deviation of the best fitnesses
mean_best_fitness = np.mean(best_fitnesses)
std_best_fitness = np.std(best_fitnesses)

print(f"\nMean of best fitnesses: {mean_best_fitness}")
print(f"Standard deviation of best fitnesses: {std_best_fitness}")
