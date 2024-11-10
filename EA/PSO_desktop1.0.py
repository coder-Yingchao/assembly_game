import random
import pandas as pd
from deap import base, creator, tools
import networkx as nx
import numpy as np

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

# Step 2: Define the PSO minimizing sum of mean and max fatigue
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))  # Minimizing the objective
creator.create("Particle", list, fitness=creator.FitnessMin, speed=list, best=None, best_fitness=None)

def generate_valid_sequence(graph):
    """Generate a valid task sequence that satisfies the DAG dependencies."""
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

def alternating_turn_constraint(assignments):
    """Check if the assignments alternate between human (1) and robot (0)."""
    for i in range(1, len(assignments)):
        if assignments[i] == assignments[i - 1]:
            return False  # Consecutive tasks assigned to the same entity
    return True

def init_particle():
    """Initialize a particle with a valid task sequence and assignments."""
    sequence = generate_valid_sequence(G)
    assignments = [random.randint(0, 1) for _ in sequence]

    # Ensure assignments alternate
    for i in range(1, len(assignments)):
        if assignments[i] == assignments[i - 1]:
            assignments[i] = 1 - assignments[i]

    particle = creator.Particle(sequence + assignments)
    particle.speed = [0] * len(particle)
    particle.best = particle[:]
    particle.best_fitness = None
    return particle

def eval_fatigue(individual):
    """
    Evaluate an individual solution.
    - `individual` is a list where first N elements are task sequence, next N are assignments.
    - We calculate the sum of mean and max human muscle fatigue based on task assignments.
    """
    N = len(individual) // 2
    sequence = individual[:N]
    assignments = individual[N:]

    # Penalize invalid sequences
    if not is_valid_sequence(sequence, G):
        return float('inf'),  # Invalid sequence

    # Penalize if assignments do not alternate
    if not alternating_turn_constraint(assignments):
        return float('inf'),  # Violates alternation constraint

    muscle_fatigue_list = []  # Store the individual muscle fatigue for calculating max later

    for task, assigned_to in zip(sequence, assignments):
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

def sigmoid(x):
    """Sigmoid function for mapping velocities to probabilities."""
    return 1 / (1 + np.exp(-x))

def update_particle(particle, best, phi1, phi2):
    """Update the particle's velocity and position."""
    u1 = np.random.rand(len(particle))
    u2 = np.random.rand(len(particle))

    v_u1 = phi1 * u1 * (np.array(particle.best) - np.array(particle))
    v_u2 = phi2 * u2 * (np.array(best) - np.array(particle))
    particle.speed = particle.speed + v_u1 + v_u2

    # Update position (apply velocity)
    N = len(particle) // 2  # Number of tasks
    for i in range(N):
        if random.random() < sigmoid(particle.speed[i]):
            # Swap tasks in the sequence part
            idx = int(abs(particle.speed[i])) % N
            particle[i], particle[idx] = particle[idx], particle[i]
    for i in range(N, len(particle)):
        if random.random() < sigmoid(particle.speed[i]):
            # Flip assignment bit
            particle[i] = 1 - particle[i]

# Step 3: Create PSO toolbox
toolbox = base.Toolbox()
toolbox.register("particle", init_particle)
toolbox.register("population", tools.initRepeat, list, toolbox.particle)
toolbox.register("update", update_particle, phi1=2.0, phi2=2.0)
toolbox.register("evaluate", eval_fatigue)

# Step 4: Initialize the swarm
population = toolbox.population(n=100)  # PSO typically uses smaller populations

# Step 5: Run the PSO algorithm
NGEN = 1000  # Number of generations
best = None

for gen in range(NGEN):
    for particle in population:
        # Evaluate fitness
        fitness = toolbox.evaluate(particle)
        particle.fitness.values = fitness

        # Update personal best
        if not particle.best_fitness or fitness[0] < particle.best_fitness[0]:
            particle.best = particle[:]
            particle.best_fitness = fitness

    # Update global best
    population.sort(key=lambda x: x.fitness.values[0])
    if not best or population[0].fitness.values[0] < best.fitness.values[0]:
        best = creator.Particle(population[0][:])
        best.fitness.values = population[0].fitness.values

    for particle in population:
        toolbox.update(particle, best)

    # Optional: print the best fitness in each generation
    if gen % 100 == 0:
        print(f"Generation {gen}: Best Fitness = {best.fitness.values[0]}")

# Step 6: Get the best solution
print("Best individual (task sequence and assignments):", best)
print("Sum of mean and max fatigue:", best.fitness.values[0])
