import random
import pandas as pd
from deap import base, creator, tools, algorithms
import networkx as nx


# Load muscle fatigue parameters from Excel (n tasks x m muscles)
# Assuming the Excel file contains a table with n tasks and m=20 muscle fatigue columns for each task
fatigue_data = pd.read_excel('../assets/random_numbers.xlsx')  # Load the entire n x m table

# Step 1: Define the task dependencies using a directed graph (DAG)
G = nx.DiGraph()
G.add_edges_from([(1, 2), (1, 3), (1, 4), (1, 5), (2, 6),(3, 6),(4, 6),(5, 6),(2, 7),(3, 7),(4, 7),(5, 7),(6,8),(7,8)])  # Example task dependencies
G.add_node(5)  # Independent task

tasks = list(G.nodes)  # Tasks are represented as a list of integers [1, 2, 3, 4, 5]

# Step 2: Define the Genetic Algorithm minimizing sum of mean and max fatigue
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))  # Minimizing the objective
creator.create("Individual", list, fitness=creator.FitnessMin)

def is_valid_sequence(sequence, graph):
    """Check if the task sequence respects the dependencies defined in the graph."""
    completed_tasks = set()
    for task in sequence:
        for predecessor in graph.predecessors(task):
            if predecessor not in completed_tasks:
                return False
        completed_tasks.add(task)
    return True

def generate_random_human_actions(task_count):
    """Randomly generate the human's task assignments in an alternating pattern."""
    actions = [1 if i % 2 == 0 else 0 for i in range(task_count)]
    random.shuffle(actions)  # Shuffle to randomize the human's actions
    return actions

def eval_fatigue(individual, human_actions):
    """
    Evaluate an individual solution.
    - `individual` is a list where each element represents task assignment: 1 for human, 0 for robot.
    - We calculate the sum of mean and max human muscle fatigue based on task assignments.
    """
    total_muscle_fatigue = [0] * 20  # Initialize the total fatigue for each of the 20 muscles
    muscle_fatigue_list = []  # Store the individual muscle fatigue for calculating max later

    # Iterate over the tasks and evaluate based on the human's random actions and robot's optimized actions
    for i, robot_assigned in enumerate(individual):
        task = tasks[i]  # The corresponding task based on the index
        if human_actions[i] == 1:  # Human does the task
            task_fatigue_vector = fatigue_data.iloc[task - 1].values  # Get the fatigue vector (20 muscles) for the task
            total_muscle_fatigue = [total_muscle_fatigue[j] + task_fatigue_vector[j] for j in range(20)]
            muscle_fatigue_list.append(task_fatigue_vector)  # Store the vector for later use

    # Calculate mean fatigue across all muscles and tasks assigned to the human
    if muscle_fatigue_list:
        mean_fatigue = sum(sum(fatigue_vector) for fatigue_vector in muscle_fatigue_list) / len(muscle_fatigue_list)
        # Calculate the maximum fatigue experienced by any muscle across all tasks
        max_fatigue = max(sum(fatigue_vector) for fatigue_vector in muscle_fatigue_list)
    else:
        mean_fatigue, max_fatigue = 0, 0  # No human tasks assigned

    return mean_fatigue + max_fatigue,  # Return the sum of mean and max fatigue

# Step 3: Create GA toolbox
toolbox = base.Toolbox()

# Initialize individuals with random robot task assignments (0 for robot)
def init_individual():
    # The GA optimizes only the robot's actions, human actions are randomized later
    return creator.Individual([random.randint(0, 1) for _ in range(len(tasks))])

toolbox.register("individual", init_individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Register genetic operators
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)  # Mutate the task assignment with 5% probability
toolbox.register("select", tools.selTournament, tournsize=3)  # Tournament selection
toolbox.register("evaluate", eval_fatigue)

# Step 4: Initialize the population
population = toolbox.population(n=300)

# Step 5: Run the genetic algorithm
for generation in range(500):  # Iterate over generations
    # For each generation, generate a new random human action sequence
    human_actions = generate_random_human_actions(len(tasks))

    # Evaluate each individual based on the current human actions
    for ind in population:
        ind.fitness.values = toolbox.evaluate(ind, human_actions)

    # Select the next generation of individuals
    offspring = toolbox.select(population, len(population))

    # Apply crossover and mutation to the offspring
    offspring = [toolbox.clone(ind) for ind in offspring]
    for child1, child2 in zip(offspring[::2], offspring[1::2]):
        if random.random() < 0.7:  # Crossover probability
            toolbox.mate(child1, child2)
            del child1.fitness.values
            del child2.fitness.values

    # Apply mutation
    for mutant in offspring:
        if random.random() < 0.2:  # Mutation probability
            toolbox.mutate(mutant)
            del mutant.fitness.values

    # Replace the old population with the new offspring
    population[:] = offspring

# Step 6: Get the best solution
best_individual = tools.selBest(population, 1)[0]
print("Best individual (robot task assignments):", best_individual)
print("Sum of mean and max fatigue:", eval_fatigue(best_individual, generate_random_human_actions(len(tasks)))[0])
