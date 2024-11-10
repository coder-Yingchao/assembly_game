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
 # Independent task

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

def alternating_turn_constraint(individual):
    """Check if the tasks alternate between human (1) and robot (0)."""
    for i in range(1, len(individual)):
        if individual[i] == individual[i - 1]:
            return False  # Consecutive tasks assigned to the same entity (human or robot)
    return True

def eval_fatigue(individual):
    """
    Evaluate an individual solution.
    - `individual` is a list where each element represents task assignment: 1 for human, 0 for robot.
    - We calculate the sum of mean and max human muscle fatigue based on task assignments.
    """
    # Check if the individual respects the alternating turn constraint
    # if not alternating_turn_constraint(individual):
    #     return float('inf'),  # Penalize for violating the alternating turn constraint
    # Extract the sequence of tasks from the individual
    sequence = [task for task, assigned_to in individual]

    # Penalize invalid sequences
    if not is_valid_sequence(sequence, G):
        return float('inf'),  # Invalid sequence, return a large penalty


    # total_muscle_fatigue = [0] * 20  # Initialize the total fatigue for each of the 20 muscles
    muscle_fatigue_list = []  # Store the individual muscle fatigue for calculating max later

    for i, assigned_to in enumerate(individual):
        task = tasks[i]  # The corresponding task based on the index
        if assigned_to == 1:  # Human does the task
            task_fatigue_vector = fatigue_data.iloc[task].values  # Get the fatigue vector (20 muscles) for the task
            # total_muscle_fatigue = [total_muscle_fatigue[j] + task_fatigue_vector[j] for j in range(20)]
            muscle_fatigue_list.append(task_fatigue_vector)  # Store the vector for later use
    # 累加所有 fatigue_vector
    total_fatigue_vector = [0] * 20  # 初始化一个长度为20的向量

    # 将每个 fatigue_vector 叠加到 total_fatigue_vector 中
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

# Step 3: Create GA toolbox
toolbox = base.Toolbox()

# Initialize individuals with random task assignments (0 for robot, 1 for human)
def init_individual():
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
result_population, logbook = algorithms.eaSimple(population, toolbox, cxpb=0.7, mutpb=0.2, ngen=5000, verbose=True)

# Step 6: Get the best solution
best_individual = tools.selBest(result_population, 1)[0]
print("Best individual (task assignments):", best_individual)
print("Sum of mean and max fatigue:", eval_fatigue(best_individual)[0])
