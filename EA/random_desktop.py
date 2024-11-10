import random
import pandas as pd
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


def generate_valid_sequence(graph):
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


def eval_fatigue(sequence, assignments):
    """
    Evaluate the sum of mean and max fatigue for a given task sequence and assignments.
    - `sequence` is a list of tasks.
    - `assignments` is a list where each element is 0 (robot) or 1 (human).
    """
    muscle_fatigue_list = []  # Store the individual muscle fatigue for calculating max later

    for task, assigned_to in zip(sequence, assignments):
        if assigned_to == 1:  # Human does the task
            task_fatigue_vector = fatigue_data.iloc[task - 1].values  # Adjusted index
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

    return mean_fatigue + max_fatigue  # Return the sum of mean and max fatigue


# Run simulations
num_simulations = 10
fatigue_results = []

for i in range(num_simulations):
    # Generate a valid task sequence
    sequence = generate_valid_sequence(G)

    # Randomly assign each task to human (1) or robot (0)
    assignments = [random.randint(0, 1) for _ in sequence]

    # Evaluate fatigue
    fatigue = eval_fatigue(sequence, assignments)
    fatigue_results.append(fatigue)
    print(f"Simulation {i + 1}: Sum of mean and max fatigue = {fatigue}")

# Calculate the average fatigue over all simulations
average_fatigue = sum(fatigue_results) / num_simulations
print(f"\nAverage Sum of mean and max fatigue over {num_simulations} simulations: {average_fatigue}")
