import os
import neat
import pickle
import pandas as pd

# Load configurations and winner genome
experiment_name = 'colab-exp3-game1'
enemy = 1
run_number = 0
num_inputs = 20
num_outputs = 5

# Load the winner genome
with open(os.path.join(experiment_name, f'winner-{enemy}-{run_number}.pkl'), 'rb') as input_file:
    winner = pickle.load(input_file)

# Load NEAT configuration
config_path = os.path.join(os.path.dirname(__file__), 'sandor_neat_config.ini')
config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     config_path)

# Extract node IDs from net.node_evals
node_ids = [t[0] for t in neat.nn.FeedForwardNetwork.create(winner, config).node_evals]

# Identify input, hidden, and output nodes
input_nodes = [-i-1 for i in range(num_inputs)]
hidden_nodes = [node_id for node_id in node_ids if node_id >= num_outputs]
output_nodes = [node_id for node_id in node_ids if 0 <= node_id < num_outputs]

# Initialize counters for weights between different layers
weights_input_hidden = 0
weights_hidden_output = 0
weights_input_output = 0

# Iterate over the connections in the winner genome and count weights
for connection in winner.connections.values():
    if connection.enabled:
        if connection.key[0] in input_nodes and connection.key[1] in hidden_nodes:
            weights_input_hidden += 1
        elif connection.key[0] in hidden_nodes and connection.key[1] in output_nodes:
            weights_hidden_output += 1
        elif connection.key[0] in input_nodes and connection.key[1] in output_nodes:
            weights_input_output += 1

# Number of Biases
total_biases = len(hidden_nodes) + len(output_nodes)

# Prepare the general statistics data
general_data = {
    "Statistic": [
        "Enemy",
        "Run Number",
        "Number of Inputs",
        "Number of Hidden Nodes",
        "Number of Output Nodes",
        "Total Biases",
        "Weights between Input and Hidden Nodes",
        "Weights between Hidden and Output Nodes",
        "Weights between Input and Output Nodes"
    ],
    "Value": [
        enemy,
        run_number,
        num_inputs,
        len(hidden_nodes),
        len(output_nodes),
        total_biases,
        weights_input_hidden,
        weights_hidden_output,
        weights_input_output
    ]
}

df_general = pd.DataFrame(general_data)

# Extract individual weights
weights_data = {"Source Node": [], "Target Node": [], "Weight": []}
for connection in winner.connections.values():
    if connection.enabled:
        weights_data["Source Node"].append(connection.key[0])
        weights_data["Target Node"].append(connection.key[1])
        weights_data["Weight"].append(connection.weight)

df_weights = pd.DataFrame(weights_data)

# Extract individual bias values (Assuming that bias is stored in the node's bias attribute)
biases_data = {"Node": [], "Bias": []}
for node_id in hidden_nodes + output_nodes:
    bias = winner.nodes[node_id].bias
    biases_data["Node"].append(node_id)
    biases_data["Bias"].append(bias)

df_biases = pd.DataFrame(biases_data)

# Write the data to an Excel file
output_file_path = os.path.join(experiment_name, f'network-stats-{enemy}-{run_number}.xlsx')
with pd.ExcelWriter(output_file_path, engine='xlsxwriter') as writer:
    df_general.to_excel(writer, index=False, sheet_name='General Information')
    df_weights.to_excel(writer, index=False, sheet_name='Weights')
    df_biases.to_excel(writer, index=False, sheet_name='Biases')

print(f"Network statistics have been written to {output_file_path}")
