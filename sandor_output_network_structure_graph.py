import os
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import neat
import pickle

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


# Create a Directed Graph
G = nx.DiGraph()

# Create a FeedForward network from the winner genome
net = neat.nn.FeedForwardNetwork.create(winner, config)

# Extract node IDs from net.node_evals
node_ids = [t[0] for t in net.node_evals]

# Identify input, hidden, and output nodes and add them to the graph
input_nodes = [-i-1 for i in range(num_inputs)]
output_nodes = [node_id for node_id in node_ids if 0 <= node_id < num_outputs]
hidden_nodes = [node_id for node_id in node_ids if node_id >= num_outputs]

# Create a dictionary with custom labels for the nodes
labels = {}

# Update labels for input nodes
for i, node_id in enumerate(input_nodes):
    labels[node_id] = str(i + 1)

# Update labels for hidden nodes
for i, node_id in enumerate(hidden_nodes):
    labels[node_id] = f"h-{i + 1}"

# Update labels for output nodes
output_labels = ["Left", "Right", "Jump", "Stop Jump", "Shoot"]
for i, node_id in enumerate(output_nodes):
    labels[node_id] = output_labels[i]


# Construct the graph with nodes and connections
for node_id in input_nodes + hidden_nodes + output_nodes:
    G.add_node(node_id)

for connection in winner.connections.values():
    if connection.enabled:
        G.add_edge(connection.key[0], connection.key[1])

# Define positions for each node layer
fig_width = 10
num_input_nodes = len(input_nodes)
num_hidden_nodes = len(hidden_nodes)
num_output_nodes = len(output_nodes)

spacing_input = fig_width / (num_input_nodes + 1)
spacing_hidden = fig_width / (num_hidden_nodes + 1)
spacing_output = fig_width / (num_output_nodes + 1)

layer_positions = {
    'input': {node_id: (0, (i + 1) * spacing_input) for i, node_id in enumerate(input_nodes)},
    'hidden': {node_id: (1, (i + 1) * spacing_hidden) for i, node_id in enumerate(hidden_nodes)},
    'output': {node_id: (2, (i + 1) * spacing_output) for i, node_id in enumerate(output_nodes)}
}

# Define different sizes for input, hidden, and output nodes
node_sizes = {node: 500 for node in input_nodes}
node_sizes.update({node: 2000 for node in hidden_nodes})
node_sizes.update({node: 3000 for node in output_nodes})

# Extract the sizes in the order of G.nodes()
default_size = 1000  # You can set this to whatever default size you prefer
size_list = [node_sizes.get(node, default_size) for node in G.nodes()]


# Combine all layer positions
positions = {**layer_positions['input'], **layer_positions['hidden'], **layer_positions['output']}


# Check if all nodes in the graph have assigned positions
nodes_without_positions = set(G.nodes()) - set(positions.keys())
default_position = (1, 1)  # Set this to a suitable default position
for node in nodes_without_positions:
    positions[node] = default_position
if nodes_without_positions:
    print("Nodes without assigned positions:", nodes_without_positions)
    assert False, "Some nodes do not have assigned positions"


# Draw the graph with custom labels
node_colors = {node: 'lightcoral' for node in input_nodes}
node_colors.update({node: 'lightgreen' for node in hidden_nodes})
node_colors.update({node: 'lightblue' for node in output_nodes})

# Define font styling
font_size = 10
font_color = 'k'

# Define figure size and layout
plt.figure(figsize=(12, 8))

# Draw nodes
nx.draw_networkx_nodes(G, positions, node_color=list(node_colors.values()), node_size=size_list)

# Draw edges with styling
nx.draw_networkx_edges(G, positions, alpha=0.5)
# Draw labels with font styling
nx.draw_networkx_labels(G, positions, labels=labels, font_size=font_size, font_color=font_color)

# Add title and annotations/legends
plt.title(f"Neural Network Structure for enemy {enemy} and run {run_number}")
legend_labels = [mpatches.Patch(color=color, label=label) for label, color in 
                 zip(['Input', 'Hidden', 'Output'], ['lightcoral', 'lightgreen', 'lightblue'])]
plt.legend(handles=legend_labels, loc='upper right')

# Initialize counters for weights/biases between different layers
weights_input_hidden = 0
weights_hidden_output = 0
weights_input_output = 0

# Iterate over the connections in the winner genome
for connection in winner.connections.values():
    if connection.enabled:
        # Check if the connection is between input and hidden nodes
        if connection.key[0] in input_nodes and connection.key[1] in hidden_nodes:
            weights_input_hidden += 1
        # Check if the connection is between hidden and output nodes
        elif connection.key[0] in hidden_nodes and connection.key[1] in output_nodes:
            weights_hidden_output += 1
        # Check if the connection is between input and output nodes
        elif connection.key[0] in input_nodes and connection.key[1] in output_nodes:
            weights_input_output += 1

#Nr Biases
total_biases = num_hidden_nodes+num_output_nodes

# Display the total number of weights/biases between each layer on the graph
plt.text(0.5, -1, f"W(I-H): {weights_input_hidden}", ha='center', fontsize=12)
plt.text(1.5, -1, f"W(H-O): {weights_hidden_output}", ha='center', fontsize=12)
plt.text(1, -2, f"Total weights: {weights_hidden_output+weights_input_hidden+weights_input_output} \nTotal biases: {total_biases} ({num_input_nodes} inputs + {num_hidden_nodes} hidden + {num_output_nodes} outputs)", ha='center', fontsize=12)

# In case there are direct connections between input and output layers, display that information as well
if weights_input_output > 0:
    plt.text(1, -1, f"W(I-O): {weights_input_output}", ha='center', fontsize=12)

# Show the graph
plt.show()
