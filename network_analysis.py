import os
import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Function to read tree data from file and create a NetworkX graph
def read_tree(file_path):
    G = nx.DiGraph()
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            parent, child = line.strip().split('->')
            parent = eval(parent)
            child = eval(child)
            parent_label = f"{parent[1]}"
            child_label = f"{child[1]}"
            G.add_node(parent_label, time=float(parent[2]), tweet=parent[1])
            G.add_node(child_label, time=float(child[2]), tweet=child[1])
            G.add_edge(parent_label, child_label)
    return G

# Function to calculate network metrics
def calculate_metrics(G):
    metrics = {}
    
    # Number of Nodes and Edges
    metrics['num_nodes'] = G.number_of_nodes()
    metrics['num_edges'] = G.number_of_edges()
    
    # Cascade Size
    cascade_size = G.number_of_nodes() - 1
    metrics['cascade_size'] = cascade_size
    
    # Tree Depth
    root = [node for node in G.nodes if G.in_degree(node) == 0][0]
    lengths = nx.single_source_shortest_path_length(G, root)
    tree_depth = max(lengths.values())
    metrics['tree_depth'] = tree_depth
    
    # Degree Distribution
    in_degrees = dict(G.in_degree())
    out_degrees = dict(G.out_degree())
    metrics['in_degrees'] = in_degrees
    metrics['out_degrees'] = out_degrees
    
    # Propagation Delay
    times = nx.get_node_attributes(G, 'time')
    propagation_delay = max(times.values()) - min(times.values())
    metrics['propagation_delay'] = propagation_delay
    
    # Reaction Times (assuming time is in chronological order)
    reaction_times = []
    for node in G.nodes:
        if G.in_degree(node) > 0:
            parent = list(G.predecessors(node))[0]
            reaction_time = times[node] - times[parent]
            if reaction_time >= 0:
                reaction_times.append(reaction_time)
    metrics['reaction_times'] = reaction_times
    
    # Betweenness Centrality
    betweenness_centrality = nx.betweenness_centrality(G)
    metrics['betweenness_centrality'] = betweenness_centrality
    
    # Closeness Centrality
    closeness_centrality = nx.closeness_centrality(G)
    metrics['closeness_centrality'] = closeness_centrality
    
    return metrics

# Function to analyze a directory of network files
def analyze_directory(directory):
    all_metrics = []
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if os.path.isfile(file_path):
            G = read_tree(file_path)
            metrics = calculate_metrics(G)
            metrics['filename'] = filename
            all_metrics.append(metrics)
    return pd.DataFrame(all_metrics)

# Directories for analysis
directories = {
    'fake_news_twitter15': './processed_data15/false_trees',
    'fake_news_twitter16': './processed_data16/false_trees',
    'real_news_twitter15': './processed_data15/true_trees',
    'real_news_twitter16': './processed_data16/true_trees'
}

# Analyze each directory and calculate average depth for all trees
summary = {}
for key, directory in directories.items():
    df = analyze_directory(directory)
    summary[key] = {
        'average_num_nodes': df['num_nodes'].mean(),
        'average_num_edges': df['num_edges'].mean(),
        'average_cascade_size': df['cascade_size'].mean(),
        'average_tree_depth': df['tree_depth'].mean(),
        'average_propagation_delay': df['propagation_delay'].mean(),
        'average_reaction_time': np.mean([np.mean(times) for times in df['reaction_times'] if len(times) > 0]),
        'average_betweenness_centrality': np.mean([np.mean(list(cent.values())) for cent in df['betweenness_centrality'] if len(cent.values()) > 0]),
        'average_closeness_centrality': np.mean([np.mean(list(cent.values())) for cent in df['closeness_centrality'] if len(cent.values()) > 0])
    }
    output_file = f'{key}_analysis.csv'
    df.to_csv(output_file, index=False)
    print(f'Analysis for {key} saved to {output_file}')

# Print summary for inclusion in LaTeX
for key, metrics in summary.items():
    print(f"\nSummary for {key}:")
    for metric, value in metrics.items():
        print(f"{metric.replace('_', ' ').capitalize()}: {value:.2f}")

# Example metrics for all files (calculate average depth for all trees)
for key, directory in directories.items():
    df = analyze_directory(directory)
    avg_tree_depth = df['tree_depth'].mean()
    print(f"\nAverage tree depth for {key}: {avg_tree_depth:.2f}")

# Visualization
metrics_to_plot = ['average_num_nodes', 'average_num_edges', 'average_cascade_size', 'average_tree_depth', 'average_propagation_delay', 'average_reaction_time', 'average_betweenness_centrality', 'average_closeness_centrality']

# Prepare data for plotting
plot_data = pd.DataFrame(summary).T

# Plotting
fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(15, 20))
axes = axes.flatten()

for idx, metric in enumerate(metrics_to_plot):
    plot_data[metric].plot(kind='bar', ax=axes[idx], title=metric.replace('_', ' ').capitalize())
    axes[idx].set_ylabel(metric.replace('_', ' ').capitalize())
    axes[idx].set_xticklabels(plot_data.index, rotation=45)

plt.tight_layout()
plt.show()
