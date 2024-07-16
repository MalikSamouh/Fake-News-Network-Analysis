# visualize_graph.py

import os
import networkx as nx
import matplotlib.pyplot as plt
import pickle
from data_loader import parse_tree_file

def visualize_network(G, title='Network Visualization'):
    """
    Visualize the given network.
    Args:
        G (nx.DiGraph): Directed graph to visualize.
        title (str): Title for the visualization plot.
    """
    plt.figure(figsize=(12, 12))
    pos = nx.spring_layout(G, k=0.15, iterations=20)
    
    # Draw nodes and edges with different styles
    nx.draw_networkx_nodes(G, pos, node_size=50, node_color='blue')
    nx.draw_networkx_edges(G, pos, edgelist=G.edges(), edge_color='gray')
    
    # Draw labels to understand the propagation
    labels = {node: node for node in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels, font_size=8, font_color='black')
    
    plt.title(title)
    plt.show()

def visualize_saved_graph(graph_file_path):
    """
    Visualize a saved graph file.
    Args:
        graph_file_path (str): Path to the saved graph file.
    """
    with open(graph_file_path, 'rb') as f:
        G = pickle.load(f)
    print(f"Visualizing saved graph from {graph_file_path} with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
    visualize_network(G, title=f'Visualization for {os.path.basename(graph_file_path)}')

def visualize_tree_file(tree_file_path):
    """
    Visualize the propagation tree from a tree file.
    Args:
        tree_file_path (str): Path to the tree file.
    """
    G = nx.DiGraph()
    edges = parse_tree_file(tree_file_path)
    G.add_edges_from(edges)
    
    print(f"Visualizing network for tree file {tree_file_path} with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
    visualize_network(G, title=f'Visualization for {os.path.basename(tree_file_path)}')

if __name__ == "__main__":
    # Uncomment one of the following lines to visualize a tree file or a saved graph file
    
    # Visualize a single tree file
    # tree_file_path = "processed_data15/true_trees/295152287901417472.txt"
    # visualize_tree_file(tree_file_path)
    
    # Visualize a saved graph file
    graph_file_path = "graphs16/true_graphs/498430783699554305.pkl"
    visualize_saved_graph(graph_file_path)