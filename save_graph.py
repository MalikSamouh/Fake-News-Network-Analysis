# save_graph.py

import os
import networkx as nx
import pickle
from data_loader import parse_tree_file

def build_tree_network(tree_file_path):
    """
    Build a directed graph from a single tree file.
    Args:
        tree_file_path (str): Path to the tree file.
    Returns:
        nx.DiGraph: Directed graph of the tree.
    """
    G = nx.DiGraph()
    edges = parse_tree_file(tree_file_path)
    G.add_edges_from(edges)
    
    print(f"Built network for tree file {tree_file_path} with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
    return G

def save_graph(G, output_path):
    """
    Save the given graph to a file.
    Args:
        G (nx.DiGraph): Directed graph to save.
        output_path (str): Path to save the graph.
    """
    with open(output_path, 'wb') as f:
        pickle.dump(G, f)
    print(f"Graph saved to {output_path}")

def process_directory(input_dir, output_dir):
    """
    Process all tree files in the input directory and save the graphs to the output directory.
    Args:
        input_dir (str): Directory containing tree files.
        output_dir (str): Directory to save the graphs.
    """
    os.makedirs(output_dir, exist_ok=True)
    for tree_file in os.listdir(input_dir):
        if tree_file.endswith(".txt"):
            tree_file_path = os.path.join(input_dir, tree_file)
            G = build_tree_network(tree_file_path)
            output_path = os.path.join(output_dir, tree_file.replace('.txt', '.pkl'))
            save_graph(G, output_path)

if __name__ == "__main__":
    base_input_dir = "processed_data16"
    base_output_dir = "graphs16"

    for label in ['true', 'false', 'unverified', 'non-rumor']:
        input_dir = os.path.join(base_input_dir, f"{label}_trees")
        output_dir = os.path.join(base_output_dir, f"{label}_graphs")
        process_directory(input_dir, output_dir)