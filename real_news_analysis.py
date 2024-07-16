import os
import networkx as nx
import matplotlib.pyplot as plt
import logging

# Set up logging
logging.basicConfig(filename='real_news_longest_depth_files.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

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

# Function to calculate the depth of a tree
def calculate_depth(G):
    if not G:
        return 0
    root = [node for node in G.nodes if G.in_degree(node) == 0]
    if not root:
        return 0
    root = root[0]
    lengths = nx.single_source_shortest_path_length(G, root)
    max_depth = max(lengths.values())
    return max_depth

# Function to find the longest path in a tree
def find_longest_path(G):
    root = [node for node in G.nodes if G.in_degree(node) == 0]
    if not root:
        return []
    root = root[0]
    all_paths = []
    for node in G.nodes:
        if nx.has_path(G, root, node):
            path = nx.shortest_path(G, root, node)
            all_paths.append(path)
    longest_path = max(all_paths, key=len)
    return longest_path

# Function to plot the entire propagation graph
def plot_propagation_graph(G, title, description):
    pos = nx.spring_layout(G)
    plt.figure(figsize=(12, 8))
    times = nx.get_node_attributes(G, 'time')
    labels = {node: times[node] for node in G.nodes}
    nx.draw(G, pos, labels=labels, node_size=500, node_color='lightblue', font_size=10, font_weight='bold', arrows=True)
    plt.title(title)
    plt.figtext(0.5, 0.01, description, wrap=True, horizontalalignment='center', fontsize=12)
    plt.show()

# Function to plot a graph with only the longest path
def plot_longest_path(G, path, title, description):
    subgraph = G.subgraph(path).copy()
    pos = nx.spring_layout(subgraph)
    plt.figure(figsize=(12, 8))
    path_edges = list(zip(path, path[1:]))
    times = nx.get_node_attributes(subgraph, 'time')
    labels = {node: times[node] for node in subgraph.nodes}
    nx.draw(subgraph, pos, labels=labels, node_size=500, node_color='lightblue', font_size=10, font_weight='bold', arrows=True, edge_color='red', width=2)
    plt.title(title)
    plt.figtext(0.5, 0.01, description, wrap=True, horizontalalignment='center', fontsize=12)
    plt.show()

# Function to find the top 3 files with the longest depth
def find_top_3_longest_depth(directory):
    files_with_depths = []
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if os.path.isfile(file_path):
            G = read_tree(file_path)
            depth = calculate_depth(G)
            files_with_depths.append((filename, depth))
    files_with_depths.sort(key=lambda x: x[1], reverse=True)
    return files_with_depths[:3]

# Function to read the tweet descriptions from source_tweets.txt
def read_tweet_descriptions(file_path):
    tweet_descriptions = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                tweet_id, description = parts
                tweet_descriptions[tweet_id] = description
    print(f"Read {len(tweet_descriptions)} descriptions from {file_path}")
    return tweet_descriptions

# Function to read the top 3 files from real_news_longest_depth_files.log
def read_top_files(log_file_path):
    top_files = []
    with open(log_file_path, 'r') as log_file:
        for line in log_file:
            if "File:" in line:
                parts = line.split("File: ")[1].split(", Depth: ")
                file_id = parts[0].strip().split('.')[0]  # Extracting file ID without .txt
                top_files.append(file_id)
    print(f"Top files from log: {top_files}")
    return top_files

# Function to match top files with their descriptions
def match_descriptions(log_file_path, tweet_file_path):
    top_files = read_top_files(log_file_path)
    tweet_descriptions = read_tweet_descriptions(tweet_file_path)
    
    matched_descriptions = {}
    for file_id in top_files:
        if file_id in tweet_descriptions:
            matched_descriptions[file_id] = tweet_descriptions[file_id]
        else:
            matched_descriptions[file_id] = "No description available"
    print(f"Matched descriptions: {matched_descriptions}")
    return matched_descriptions

# Directories for real news
directories = {
    'twitter15': './processed_data15/true_trees',  # Update this to the correct path if necessary
    'twitter16': './processed_data16/true_trees'   # Update this to the correct path if necessary
}

# Read tweet descriptions
tweet_descriptions_files = {
    'twitter15': './processed_data15/true_source_tweets.txt',  # Update this to the correct path if necessary
    'twitter16': './processed_data16/true_source_tweets.txt'   # Update this to the correct path if necessary
}

# Find and log the top 3 files with the longest depth and plot the longest path for each directory
for key, directory in directories.items():
    top_3_files = find_top_3_longest_depth(directory)
    tweet_descriptions_file_path = tweet_descriptions_files[key]
    matched_descriptions = match_descriptions('real_news_longest_depth_files.log', tweet_descriptions_file_path)
    logging.info(f"Top 3 files with the longest depth in {key}:")
    for file, depth in top_3_files:
        logging.info(f"File: {file}, Depth: {depth}")
        file_path = os.path.join(directory, file)
        if os.path.exists(file_path):
            G = read_tree(file_path)
            description = matched_descriptions.get(file.split('.')[0], "No description available")
            title = f"File: {file}, Depth: {depth}"
            
            # Plot the entire propagation graph
            plot_propagation_graph(G, title=f"Propagation Graph - {title}", description=description)
            
            # Find and plot the longest path
            longest_path = find_longest_path(G)
            if longest_path:
                root_tweet_id = longest_path[0]
                max_depth = len(longest_path) - 1
                description = matched_descriptions.get(root_tweet_id, description)  # Use the same description if not found
                title = f"Longest Path in File: {file}, Max Depth: {max_depth}"
                plot_longest_path(G, longest_path, title=title, description=description)

print("Top 3 files with the longest depth logged in 'real_news_longest_depth_files.log' and plots created.")
