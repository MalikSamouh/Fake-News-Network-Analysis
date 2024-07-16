import os
import networkx as nx
import pickle
import statistics
from data_loader import parse_tree_file

def analyze_graph(graph):
    """
    Analyze basic properties of the graph.
    Args:
        graph (nx.DiGraph): Directed graph to analyze.
    Returns:
        dict: Dictionary containing analysis results.
    """
    analysis_results = {}

    # Number of Nodes and Edges
    analysis_results['number_of_nodes'] = graph.number_of_nodes()
    analysis_results['number_of_edges'] = graph.number_of_edges()

    # Cascade Size
    analysis_results['cascade_size'] = graph.number_of_nodes() - 1

    # Tree Depth
    if nx.is_directed_acyclic_graph(graph):
        analysis_results['tree_depth'] = nx.dag_longest_path_length(graph)
    else:
        analysis_results['tree_depth'] = "Graph contains a cycle"

    # Degree Distribution
    in_degrees = [degree for node, degree in graph.in_degree()]
    out_degrees = [degree for node, degree in graph.out_degree()]
    analysis_results['in_degree_distribution'] = in_degrees
    analysis_results['out_degree_distribution'] = out_degrees

    # Propagation Delay
    delays = [float(data['delay']) for u, v, data in graph.edges(data=True)]
    analysis_results['propagation_delay_min'] = min(delays)
    analysis_results['propagation_delay_mean'] = statistics.mean(delays)
    analysis_results['propagation_delay_max'] = max(delays)

    # Reaction Times
    reaction_times = [float(data['delay']) for u, v, data in graph.edges(data=True)]
    analysis_results['reaction_time_min'] = min(reaction_times)
    analysis_results['reaction_time_mean'] = statistics.mean(reaction_times)
    analysis_results['reaction_time_max'] = max(reaction_times)

    # Betweenness Centrality
    betweenness_centrality = nx.betweenness_centrality(graph)
    analysis_results['betweenness_centrality'] = betweenness_centrality

    # Closeness Centrality
    closeness_centrality = nx.closeness_centrality(graph)
    analysis_results['closeness_centrality'] = closeness_centrality

    return analysis_results

def save_analysis_results(label, tweet_id, results, output_dir):
    """
    Save analysis results to a file.
    Args:
        label (str): Label of the tweet (true, false, etc.)
        tweet_id (str): Tweet ID
        results (dict): Analysis results
        output_dir (str): Directory to save the results
    """
    label_output_dir = os.path.join(output_dir, label)
    os.makedirs(label_output_dir, exist_ok=True)
    file_path = os.path.join(label_output_dir, f"{label}_{tweet_id}_analysis.txt")
    with open(file_path, 'w') as file:
        for key, value in results.items():
            file.write(f"{key}: {value}\n")
    print(f"Analysis results saved to {file_path}")

def process_graphs(graph_dir, output_dir):
    """
    Process all graphs in the directory and save analysis results.
    Args:
        graph_dir (str): Directory containing graph files
        output_dir (str): Directory to save the analysis results
    """
    for label in ['true', 'false', 'unverified', 'non-rumor']:
        label_dir = os.path.join(graph_dir, f"{label}_graphs")
        for graph_file in os.listdir(label_dir):
            if graph_file.endswith(".pkl"):
                graph_file_path = os.path.join(label_dir, graph_file)
                with open(graph_file_path, 'rb') as f:
                    graph = pickle.load(f)
                tweet_id = graph_file.replace('.pkl', '')
                print(f"Processing {label} tweet ID {tweet_id}...")
                results = analyze_graph(graph)
                save_analysis_results(label, tweet_id, results, output_dir)

def aggregate_results(output_dir):
    """
    Aggregate results from all analysis files in the output directory.
    Args:
        output_dir (str): Directory containing analysis result files
    Returns:
        dict: Dictionary containing aggregated results
    """
    aggregated_results = {}
    for label in ['true', 'false', 'unverified', 'non-rumor']:
        label_dir = os.path.join(output_dir, label)
        label_results = []
        for file_name in os.listdir(label_dir):
            if file_name.endswith("_analysis.txt"):
                file_path = os.path.join(label_dir, file_name)
                with open(file_path, 'r') as file:
                    result = {}
                    for line in file:
                        if ': ' in line:
                            key, value = line.strip().split(': ', 1)
                            if value == "Graph contains a cycle":
                                result[key] = value
                            else:
                                try:
                                    result[key] = eval(value)
                                except:
                                    result[key] = value
                    label_results.append(result)
        aggregated_results[label] = label_results
    return aggregated_results

def compare_labels(aggregated_results):
    """
    Compare analysis results across different labels.
    Args:
        aggregated_results (dict): Dictionary containing aggregated results for each label
    Returns:
        dict: Dictionary containing comparison results
    """
    comparison_results = {}
    for metric in aggregated_results['true'][0].keys():
        comparison_results[metric] = {}
        for label in aggregated_results.keys():
            metric_values = [result[metric] for result in aggregated_results[label] if isinstance(result[metric], (int, float))]
            if metric_values:
                comparison_results[metric][label] = {
                    'min': min(metric_values) if metric_values else None,
                    'mean': statistics.mean(metric_values) if metric_values else None,
                    'max': max(metric_values) if metric_values else None
                }
            else:
                # Handle list type metrics
                list_metric_values = [item for sublist in [result[metric] for result in aggregated_results[label] if isinstance(result[metric], list)] for item in sublist]
                if list_metric_values:
                    comparison_results[metric][label] = {
                        'min': min(list_metric_values) if list_metric_values else None,
                        'mean': statistics.mean(list_metric_values) if list_metric_values else None,
                        'max': max(list_metric_values) if list_metric_values else None
                    }
    return comparison_results

def save_comparison_results(comparison_results, output_path):
    """
    Save comparison results to a file.
    Args:
        comparison_results (dict): Dictionary containing comparison results
        output_path (str): Path to save the comparison results
    """
    with open(output_path, 'w') as file:
        for metric, results in comparison_results.items():
            file.write(f"Metric: {metric}\n")
            for label, stats in results.items():
                file.write(f"  {label}:\n")
                file.write(f"    Min: {stats['min']}\n")
                file.write(f"    Mean: {stats['mean']}\n")
                file.write(f"    Max: {stats['max']}\n")
    print(f"Comparison results saved to {output_path}")

if __name__ == "__main__":
    graph_dir = "graphs16"
    output_dir = "analysis_results16"
    comparison_output_path = "comparison_results16.txt"
    
    print("Starting analysis...")
    process_graphs(graph_dir, output_dir)
    print("Analysis complete. Starting label-based comparison...")
    aggregated_results = aggregate_results(output_dir)
    comparison_results = compare_labels(aggregated_results)
    save_comparison_results(comparison_results, comparison_output_path)
    print("Label-based comparison complete. Check the output file for results.")