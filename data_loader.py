# data_loader.py

def load_labels(label_file_path):
    """
    Load labels from the label file.
    Args:
        label_file_path (str): Path to the label file.
    Returns:
        dict: Dictionary with labels as keys and list of tweet IDs as values.
    """
    labels = {}
    with open(label_file_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            label, tweet_id = line.strip().split(':')
            if label not in labels:
                labels[label] = []
            labels[label].append(tweet_id)
    return labels

def load_source_tweets(source_tweets_file_path):
    """
    Load source tweets from the source tweets file.
    Args:
        source_tweets_file_path (str): Path to the source tweets file.
    Returns:
        dict: Dictionary with tweet IDs as keys and tweet content as values.
    """
    source_tweets = {}
    with open(source_tweets_file_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            tweet_id, content = line.strip().split('\t')
            source_tweets[tweet_id] = content
    return source_tweets

def parse_tree_file(tree_file_path):
    """
    Parse the tree file to extract edges.
    Args:
        tree_file_path (str): Path to the tree file.
    Returns:
        list: List of edges represented as (parent, child, attributes).
    """
    edges = []
    with open(tree_file_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            parent, child = line.strip().split('->')
            parent = eval(parent.strip())
            child = eval(child.strip())
            edges.append((parent[0], child[0], {'delay': float(child[2])}))  # Use UID for nodes and ensure delay is float
    return edges