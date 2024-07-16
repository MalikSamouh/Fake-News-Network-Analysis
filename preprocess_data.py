# preprocess_data.py

import os
from data_loader import load_labels, load_source_tweets

def separate_data_by_labels(label_file_path, source_tweets_file_path, tree_files_path, output_dir):
    """
    Separate data by labels and store them in specified directories.
    Args:
        label_file_path (str): Path to the label file.
        source_tweets_file_path (str): Path to the source tweets file.
        tree_files_path (str): Path to the directory containing tree files.
        output_dir (str): Directory to store the separated data.
    """
    labels = load_labels(label_file_path)
    source_tweets = load_source_tweets(source_tweets_file_path)
    
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory {output_dir} created.")

    # Separate label file data
    for label, tweet_ids in labels.items():
        with open(os.path.join(output_dir, f"{label}_labels.txt"), 'w') as file:
            for tweet_id in tweet_ids:
                file.write(f"{label}:{tweet_id}\n")
        print(f"Separated label data for {label}.")

    # Separate source tweets
    for label, tweet_ids in labels.items():
        with open(os.path.join(output_dir, f"{label}_source_tweets.txt"), 'w') as file:
            for tweet_id in tweet_ids:
                if tweet_id in source_tweets:
                    file.write(f"{tweet_id}\t{source_tweets[tweet_id]}\n")
        print(f"Separated source tweets for {label}.")
    
    # Separate tree files
    for label, tweet_ids in labels.items():
        label_tree_dir = os.path.join(output_dir, f"{label}_trees")
        os.makedirs(label_tree_dir, exist_ok=True)
        for tweet_id in tweet_ids:
            tree_file_path = os.path.join(tree_files_path, f"{tweet_id}.txt")
            if os.path.exists(tree_file_path):
                os.system(f"cp {tree_file_path} {label_tree_dir}")
        print(f"Separated tree files for {label}.")

    print("Data separation complete. Check the output directory for separated files.")

if __name__ == "__main__":
    label_file_path = "rumor_detection_acl2017/twitter16/label.txt"
    source_tweets_file_path = "rumor_detection_acl2017/twitter16/source_tweets.txt"
    tree_files_path = "rumor_detection_acl2017/twitter16/tree"
    output_dir = "processed_data16"

    separate_data_by_labels(label_file_path, source_tweets_file_path, tree_files_path, output_dir)