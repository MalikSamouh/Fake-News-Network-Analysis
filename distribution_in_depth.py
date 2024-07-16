import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import matplotlib.pyplot as plt

# Function to read tweet descriptions and timestamps from file
def read_tweet_descriptions_with_timestamps(file_path):
    tweet_descriptions = []
    timestamps = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 3:
                tweet_id, description, timestamp = parts
                tweet_descriptions.append(description)
                timestamps.append(timestamp)
    return tweet_descriptions, timestamps

# Function to filter tweets containing specific keywords
def filter_tweets(descriptions, timestamps, keyword):
    filtered_descriptions = []
    filtered_timestamps = []
    for description, timestamp in zip(descriptions, timestamps):
        if keyword.lower() in description.lower():
            filtered_descriptions.append(description)
            filtered_timestamps.append(timestamp)
    return filtered_descriptions, filtered_timestamps

# Function to aggregate descriptions by month
def aggregate_by_month(descriptions, timestamps):
    # Ensure timestamps are datetime objects
    timestamps = pd.to_datetime(timestamps)
    data = pd.DataFrame({'description': descriptions, 'timestamp': timestamps})
    data['month'] = data['timestamp'].dt.to_period('M')
    monthly_aggregated = data.groupby('month')['description'].apply(lambda x: ' '.join(x)).reset_index()
    return monthly_aggregated

# Function to apply LDA topic modeling by month
def apply_lda_by_month(monthly_data, num_topics=5):
    topic_distributions = []
    vectorizer = CountVectorizer(stop_words='english')
    lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
    
    for descriptions in monthly_data['description']:
        if descriptions.strip():  # Check if there are descriptions to process
            try:
                X = vectorizer.fit_transform([descriptions])
                if X.shape[1] > 0:  # Ensure the matrix is not empty
                    lda.fit(X)
                    topic_distribution = np.mean(lda.transform(X), axis=0)
                    topic_distributions.append(topic_distribution)
                else:
                    topic_distributions.append([0] * num_topics)
            except:
                topic_distributions.append([0] * num_topics)
        else:
            topic_distributions.append([0] * num_topics)
    
    if len(vectorizer.get_feature_names_out()) > 0:
        return topic_distributions, vectorizer.get_feature_names_out(), lda.components_
    else:
        return topic_distributions, [], []

# Function to get topic distribution over time
def get_topic_distribution_over_time(monthly_data, topic_distributions):
    topic_dist_df = pd.DataFrame(topic_distributions, columns=[f'Topic {i}' for i in range(len(topic_distributions[0]))])
    topic_dist_df['month'] = monthly_data['month'].astype(str)
    return topic_dist_df

# Function to plot topic distribution over time
def plot_topic_distribution_over_time(months, topic_distributions, title, topic_labels):
    fig, ax = plt.subplots(figsize=(12, 8))
    for i in range(len(topic_labels)):
        ax.plot(months, topic_distributions[:, i], label=topic_labels[i])
    ax.set_xlabel('Month')
    ax.set_ylabel('Proportion')
    ax.set_title(title)
    ax.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Paths to the data files
data_paths = {
    'fake_news_twitter15': './processed_data15/false_source_tweets.txt',
    'real_news_twitter15': './processed_data15/true_source_tweets.txt',
    'fake_news_twitter16': './processed_data16/false_source_tweets.txt',
    'real_news_twitter16': './processed_data16/true_source_tweets.txt'
}

# Read tweet descriptions and timestamps
descriptions_timestamps = {key: read_tweet_descriptions_with_timestamps(path) for key, path in data_paths.items()}

# Filter tweets mentioning "Donald Trump"
keyword = "Donald Trump"
filtered_descriptions_timestamps = {key: filter_tweets(*descriptions_timestamps[key], keyword) for key in descriptions_timestamps}

# Aggregate descriptions by month
monthly_aggregated = {key: aggregate_by_month(*filtered_descriptions_timestamps[key]) for key in filtered_descriptions_timestamps}

# Apply LDA topic modeling by month
num_topics = 5
topic_distributions = {key: apply_lda_by_month(monthly_data, num_topics) for key, monthly_data in monthly_aggregated.items()}

# Get topic distribution over time
topic_dist_over_time = {key: get_topic_distribution_over_time(monthly_aggregated[key], topic_distributions[key][0]) for key in topic_distributions}

# Plotting the topic distributions for fake news and real news over time
topic_labels = [
    "Politics and Policies",
    "Scandals and Controversies",
    "Media Coverage",
    "Public Reactions",
    "General News"
]

# Generate plots
months_15 = monthly_aggregated['fake_news_twitter15']['month']
months_16 = monthly_aggregated['fake_news_twitter16']['month']

plot_topic_distribution_over_time(months_15, np.array(topic_dist_over_time['fake_news_twitter15'].drop('month', axis=1)), 'Fake News Topic Distribution Over Time for Twitter15', topic_labels)
plot_topic_distribution_over_time(months_15, np.array(topic_dist_over_time['real_news_twitter15'].drop('month', axis=1)), 'Real News Topic Distribution Over Time for Twitter15', topic_labels)

plot_topic_distribution_over_time(months_16, np.array(topic_dist_over_time['fake_news_twitter16'].drop('month', axis=1)), 'Fake News Topic Distribution Over Time for Twitter16', topic_labels)
plot_topic_distribution_over_time(months_16, np.array(topic_dist_over_time['real_news_twitter16'].drop('month', axis=1)), 'Real News Topic Distribution Over Time for Twitter16', topic_labels)

# Print top words and summaries for each topic for both Twitter15 and Twitter16
for i in range(num_topics):
    print(f"\nTopic {i} ({topic_labels[i]}):")
    print(f"Fake News Twitter15 Top Words: {', '.join(topic_distributions['fake_news_twitter15'][1][i])}")
    print(f"Real News Twitter15 Top Words: {', '.join(topic_distributions['real_news_twitter15'][1][i])}")
    print(f"Fake News Twitter16 Top Words: {', '.join(topic_distributions['fake_news_twitter16'][1][i])}")
    print(f"Real News Twitter16 Top Words: {', '.join(topic_distributions['real_news_twitter16'][1][i])}")
    print(f"Summary: This topic generally covers {topic_labels[i].lower()}.\n")
