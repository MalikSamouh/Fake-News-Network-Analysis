import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import matplotlib.pyplot as plt

# Function to read tweet descriptions from file
def read_tweet_descriptions(file_path):
    tweet_descriptions = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                tweet_id, description = parts
                tweet_descriptions.append(description)
    return tweet_descriptions

# Function to extract topics using LDA
def extract_topics(descriptions, num_topics=5):
    vectorizer = CountVectorizer(stop_words='english')
    X = vectorizer.fit_transform(descriptions)
    lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
    lda.fit(X)
    topic_labels = np.argmax(lda.transform(X), axis=1)
    topic_words = {i: [vectorizer.get_feature_names_out()[index] for index in np.argsort(lda.components_[i])[-10:]] for i in range(num_topics)}
    return topic_labels, topic_words

# Function to compare topics between fake and real news
def compare_topics(fake_descriptions, real_descriptions, num_topics=5):
    # Extract topics for fake news
    fake_topic_labels, fake_topic_words = extract_topics(fake_descriptions, num_topics)
    fake_topic_counts = np.bincount(fake_topic_labels)
    
    # Extract topics for real news
    real_topic_labels, real_topic_words = extract_topics(real_descriptions, num_topics)
    real_topic_counts = np.bincount(real_topic_labels)
    
    # Normalize topic counts
    fake_topic_dist = fake_topic_counts / sum(fake_topic_counts)
    real_topic_dist = real_topic_counts / sum(real_topic_counts)
    
    return fake_topic_dist, real_topic_dist, fake_topic_words, real_topic_words

# Directories for analysis
fake_news_files = {
    'twitter15': './processed_data15/false_source_tweets.txt',
    'twitter16': './processed_data16/false_source_tweets.txt'
}

real_news_files = {
    'twitter15': './processed_data15/true_source_tweets.txt',
    'twitter16': './processed_data16/true_source_tweets.txt'
}

# Read tweet descriptions
fake_news_descriptions_15 = read_tweet_descriptions(fake_news_files['twitter15'])
real_news_descriptions_15 = read_tweet_descriptions(real_news_files['twitter15'])

fake_news_descriptions_16 = read_tweet_descriptions(fake_news_files['twitter16'])
real_news_descriptions_16 = read_tweet_descriptions(real_news_files['twitter16'])

# Compare topics
num_topics = 5
fake_topic_dist_15, real_topic_dist_15, fake_topic_words_15, real_topic_words_15 = compare_topics(fake_news_descriptions_15, real_news_descriptions_15, num_topics)
fake_topic_dist_16, real_topic_dist_16, fake_topic_words_16, real_topic_words_16 = compare_topics(fake_news_descriptions_16, real_news_descriptions_16, num_topics)

# Labeling topics manually based on the top words
topic_labels = [
    "Politics and Scandals",
    "Criminal Activities",
    "Violence and Disasters",
    "Celebrity and Entertainment",
    "General News"
]

# Plotting function
def plot_topic_distribution(fake_dist, real_dist, title):
    topics = [topic_labels[i] for i in range(num_topics)]
    x = np.arange(len(topics))

    fig, ax = plt.subplots()
    bar_width = 0.35

    bar1 = ax.bar(x - bar_width/2, fake_dist, bar_width, label='Fake News')
    bar2 = ax.bar(x + bar_width/2, real_dist, bar_width, label='Real News')

    ax.set_xlabel('Topics')
    ax.set_ylabel('Proportion')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(topics, rotation=45, ha="right")
    ax.legend()

    plt.show()

# Plotting the topic distributions for Twitter15
plot_topic_distribution(fake_topic_dist_15, real_topic_dist_15, 'Topic Distribution Comparison for Twitter15')

# Plotting the topic distributions for Twitter16
plot_topic_distribution(fake_topic_dist_16, real_topic_dist_16, 'Topic Distribution Comparison for Twitter16')

# Plotting the topic distributions over time for fake news
plot_topic_distribution(fake_topic_dist_15, fake_topic_dist_16, 'Fake News Topic Distribution Over Time')

# Plotting the topic distributions over time for real news
plot_topic_distribution(real_topic_dist_15, real_topic_dist_16, 'Real News Topic Distribution Over Time')

# Print top words and summaries for each topic for both Twitter15 and Twitter16
for i in range(num_topics):
    print(f"\nTopic {i} ({topic_labels[i]}):")
    print(f"Fake News Twitter15 Top Words: {', '.join(fake_topic_words_15[i])}")
    print(f"Real News Twitter15 Top Words: {', '.join(real_topic_words_15[i])}")
    print(f"Fake News Twitter16 Top Words: {', '.join(fake_topic_words_16[i])}")
    print(f"Real News Twitter16 Top Words: {', '.join(real_topic_words_16[i])}")
    print(f"Summary: This topic generally covers {topic_labels[i].lower()}.\n")
