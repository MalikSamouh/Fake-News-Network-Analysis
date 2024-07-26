# Fake News Analysis Project

## Overview

This is the official documentation for the Fake News Analysis Project. The project aims to analyze social media networks to detect and understand the spread of fake news, providing insights that could help mitigate its impact and enhance the reliability of online information.

## Research Objectives

1. **Clean and Label Dataset**: Categorize news into real, fake, rumor unverified, and rumor verified.
2. **Visualize News Propagation**: Construct interaction networks of tweets.
3. **Analyze Spread Patterns**: Compare speed, reach, and depth of fake news versus real news.

## Key Questions

1. Does fake news spread more than real news on Twitter?
2. Do people with a higher following spread fake news?
3. How do emotions affect reactions towards tweets?

## Methodology

### Data Collection
We used Twitter15 and Twitter16 datasets, collected via the Twitter API.

### Data Labeling and Analysis
Data was labeled to categorize tweets into real and fake news, then analyzed to identify differences in propagation.

### Network Construction and Visualization
Interaction networks were constructed to visualize tweet propagation, focusing on nodes (users) and edges (interactions).

### Sentiment and Emotion Analysis
Performed to understand emotional triggers causing users to share or react to tweets.

## Insights and Mitigations

1. **Prevalence**: Fake news is more prevalent and engages more users.
2. **Propagation**: Fake news spreads more widely and penetrates deeper into the network.
3. **Delay and Reaction**: Fake news has a longer propagation delay and higher reaction time.
4. **Emotional Impact**: Emotions like disgust, joy, and fear significantly impact the spread of fake news.

### Mitigation Strategies
Implement fact-checking mechanisms, label fake content, and restrict sharing and retweeting of identified fake news.

## Setup Instructions

1. **Clone the repository**:
   ```bash
   git clone <repository_url>
   cd fake-news-analysis
   ```

2. **Create a virtual environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts ctivate`
   ```

3. **Install the dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Scripts

1. **Preprocess the data**:
   ```bash
   python preprocess_data.py
   ```

2. **Analyze the fake news**:
   ```bash
   python fake_news_analysis.py
   ```

3. **Analyze the real news**:
   ```bash
   python real_news_analysis.py
   ```

4. **Perform network analysis**:
   ```bash
   python network_analysis.py
   ```

5. **Visualize the graphs**:
   ```bash
   python visualize_graph.py
   ```
