import pandas as pd
from bertopic import BERTopic

# Example data (replace with your own data)
data = ["Your first text document here.",
        "Another document here.",
        "And one more document for example purposes."]

df = pd.DataFrame(data, columns=['Text'])


# Initialize BERTopic model
model = BERTopic()

# Fit BERTopic model on your data
topics, probabilities = model.fit_transform(df['Text'])


# Visualize topics
model.visualize_topics()


# Get topics and their top keywords
topics = model.get_topics()

# Print topics and their top words
for topic_idx, top_words in topics.items():
    print(f"Topic {topic_idx}: {', '.join(top_words)}")
