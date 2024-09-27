import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils import resample
from sklearn.pipeline import Pipeline

# transformer model

# get data: download csv from https://www.kaggle.com/datasets/crowdflower/first-gop-debate-twitter-sentiment?resource=download

def load_and_clean_data(file_path):
    data = pd.read_csv(file_path)
    data = data[["text", "sentiment"]]
    data["text"] = data["text"].apply(clean_text)
    return data

def clean_text(text):
    text = re.sub(r'@[A-Za-z0-9]+', '', text)
    text = re.sub(r'#', '', text)
    text = re.sub(r'RT[\s]+', '', text)
    text = re.sub(r'https?:\/\/\s+', '', text)
    text = re.sub(r':+', '', text)
    text = re.sub(r'--+', '', text)
    text = re.sub(r'http', '', text)
    return text.lower()  # Convert to lowercase

def create_wordcloud(data, visualize=True):
    allwords = ' '.join([twts for twts in data["text"]])
    wordcloud = WordCloud(width=2500, height=2000, stopwords=STOPWORDS, background_color="white", random_state=21).generate(allwords)
    
    if visualize:
        plt.figure(1, figsize=(8, 8))
        plt.imshow(wordcloud)
        plt.axis('off')
        plt.show()
    
    return wordcloud

def prepare_data_for_model(data):
    # Keep only Positive and Negative sentiments
    data = data[data['sentiment'].isin(['Positive', 'Negative'])]
    
    # Map sentiment labels to integers
    sentiment_map = {'Negative': 0, 'Positive': 1}
    data['sentiment'] = data['sentiment'].map(sentiment_map)
    
    # Balance the dataset
    positive = data[data.sentiment == 1]
    negative = data[data.sentiment == 0]
    
    # Undersample the majority class
    if len(positive) > len(negative):
        positive = resample(positive, n_samples=len(negative), random_state=42)
    else:
        negative = resample(negative, n_samples=len(positive), random_state=42)
    
    # Combine the balanced dataset
    data_balanced = pd.concat([positive, negative])
    
    return data_balanced

def create_and_evaluate_model(X_train, y_train, X_test, y_test):
    # Create a pipeline with TF-IDF vectorizer and Logistic Regression
    model = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
        ('clf', LogisticRegression(random_state=42, max_iter=1000, C=1.0))
    ])

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Negative', 'Positive']))

    return model

def main(visualize=True):
    # Load and clean data
    data = load_and_clean_data("Sentiment.csv")
    print(data.head())

    # Create word cloud
    wordcloud = create_wordcloud(data, visualize)

    # Prepare data for model
    data_balanced = prepare_data_for_model(data)

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(data_balanced['text'], data_balanced['sentiment'], test_size=0.2, random_state=42)

    # Train and evaluate the model
    model = create_and_evaluate_model(X_train, y_train, X_test, y_test)

    # Test custom text
    custom_tests = [
        "happy, funny, good president",
        "unhappy, deadly, hate, bad president",
        "I love this product, it's amazing!",
        "This movie was terrible, I hated every minute of it.",
        "The service was excellent and the staff was very friendly.",
        "The food was cold and the waiter was rude."
    ]
    
    custom_predictions = model.predict(custom_tests)
    
    sentiment_map = {0: 'Negative', 1: 'Positive'}
    for test, prediction in zip(custom_tests, custom_predictions):
        print(f"Sentiment of '{test}': {sentiment_map[prediction]}")

    return wordcloud, model

if __name__ == "__main__":
    main(visualize=False)  # Set to False to skip visualization