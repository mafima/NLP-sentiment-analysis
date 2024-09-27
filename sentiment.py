import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from nltk.classify import SklearnClassifier
from sklearn.model_selection import train_test_split
from textblob import TextBlob
from wordcloud import WordCloud, STOPWORDS
import re

# Sentiment Analysis 
# train on text using nltk and predict sentiment (good or bad)
# analyzing the sentiment of a given text

# data used: kaggle get data: download csv from https://www.kaggle.com/datasets/crowdflower/first-gop-debate-twitter-sentiment?resource=download

def load_and_clean_data(file_path):
    data = pd.read_csv(file_path)
    data = data[["text", "sentiment"]]
    data["text"] = data["text"].apply(clean_text)
    data["subjectivity"] = data["text"].apply(get_subjectivity)
    data["polarity"] = data["text"].apply(get_polarity)
    return data

def clean_text(text):
    text = re.sub(r'@[A-Za-z0-9]+', '', text) # remove @ from tweet
    text = re.sub(r'#', '', text)
    text = re.sub(r'RT[\s]+', '', text)
    text = re.sub(r'https?:\/\/\s+', '', text) # links are not helpful
    text = re.sub(r':+', '', text)
    text = re.sub(r'--+', '', text)
    text = re.sub(r'http', '', text)
    return text

def get_subjectivity(text):
    return TextBlob(text).sentiment.subjectivity

def get_polarity(text):
    return TextBlob(text).sentiment.polarity

def create_wordcloud(data, visualize=True):
    allwords = ' '.join([twts for twts in data["text"]])
    wordcloud = WordCloud(width=2500, height=2000, stopwords=STOPWORDS, background_color="white", random_state=21).generate(allwords)
    
    if visualize:
        plt.figure(1, figsize=(8, 8))
        plt.imshow(wordcloud)
        plt.axis('off')
        plt.show()
    
    return wordcloud

def prepare_training_data(train):
    text = []
    stopwords_set = set(stopwords.words("english"))
    for _, row in train.iterrows():
        words_filtered = [e.lower() for e in row.text.split() if len(e) >= 3]
        words_cleaned = [word for word in words_filtered
            if 'http' not in word
            and not word.startswith('@')
            and not word.startswith('#')
            and word != 'RT']
        words_without_stopwords = [word for word in words_cleaned if not word in stopwords_set]
        text.append((words_without_stopwords, row.sentiment))
    return text

def get_word_features(text):
    all_words = []
    for (words, sentiment) in text:
        all_words.extend(words)
    wordlist = nltk.FreqDist(all_words)
    return wordlist.keys()

def extract_features(document):
    document_words = set(document)
    features = {}
    for word in word_features:
        features['contains(%s)' % word] = (word in document_words)
    return features

def train_classifier(training_set):
    return nltk.NaiveBayesClassifier.train(training_set)

def evaluate_classifier(classifier, test_neg, test_pos):
    neg_cnt = sum(1 for obj in test_neg if classifier.classify(extract_features(obj.split())) == 'Negative')
    pos_cnt = sum(1 for obj in test_pos if classifier.classify(extract_features(obj.split())) == 'Positive')
    print('[Negative]: %s/%s ' % (len(test_neg), neg_cnt))
    print('[Positive]: %s/%s ' % (len(test_pos), pos_cnt))

def main(visualize=False):
    # Load and clean data
    data = load_and_clean_data("Sentiment.csv")
    print(data.head())

    # Create word cloud
    wordcloud = create_wordcloud(data, visualize)

    # Split data into train and test sets
    train, test = train_test_split(data, test_size=0.1)
    train = train[train.sentiment != "Neutral"]

    # Prepare training data
    text = prepare_training_data(train)

    # Extract word features
    global word_features
    word_features = get_word_features(text)

    # Train classifier
    training_set = nltk.classify.apply_features(extract_features, text)
    classifier = train_classifier(training_set)

    # Evaluate classifier
    test_neg = test[test['sentiment'] == 'Negative']['text']
    test_pos = test[test['sentiment'] == 'Positive']['text']
    evaluate_classifier(classifier, test_neg, test_pos)

    # Test custom text
    custom_test = "unhappy, deadly, hate, bad president"
    print("neg "+classifier.classify(extract_features(custom_test.split())))

    return wordcloud, classifier

if __name__ == "__main__":
    main(visualize=False)  # Set to False to skip visualization