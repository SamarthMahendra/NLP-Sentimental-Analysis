# models.py

from sentiment_data import *
from utils import *
import numpy as np
import math
import random

from collections import Counter

class FeatureExtractor(object):
    """
    Feature extraction base type. Takes a sentence and returns an indexed list of features.
    """
    def get_indexer(self):
        raise Exception("Don't call me, call my subclasses")

    def extract_features(self, sentence: List[str], add_to_indexer: bool=False) -> Counter:
        """
        Extract features from a sentence represented as a list of words. Includes a flag add_to_indexer to
        :param sentence: words in the example to featurize
        :param add_to_indexer: True if we should grow the dimensionality of the featurizer if new features are encountered.
        At test time, any unseen features should be discarded, but at train time, we probably want to keep growing it.
        :return: A feature vector. We suggest using a Counter[int], which can encode a sparse feature vector (only
        a few indices have nonzero value) in essentially the same way as a map. However, you can use whatever data
        structure you prefer, since this does not interact with the framework code.
        """
        raise Exception("Don't call me, call my subclasses")


class UnigramFeatureExtractor(FeatureExtractor):
    """
    Extracts unigram bag-of-words features from a sentence. It's up to you to decide how you want to handle counts
    and any additional preprocessing you want to do.
    """
    def __init__(self, indexer: Indexer):
        self.indexer = indexer

    def get_indexer(self):
        return self.indexer

    def extract_features(self, sentence: List[str], add_to_indexer: bool=False) -> Counter:
        features = Counter()
        # lower case and remove punctuation
        sentence = [word.lower() for word in sentence if word.isalnum()]

        #
        for word in sentence:
            # Decide whether to use counts or binary presence
            # Here, we'll use binary features (0/1 for absence/presence)
            if add_to_indexer:
                idx = self.indexer.add_and_get_index(word)
            else:
                idx = self.indexer.index_of(word)
            if idx != -1:
                features[idx] = 1  # Set to 1 for presence
        return features




class BigramFeatureExtractor(FeatureExtractor):
    """
    Bigram feature extractor analogous to the unigram one.
    """
    def __init__(self, indexer: Indexer):
        self.indexer = indexer

    def get_indexer(self):
        return self.indexer

    def extract_features(self, sentence: List[str], add_to_indexer: bool=False) -> Counter:

        features = Counter()
        # lower case and remove punctuation
        sentence = [word.lower() for word in sentence if word.isalnum()]
        # remove stopwords
        from nltk.corpus import stopwords
        stop_words = set(stopwords.words('english'))
        sentence = [word for word in sentence if word not in stop_words]
        for i in range(len(sentence) - 1):
            bigram = sentence[i] + ' ' + sentence[i + 1]
            if add_to_indexer:
                idx = self.indexer.add_and_get_index(bigram)
            else:
                idx = self.indexer.index_of(bigram)
            if idx != -1:
                features[idx] += 1
        return features


class BetterFeatureExtractor(FeatureExtractor):
    """
    Better feature extractor...try whatever you can think of!
    """
    def __init__(self, indexer: Indexer):
        self.indexer = indexer

    def get_indexer(self):
        return self.indexer

    def extract_features(self, sentence: List[str], add_to_indexer: bool=False) -> Counter:
        features = Counter()
        # lower case and remove punctuation
        sentence = [word.lower() for word in sentence if word.isalnum()]
        # remove stopwords
        from nltk.corpus import stopwords
        stop_words = set(stopwords.words('english'))
        sentence = [word for word in sentence if word not in stop_words]
        # Unigram features
        for word in sentence:
            if add_to_indexer:
                idx = self.indexer.add_and_get_index(word)
            else:
                idx = self.indexer.index_of(word)
            if idx != -1:
                features[idx] += 1
            # character tr-gram features
            for i in range(len(word) - 2):
                trigram = word[i:i+3]
                if add_to_indexer:
                    idx = self.indexer.add_and_get_index(trigram)
                else:
                    idx = self.indexer.index_of(trigram)
                if idx != -1:
                    features[idx] += 1

        # Bigram features
        for i in range(len(sentence) - 1):
            bigram = sentence[i] + ' ' + sentence[i + 1]
            if add_to_indexer:
                idx = self.indexer.add_and_get_index(bigram)
            else:
                idx = self.indexer.index_of(bigram)
            if idx != -1:
                features[idx] += 1

        return features




class SentimentClassifier(object):
    """
    Sentiment classifier base type
    """
    def predict(self, sentence: List[str]) -> int:
        """
        :param sentence: words (List[str]) in the sentence to classify
        :return: Either 0 for negative class or 1 for positive class
        """
        raise Exception("Don't call me, call my subclasses")


class TrivialSentimentClassifier(SentimentClassifier):
    """
    Sentiment classifier that always predicts the positive class.
    """
    def predict(self, sentence: List[str]) -> int:
        return 1


class PerceptronClassifier(SentimentClassifier):
    """
    Implement this class -- you should at least have init() and implement the predict method from the SentimentClassifier
    superclass. Hint: you'll probably need this class to wrap both the weight vector and featurizer -- feel free to
    modify the constructor to pass these in.
    """
    def __init__(self, weights: np.ndarray, feat_extractor: FeatureExtractor):
        self.weights = weights
        self.feat_extractor = feat_extractor


    def predict(self, sentence: List[str]) -> int:
        features = self.feat_extractor.extract_features(sentence, add_to_indexer=False)
        score = 0.0
        for idx, value in features.items():
            score += self.weights[idx] * value
        # Return 1 if score >= 0, else 0
        return 1 if score >= 0 else 0




class LogisticRegressionClassifier(SentimentClassifier):
    """
    Implement this class -- you should at least have init() and implement the predict method from the SentimentClassifier
    superclass. Hint: you'll probably need this class to wrap both the weight vector and featurizer -- feel free to
    modify the constructor to pass these in.
    """
    def __init__(self, weights: np.ndarray, feat_extractor: FeatureExtractor):
        self.weights = weights
        self.feat_extractor = feat_extractor

    def predict(self, sentence: List[str]) -> int:
        features = self.feat_extractor.extract_features(sentence, add_to_indexer=False)
        score = 0.0
        for idx, value in features.items():
            score += self.weights[idx] * value
        probability = 1 / (1 + math.exp(-score))
        return 1 if probability >= 0.5 else 0



def train_perceptron(train_exs: List[SentimentExample], feat_extractor: FeatureExtractor, num_epochs: int = 20) -> PerceptronClassifier:
    """
    Train a classifier with the perceptron algorithm.
    :param train_exs: training set, List of SentimentExample objects
    :param feat_extractor: feature extractor to use
    :param num_epochs: number of epochs to train for
    :return: trained PerceptronClassifier model
    """
    indexer = feat_extractor.get_indexer()
    # First pass: build the feature indexer
    for ex in train_exs:
        feat_extractor.extract_features(ex.words, add_to_indexer=True)

    # Initialize weights (numpy array)
    weights = np.zeros(len(indexer))

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        # Shuffle the training data at the beginning of each epoch
        random.shuffle(train_exs)
        for ex in train_exs:
            # Extract features without adding to indexer
            features = feat_extractor.extract_features(ex.words, add_to_indexer=False)
            # Compute the score
            score = 0.0
            for idx, value in features.items():
                score += weights[idx] * value
            # Predict label
            prediction = 1 if score >= 0 else 0
            # Update weights if prediction is incorrect
            if prediction != ex.label:
                # Update rule: weights = weights + learning_rate * (true_label - predicted_label) * features
                # Since learning_rate is 1 and labels are 0 or 1, the update simplifies
                for idx, value in features.items():
                    # For perceptron, the update is +/- feature value
                    if ex.label == 1:
                        weights[idx] += value  # Promote features for positive class
                    else:
                        weights[idx] -= value  # Demote features for negative class

    return PerceptronClassifier(weights, feat_extractor)


def train_logistic_regression(train_exs: List[SentimentExample], feat_extractor: FeatureExtractor, num_epochs: int = 20, learning_rate: float = 0.1) -> LogisticRegressionClassifier:
    """
    Train a logistic regression model.
    :param train_exs: training set, List of SentimentExample objects
    :param feat_extractor: feature extractor to use
    :return: trained LogisticRegressionClassifier model
    """
    indexer = feat_extractor.get_indexer()
    # First pass: build the feature indexer
    for ex in train_exs:
        feat_extractor.extract_features(ex.words, add_to_indexer=True)

    # Initialize weights (numpy array)
    weights = np.zeros(len(indexer))

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        # Shuffle the training data at the beginning of each epoch
        random.shuffle(train_exs)
        for ex in train_exs:
            # Extract features without adding to indexer
            features = feat_extractor.extract_features(ex.words, add_to_indexer=False)
            # Compute the score
            score = 0.0
            for idx, value in features.items():
                score += weights[idx] * value
            # Compute the prediction probability using sigmoid function
            probability = 1 / (1 + math.exp(-score))
            # Compute the error (difference between true label and predicted probability)
            error = ex.label - probability
            # Update weights using gradient descent
            for idx, value in features.items():
                weights[idx] += learning_rate * error * value

    return LogisticRegressionClassifier(weights, feat_extractor)


def train_model(args, train_exs: List[SentimentExample], dev_exs: List[SentimentExample]) -> SentimentClassifier:
    """
    Main entry point for your modifications. Trains and returns one of several models depending on the args
    passed in from the main method. You may modify this function, but probably will not need to.
    :param args: args bundle from sentiment_classifier.py
    :param train_exs: training set, List of SentimentExample objects
    :param dev_exs: dev set, List of SentimentExample objects. You can use this for validation throughout the training
    process, but you should *not* directly train on this data.
    :return: trained SentimentClassifier model, of whichever type is specified
    """
    # Initialize feature extractor
    if args.model == "TRIVIAL":
        feat_extractor = None
    elif args.feats == "UNIGRAM":
        # Add additional preprocessing code here
        feat_extractor = UnigramFeatureExtractor(Indexer())
    elif args.feats == "BIGRAM":
        # Add additional preprocessing code here
        feat_extractor = BigramFeatureExtractor(Indexer())
    elif args.feats == "BETTER":
        # Add additional preprocessing code here
        feat_extractor = BetterFeatureExtractor(Indexer())
    else:
        raise Exception("Pass in UNIGRAM, BIGRAM, or BETTER to run the appropriate system")

    # Train the model
    if args.model == "TRIVIAL":
        model = TrivialSentimentClassifier()
    elif args.model == "PERCEPTRON":
        model = train_perceptron(train_exs, feat_extractor, num_epochs=4)
    elif args.model == "LR":
        model = train_logistic_regression(train_exs, feat_extractor, num_epochs=5)
    else:
        raise Exception("Pass in TRIVIAL, PERCEPTRON, or LR to run the appropriate system")
    return model