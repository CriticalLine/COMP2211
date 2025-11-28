import sys
import os
import numpy as np
from scipy.sparse import csr_matrix
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

if __name__ == '__main__':
    print("Public Accuracy Test Case 5 - Tests function 'predict' in NaiveBayesClassifier")
    with HiddenPrints():
        from pa1_task import NaiveBayesClassifier
        
        # Initialize the classifier with pre-computed probabilities
        # nb_classifier = NaiveBayesClassifier()
        # nb_classifier.class_probs = np.array([0.4, 0.6])  # P(c_0) = 0.4, P(c_1) = 0.6
        # nb_classifier.word_probs = np.array([
        #     [0.1, 0.3],  # P(w_1|c_0) = 0.1, P(w_1|c_1) = 0.3
        #     [0.2, 0.1],  # P(w_2|c_0) = 0.2, P(w_2|c_1) = 0.1
        #     [0.3, 0.2]   # P(w_3|c_0) = 0.3, P(w_3|c_1) = 0.2
        # ])
        full_review = pd.read_pickle("todo1_test.pkl")
        labels = np.load("todo2_test.npy")
        
        vectorizer = CountVectorizer(min_df=3, max_df=0.8)
        bow_matrix = vectorizer.fit_transform(full_review)
        
        train_bow, test_bow, train_labels, test_labels  = train_test_split(bow_matrix, labels, test_size=0.2, random_state=42)
        # Test data: 3 documents with counts of the 3 words
        # X_test = csr_matrix([
        #     [2, 1, 0],  # doc 1: 2 of word1, 1 of word2, 0 of word3 -> should be class 1
        #     [0, 3, 1],  # doc 2: 0 of word1, 3 of word2, 1 of word3 -> should be class 0
        #     [1, 1, 1]   # doc 3: 1 each of word1, word2, word3 -> could go either way
        # ])
        nb_classifier = NaiveBayesClassifier()
        nb_classifier.class_probs = np.array([0.39665653, 0.60334347])
        nb_classifier.word_probs = np.load("todo4_test.npy")
        # Expected predictions (calculated by hand)
        expected = np.load("todo5_test.npy")
        # expected = np.array([1, 0, 1])
        
        # Student's prediction
        student_result = nb_classifier.predict(test_bow)
    
    # Check if results match
    if np.array_equal(student_result, expected):
        print(True)
    else:
        print("predict does not produce the expected result. Check first 10 elements below as reference.")
        print(f"Expected: {expected[:10]}")
        print(f"Got: {student_result[:10]}")