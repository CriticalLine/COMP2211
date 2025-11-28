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
    print("Public Accuracy Test Case 4 - Tests function 'compute_word_prob' in NaiveBayesClassifier")
    with HiddenPrints():
        from pa1_task import NaiveBayesClassifier
        
        # Create a simple bag-of-words matrix (4 documents, 3 words)
        full_review = pd.read_pickle("todo1_test.pkl")
        labels = np.load("todo2_test.npy")
        # X = csr_matrix([
        #     [1, 2, 0],  # doc 1: 1 of word1, 2 of word2, 0 of word3
        #     [0, 1, 3],  # doc 2: 0 of word1, 1 of word2, 3 of word3
        #     [2, 0, 1],  # doc 3: 2 of word1, 0 of word2, 1 of word3
        #     [1, 1, 1]   # doc 4: 1 each of word1, word2, word3
        # ])
        
        # Binary labels: 0 for negative, 1 for positive
        vectorizer = CountVectorizer(min_df=3, max_df=0.8)
        bow_matrix = vectorizer.fit_transform(full_review)
        
        train_bow, test_bow, train_labels, test_labels  = train_test_split(bow_matrix, labels, test_size=0.2, random_state=42)
        # y = np.array([0, 0, 1, 1])
        
        # Initialize the classifier and set up necessary attributes
        nb_classifier = NaiveBayesClassifier()
        nb_classifier.delta = np.load("todo4_delta.npy")
        nb_classifier.vocabulary_size = 1975
        # nb_classifier.vocabulary_size = 3
        # nb_classifier.delta = np.array([[1, 0], [1, 0], [0, 1], [0, 1]])
        
        # Expected output: P(w_k|c_j) matrix
        # For class 0:
        # P(w_1|c_0) = (1 + 1)/(3 + 8) = 2/11
        # P(w_2|c_0) = (1 + 3)/(3 + 8) = 4/11
        # P(w_3|c_0) = (1 + 3)/(3 + 8) = 4/11
        # For class 1:
        # P(w_1|c_1) = (1 + 3)/(3 + 6) = 4/9
        # P(w_2|c_1) = (1 + 1)/(3 + 6) = 2/9
        # P(w_3|c_1) = (1 + 2)/(3 + 6) = 3/9
        expected = np.load("todo4_test.npy")
        # expected = np.array([
        #     [2/11, 4/9],
        #     [4/11, 2/9],
        #     [4/11, 3/9]
        # ])
        
        # Call the method directly
        student_result=nb_classifier.compute_word_prob(train_bow, train_labels)
        # student_result = nb_classifier.compute_word_prob(X, y)
    
    # Check if results match within a small tolerance
    if np.allclose(student_result, expected, rtol=1e-5):
        print(True)
    else:
        print("compute_word_prob does not produce the expected result.")
        print(f"The error is more than the tolerance of 1e-5.")
        # print(f"Expected: {expected}")
        # print(f"Got: {student_result}")