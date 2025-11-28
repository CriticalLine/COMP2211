import sys
import os
import numpy as np
import time
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
    print("Public Efficiency Test Case 3 - Tests function 'compute_word_prob' in NaiveBayesClassifier")
    try:
        with HiddenPrints():
            from pa1_task import NaiveBayesClassifier
            
            # Initialize the classifier with pre-computed probabilities
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
            
            # Time the prediction
            start_time = time.time()
            word_probs = nb_classifier.compute_word_prob(train_bow, train_labels)
            end_time = time.time()
            print(word_probs)
            execution_time = end_time - start_time
            
            # Reasonable threshold for a vectorized implementation
            time_threshold = 0.05  # seconds 0.1
        
        if execution_time <= time_threshold:
            print(True)
            # print("Execution time:", execution_time, "seconds (threshold:", time_threshold, "seconds)")
        else:
            print(False)
            print(f"compute_word_prob is too slow. Execution time: {execution_time:.6f} seconds (threshold: {time_threshold} seconds)")
            print("Consider using vectorized operations instead of loops for better performance.")
    
    except Exception as e:
        print(False)
        print(f"Error during execution: {e}")