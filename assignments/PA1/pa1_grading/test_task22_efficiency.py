import sys
import os
import numpy as np
import time
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

if __name__ == '__main__':
    print("Public Efficiency Test Case 6 - Tests function 'find_top_words'")
    try:
        with HiddenPrints():
            from pa1_task import find_top_words
            
            # Create a large word probability matrix (100,000 words)
            full_review = pd.read_pickle("todo1_test.pkl")
        
            # Create log odds for 5 words
            log_odds = np.load("todo6_test.npy")
            # log_odds = np.array([1.5, -0.8, 0.2, 2.0, -1.2])
            expected = np.load("todo7_test.npy", allow_pickle=True)

            # Vocabulary
            vectorizer = CountVectorizer(min_df=3, max_df=0.8)
            bow_matrix = vectorizer.fit_transform(full_review)
            vocab = np.array(vectorizer.get_feature_names_out())
            # vocab = np.array(['excellent', 'terrible', 'okay', 'amazing', 'horrible'])
            
            # Test for positive indicators (top 2)
            # pos_words, pos_scores, pos_idx = find_top_words(log_odds, vocab, top_k=-1, indicator=True)
            
            # Time the computation
            start_time = time.time()
            pos_words, pos_scores, pos_idx = find_top_words(log_odds, vocab, top_k=-1, indicator=True)
            pos_words, pos_scores, pos_idx = find_top_words(log_odds, vocab, top_k=-1, indicator=False)
            end_time = time.time()
            print(pos_words, pos_scores, pos_idx, pos_words, pos_scores, pos_idx)
            execution_time = end_time - start_time
            
            # Reasonable threshold for a vectorized implementation
            time_threshold = 0.0005  # seconds 0.2
        
        if execution_time <= time_threshold:
            print(True)
            # print("Execution time:", execution_time, "seconds (threshold:", time_threshold, "seconds)")
        else:
            print(False)
            print(f"find_top_words is too slow. Execution time: {execution_time:.6f} seconds (threshold: {time_threshold} seconds)")
            print("Make sure you're using numpy's vectorized operations.")
    
    except Exception as e:
        print(False)
        print(f"Error during execution: {e}")