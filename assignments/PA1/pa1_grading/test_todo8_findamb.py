import sys
import os
import numpy as np
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
    print("Public Accuracy Test Case 8 - Tests function 'find_ambiguous_words'")
    with HiddenPrints():
        from pa1_task import find_ambiguous_words
        
        full_review = pd.read_pickle("todo1_test.pkl")
        
        # Create log odds for 5 words
        log_odds = np.load("todo6_test.npy")
        # log_odds = np.array([1.5, -0.8, 0.01, 2.0, -0.05])
        expected = np.load("todo8_test.npy", allow_pickle=True)

        # Vocabulary
        vectorizer = CountVectorizer(min_df=3, max_df=0.8)
        bow_matrix = vectorizer.fit_transform(full_review)
        vocab = np.array(vectorizer.get_feature_names_out())
        
        # Top 2 most ambiguous words (closest to zero log odds)
        ambiguous_words, ambiguous_scores = find_ambiguous_words(log_odds, vocab, top_k=-1)
        # ambiguous_words, ambiguous_scores = find_ambiguous_words(log_odds, vocab, top_k=2)
        
        # Expected results
        expected_words = np.array(expected[0])
        expected_scores = np.asarray(expected[1], dtype=float)
    
    # Check if results match
    
    if_correct = np.array_equal(ambiguous_words[:2], expected_words[:2]) and np.allclose(ambiguous_scores, expected_scores, rtol=1e-5)
    # scores_correct = np.allclose(ambiguous_scores, expected_scores, rtol=1e-5)

    if if_correct:
        print(True)
    else:
        print("find_ambiguous_words does not produce the expected result. check first 10 elements below as reference.")
        print(f"Expected words: {expected_words[:10]}, got: {ambiguous_words[:10]}")