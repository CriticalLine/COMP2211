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
    print("Public Accuracy Test Case 7 - Tests function 'find_top_words'")
    with HiddenPrints():
        from pa1_task import find_top_words
        
        full_review = pd.read_pickle("todo1_test.pkl")
        
        # Create log odds for 5 words
        log_odds = np.load("todo6_test.npy")
        # log_odds = np.array([1.5, -0.8, 0.2, 2.0, -1.2])
        expected = np.load("todo7_test.npy",allow_pickle=True)
        
        # Vocabulary
        vectorizer = CountVectorizer(min_df=3, max_df=0.8)
        bow_matrix = vectorizer.fit_transform(full_review)
        vocab = np.array(vectorizer.get_feature_names_out())
        # vocab = np.array(['excellent', 'terrible', 'okay', 'amazing', 'horrible'])
        
        # Test for positive indicators (top 2)
        pos_words, pos_scores, pos_idx = find_top_words(log_odds, vocab, top_k=-1, indicator=True)
        
        # Expected results for positive
        expected_pos_words = np.array(expected[0])
        expected_pos_scores = np.asarray(expected[1], dtype=float)
        expected_pos_idx = np.array(expected[2])
        
        # Test for negative indicators (top 2)
        neg_words, neg_scores, neg_idx = find_top_words(log_odds, vocab, top_k=-1, indicator=False)
        
        # Expected results for negative
        expected_neg_words = np.array(expected[3])
        expected_neg_scores = np.asarray(expected[4], dtype=float)
        expected_neg_idx = np.array(expected[5])

    # Check positive results
    pos_correct = np.array_equal(pos_words[:5], expected_pos_words[:5]) and np.allclose(pos_scores, expected_pos_scores, atol=1e-5) and np.array_equal(pos_idx[:5], expected_pos_idx[:5])

    # Check negative results
    neg_correct = np.array_equal(neg_words[:5], expected_neg_words[:5]) and np.allclose(neg_scores, expected_neg_scores, atol=1e-5) and np.array_equal(neg_idx[:5], expected_neg_idx[:5])

    if pos_correct and neg_correct:
        print(True)
    else:
        print("find_top_words does not produce the expected result. Check first 10 elements below as reference.")
        if not pos_correct:
            print("Issue with positive indicators:")
            print(f"Expected words: {expected_pos_words[:10]}, got: {pos_words[:10]}")
            print(f"Expected scores: {expected_pos_scores[:10]}, got: {pos_scores[:10]}")
        if not neg_correct:
            print("Issue with negative indicators:")
            print(f"Expected words: {expected_neg_words[:10]}, got: {neg_words[:10]}")
            print(f"Expected scores: {expected_neg_scores[:10]}, got: {neg_scores[:10]}")