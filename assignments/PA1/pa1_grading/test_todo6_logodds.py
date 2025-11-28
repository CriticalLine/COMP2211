import sys
import os
import numpy as np
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
    print("Public Accuracy Test Case 6 - Tests function 'compute_log_odds'")
    with HiddenPrints():
        from pa1_task import compute_log_odds
        
        # Create word probabilities for 3 words
        # format: [P(w|c=0), P(w|c=1)]
        # word_probs = np.array([
        #     [0.1, 0.3],  # word 1 is more likely in positive class
        #     [0.2, 0.1],  # word 2 is more likely in negative class
        #     [0.3, 0.3]   # word 3 is equally likely in both classes
        # ])
        
        # Expected log odds: log(P(w|c=1)/P(w|c=0))
        expected = np.load("todo6_test.npy")
        # expected = np.array([
        #     np.log(0.3/0.1),  # positive log odds (positive word)
        #     np.log(0.1/0.2),  # negative log odds (negative word)
        #     np.log(0.3/0.3)   # zero log odds (neutral word)
        # ])
        word_probs = np.load("todo4_test.npy")
        # Student's output
        student_result = compute_log_odds(word_probs)
    
    # Check if results match within a small tolerance
    if np.allclose(student_result, expected, rtol=1e-5):
        print(True)
    else:
        print("compute_log_odds does not produce the expected result. Check first 10 elements below as reference.")
        print(f"Expected: {expected[:10]}")
        print(f"Got: {student_result[:10]}")