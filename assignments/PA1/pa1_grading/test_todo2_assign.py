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
    print("Public Accuracy Test Case 2 - Tests function 'label_assign'")
    with HiddenPrints():
        from pa1_task import label_assign

        df_data = pd.read_csv("private.csv")
        expected = np.load("todo2_test.npy")
        # Test data with various ratings
        # test_data = np.array([
        #     [1.0, 'bad', 'review', 1],
        #     [2.0, 'poor', 'review', 2],
        #     [4.0, 'good', 'review', 3],
        #     [5.0, 'excellent', 'review', 4]
        # ])
        data = df_data.to_numpy()
        student_result = label_assign(data)
        # Expected output: 0 for ratings 1.0 and 2.0, 1 for ratings 4.0 and 5.0
        # expected = np.array([0, 0, 1, 1])
        
        # Student's output
        # student_result = label_assign(test_data)

    # Check if results match
    if np.array_equal(student_result, expected):
        print(True)
    else:
        print("label_assign does not produce the expected result. Check first 10 elements below as reference.")
        print(f"Expected: {expected[:10]}")
        print(f"Got: {student_result[:10]}")