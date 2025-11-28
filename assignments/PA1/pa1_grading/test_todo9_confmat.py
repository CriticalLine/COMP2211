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
    print("Public Accuracy Test Case 9 - Tests function 'compute_confusion_matrix'")
    with HiddenPrints():
        # Import student's function
        from pa1_task import compute_confusion_matrix
        
        full_review = pd.read_pickle("todo1_test.pkl")
        labels = np.load("todo2_test.npy")
        # Test data: ground truth and prediction
        vectorizer = CountVectorizer(min_df=3, max_df=0.8)
        bow_matrix = vectorizer.fit_transform(full_review)
        train_bow, test_bow, train_labels, test_labels  = train_test_split(bow_matrix, labels, test_size=0.2, random_state=42)
        
        y_pred = np.load("todo5_test.npy")
        y_true = test_labels
        # y_true = np.array([0, 0, 1, 1, 0, 1, 1, 0])
        # y_pred = np.array([0, 1, 1, 0, 0, 1, 0, 1])
        # Expected confusion matrix:
        # True negatives: y_true=0, y_pred=0 -> indices 0,4 => 2
        # False positives: y_true=0, y_pred=1 -> indices 1,7 => 2
        # False negatives: y_true=1, y_pred=0 -> indices 3,6 => 2
        # True positives: y_true=1, y_pred=1 -> indices 2,5 => 2
        expected = np.array([[66, 11],
                             [15, 72]])
        # expected = np.array([[2, 2], [2, 2]])
        student = compute_confusion_matrix(y_true, y_pred)
    if np.array_equal(student, expected):
        print(True)
    else:
        print("compute_confusion_matrix does not produce the expected result.")
        if student[0][0] != expected[0][0]:
            print("True Negatives not pass.")
        if student[0][1] != expected[0][1]:
            print("False Positives not pass.")
        if student[1][0] != expected[1][0]:
            print("False Negatives not pass.")
        if student[1][1] != expected[1][1]:
            print("True Positives not pass.")
        # print(f"Expected:\n{expected}\nGot:\n{student}")