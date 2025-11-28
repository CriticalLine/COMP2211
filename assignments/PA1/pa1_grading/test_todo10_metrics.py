import sys
import os
import numpy as np

class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

if __name__ == '__main__':
    print("Public Accuracy Test Case 10 - Tests function 'calculate_metrics'")
    with HiddenPrints():
        from pa1_task import calculate_metrics
        # Confusion matrix: [[TN, FP],[FN, TP]]
        conf_matrix = np.array([[66, 11],
                             [15, 72]])
        # conf_matrix = np.array([[50, 10], [5, 35]])
        # Precision: TP/(TP+FP) = 35/(35+10) = 0.7778
        # Recall: TP/(TP+FN) = 35/(35+5) = 0.875
        # F1: 2*P*R/(P+R) = 2*0.7778*0.875/(0.7778+0.875) = 0.8228
        expected = (0.8675, 0.8276, 0.8471)
        precision, recall, f1 = calculate_metrics(conf_matrix)
    # Allow small tolerance due to float division
    if np.allclose([precision, recall, f1], expected, atol=1e-3):
        print(True)
    else:
        print("calculate_metrics does not produce expected values.")
        if precision != expected[0]:
            print("incorrect precision")
        if recall != expected[1]:
            print("incorrect recall")
        if f1 != expected[2]:
            print("incorrect f1")
        # print(f"Expected: precision {expected[0]:.4f}, recall {expected[1]:.4f}, f1 {expected[2]:.4f}")
        # print(f"Got: precision {precision:.4f}, recall {recall:.4f}, f1 {f1:.4f}")