import sys
import os
import numpy as np
import time

class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

if __name__ == '__main__':
    print("Public Efficiency Test Case 8 - Tests function 'compute_confusion_matrix'")
    try:
        with HiddenPrints():
            from pa1_task import compute_confusion_matrix
            # Large arrays (1 million samples)
            rng = np.random.default_rng(42)
            y_true = rng.integers(0, 2, size=1_000_000)
            # Slightly noisy prediction to ensure both classes present
            y_pred = y_true.copy()
            flip_indices = rng.choice(1_000_000, size=100_000, replace=False)
            y_pred[flip_indices] = 1 - y_pred[flip_indices]
            start = time.time()
            result = compute_confusion_matrix(y_true, y_pred)
            end = time.time()
            print(result)
        elapsed = end-start
        time_threshold = 0.01  # seconds 0.3
        if elapsed <= time_threshold:
            print(True)
            # print("Execution time:", elapsed, "seconds (threshold:", time_threshold, "seconds)")
        else:
            print(False)
            print(f"Execution time: {elapsed:.3f} seconds (threshold: {time_threshold} seconds)")
            print("compute_confusion_matrix is too slow. Make sure you're using numpy's vectorized operations instead of for-loops.")
    except Exception as e:
        print(False)
        print(f"Error during execution: {e}")