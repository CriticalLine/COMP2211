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
    print("Public Efficiency Test Case 5 - Tests function 'compute_log_odds'")
    try:
        with HiddenPrints():
            from pa1_task import compute_log_odds
            
            # Create a large word probability matrix (100,000 words)
            word_probs = np.random.random((100000, 2)) * 0.5 + 0.1  # values between 0.1 and 0.6
            
            # Time the computation
            start_time = time.time()
            log_odds = compute_log_odds(word_probs)
            end_time = time.time()
            print(log_odds)
            execution_time = end_time - start_time
            
            # Reasonable threshold for a vectorized implementation
            time_threshold = 0.002  # seconds 0.05
        
        if execution_time <= time_threshold:
            print(True)
            # print("Execution time:", execution_time, "seconds (threshold:", time_threshold, "seconds)")
        else:
            print(False)
            print(f"compute_log_odds is too slow. Execution time: {execution_time:.6f} seconds (threshold: {time_threshold} seconds)")
            print("Make sure you're using numpy's vectorized operations.")
    
    except Exception as e:
        print(False)
        print(f"Error during execution: {e}")