import sys
import os
import numpy as np
import time
from scipy.sparse import csr_matrix

class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

if __name__ == '__main__':
    print("Public Efficiency Test Case 4 - Tests function 'predict' in NaiveBayesClassifier")
    try:
        with HiddenPrints():
            from pa1_task import NaiveBayesClassifier
            
            # Initialize the classifier with pre-computed probabilities
            nb_classifier = NaiveBayesClassifier()
            nb_classifier.class_probs = np.array([0.4, 0.6])
            
            # Create word probabilities for 100 words
            nb_classifier.word_probs = np.random.random((100, 2))
            
            # Create a large test set (1000 documents)
            # Each document has sparse representation (most words don't appear)
            row_indices = []
            col_indices = []
            data_values = []
            
            for i in range(1000):
                # Each document has about 10 random words
                word_indices = np.random.choice(100, size=10, replace=False)
                word_counts = np.random.randint(1, 5, size=10)  # 1-4 occurrences of each word
                
                for j, word_idx in enumerate(word_indices):
                    row_indices.append(i)
                    col_indices.append(word_idx)
                    data_values.append(word_counts[j])
            
            X_test_large = csr_matrix((data_values, (row_indices, col_indices)), shape=(1000, 100))
            
            # Time the prediction
            start_time = time.time()
            predicts = nb_classifier.predict(X_test_large)
            end_time = time.time()
            print(predicts)
            execution_time = end_time - start_time
            
            # Reasonable threshold for a vectorized implementation
            time_threshold = 0.002  # seconds 0.1
        
        if execution_time <= time_threshold:
            print(True)
            # print("Execution time:", execution_time, "seconds (threshold:", time_threshold, "seconds)")
        else:
            print(False)
            print(f"predict is too slow. Execution time: {execution_time:.6f} seconds (threshold: {time_threshold} seconds)")
            print("Consider using vectorized operations instead of loops for better performance.")
    
    except Exception as e:
        print(False)
        print(f"Error during execution: {e}")