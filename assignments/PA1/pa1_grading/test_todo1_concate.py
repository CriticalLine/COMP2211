import sys
import os
import pandas as pd
import numpy as np

class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

if __name__ == "__main__":
    print("Public Accuracy Test Case 1 - Tests function 'concatenate_title_and_text'")
    with HiddenPrints():
        
        # # Create a simple test DataFrame
        # test_df = pd.DataFrame({
        #     'rating': [5.0, 1.0],
        #     'title': ['great game', 'terrible experience'],
        #     'text': ['loved playing it', 'waste of money']
        # })
        
        # Read DF Data from csv
        df_data = pd.read_csv("private.csv")
        expected = pd.read_pickle("todo1_test.pkl")
        
        # turn all the data into string
        df_data['title'] = df_data['title'].astype(str)
        df_data['text'] = df_data['text'].astype(str)
        
        # 1. turn into lower cases
        df_data['title'] = df_data['title'].str.lower()
        df_data['text'] = df_data['text'].str.lower()

        # 2. Remove punctuation, numbers, and special characters from the text
        df_data['title'] = df_data['title'].str.replace(r'[^\w\s]', '', regex=True)
        df_data['title'] = df_data['title'].str.replace(r'\d+', '', regex=True)
        df_data['text'] = df_data['text'].str.replace(r'[^\w\s]', '', regex=True)
        df_data['text'] = df_data['text'].str.replace(r'\d+', '', regex=True)
        
        # Expected output (title repeated twice + text)
        # expected = pd.Series([
        #     'great game great game loved playing it',
        #     'terrible experience terrible experience waste of money'
        # ])
        
        # Student's output
        from pa1_task import concatenate_title_and_text
        student_result = concatenate_title_and_text(df_data)
    
    # Check if results match
    if student_result.equals(expected):
        print(True)
    else:
        print("concatenate_title_and_text does not produce the expected result. Check first row below as reference.")
        print(f"Expected: {expected.iloc[0]}")
        print(f"Got: {student_result.iloc[0]}")