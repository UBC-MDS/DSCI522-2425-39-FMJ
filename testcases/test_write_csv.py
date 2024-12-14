# test_write_csv.py
# author: Forgive Agbesi
# date: 2024-12-11

import pytest
import os
import unittest
import pandas as pd
from src.write_csv import write_csv
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

class TestWriteCSV(unittest.TestCase):
    
    def setUp(self):
        """Create a temporary directory and sample DataFrame for testing."""
        self.test_dir = 'test_dir'
        os.makedirs(self.test_dir, exist_ok=True)
        
        self.sample_data = {
            'Column1': [1, 2, 3],
            'Column2': [4, 5, 6]
        }
        self.sample_df = pd.DataFrame(self.sample_data)

    def test_invalid_filename_extension(self):
        """Test that a ValueError is raised if the filename doesn't end with '.csv'."""
        filename = 'test_file.txt'
        with self.assertRaises(ValueError):
            write_csv(self.sample_df, self.test_dir, filename)    
        
    def tearDown(self):
        """Clean up by removing the test directory and files created."""
        for filename in os.listdir(self.test_dir):
            file_path = os.path.join(self.test_dir, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
        os.rmdir(self.test_dir)

    def test_empty_dataframe(self):
        """Test that a ValueError is raised if the DataFrame is empty."""
        empty_df = pd.DataFrame()
        filename = 'test_file.csv'
        with self.assertRaises(ValueError):
            write_csv(empty_df, self.test_dir, filename)

    def test_write_valid_csv(self):
        """Test that the DataFrame is correctly saved as a CSV file."""
        filename = 'test_file.csv'
        write_csv(self.sample_df, self.test_dir, filename)
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, filename)))
    
    def test_invalid_dataframe(self):
        """Test that a TypeError is raised if the input is not a pandas DataFrame."""
        invalid_df = "invalid"
        filename = 'test_file.csv'
        with self.assertRaises(TypeError):
            write_csv(invalid_df, self.test_dir, filename)

    def test_non_existing_directory(self):
        """Test that a FileNotFoundError is raised if the directory doesn't exist."""
        invalid_dir = 'invalid_dir'
        filename = 'test_file.csv'
        with self.assertRaises(FileNotFoundError):
            write_csv(self.sample_df, invalid_dir, filename)

if __name__ == "__main__":
    unittest.main()
    
    

