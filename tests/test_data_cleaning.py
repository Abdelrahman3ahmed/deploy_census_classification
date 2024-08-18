import unittest
import pandas as pd
from ..src.data_cleaning import clean_data

class TestDataCleaning(unittest.TestCase):
    def test_clean_data(self):
        clean_data('data/census.csv', 'data/clean_census.csv')
        data = pd.read_csv('data/clean_census.csv')
        self.assertFalse(data.isnull().values.any())

if __name__ == "__main__":
    unittest.main()

