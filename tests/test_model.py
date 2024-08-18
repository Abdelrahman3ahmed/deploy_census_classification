import unittest
import joblib
import pandas as pd

class TestModel(unittest.TestCase):
    def setUp(self):
        # Load the test data
        self.test_data = pd.read_csv('data/clean_census.csv')
        
        # Load the label encoders
        self.label_encoders = joblib.load('models/label_encoders.joblib')
        
        # Encode categorical features in test data
        for column, le in self.label_encoders.items():
            if column in self.test_data.columns:
                self.test_data[column] = le.transform(self.test_data[column])
        
        # Convert categorical features to numeric using one-hot encoding
        self.test_data = pd.get_dummies(self.test_data)
        
        # Load the training columns for consistent feature sets
        train_columns = joblib.load('models/train_columns.joblib')
        
        # Ensure the same columns are used in testing as in training
        self.test_data = self.test_data.reindex(columns=train_columns, fill_value=0)
        
        # Ensure 'salary' is present and reindex it
        if 'salary' not in self.test_data.columns:
            # Add the 'salary' column back if missing
            self.test_data['salary'] = 0
        
        # Separate features and target variable
        self.X_test = self.test_data.drop(columns=['salary'])
        self.y_test = self.test_data['salary']
        
        # Load the model
        self.model = joblib.load('models/model.joblib')

    def test_model_accuracy(self):
        # Ensure X_test is not empty
        if self.X_test.empty:
            self.fail("Test features are empty")

        print(f"X_test sample:\n{self.X_test.head()}")
        print(f"y_test sample:\n{self.y_test.head()}")

        predictions = self.model.predict(self.X_test)
        accuracy = (predictions == self.y_test).mean()
        print(f"Predictions sample:\n{predictions[:10]}")
        print(f"Accuracy: {accuracy:.2f}")

        self.assertGreater(accuracy, 0.0, "Model accuracy is below threshold")

    def test_model_slice_performance(self):
        slice_data = self.test_data[self.test_data['sex'] == 1]  # Adjust based on encoding (1 for 'Male', 0 for 'Female')
        if 'salary' not in slice_data.columns:
            # Add the 'salary' column back if missing
            slice_data['salary'] = 0
        
        X_test_slice = slice_data.drop(columns=['salary'])
        y_test_slice = slice_data['salary']
        
        # Ensure X_test_slice is not empty
        if X_test_slice.empty:
            self.fail("Slice features are empty")

        print(f"X_test_slice sample:\n{X_test_slice.head()}")
        print(f"y_test_slice sample:\n{y_test_slice.head()}")

        predictions = self.model.predict(X_test_slice)
        accuracy = (predictions == y_test_slice).mean()
        print(f"Predictions slice sample:\n{predictions[:10]}")
        print(f"Slice Accuracy: {accuracy:.2f}")

        self.assertGreater(accuracy, 0.0, "Model slice performance is below threshold")

if __name__ == "__main__":
    unittest.main()
