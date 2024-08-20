import unittest
import joblib
import pandas as pd
from huggingface_hub import hf_hub_download
from ucimlrepo import fetch_ucirepo  # Import fetch_ucirepo
from sklearn.preprocessing import LabelEncoder  # Import LabelEncoder


class TestModel(unittest.TestCase):
    def setUp(self):
        # Fetch dataset
        census_income = fetch_ucirepo(id=20)

        # Extract features and targets
        self.X_test = census_income.data.features
        self.y_test = census_income.data.targets

        # Initialize label encoders
        self.label_encoders = {}

        # Create a DataFrame from the features for further processing
        self.test_data = pd.DataFrame(self.X_test)

        # Encode categorical features
        for column in self.test_data.select_dtypes(include=['object']).columns:
            le = LabelEncoder()
            self.test_data[column] = le.fit_transform(self.test_data[column])
            self.label_encoders[column] = le

        # Convert categorical features to numeric using one-hot encoding
        self.test_data = pd.get_dummies(self.test_data)

        # Load the training columns for consistent feature sets
        train_columns = joblib.load('models/train_columns.joblib')

        # Ensure the same columns are used in testing as in training
        self.test_data = self.test_data.reindex(columns=train_columns, fill_value=0)

        # Update self.X_test with processed test data
        self.X_test = self.test_data
        
        # Convert self.y_test to a Series, ensuring it is a one-dimensional array
        self.y_test = pd.Series(self.y_test.squeeze())

        # Download the model file from the Hugging Face Hub
        model_file = hf_hub_download(repo_id="Abdelrahman39/cenusus", filename="model.joblib")

        # Load the model using joblib
        self.model = joblib.load(model_file)


    def test_model_accuracy(self):
        print("test data full",  self.test_data)
        # Ensure X_test is not empty
        if self.X_test.empty:
            self.fail("Test features are empty")

        print(f"X_test sample:\n{self.X_test.head()}")
        print(f"y_test sample:\n{self.y_test.head()}")

        # Make predictions
        predictions = self.model.predict(self.X_test)
        accuracy = (predictions == self.y_test).mean()
        print(f"y_test sample:\n{predictions[:10]}")
        print(f"Predictions sample:\n{predictions[:10]}")
        print(f"Accuracy: {accuracy:.2f}")

        self.assertGreater(accuracy, 0.0, "Model accuracy is below threshold")

    def test_model_slice_performance(self):
        # Ensure 'sex' column exists and is correctly encoded in X_test
        if 'sex' not in self.X_test.columns:
            self.fail("Column 'sex' is missing from test data")

        # Slice the test data based on the 'sex' column
        X_test_slice = self.X_test[self.X_test['sex'] == 1]  # Adjust based on encoding (1 for 'Male', 0 for 'Female')
        y_test_slice = self.y_test[self.X_test['sex'] == 1]

        # Ensure X_test_slice is not empty
        if X_test_slice.empty:
            self.fail("Slice features are empty")

        print(f"X_test_slice sample:\n{X_test_slice.head()}")
        print(f"y_test_slice sample:\n{y_test_slice.head()}")

        # Make predictions
        predictions = self.model.predict(X_test_slice)

        # Convert predictions to the same format as y_test_slice for comparison
        # predictions_mapped = pd.Series(predictions).map(lambda x: 1 if x == '>50K' else 0)

        # Reset index to ensure they are comparable
        predictions_mapped = predictions
        predictions_mapped.reset_index(drop=True, inplace=True)
        
        y_test_slice.reset_index(drop=True, inplace=True)

        # Calculate accuracy
        accuracy = (predictions_mapped == y_test_slice).mean()
        print(f"real slice sample:\n{y_test_slice[:10]}")
        print(f"Predictions slice sample:\n{predictions_mapped[:10]}")
        print(f"Slice Accuracy: {accuracy:.2f}")

        self.assertGreater(accuracy, 0.0, "Model slice performance is below threshold")



if __name__ == "__main__":
    unittest.main()
