import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib

def train_model(input_path: str, model_path: str, encoders_path: str):
    data = pd.read_csv(input_path)
    
    label_encoders = {}
    for column in data.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        data[column] = le.fit_transform(data[column])
        label_encoders[column] = le

    X = data.drop('income', axis=1)
    y = data['income']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    
    joblib.dump(model, model_path)
    joblib.dump(label_encoders, encoders_path)

if __name__ == "__main__":
    train_model('data/clean_census.csv', 'models/model.joblib', 'models/label_encoders.joblib')

