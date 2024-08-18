from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI()

model = joblib.load('/app/models/model.joblib')
label_encoders = joblib.load('/app/models/label_encoders.joblib')

class PredictionInput(BaseModel):
    age: int
    workclass: str
    fnlwgt: int
    education: str
    education_num: int
    marital_status: str
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int
    capital_loss: int
    hours_per_week: int
    native_country: str

    class Config:
        schema_extra = {
            "example": {
                "age": 25,
                "workclass": "Private",
                "fnlwgt": 226802,
                "education": "11th",
                "education_num": 7,
                "marital_status": "Never-married",
                "occupation": "Machine-op-inspct",
                "relationship": "Own-child",
                "race": "Black",
                "sex": "Male",
                "capital_gain": 0,
                "capital_loss": 0,
                "hours_per_week": 40,
                "native_country": "United-States"
            }
        }

@app.get("/")
def read_root():
    return {"message": "Welcome to the Census Classification API"}

@app.post("/predict/")
def predict(input_data: PredictionInput):
    input_dict = input_data.dict()
    for column, le in label_encoders.items():
        input_dict[column] = le.transform([input_dict[column]])[0]
    data = [list(input_dict.values())]
    prediction = model.predict(data)
    return {"prediction": prediction[0]}

