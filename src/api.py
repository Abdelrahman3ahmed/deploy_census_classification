from fastapi import FastAPI , HTTPException
from pydantic import BaseModel
from huggingface_hub import hf_hub_download
import joblib

app = FastAPI()

# model = joblib.load('/app/models/model.joblib')
model_file = hf_hub_download(repo_id="Abdelrahman39/cenusus", filename="model.joblib")

# Load the model using joblib
model = joblib.load(model_file)

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
    
    # Example mapping of keys if necessary
    column_mapping = {
        'marital_status': 'marital-status',
        'native_country': 'native-country'
        # Add other mappings as necessary
    }
    

    print("the date input before mapping",input_dict)
    # Transform keys if needed
    for key in list(input_dict.keys()):
        if key in column_mapping:
            new_key = column_mapping[key]
            input_dict[new_key] = input_dict.pop(key)
    print("the date input ",input_dict)
    try:
        for column, le in label_encoders.items():
            if column in input_dict:
                input_dict[column] = le.transform([input_dict[column]])[0]
            else:
                raise ValueError(f"Column {column} not found in input data")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    print("the date input for model",input_dict)
    prediction = model.predict([list(input_dict.values())])
    return {"prediction": prediction[0]}

