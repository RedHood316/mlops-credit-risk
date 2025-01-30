
from fastapi import FastAPI
import mlflow.pyfunc
import pandas as pd
from pydantic import BaseModel

# Load the trained MLflow model
model_path = "C:/Github Projects/mlops-credit-risk/mlruns/0/<run_id>/artifacts/model"  # Replace <run_id> with your actual MLflow run ID
model = mlflow.pyfunc.load_model(model_path)

app = FastAPI()

class InputData(BaseModel):
    features: list[float]

@app.post("/predict")
def predict(data: InputData):
    df = pd.DataFrame([data.features])
    prediction = model.predict(df)
    return {"prediction": int(prediction[0])}
