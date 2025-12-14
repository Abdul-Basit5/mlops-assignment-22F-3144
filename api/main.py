from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import os

app = FastAPI()

class InputData(BaseModel):
    text: str

# Correct dynamic path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "airflow", "dags", "project", "models", "model.pkl")
model = joblib.load(MODEL_PATH)

@app.get("/")
def home():
    return {"message": "API is running"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(data: InputData):
    text = data.text
    pred = model.predict([text])[0]
    return {"prediction": pred}
