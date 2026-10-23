import os
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="API MLOps - Predicción de Diabetes")

DATABRICKS_URL = os.getenv("DATABRICKS_URL")
DATABRICKS_TOKEN = os.getenv("DATABRICKS_TOKEN")

class Paciente(BaseModel):
    Pregnancies: float
    Glucose: float
    BloodPressure: float
    SkinThickness: float
    Insulin: float
    BMI: float
    DiabetesPedigreeFunction: float
    Age: float

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predecir")
def predecir(p: Paciente):
    payload = {"dataframe_records": [p.model_dump()]}
    headers = {"Authorization": f"Bearer {DATABRICKS_TOKEN}"}
    r = requests.post(DATABRICKS_URL, json=payload, headers=headers, timeout=120)
    if r.status_code != 200:
        raise HTTPException(status_code=r.status_code, detail=r.text)
    return r.json()