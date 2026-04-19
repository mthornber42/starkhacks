from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import json
from model import predict

app = FastAPI()

DATA_STORE = []

class Sample(BaseModel):
    a_x: float
    a_y: float
    a_z: float
    g_x: float
    g_y: float
    g_z: float

class IMUData(BaseModel):
    samples: List[Sample]


@app.post("/imu")
def receive(data: IMUData):
    json_data = data.model_dump()
    # prediction = mollyend.molly_predict(json_data)
    prediction = predict(json_data)
    DATA_STORE.append(prediction)
    return {"status": "ok", "count": len(DATA_STORE)}

@app.get("/latest")
def latest():
    return DATA_STORE[-1] if DATA_STORE else {}

@app.get("/all")
def all_data():
    return DATA_STORE