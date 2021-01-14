import json
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from models.classical_models import SentimixModel
from pydantic import BaseModel

app = FastAPI(
    title="Sentimix",
    description="App to predict the sentiment of a given code-mixed tweet( Hindi + English) using Machine Learning",
    version="0.0.1",
)


class PredictRequest(BaseModel):
    data: List[str]

    class Config:
        schema_extra = {"example": {"data": ["bhai, we love you!"]}}


class PredictResponse(BaseModel):
    data: List[dict]

    class Config:
        schema_extra = {
            "example": {
                "data": [{"sentiment": "positive", "input": "bhai, we love you!"}]
            }
        }


with open(Path.cwd() / "api" / "model_registry.json") as f:
    model_registry = json.load(f)


@app.get("/")
async def root():
    return {"message": "Hello World"}


model1 = SentimixModel(
    model_path=model_registry["sentimix_logistic_regression"]["path"]
)
model1_labels = model_registry["sentimix_logistic_regression"]["labels"]


@app.post("/predict", response_model=PredictResponse)
def predict(input: PredictRequest):
    X = np.array(input.data)
    y_pred = model1.predict(X).tolist()
    y_pred = [
        {"sentiment": model1_labels[i], "input": input.data[n]}
        for n, i in enumerate(y_pred)
    ]
    result = PredictResponse(data=y_pred)
    return result
