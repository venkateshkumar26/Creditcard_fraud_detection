import joblib
from fastapi import FastAPI
import numpy as np

app=FastAPI()

model=joblib.load("models/model.pkl")

THRESHOLD=0.2

from pydantic import BaseModel

class Input(BaseModel):
    features:list[float]

@app.post("/fraud_detector")
def predict(features:Input):
    X=np.array(features.features).reshape(1,-1)
    if len(features.features)!=30:
        return {"error":"Expected 30 features"}
    prob=model.predict_proba(X)[0][1]
    pred=int(prob>=THRESHOLD)
    return {
        "fraud_probability":round(float(prob),4),
        "prediction":"fraud" if pred==1 else "not_fraud",
        "threshold_used":THRESHOLD
    }
    