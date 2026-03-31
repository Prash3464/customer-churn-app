from fastapi import FastAPI
import joblib
import pandas as pd
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# load Model
model = joblib.load("bank_churn.pkl")

# Request schema
class CustomerData(BaseModel):
    CreditScore: float
    Gender: str
    Age: float
    Tenure: float
    Balance: float
    NumOfProducts: float
    HasCrCard: float
    IsActiveMember: float
    EstimatedSalary: float


@app.get("/")
def home():
    return {"message":"API running sucessfully."}


@app.post("/predict")
def predict(data: CustomerData):

    # Convert input to DataFrame
    df = pd.DataFrame([data.dict()])
   
    prediction = model.predict(df)[0]
    prob = model.predict_proba(df)[0]
    print(prob)
    stay_prob = prob[0]
    exit_prob = prob[1]

    if stay_prob > exit_prob:
        result = "Customer will Stay"
        confidence = round(stay_prob * 100, 2)
    else:
        result = "Customer will Exit"
        confidence = round(exit_prob * 100, 2)

    return {
        "prediction": int(prediction),
        "result": result,
        "confidence": f"{confidence}%"
    }