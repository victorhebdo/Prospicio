from imp import load_module
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from prospicio.predict import predict


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


@app.get('/')
def root():
    return "Hello"


@app.get('/predict')
def get_predict(
    country_code,
    employee_range,
    min_revenues,
    traffic_monthly,
    industries_cleaned
):
    tmp = [{
            "country_code": country_code,
            "employee_range": float(employee_range),
            "min_revenues": float(min_revenues),
            "traffic.monthly":float(traffic_monthly),
            "industries_cleaned": set(industries_cleaned.split(','))
            }]
    X_new = pd.DataFrame(data=tmp)
    return {'prediction': int(predict(X_new)[0])}
