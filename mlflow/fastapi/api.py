import mlflow
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
import os

# Set MLflow Tracking URI from Docker environment
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://tracking_server:5000")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

app = FastAPI()

# Load the latest model dynamically
model_name = "random_forest_model"
latest_version = mlflow.MlflowClient().get_latest_versions(model_name, stages=["Production"])[0].version
model_uri = f"models:/{model_name}/{latest_version}"
model = mlflow.sklearn.load_model(model_uri)

# Load the pre-trained scaler and feature order
scaler = joblib.load("scaler.pkl")
feature_order = joblib.load("feature_order.pkl")

# Define API input model
class EmployeeData(BaseModel):
    Age: int
    Gender: str
    BusinessTravel: str
    MaritalStatus: str
    EducationField: str
    Department: str
    OverTime: str
    DistanceFromHome: int
    MonthlyIncome: int
    NumCompaniesWorked: int
    TotalWorkingYears: int
    YearsAtCompany: int
    JobSatisfaction: int
    EnvironmentSatisfaction: int
    WorkLifeBalance: int

@app.post("/predict")
def predict(employee: EmployeeData):
    # Convert input data to DataFrame
    employee_dict = employee.dict()
    df_input = pd.DataFrame([employee_dict])

    # Encode categorical variables
    mappings = {
        'Gender': {'Male': 1, 'Female': 0},
        'OverTime': {'Yes': 1, 'No': 0},
        'BusinessTravel': {'Non-Travel': 0, 'Travel_Rarely': 1, 'Travel_Frequently': 2}
    }
    
    for col, mapping in mappings.items():
        df_input[col] = df_input[col].map(mapping)

    # One-hot encode categorical features (ensure missing columns are handled)
    df_input = pd.get_dummies(df_input)
    
    # Ensure feature order consistency
    df_input = df_input.reindex(columns=feature_order, fill_value=0)

    # Scale features
    df_input_scaled = scaler.transform(df_input)

    # Predict
    prediction = model.predict(df_input_scaled)[0]

    return {"prediction": prediction, "attrition": "Yes" if prediction == 1 else "No"}
