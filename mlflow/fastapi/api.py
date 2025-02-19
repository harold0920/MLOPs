import mlflow
import pandas as pd
import numpy as np
import joblib
import requests
from fastapi import FastAPI
from mlflow.tracking import MlflowClient
from pydantic import BaseModel
from io import StringIO

# Initialize FastAPI
app = FastAPI()

# Set MLflow tracking URI (inside Docker, use container name)
mlflow.set_tracking_uri("http://mlflow_server:5000")
client = MlflowClient()

# Load feature order and scaler
scaler = joblib.load("/app/scaler.pkl")
feature_order = joblib.load("/app/feature_order.pkl")

# Load dataset structure from GitHub
def load_dataset_structure():
    """Loads Employee dataset from GitHub to get correct feature names."""
    url = "https://raw.githubusercontent.com/harold0920/MLOPs/main/Employee.csv"
    response = requests.get(url)
    
    if response.status_code != 200:
        raise FileNotFoundError(f"Failed to download file from {url}. Status code: {response.status_code}")

    # Read CSV and drop unnecessary columns
    data = pd.read_csv(StringIO(response.text))
    data.drop(columns=['EmployeeCount', 'StandardHours', 'JobRole', 'Over18', 'EmployeeNumber'], inplace=True)
    return data

# Retrieve correct feature names
employee_data = load_dataset_structure()
feature_names = employee_data.drop(columns=["Attrition"]).columns.tolist()

# Load the latest production model
def get_model():
    """Retrieves the latest production model from MLflow."""
    model_name = "random_forest_model"
    
    # Fetch all versions of the model
    versions = client.search_model_versions(f"name='{model_name}'")

    # Manually filter for the latest version in 'Production'
    production_versions = [
        v for v in versions if v.current_stage == "Production"
    ]

    if not production_versions:
        raise ValueError("No model found in Production stage. Train and deploy a model first.")

    latest_version = max(production_versions, key=lambda v: int(v.version))

    # Load model from MLflow
    model_uri = latest_version.source
    return mlflow.sklearn.load_model(model_uri)

# API input schema
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

# Preprocessing function
def preprocess_data(employee: EmployeeData):
    """Converts input JSON into a formatted DataFrame for prediction."""
    employee_dict = employee.dict()
    df_input = pd.DataFrame([employee_dict])

    # Mapping categorical values
    mappings = {
        'BusinessTravel': {'Non-Travel': 0, 'Travel_Rarely': 1, 'Travel_Frequently': 2},
        'Gender': {'Male': 1, 'Female': 0},
        'OverTime': {'Yes': 1, 'No': 0}
    }

    for col, mapping in mappings.items():
        if col in df_input:
            df_input[col] = df_input[col].replace(mapping)

    # One-hot encoding categorical features
    df_input = pd.get_dummies(df_input, columns=['MaritalStatus', 'EducationField', 'Department'])

    # Ensure all feature columns exist
    for col in feature_order:
        if col not in df_input:
            df_input[col] = 0  # Add missing columns

    # Reorder columns to match training data
    df_input = df_input[feature_order]

    # Scale input data
    return scaler.transform(df_input)

# Prediction endpoint
@app.post("/predict")
def predict(employee: EmployeeData):
    """Predicts attrition risk based on employee data."""
    model = get_model()
    X_scaled = preprocess_data(employee)
    
    prediction = model.predict(X_scaled).tolist()
    return {"prediction": prediction[0], "attrition": "Yes" if prediction[0] == 1 else "No"}
