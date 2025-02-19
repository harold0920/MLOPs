import mlflow
import pandas as pd
import numpy as np
import requests
import joblib
from fastapi import FastAPI
from mlflow.tracking import MlflowClient
from pydantic import BaseModel
from io import StringIO

app = FastAPI()
mlflow.set_tracking_uri("http://localhost:5000")
client = MlflowClient()

### **Step 1: Load Dataset Structure for Feature Names**
def load_dataset_structure():
    """Loads Employee dataset from GitHub to get correct feature names."""
    url = "https://raw.githubusercontent.com/harold0920/MLOPs/main/Employee.csv"
    
    response = requests.get(url)
    if response.status_code != 200:
        raise FileNotFoundError(f"Failed to download file from {url}. Status code: {response.status_code}")

    # Read CSV
    data = pd.read_csv(StringIO(response.text))

    # Drop unnecessary columns
    data.drop(columns=['EmployeeCount', 'StandardHours', 'JobRole', 'Over18', 'EmployeeNumber'], inplace=True)

    return data

# **Retrieve correct feature names**
employee_data = load_dataset_structure()
feature_names = employee_data.drop(columns=["Attrition"]).columns.tolist()

### **Step 2: Load Model and Scaler from MLflow**
model = None  # Lazy loading
scaler = joblib.load("scaler.pkl")  # Load pre-trained scaler
feature_order = joblib.load("feature_order.pkl")  # Load correct feature order

def get_model():
    """Loads the latest production model from MLflow."""
    global model
    if model is None:
        print("⚡ Loading model from MLflow...")
        model_name = "random_forest_model"

        # Instead of get_latest_versions(), use search_model_versions()
        versions = client.search_model_versions(f"name='{model_name}'")
        latest_version = max(versions, key=lambda v: int(v.version))  # Get the highest version

        # Use latest version URI
        model_uri = latest_version.source
        model = mlflow.sklearn.load_model(model_uri)
        print("✅ Model loaded successfully!")
    return model

### **Step 3: Define API Input Schema (Raw Employee Data)**
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

### **Step 4: FastAPI Prediction Endpoint**
@app.post("/predict")
def predict(employee: EmployeeData):
    model = get_model()
    if model is None:
        return {"error": "🚨 No model available. Train and register a model first."}

    # Convert EmployeeData to DataFrame
    employee_dict = employee.dict()
    df_input = pd.DataFrame([employee_dict])

    ### **Step 5: Preprocess Raw Data (Convert Categorical to Numeric)**
    mappings = {
        'BusinessTravel': {'Non-Travel': 0, 'Travel_Rarely': 1, 'Travel_Frequently': 2},
        'Gender': {'Male': 1, 'Female': 0},
        'OverTime': {'Yes': 1, 'No': 0}
    }

    for col, mapping in mappings.items():
        if col in df_input:
            df_input[col] = df_input[col].replace(mapping)

    # One-hot encode categorical variables
    df_input = pd.get_dummies(df_input, columns=['MaritalStatus', 'EducationField', 'Department'])

    # Ensure all feature columns exist
    for col in feature_order:
        if col not in df_input:
            df_input[col] = 0  # Add missing columns

    # Reorder columns to match training data
    df_input = df_input[feature_order]

    ### **Step 6: Scale Input using pre-trained scaler**
    X_scaled = scaler.transform(df_input)

    ### **Step 7: Predict**
    prediction = model.predict(X_scaled).tolist()

    return {"prediction": prediction[0], "attrition": "Yes" if prediction[0] == 1 else "No"}
