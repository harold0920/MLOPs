from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import List, Optional
import mlflow
import pandas as pd
import numpy as np
import joblib
from mlflow.tracking import MlflowClient

# Initialize FastAPI
app = FastAPI(
    title="ML Model API",
    description="API for registering ML models and making predictions using MLflow.",
    version="1.0.0"
)

# ------------------------------
# MLflow Setup & Model Loading
# ------------------------------

# Set MLflow tracking URI (Inside Docker, reference container name)
mlflow.set_tracking_uri("http://mlflow_server:5000")
client = MlflowClient()

# Load feature order and scaler
scaler = joblib.load("/app/scaler.pkl")
feature_order = joblib.load("/app/feature_order.pkl")

def get_model():
    """Loads the latest version of the model from MLflow registry."""
    model_name = "random_forest_model"

    # Fetch all model versions
    versions = client.search_model_versions(f"name='{model_name}'")

    if not versions:
        raise ValueError("No model found in the registry. Train and deploy a model first.")

    # Get the latest version based on the version number
    latest_version = max(versions, key=lambda v: int(v.version))

    # Load model from MLflow
    model_uri = latest_version.source
    print(f"Loading model from: {model_uri}")
    return mlflow.pyfunc.load_model(model_uri)

# Load the model once at startup
model = get_model()

# ------------------------------
# Pydantic Model Definitions
# ------------------------------

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

# ------------------------------
# Data Preprocessing
# ------------------------------

def preprocess_data(employee: EmployeeData):
    """Converts input JSON into a formatted DataFrame for prediction."""
    employee_dict = employee.dict()
    df_input = pd.DataFrame([employee_dict])

    # Encode categorical variables
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

# ------------------------------
# API Endpoints
# ------------------------------

@app.get("/", summary="API Root", tags=["Root"])
async def root():
    """Welcome endpoint for the ML API."""
    return {"message": "Welcome to the ML Model API"}

@app.post("/predict", summary="Predict attrition", tags=["Prediction"])
async def predict(employee: EmployeeData):
    """Predicts attrition risk based on employee data."""
    global model

    if model is None:
        raise HTTPException(status_code=500, detail="🚨 No model available. Train and register a model first.")

    X_scaled = preprocess_data(employee)
    prediction = model.predict(X_scaled).tolist()

    return {"prediction": prediction[0], "attrition": "Yes" if prediction[0] == 1 else "No"}



# import mlflow
# import pandas as pd
# import numpy as np
# import joblib
# import requests
# from fastapi import FastAPI, HTTPException
# from mlflow.tracking import MlflowClient
# from pydantic import BaseModel
# from io import StringIO

# # Initialize FastAPI
# app = FastAPI()

# # Set MLflow tracking URI
# mlflow.set_tracking_uri("http://mlflow_server:5000")
# client = MlflowClient()

# # Load preprocessing artifacts
# scaler = joblib.load("/app/scaler.pkl")  # StandardScaler for input data
# feature_order = joblib.load("/app/feature_order.pkl")  # Feature order to match training

# # Load dataset structure for feature extraction
# def load_dataset_structure():
#     """Loads Employee dataset structure from GitHub to retrieve feature names."""
#     url = "https://raw.githubusercontent.com/harold0920/MLOPs/main/Employee.csv"
#     response = requests.get(url)

#     if response.status_code != 200:
#         raise FileNotFoundError(f"Failed to download dataset structure. Status: {response.status_code}")

#     data = pd.read_csv(StringIO(response.text))
#     drop_columns = ["EmployeeCount", "StandardHours", "JobRole", "Over18", "EmployeeNumber"]
#     data.drop(columns=drop_columns, inplace=True, errors="ignore")

#     return data

# # Retrieve feature names
# employee_data = load_dataset_structure()
# feature_names = employee_data.drop(columns=["Attrition"]).columns.tolist()

# # Function to load the latest trained model
# def get_model():
#     """Loads the latest trained model from MLflow registry."""
#     model_name = "random_forest_model"

#     # Fetch all versions of the model
#     versions = client.search_model_versions(f"name='{model_name}'")

#     if not versions:
#         raise ValueError("🚨 No model found. Train and log a model first.")

#     # Get the latest version based on highest version number
#     latest_version = max(versions, key=lambda v: int(v.version))

#     # Load model from MLflow
#     model_uri = latest_version.source
#     print(f"✅ Loading model from: {model_uri}")
#     return mlflow.sklearn.load_model(model_uri)

# # API input schema
# class EmployeeData(BaseModel):
#     Age: int
#     Gender: str
#     BusinessTravel: str
#     MaritalStatus: str
#     EducationField: str
#     Department: str
#     OverTime: str
#     DistanceFromHome: int
#     MonthlyIncome: int
#     NumCompaniesWorked: int
#     TotalWorkingYears: int
#     YearsAtCompany: int
#     JobSatisfaction: int
#     EnvironmentSatisfaction: int
#     WorkLifeBalance: int

# # Preprocessing function
# def preprocess_data(employee: EmployeeData):
#     """Preprocesses input JSON into a formatted DataFrame for prediction."""
#     employee_dict = employee.dict()
#     df_input = pd.DataFrame([employee_dict])

#     # Categorical mappings
#     mappings = {
#         'BusinessTravel': {'Non-Travel': 0, 'Travel_Rarely': 1, 'Travel_Frequently': 2},
#         'Gender': {'Male': 1, 'Female': 0},
#         'OverTime': {'Yes': 1, 'No': 0}
#     }

#     # Convert categorical values to numeric
#     for col, mapping in mappings.items():
#         if col in df_input:
#             df_input[col] = df_input[col].replace(mapping)

#     # One-hot encoding for categorical features
#     df_input = pd.get_dummies(df_input, columns=['MaritalStatus', 'EducationField', 'Department'])

#     # Ensure all feature columns exist
#     for col in feature_order:
#         if col not in df_input:
#             df_input[col] = 0  # Add missing columns

#     # Reorder columns to match training data
#     df_input = df_input[feature_order]

#     # Scale input data using the trained scaler
#     return scaler.transform(df_input)

# # Prediction endpoint
# @app.post("/predict")
# def predict(employee: EmployeeData):
#     """Predicts attrition risk based on employee data."""
#     try:
#         model = get_model()
#         X_scaled = preprocess_data(employee)
#         prediction = model.predict(X_scaled).tolist()
#         return {"prediction": prediction[0], "attrition": "Yes" if prediction[0] == 1 else "No"}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

# import mlflow
# import pandas as pd
# import numpy as np
# import joblib
# import requests
# from fastapi import FastAPI
# from mlflow.tracking import MlflowClient
# from pydantic import BaseModel
# from io import StringIO

# # Initialize FastAPI
# app = FastAPI()

# # Set MLflow tracking URI (inside Docker, use container name)
# mlflow.set_tracking_uri("http://mlflow_server:5000")
# client = MlflowClient()

# # Load feature order and scaler
# scaler = joblib.load("/app/scaler.pkl")
# feature_order = joblib.load("/app/feature_order.pkl")

# # Load dataset structure from GitHub
# def load_dataset_structure():
#     """Loads Employee dataset from GitHub to get correct feature names."""
#     url = "https://raw.githubusercontent.com/harold0920/MLOPs/main/Employee.csv"
#     response = requests.get(url)
    
#     if response.status_code != 200:
#         raise FileNotFoundError(f"Failed to download file from {url}. Status code: {response.status_code}")

#     # Read CSV and drop unnecessary columns
#     data = pd.read_csv(StringIO(response.text))
#     data.drop(columns=['EmployeeCount', 'StandardHours', 'JobRole', 'Over18', 'EmployeeNumber'], inplace=True)
#     return data

# # Retrieve correct feature names
# employee_data = load_dataset_structure()
# feature_names = employee_data.drop(columns=["Attrition"]).columns.tolist()

# # Load the latest production model
# # def get_model():
# #     """Retrieves the latest model in 'Production' stage from MLflow."""
# #     model_name = "random_forest_model"

# #     # Fetch all versions of the model
# #     versions = client.search_model_versions(f"name='{model_name}'")

# #     # Manually filter for the latest version in 'Production' (since we cannot filter in query)
# #     production_versions = [
# #         v for v in versions if getattr(v, "current_stage", None) == "Production"
# #     ]

# #     if not production_versions:
# #         raise ValueError("No model found in Production stage. Train and deploy a model first.")

# #     # Get the latest version in production
# #     latest_version = max(production_versions, key=lambda v: int(v.version))

# #     # Load model from MLflow
# #     model_uri = latest_version.source
# #     print(f"Loading model from: {model_uri}")
# #     return mlflow.sklearn.load_model(model_uri)

# client = MlflowClient()

# def get_model():
#     """Loads the latest available model from MLflow registry."""
#     model_name = "random_forest_model"

#     # Fetch all versions of the model
#     versions = client.search_model_versions(f"name='{model_name}'")

#     if not versions:
#         raise ValueError("🚨 No model found. Train and deploy a model first.")

#     # Get the latest version (highest version number)
#     latest_version = max(versions, key=lambda v: int(v.version))

#     # Load model from MLflow
#     model_uri = latest_version.source
#     print(f"✅ Loading model from: {model_uri}")
#     return mlflow.pyfunc.load_model(model_uri)


# # API input schema
# class EmployeeData(BaseModel):
#     Age: int
#     Gender: str
#     BusinessTravel: str
#     MaritalStatus: str
#     EducationField: str
#     Department: str
#     OverTime: str
#     DistanceFromHome: int
#     MonthlyIncome: int
#     NumCompaniesWorked: int
#     TotalWorkingYears: int
#     YearsAtCompany: int
#     JobSatisfaction: int
#     EnvironmentSatisfaction: int
#     WorkLifeBalance: int

# # Preprocessing function
# def preprocess_data(employee: EmployeeData):
#     """Converts input JSON into a formatted DataFrame for prediction."""
#     employee_dict = employee.dict()
#     df_input = pd.DataFrame([employee_dict])

#     # Mapping categorical values
#     mappings = {
#         'BusinessTravel': {'Non-Travel': 0, 'Travel_Rarely': 1, 'Travel_Frequently': 2},
#         'Gender': {'Male': 1, 'Female': 0},
#         'OverTime': {'Yes': 1, 'No': 0}
#     }

#     for col, mapping in mappings.items():
#         if col in df_input:
#             df_input[col] = df_input[col].replace(mapping)

#     # One-hot encoding categorical features
#     df_input = pd.get_dummies(df_input, columns=['MaritalStatus', 'EducationField', 'Department'])

#     # Ensure all feature columns exist
#     for col in feature_order:
#         if col not in df_input:
#             df_input[col] = 0  # Add missing columns

#     # Reorder columns to match training data
#     df_input = df_input[feature_order]

#     # Scale input data
#     return scaler.transform(df_input)

# # Prediction endpoint
# @app.post("/predict")
# def predict(employee: EmployeeData):
#     """Predicts attrition risk based on employee data."""
#     model = get_model()
#     X_scaled = preprocess_data(employee)
    
#     prediction = model.predict(X_scaled).tolist()
#     return {"prediction": prediction[0], "attrition": "Yes" if prediction[0] == 1 else "No"}

