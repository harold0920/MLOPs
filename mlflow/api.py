from fastapi import FastAPI
import mlflow
from mlflow.tracking import MlflowClient
from pydantic import BaseModel
import pandas as pd
import numpy as np

app = FastAPI()
mlflow.set_tracking_uri("http://localhost:5000")
client = MlflowClient()

# Load model from MLflow Registry
def load_production_model():
    model_name = "random_forest_model"
    try:
        latest_version = client.get_latest_versions(model_name, stages=["Production"])[0]
        model_uri = f"models:/{model_name}/{latest_version.version}"
        return mlflow.sklearn.load_model(model_uri)
    except IndexError:
        return None

model = load_production_model()


# ✅ **1. Prediction Endpoint**
class InputData(BaseModel):
    X: list

@app.post("/predict")
def predict(data: InputData):
    if model is None:
        return {"error": "🚨 No model available. Train and register a model first."}

    X_input = np.array(data.X)
    predictions = model.predict(X_input).tolist()
    return {"predictions": predictions}


# ✅ **2. Train & Register Model**
@app.post("/train")
def train_and_register_model():
    # Load dataset
    url = "https://raw.githubusercontent.com/harold0920/MLOPs/main/Employee.csv"
    data = pd.read_csv(url)

    # Feature Engineering
    data.drop(columns=['EmployeeCount', 'StandardHours', 'JobRole', 'Over18', 'EmployeeNumber'], inplace=True)
    y = data.pop("Attrition").replace({"Yes": 1, "No": 0})
    X = pd.get_dummies(data)

    # Train model
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    
    with mlflow.start_run():
        mlflow.log_params({"n_estimators": 100, "max_depth": 5, "random_state": 42})
        model.fit(X_train, y_train)
        accuracy = accuracy_score(y_test, model.predict(X_test))
        mlflow.log_metric("accuracy", accuracy)
        
        # Log and register model
        model_uri = mlflow.sklearn.log_model(model, "model")
        registered_model = mlflow.register_model(model_uri, "random_forest_model")

    return {"message": "✅ Model trained and registered successfully!", "accuracy": accuracy}


# ✅ **3. Move Model to Production**
@app.post("/promote")
def promote_model(version: int):
    model_name = "random_forest_model"
    try:
        client.transition_model_version_stage(name=model_name, version=version, stage="Production")
        return {"message": f"🚀 Model version {version} moved to Production!"}
    except Exception as e:
        return {"error": str(e)}
