import os
import mlflow
import mlflow.sklearn
import pandas as pd
import requests
import joblib
import logging
import numpy as np
from io import StringIO
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

# Configure Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Set MLflow tracking URI
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("employee_attrition_experiment")


### **Step 1: Load & Transform Data**
def load_and_transform_data():
    """Load and transform the dataset from GitHub."""
    url = "https://raw.githubusercontent.com/harold0920/MLOPs/main/Employee.csv"

    try:
        response = requests.get(url)
        response.raise_for_status()
        data = pd.read_csv(StringIO(response.text))
    except requests.exceptions.RequestException as e:
        logging.error(f"Failed to fetch data: {e}")
        raise

    # Drop unnecessary columns
    drop_columns = ["EmployeeCount", "StandardHours", "JobRole", "Over18", "EmployeeNumber"]
    data.drop(columns=drop_columns, inplace=True, errors="ignore")

    logging.info("Data loaded and preprocessed successfully.")
    return data


### **Step 2: Feature Engineering**
def preprocess_features(employee_data: pd.DataFrame):
    """Preprocess and engineer features from raw employee data."""
    categorical_mappings = {
        "Attrition": {"Yes": 1, "No": 0},
        "BusinessTravel": {"Non-Travel": 0, "Travel_Rarely": 1, "Travel_Frequently": 2},
        "Gender": {"Male": 1, "Female": 0},
        "OverTime": {"Yes": 1, "No": 0},
    }

    for col, mapping in categorical_mappings.items():
        if col in employee_data:
            employee_data[col] = employee_data[col].replace(mapping)

    # One-hot encoding for categorical columns
    categorical_features = ["MaritalStatus", "EducationField", "Department"]
    employee_data = pd.get_dummies(employee_data, columns=categorical_features, drop_first=True)

    # Separate features and target
    y = employee_data.pop("Attrition")
    X = employee_data

    # Standardize features
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    # Save the scaler and feature order
    joblib.dump(scaler, "scaler.pkl")
    joblib.dump(X.columns.tolist(), "feature_order.pkl")

    logging.info("Feature engineering completed. Scaler saved.")
    return X_scaled, y


### **Step 3: Train Model and Log to MLflow**
def train_model():
    """Train and log a RandomForest model in MLflow."""
    try:
        # Load and preprocess data
        employee_data = load_and_transform_data()
        X, y = preprocess_features(employee_data)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # Define hyperparameters
        hyperparams = {
            "n_estimators": 200,
            "max_depth": 10,
            "min_samples_split": 3,
            "min_samples_leaf": 2,
            "random_state": 42,
        }

        # Start MLflow Run
        with mlflow.start_run():
            # Train model
            model = RandomForestClassifier(**hyperparams)
            model.fit(X_train, y_train)

            # Make predictions
            y_pred = model.predict(X_test)

            # Compute accuracy
            accuracy = accuracy_score(y_test, y_pred)

            # Log hyperparameters and metrics
            mlflow.log_params(hyperparams)
            mlflow.log_metric("accuracy", accuracy)

            # Log classification report
            report = classification_report(y_test, y_pred, output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            report_df.to_csv("classification_report.csv", index=True)
            mlflow.log_artifact("classification_report.csv", artifact_path="artifacts")
            mlflow.log_artifact("random_forest_model.pkl", artifact_path="artifacts")

            # Save and log model
            model_path = "random_forest_model.pkl"
            joblib.dump(model, model_path)
            mlflow.sklearn.log_model(model, "random_forest_model")

            logging.info(f"Model trained and logged successfully with Accuracy: {accuracy:.4f}")

        return model

    except Exception as e:
        logging.error(f"Error in model training: {e}")
        raise


if __name__ == "__main__":
    train_model()
