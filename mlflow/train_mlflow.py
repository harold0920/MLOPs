import mlflow
import mlflow.sklearn
import pandas as pd
import requests
import joblib
import numpy as np
from io import StringIO
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

# Set MLflow tracking URI
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("employee_attrition_experiment")


### **Step 1: Load & Transform Data**
def load_and_transform_data():
    """Load and transform the dataset from GitHub."""
    url = "https://raw.githubusercontent.com/harold0920/MLOPs/main/Employee.csv"
    
    response = requests.get(url)
    if response.status_code != 200:
        raise FileNotFoundError(f"Failed to download file from {url}. Status code: {response.status_code}")
    
    # Read the CSV file
    data = pd.read_csv(StringIO(response.text))

    # Drop unnecessary columns
    data.drop(columns=['EmployeeCount', 'StandardHours', 'JobRole', 'Over18', 'EmployeeNumber'], inplace=True)
    
    return data


### **Step 2: Feature Engineering**
def preprocess_features(employee_data: pd.DataFrame):
    """Preprocess and engineer features from raw employee data."""
    
    # Convert categorical variables to numeric mappings
    mappings = {
        'Attrition': {'Yes': 1, 'No': 0},
        'BusinessTravel': {'Non-Travel': 0, 'Travel_Rarely': 1, 'Travel_Frequently': 2},
        'Gender': {'Male': 1, 'Female': 0},
        'OverTime': {'Yes': 1, 'No': 0}
    }
    
    for col, mapping in mappings.items():
        employee_data[col] = employee_data[col].replace(mapping)

    # One-hot encode categorical variables
    employee_data = pd.get_dummies(employee_data, columns=['MaritalStatus', 'EducationField', 'Department'])

    # Separate features and target
    y = employee_data['Attrition']
    X = employee_data.drop(columns=['Attrition'])

    # Scale features
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    
    return X_scaled, y


### **Step 3: Train Model and Log to MLflow**
def train_model():
    """Train and log a RandomForest model in MLflow."""
    
    # Load and preprocess data
    employee_data = load_and_transform_data()
    X, y = preprocess_features(employee_data)

    # Save the StandardScaler for inference
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    # Save the scaler and feature order
    joblib.dump(scaler, "scaler.pkl")
    joblib.dump(X.columns.tolist(), "feature_order.pkl")  # Save column order

    print("✅ Saved scaler as 'scaler.pkl'")
    print("✅ Saved feature order as 'feature_order.pkl'")

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

    # Define hyperparameters
    hyperparams = {
        "n_estimators": 200,
        "max_depth": 10,
        "min_samples_split": 3,
        "min_samples_leaf": 2,
        "random_state": 42
    }

    # Start MLflow Run
    with mlflow.start_run():
        # Train Random Forest model
        model = RandomForestClassifier(**hyperparams)
        model.fit(X_train, y_train)

        # Make predictions
        y_pred = model.predict(X_test)

        # Compute accuracy
        accuracy = accuracy_score(y_test, y_pred)

        # Log hyperparameters and metrics
        mlflow.log_params(hyperparams)
        mlflow.log_metric("accuracy", accuracy)

        # Log classification report as an artifact
        report = classification_report(y_test, y_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        report_path = "classification_report.csv"
        report_df.to_csv(report_path)
        mlflow.log_artifact(report_path)

        # Save model locally
        model_path = "random_forest_model.pkl"
        joblib.dump(model, model_path)

        # Log model to MLflow
        mlflow.sklearn.log_model(model, "random_forest_model")

        print(f"✅ Model logged successfully with Accuracy: {accuracy}")

    return model


if __name__ == "__main__":
    train_model()