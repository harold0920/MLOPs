import base64
from io import StringIO
from typing import Any, Dict, Tuple
import requests

from dagster import (
    AssetOut,
    IOManager,
    MetadataValue,
    Output,
    asset,
    io_manager,
    multi_asset,
    AutomationCondition,
)



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from dagster import asset

# Step 1: Data Transformation
@asset(name="employee_data")
def load_and_transform_data() -> pd.DataFrame:
    """Load and transform the dataset from GitHub."""
    # URL to the raw CSV file on GitHub
    url = "https://raw.githubusercontent.com/harold0920/MLOPs/main/Employee.csv"
    
    # Download the file
    response = requests.get(url)
    if response.status_code != 200:
        raise FileNotFoundError(f"Failed to download file from {url}. Status code: {response.status_code}")
    
    # Read the CSV file into a DataFrame
    data = pd.read_csv(StringIO(response.text))
    
    # Drop unnecessary columns
    data.drop(columns=['EmployeeCount', 'StandardHours', 'JobRole', 'Over18', 'EmployeeNumber'], inplace=True)
    
    return data

# Step 2: Feature Engineering
@asset(name="preprocessed_data")
def preprocess_features(employee_data: pd.DataFrame):
    """Preprocess and engineer features."""
    # Plot histograms for each feature
    employee_data.hist(bins=20, figsize=(15, 10))
    plt.tight_layout()
    plt.show()

    # Correlation pair plot
    sns.pairplot(employee_data)
    plt.show()

    # Convert categorical columns to numerical
    categorical_cols = employee_data.select_dtypes(include=['object']).columns
    employee_data = pd.get_dummies(employee_data, columns=categorical_cols, drop_first=True)
    
    # Split features and target
    X = employee_data.drop('Attrition', axis=1)
    y = employee_data['Attrition'].apply(lambda x: 1 if x == 'Yes' else 0)
    return {"features": X, "target": y}

# Step 3: Model Training
@asset(name="trained_model")
def train_model(preprocessed_data: dict) -> RandomForestClassifier:
    """Train a Random Forest model."""
    X = preprocessed_data["features"]
    y = preprocessed_data["target"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate the model
    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))
    
    return model

# Step 4: Model Inference
@asset(name="model_predictions")
def predict_new_data(trained_model: RandomForestClassifier, preprocessed_data: dict) -> pd.Series:
    """Use the trained model to predict new data."""
    new_data = preprocessed_data["features"].sample(5, random_state=42)
    predictions = trained_model.predict(new_data)
    return predictions

