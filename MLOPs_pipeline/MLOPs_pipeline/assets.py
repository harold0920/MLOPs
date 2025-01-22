import base64
from io import StringIO
from typing import Any, Dict, Tuple
import requests
from dagster import Definitions

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

class LocalCSVIOManager(IOManager):
    """
    A custom IOManager to handle saving and loading CSV files locally.
    """

    def handle_output(self, context, obj: pd.DataFrame) -> None:
        """
        Save a Pandas DataFrame to a CSV file.
        Args:
            context: The context object provided by Dagster.
            obj (pd.DataFrame): The DataFrame to save.
        """
        obj.to_csv(f"{context.asset_key.path[-1]}.csv")

    def load_input(self, context) -> pd.DataFrame:
        """
        Load a Pandas DataFrame from a CSV file.
        Args:
            context: The context object provided by Dagster.
        Returns:
            pd.DataFrame: The loaded DataFrame.
        """
        return pd.read_csv(f"{context.asset_key.path[-1]}.csv")


@io_manager
def local_csv_io_manager() -> LocalCSVIOManager:
    """
    Instantiate the custom CSV IOManager.
    Returns:
        LocalCSVIOManager: An instance of the custom IOManager.
    """
    return LocalCSVIOManager()


# Step 1: Data Transformation
@asset(name="employee_data")
def load_and_transform_data() -> pd.DataFrame:
    """Load and transform the dataset from GitHub."""
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
from sklearn.preprocessing import StandardScaler

@asset(name="preprocessed_data")
def preprocess_features(employee_data: pd.DataFrame) -> dict:
    """
    Preprocess and engineer features from the raw employee data, including scaling.
    
    Args:
        employee_data (pd.DataFrame): The raw employee data loaded from CSV.
        
    Returns:
        dict: A dictionary containing scaled features (X) and the target variable (y).
    """
    # Step 1: Map categorical columns to numerical values
    Attrition_mapping = {'Yes': 1, 'No': 0}
    employee_data['Attrition'] = employee_data['Attrition'].replace(Attrition_mapping)

    BusinessTravel_mapping = {'Non-Travel': 0, 'Travel_Rarely': 1, 'Travel_Frequently': 2}
    employee_data['BusinessTravel'] = employee_data['BusinessTravel'].replace(BusinessTravel_mapping)

    Gender_mapping = {'Male': 1, 'Female': 0}
    employee_data['Gender'] = employee_data['Gender'].replace(Gender_mapping)

    OverTime_mapping = {'Yes': 1, 'No': 0}
    employee_data['OverTime'] = employee_data['OverTime'].replace(OverTime_mapping)

    # Step 2: One-hot encode selected categorical columns
    employee_data = pd.get_dummies(employee_data, columns=['MaritalStatus', 'EducationField', 'Department'])

    # Step 3: Separate features and target
    y = employee_data['Attrition']
    X = employee_data.drop(columns=['Attrition'])

    # Step 4: Scale the features
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    # Debugging Outputs
    print(f"Shape of scaled features (X): {X_scaled.shape}")
    print(f"Shape of target (y): {y.shape}")
    print("First few rows of scaled features:")
    print(X_scaled.head())

    return {"features": X_scaled, "target": y}



# Step 3: Model Training
@asset(name="trained_model")
def train_model(preprocessed_data: dict) -> RandomForestClassifier:
    """Train a Random Forest model."""
    X = preprocessed_data["features"]
    y = preprocessed_data["target"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Train the model
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
    # Generate predictions and convert to Pandas Series
    predictions = pd.Series(trained_model.predict(new_data), index=new_data.index)
    return predictions


# Register assets in Dagster Definitions
defs = Definitions(
    assets=[
        load_and_transform_data,
        preprocess_features,
        train_model,
        predict_new_data,
    ]
)