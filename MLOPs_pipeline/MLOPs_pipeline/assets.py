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
    
    print("Raw DataFrame:\n", employee_data)
    return employee_data
    
    # # Plot histograms for each feature
    # employee_data.hist(bins=20, figsize=(15, 10))
    # plt.tight_layout()
    # plt.show()
    # # # Correlation pair plot
    # # sns.pairplot(employee_data)
    # # plt.show()
    
    # # Debugging: Check dataset structure
    # print("Columns in the dataset:", employee_data.columns)
    # print("First few rows:", employee_data.head())

    # # Convert categorical columns to numerical
    # categorical_cols = employee_data.select_dtypes(include=['object']).columns
    # employee_data = pd.get_dummies(employee_data, columns=categorical_cols, drop_first=True)
    

    # # Split features and target
    # y = employee_data['Attrition'].apply(lambda x: 1 if x == 'Yes' else 0)
    # X = employee_data.drop('Attrition', axis=1)
    # return {"features": X, "target": y}

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

