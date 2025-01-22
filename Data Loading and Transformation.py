import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

#Data Transformation
def load_and_transform_data(filepath):
    """Load and transform the dataset."""
    data = pd.read_csv(filepath)
    # Drop unnecessary columns
    data.drop(columns=['EmployeeCount', 'StandardHours', 'JobRole', 'Over18', 'EmployeeNumber'], inplace=True)
    return data