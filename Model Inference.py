import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Model Inference
def predict_new_data(model, new_data):
    """Use the trained model to predict new data."""
    predictions = model.predict(new_data)
    return predictions