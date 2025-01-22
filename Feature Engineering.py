import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Feature Engineering
def preprocess_features(data):
    """Preprocess and engineer features."""
    # Plot histograms for each feature
    data.hist(bins=20, figsize=(15, 10))
    plt.tight_layout()
    plt.show()

    # Correlation pair plot
    sns.pairplot(data)
    plt.show()

    # Convert categorical columns to numerical
    categorical_cols = data.select_dtypes(include=['object']).columns
    data = pd.get_dummies(data, columns=categorical_cols, drop_first=True)
    
    # Split features and target
    X = data.drop('Attrition', axis=1)
    y = data['Attrition'].apply(lambda x: 1 if x == 'Yes' else 0)
    return X, y