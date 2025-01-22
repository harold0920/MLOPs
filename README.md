# Employee Attrition Analysis and Prediction

This project utilizes machine learning to predict employee attrition and provides actionable insights to reduce employee turnover. It follows a modular pipeline structure, with each step designed to transform raw data into predictions through data transformation, feature engineering, model training, and inference.

## Project Structure

- **`data_transformation.py`**: Handles data loading and preprocessing, such as removing unnecessary columns.
- **`feature_engineering.py`**: Conducts feature preprocessing and engineering, including generating histograms and correlation plots.
- **`model_training.py`**: Trains a Random Forest model and evaluates its performance.
- **`model_inference.py`**: Provides functions for making predictions on new data using the trained model.

## Requirements

The project requires the following Python libraries:

- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scikit-learn`

You can install these dependencies using:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

## Data

The dataset should be in CSV format and must contain the following columns:

- `Attrition` (target variable: 'Yes' or 'No')
- Other relevant features for employee analysis.

Columns like `EmployeeCount`, `StandardHours`, `JobRole`, `Over18`, and `EmployeeNumber` are automatically removed during preprocessing.


    ```

## Features

- **Data Visualization**: Generates histograms and correlation plots during feature engineering.
- **Random Forest Model**: Provides a robust and interpretable classification model.
- **Modular Design**: Each pipeline step is separated for better reusability and clarity.

## Outputs

- Model accuracy and classification report are displayed during training.
- Predictions are output for new data samples during inference.



