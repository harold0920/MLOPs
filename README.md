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

## How to Use

1. **Clone the repository**:
    ```bash
    git clone <repository_url>
    cd <repository_folder>
    ```

2. **Prepare the dataset**:
    Place your dataset in the project directory and update the `filepath` variable in the example usage.

3. **Run the pipeline**:
    You can execute the pipeline in sequence as follows:

    ```python
    # Example Usage
    if __name__ == "__main__":
        # Load and transform data
        filepath = 'Employee.csv'  # Replace with actual path
        data = load_and_transform_data(filepath)

        # Preprocess features
        X, y = preprocess_features(data)

        # Train model
        trained_model = train_model(X, y)

        # Example of inference (using a subset of the data as new data)
        sample_data = X.sample(5, random_state=42)
        predictions = predict_new_data(trained_model, sample_data)
        print("Predictions:", predictions)
    ```

## Features

- **Data Visualization**: Generates histograms and correlation plots during feature engineering.
- **Random Forest Model**: Provides a robust and interpretable classification model.
- **Modular Design**: Each pipeline step is separated for better reusability and clarity.

## Outputs

- Model accuracy and classification report are displayed during training.
- Predictions are output for new data samples during inference.



