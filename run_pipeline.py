from data_transformation import load_and_transform_data
from feature_engineering import preprocess_features
from model_training import train_model
from model_inference import predict_new_data


# Example Usage
if __name__ == "__main__":
    # Load and transform data
    filepath = 'Employee.csv'
    data = load_and_transform_data(filepath)

    # Preprocess features
    X, y = preprocess_features(data)

    # Train model
    trained_model = train_model(X, y)

    # inference on new data
    sample_data = X.sample(5, random_state=42)
    predictions = predict_new_data(trained_model, sample_data)
    print("Predictions:", predictions)
