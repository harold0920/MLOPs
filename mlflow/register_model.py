import mlflow
from mlflow.tracking import MlflowClient

# Set MLflow Tracking URI
mlflow.set_tracking_uri("http://localhost:5000")

# Define the run ID
run_id = "548a92d42399420a81bf30cdc40e61b9"
model_name = "random_forest_model"

# Register the Model
model_uri = f"runs:/{run_id}/model"
client = MlflowClient()

try:
    registered_model = mlflow.register_model(model_uri, model_name)
    print(f"🎯 Model registered successfully! Model URI: {model_uri}")
except Exception as e:
    print(f"❌ Model registration failed: {e}")
    exit()

# Move Model to "Production"
try:
    version = client.get_latest_versions(model_name)[0].version
    client.transition_model_version_stage(name=model_name, version=version, stage="Production")
    print(f"🚀 Model version {version} moved to Production!")
except Exception as e:
    print(f"❌ Failed to transition model to Production: {e}")
    exit()

# Verify Registered Models
models = client.search_registered_models()
for model in models:
    print(f"✅ Model Name: {model.name}, Versions: {[v.version for v in model.latest_versions]}")
