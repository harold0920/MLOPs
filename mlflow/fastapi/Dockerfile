# FastAPI Dockerfile
# FROM python:3.12-slim-bookworm

# WORKDIR /app
# COPY requirements.txt .
# COPY scaler.pkl /app/
# COPY feature_order.pkl /app/
# COPY api.py /app/

# RUN pip install --no-cache-dir -r requirements.txt

# CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]

# Use Python as the base image
FROM python:3.12

# Set the working directory inside the container
WORKDIR /app

COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r /app/requirements.txt

# # Expose the FastAPI port
# EXPOSE 8000

# Start FastAPI
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
