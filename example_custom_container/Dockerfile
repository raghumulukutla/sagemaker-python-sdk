# Custom container Dockerfile for SageMaker ModelBuilder
# This example shows how to create a custom container for model serving

FROM python:3.12-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /opt/ml/model

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy model artifacts and inference code
# ModelBuilder will handle copying model artifacts to /opt/ml/model/
# Your custom inference code should be in /opt/ml/model/code/
COPY code/ /opt/ml/model/code/

# Set environment variables for SageMaker
ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV SAGEMAKER_PROGRAM=inference.py

# Expose port for SageMaker
EXPOSE 8080

# SageMaker will call the handler function in inference.py
# The handler function signature depends on your model_server choice
CMD ["python", "/opt/ml/model/code/inference.py"]
