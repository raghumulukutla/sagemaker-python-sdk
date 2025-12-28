# FastAPI Server Dockerfile for SageMaker ModelBuilder with SMD
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

# Copy inference code
COPY code/ /opt/ml/model/code/

# Set environment variables
ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV SAGEMAKER_PROGRAM=inference.py

# Expose port 8080 (SageMaker default)
EXPOSE 8080

# Run FastAPI server with uvicorn
CMD ["python", "/opt/ml/model/code/inference.py"]
