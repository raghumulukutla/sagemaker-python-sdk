# FastAPI Server with SageMaker ModelBuilder and SMD

This example demonstrates how to create a FastAPI server that works with ModelBuilder's SMD (SageMaker Distribution) and CustomOrchestrator.

## Overview

FastAPI provides a modern, fast web framework for building APIs. When combined with ModelBuilder and SMD, you get:

- **FastAPI** - Modern async web framework
- **ModelBuilder** - Automated model packaging and deployment
- **SMD** - SageMaker Distribution container
- **CustomOrchestrator** - Custom inference logic

## Architecture

```
┌─────────────────────────────────────────┐
│  SageMaker Endpoint                     │
│  ┌───────────────────────────────────┐  │
│  │  FastAPI Server (Port 8080)      │  │
│  │  ┌─────────────────────────────┐ │  │
│  │  │  CustomOrchestrator         │ │  │
│  │  │  - handle() method           │ │  │
│  │  └─────────────────────────────┘ │  │
│  │  ┌─────────────────────────────┐ │  │
│  │  │  Model (loaded from serve.pkl)│ │  │
│  │  └─────────────────────────────┘ │  │
│  └───────────────────────────────────┘  │
└─────────────────────────────────────────┘
```

## Features

- ✅ FastAPI async/await support
- ✅ SageMaker inference contract (`/invocations`)
- ✅ Custom endpoints (`/predict`, `/ping`, `/model-info`)
- ✅ Automatic model loading from ModelBuilder artifacts
- ✅ Integrity checking for model artifacts
- ✅ Support for both sync and async CustomOrchestrators

## Quick Start

### 1. Build FastAPI Container

```bash
cd fastapi_smd
docker build -t fastapi-model:latest .
```

### 2. Push to ECR

```bash
# Create ECR repository
aws ecr create-repository --repository-name fastapi-model --region us-west-2

# Login to ECR
aws ecr get-login-password --region us-west-2 | \
  docker login --username AWS --password-stdin \
  123456789012.dkr.ecr.us-west-2.amazonaws.com

# Tag and push
docker tag fastapi-model:latest \
  123456789012.dkr.ecr.us-west-2.amazonaws.com/fastapi-model:latest

docker push 123456789012.dkr.ecr.us-west-2.amazonaws.com/fastapi-model:latest
```

### 3. Use with ModelBuilder

```python
from sagemaker.serve.model_builder import ModelBuilder
from sagemaker.serve.spec.inference_base import AsyncCustomOrchestrator
from sagemaker.serve.utils.types import ModelServer

class MyOrchestrator(AsyncCustomOrchestrator):
    async def handle(self, data, context=None):
        # Your inference logic
        return self.model.predict(data)

orchestrator = MyOrchestrator(my_model)

model_builder = ModelBuilder(
    model=my_model,
    inference_spec=orchestrator,
    image_uri="123456789012.dkr.ecr.us-west-2.amazonaws.com/fastapi-model:latest",
    model_server=ModelServer.SMD,
    role_arn="arn:aws:iam::123456789012:role/SageMakerExecutionRole",
    instance_type="ml.m5.xlarge"
)

model = model_builder.build()
endpoint = model_builder.deploy()
```

## API Endpoints

### `/invocations` (SageMaker Standard)
- **Method**: POST
- **Purpose**: SageMaker inference endpoint
- **Content-Type**: application/json
- **Response**: JSON with prediction

```python
endpoint.invoke(data={"features": [1.0, 2.0, 3.0]})
```

### `/predict` (Custom)
- **Method**: POST
- **Purpose**: Alternative prediction endpoint
- **Request Body**: `{"data": [...]}`
- **Response**: `{"prediction": ..., "status": "success"}`

### `/ping` (Health Check)
- **Method**: GET
- **Purpose**: Health check
- **Response**: `{"status": "healthy"}`

### `/model-info` (Model Information)
- **Method**: GET
- **Purpose**: Get model information
- **Response**: Model metadata and status

## CustomOrchestrator Examples

### Async CustomOrchestrator

```python
from sagemaker.serve.spec.inference_base import AsyncCustomOrchestrator

class FastAPIOrchestrator(AsyncCustomOrchestrator):
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    async def handle(self, data, context=None):
        # Preprocess
        processed = self.preprocess(data)
        
        # Predict
        result = self.model.predict(processed)
        
        # Postprocess
        return self.postprocess(result)
    
    def preprocess(self, data):
        # Your preprocessing logic
        return data
    
    def postprocess(self, prediction):
        # Your postprocessing logic
        return prediction
```

### Sync CustomOrchestrator

```python
from sagemaker.serve.spec.inference_base import CustomOrchestrator

class SyncOrchestrator(CustomOrchestrator):
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    def handle(self, data, context=None):
        # Sync inference
        return self.model.predict(data)
```

## Testing Locally

### Option 1: Local Container Mode

```python
model_builder = ModelBuilder(
    model=my_model,
    inference_spec=orchestrator,
    image_uri="fastapi-model:latest",  # Local image
    model_server=ModelServer.SMD,
    mode=Mode.LOCAL_CONTAINER,
    role_arn="arn:aws:iam::...",
)

local_endpoint = model_builder.deploy()
result = local_endpoint.invoke(data={"features": [1, 2, 3]})
```

### Option 2: Direct FastAPI Testing

```bash
# Run container locally
docker run -p 8080:8080 \
  -v $(pwd)/model:/opt/ml/model \
  fastapi-model:latest

# Test endpoints
curl http://localhost:8080/ping
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{"data": [1.0, 2.0, 3.0]}'
```

## File Structure

```
fastapi_smd/
├── Dockerfile                    # Container definition
├── requirements.txt             # Python dependencies
├── code/
│   └── inference.py            # FastAPI server code
├── model_builder_fastapi_example.py  # Usage examples
└── README.md                   # This file
```

## How It Works

1. **ModelBuilder packages your model**:
   - Saves model/orchestrator to `serve.pkl`
   - Creates metadata.json for integrity checking
   - Uploads to S3

2. **FastAPI server starts**:
   - Loads model from `/opt/ml/model/code/serve.pkl`
   - Performs integrity check
   - Initializes CustomOrchestrator or model

3. **Inference request**:
   - SageMaker sends request to `/invocations`
   - FastAPI receives request
   - Calls CustomOrchestrator.handle() or model.predict()
   - Returns prediction

## Environment Variables

- `SAGEMAKER_INFERENCE_CODE_DIRECTORY` - Path to code directory (default: `/opt/ml/model/code`)
- `SAGEMAKER_BIND_TO_PORT` - Port to bind (default: `8080`)
- `LOG_LEVEL` - Logging level (default: `INFO`)

## Advantages of FastAPI

1. **Async Support**: Native async/await for better performance
2. **Type Safety**: Pydantic models for request/response validation
3. **Auto Documentation**: Swagger UI at `/docs`
4. **Modern**: Built on Starlette and Pydantic
5. **Fast**: One of the fastest Python frameworks

## Troubleshooting

### Model not loading
- Check CloudWatch logs for errors
- Verify `serve.pkl` exists in `/opt/ml/model/code/`
- Ensure ModelBuilder packaged artifacts correctly

### Port conflicts
- Ensure port 8080 is exposed in Dockerfile
- Check `SAGEMAKER_BIND_TO_PORT` environment variable

### Async errors
- Ensure CustomOrchestrator.handle() is async if using AsyncCustomOrchestrator
- Check that model operations are thread-safe

## References

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [SageMaker ModelBuilder](https://docs.aws.amazon.com/sagemaker/latest/dg/model-building.html)
- [CustomOrchestrator Guide](https://docs.aws.amazon.com/sagemaker/latest/dg/model-building.html)
