# What is CustomOrchestrator?

## Overview

**CustomOrchestrator** is a class that lets you define **your own custom inference logic** without being tied to a specific model server (like TorchServe, Triton, etc.). It's a way to have complete control over how your model handles inference requests.

## Two Types

### 1. `CustomOrchestrator` (Synchronous)
```python
from sagemaker.serve.spec.inference_base import CustomOrchestrator

class MyOrchestrator(CustomOrchestrator):
    def handle(self, data, context=None):
        # Your custom inference logic here
        return result
```

### 2. `AsyncCustomOrchestrator` (Asynchronous)
```python
from sagemaker.serve.spec.inference_base import AsyncCustomOrchestrator

class MyAsyncOrchestrator(AsyncCustomOrchestrator):
    async def handle(self, data, context=None):
        # Your async inference logic here
        return result
```

## Key Characteristics

### What CustomOrchestrator Provides:

1. **Single Entry Point**: One `handle()` method that receives all inference requests
2. **Complete Control**: You decide how to process input, run inference, and format output
3. **Model Server Independence**: Not tied to TorchServe, Triton, etc. - you write your own logic
4. **Boto3 Client Access**: Built-in SageMaker runtime client for calling other endpoints
5. **Automatic SMD**: ModelBuilder automatically uses SMD (SageMaker Distribution) container

### How It Works:

```
┌─────────────────────────────────────────┐
│  SageMaker Endpoint                     │
│  ┌───────────────────────────────────┐ │
│  │  SMD Container                     │ │
│  │  ┌───────────────────────────────┐ │ │
│  │  │  inference.py (auto-generated)│ │ │
│  │  │  - Loads serve.pkl           │ │ │
│  │  │  - Calls orchestrator.handle()│ │ │
│  │  └───────────────────────────────┘ │ │
│  │  ┌───────────────────────────────┐ │ │
│  │  │  Your CustomOrchestrator      │ │ │
│  │  │  - handle(data) method         │ │ │
│  │  │  - Your custom logic          │ │ │
│  │  └───────────────────────────────┘ │ │
│  └───────────────────────────────────┘ │
└─────────────────────────────────────────┘
```

## Comparison: CustomOrchestrator vs InferenceSpec

### InferenceSpec (Standard)
```python
from sagemaker.serve.spec.inference_spec import InferenceSpec

class MyInferenceSpec(InferenceSpec):
    def load(self, model_dir):
        # Load model from directory
        return model
    
    def invoke(self, input_object, model):
        # Run inference
        return model.predict(input_object)
```

**Characteristics:**
- Works with specific model servers (TorchServe, Triton, etc.)
- Separates loading (`load`) and inference (`invoke`)
- Model server handles the request routing
- More structured, less flexible

### CustomOrchestrator
```python
from sagemaker.serve.spec.inference_base import CustomOrchestrator

class MyOrchestrator(CustomOrchestrator):
    def handle(self, data, context=None):
        # Everything in one method - complete control
        # Load model, preprocess, predict, postprocess
        return result
```

**Characteristics:**
- Works with SMD container only
- Single `handle()` method does everything
- You control the entire request flow
- More flexible, less structure

## Example: CustomOrchestrator

```python
from sagemaker.serve.spec.inference_base import AsyncCustomOrchestrator
from sagemaker.serve.model_builder import ModelBuilder
from sagemaker.serve.utils.types import ModelServer

class MyCustomOrchestrator(AsyncCustomOrchestrator):
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    async def handle(self, data, context=None):
        """
        Handle inference request.
        
        Args:
            data: Raw input data (dict, list, bytes, etc.)
            context: Optional context (not commonly used)
            
        Returns:
            Prediction result
        """
        # 1. Preprocess input
        processed_input = self.preprocess(data)
        
        # 2. Run inference
        prediction = self.model.predict(processed_input)
        
        # 3. Postprocess output
        result = self.postprocess(prediction)
        
        return result
    
    def preprocess(self, data):
        """Custom preprocessing logic."""
        if isinstance(data, dict):
            return data.get('features', data.get('data', data))
        return data
    
    def postprocess(self, prediction):
        """Custom postprocessing logic."""
        # Format output as needed
        return {"prediction": prediction.tolist()}

# Use with ModelBuilder
model_builder = ModelBuilder(
    model=my_trained_model,
    inference_spec=MyCustomOrchestrator(my_trained_model),
    # model_server=ModelServer.SMD,  # Automatically set!
    role_arn="arn:aws:iam::...",
    instance_type="ml.m5.xlarge"
)

model = model_builder.build()
endpoint = model_builder.deploy()
```

## What Happens Behind the Scenes

### 1. ModelBuilder Packaging (`prepare_for_smd`):
```python
# ModelBuilder does this:
1. Saves your CustomOrchestrator to serve.pkl
2. Copies custom_execution_inference.py → inference.py
3. Captures dependencies
4. Generates secret key for integrity checking
5. Uploads everything to S3
```

### 2. Container Startup (`custom_execution_inference.py`):
```python
# In the container:
1. Loads serve.pkl (your CustomOrchestrator)
2. Runs integrity check
3. Exposes handler() function that calls orchestrator.handle()
```

### 3. Inference Request:
```python
# When request comes in:
1. SageMaker sends request to /invocations
2. inference.py handler() receives request
3. Calls custom_orchestrator.handle(request.body)
4. Your handle() method processes it
5. Returns result
```

## When to Use CustomOrchestrator

### ✅ Use CustomOrchestrator when:
- You need complete control over inference flow
- You want custom preprocessing/postprocessing
- You need to call multiple models or services
- You want async/await support
- You need complex request routing logic
- You're building a custom API (like FastAPI)

### ❌ Don't use CustomOrchestrator when:
- You want to use a specific model server (TorchServe, Triton, etc.)
- You prefer the structured load/invoke pattern
- You want framework-specific optimizations
- You need model server features (batching, multi-model, etc.)

## Key Differences Summary

| Feature | InferenceSpec | CustomOrchestrator |
|--------|---------------|-------------------|
| **Model Server** | TorchServe, Triton, etc. | SMD only |
| **Methods** | `load()`, `invoke()` | `handle()` |
| **Control** | Structured | Complete |
| **Flexibility** | Medium | High |
| **Use Case** | Standard inference | Custom logic/APIs |
| **Container** | Model server specific | SageMaker Distribution |

## Real-World Example: FastAPI Server

```python
class FastAPIOrchestrator(AsyncCustomOrchestrator):
    """Custom orchestrator for FastAPI server."""
    
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    async def handle(self, data, context=None):
        # Handle different input formats
        if isinstance(data, bytes):
            import json
            data = json.loads(data.decode('utf-8'))
        
        # Extract features
        features = data.get('features', data.get('data', data))
        
        # Run inference
        result = self.model.predict(features)
        
        # Format response
        return {
            "prediction": result.tolist(),
            "status": "success"
        }
```

## Built-in Features

### 1. SageMaker Runtime Client
```python
class MyOrchestrator(CustomOrchestrator):
    async def handle(self, data, context=None):
        # Access SageMaker runtime client
        # (useful for calling other endpoints)
        response = self.client.invoke_endpoint(
            EndpointName="other-endpoint",
            Body=data
        )
        return response
```

### 2. Integrity Checking
- ModelBuilder automatically adds integrity checks
- Prevents tampering with model artifacts
- Uses secret key and hash validation

### 3. Dependency Management
- ModelBuilder captures dependencies automatically
- Packages them with your orchestrator
- Ensures consistent environment

## Code Flow Diagram

```
User Code
    ↓
CustomOrchestrator.handle(data)
    ↓
Your Custom Logic
    ├─→ Preprocess
    ├─→ Model Inference
    ├─→ Postprocess
    └─→ Return Result
        ↓
SageMaker Endpoint
    ↓
Client receives response
```

## Summary

**CustomOrchestrator** is a way to:
- Write your own inference logic
- Have complete control over request handling
- Use async/await for better performance
- Build custom APIs (like FastAPI)
- Avoid being tied to specific model servers

It's automatically packaged by ModelBuilder and deployed using SMD (SageMaker Distribution) containers, giving you maximum flexibility while still benefiting from SageMaker's deployment infrastructure.
