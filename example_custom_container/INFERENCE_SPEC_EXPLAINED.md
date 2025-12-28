# What is InferenceSpec?

## Overview

**InferenceSpec** is an abstract base class that defines a **standardized interface** for loading models and running inference. It separates model loading from inference execution, making it compatible with various model servers (TorchServe, Triton, MMS, etc.).

## Definition

```python
from sagemaker.serve.spec.inference_spec import InferenceSpec

class InferenceSpec(abc.ABC):
    @abc.abstractmethod
    def load(self, model_dir: str):
        """Load model from directory."""
        pass
    
    @abc.abstractmethod
    def invoke(self, input_object: object, model: object):
        """Run inference with loaded model."""
        pass
    
    def preprocess(self, input_data: object):
        """Optional preprocessing."""
        pass
    
    def postprocess(self, predictions: object):
        """Optional postprocessing."""
        pass
    
    def prepare(self, *args, **kwargs):
        """Optional preparation step."""
        pass
```

## Key Characteristics

### Two Main Methods:

1. **`load(model_dir)`** - Loads the model from a directory
   - Called **once** when the container starts
   - Returns the loaded model object
   - Model is kept in memory for subsequent requests

2. **`invoke(input_object, model)`** - Runs inference
   - Called **for each request**
   - Receives the input and the loaded model
   - Returns prediction result

### How It Works:

```
Container Startup:
    ↓
inference_spec.load(model_dir)  ← Called ONCE
    ↓
Model loaded and kept in memory
    ↓
─────────────────────────────────
Request 1:
    ↓
inference_spec.invoke(input1, model)  ← Called for each request
    ↓
Return prediction1

Request 2:
    ↓
inference_spec.invoke(input2, model)  ← Same model object reused
    ↓
Return prediction2
```

## Example: Basic InferenceSpec

```python
from sagemaker.serve.spec.inference_spec import InferenceSpec
import pickle
import numpy as np

class MyInferenceSpec(InferenceSpec):
    """Simple InferenceSpec for a scikit-learn model."""
    
    def load(self, model_dir: str):
        """
        Load model from directory.
        Called once when container starts.
        """
        model_path = f"{model_dir}/model.pkl"
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        return model
    
    def invoke(self, input_object: object, model: object):
        """
        Run inference.
        Called for each request.
        """
        # Preprocess if needed
        if isinstance(input_object, dict):
            features = input_object.get('features', input_object.get('data'))
        else:
            features = input_object
        
        # Convert to numpy array
        if not isinstance(features, np.ndarray):
            features = np.array(features)
        
        # Reshape if single sample
        if len(features.shape) == 1:
            features = features.reshape(1, -1)
        
        # Run prediction
        prediction = model.predict(features)
        
        return prediction.tolist()
```

## How Model Servers Use InferenceSpec

### Example: TorchServe

```python
# In TorchServe container:
inference_spec = load_from_serve_pkl()  # Load InferenceSpec
model = inference_spec.load(model_dir)  # Load model ONCE

# For each request:
prediction = inference_spec.invoke(input_data, model)  # Run inference
```

### Example: Triton

```python
# In Triton container:
class TritonPythonModel:
    def initialize(self, args):
        inference_spec = load_from_serve_pkl()
        self.model = inference_spec.load(model_dir)  # Load ONCE
    
    def execute(self, requests):
        for request in requests:
            input_data = parse_request(request)
            output = inference_spec.invoke(input_data, self.model)  # Invoke per request
            return output
```

## Comparison: InferenceSpec vs CustomOrchestrator

| Feature | InferenceSpec | CustomOrchestrator |
|--------|---------------|-------------------|
| **Methods** | `load()`, `invoke()` | `handle()` |
| **Model Loading** | Separate method | In handle() |
| **Model Servers** | TorchServe, Triton, MMS, etc. | SMD only |
| **Structure** | Two-phase (load, then invoke) | Single-phase (handle does everything) |
| **Use Case** | Standard inference patterns | Custom logic/APIs |
| **Flexibility** | Medium (structured) | High (complete control) |

## When to Use InferenceSpec

### ✅ Use InferenceSpec when:
- You want to use a specific model server (TorchServe, Triton, etc.)
- You prefer the structured load/invoke pattern
- You want model server features (batching, multi-model, etc.)
- You need framework-specific optimizations
- You want separation between loading and inference

### ❌ Don't use InferenceSpec when:
- You need complete control over request handling
- You want a single entry point (use CustomOrchestrator)
- You're building a custom API (use CustomOrchestrator)
- You need async/await with full control

## Complete Example

```python
from sagemaker.serve.spec.inference_spec import InferenceSpec
from sagemaker.serve.model_builder import ModelBuilder
from sagemaker.serve.utils.types import ModelServer
import pickle
import numpy as np

class MyInferenceSpec(InferenceSpec):
    """Complete InferenceSpec example."""
    
    def load(self, model_dir: str):
        """Load model once at startup."""
        print(f"Loading model from {model_dir}")
        model_path = f"{model_dir}/model.pkl"
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        print("Model loaded successfully")
        return model
    
    def invoke(self, input_object: object, model: object):
        """Run inference for each request."""
        # Preprocess
        features = self.preprocess(input_object)
        
        # Predict
        prediction = model.predict(features)
        
        # Postprocess
        result = self.postprocess(prediction)
        
        return result
    
    def preprocess(self, input_data: object):
        """Custom preprocessing."""
        if isinstance(input_data, dict):
            return np.array(input_data.get('features', []))
        return np.array(input_data)
    
    def postprocess(self, predictions: object):
        """Custom postprocessing."""
        return {"prediction": predictions.tolist()}

# Use with ModelBuilder
model_builder = ModelBuilder(
    model=my_trained_model,
    inference_spec=MyInferenceSpec(),
    model_server=ModelServer.TORCHSERVE,  # Can use any model server!
    role_arn="arn:aws:iam::...",
    instance_type="ml.m5.xlarge"
)

model = model_builder.build()
endpoint = model_builder.deploy()
```

## How ModelBuilder Uses InferenceSpec

### 1. Packaging Phase:
```python
# ModelBuilder saves InferenceSpec to serve.pkl
save_pkl(code_path, (inference_spec, schema_builder))
```

### 2. Container Startup:
```python
# Model server loads InferenceSpec
inference_spec, schema_builder = cloudpickle.load(serve_pkl)

# Calls load() once
model = inference_spec.load(model_dir)
```

### 3. Inference Requests:
```python
# For each request, calls invoke()
prediction = inference_spec.invoke(input_data, model)
```

## Advanced: InferenceSpec with Preprocessing/Postprocessing

```python
class AdvancedInferenceSpec(InferenceSpec):
    """InferenceSpec with preprocessing and postprocessing."""
    
    def load(self, model_dir: str):
        # Load model
        model = load_model(model_dir)
        
        # Load preprocessing/postprocessing configs
        self.preprocess_config = load_config(f"{model_dir}/preprocess.json")
        self.postprocess_config = load_config(f"{model_dir}/postprocess.json")
        
        return model
    
    def invoke(self, input_object: object, model: object):
        # Use preprocess method
        processed_input = self.preprocess(input_object)
        
        # Run inference
        prediction = model.predict(processed_input)
        
        # Use postprocess method
        result = self.postprocess(prediction)
        
        return result
    
    def preprocess(self, input_data: object):
        """Custom preprocessing logic."""
        # Normalize, scale, transform, etc.
        return normalized_data
    
    def postprocess(self, predictions: object):
        """Custom postprocessing logic."""
        # Format, filter, transform, etc.
        return formatted_result
```

## Supported Model Servers

InferenceSpec works with:
- ✅ **TorchServe** - PyTorch models
- ✅ **Triton** - NVIDIA Triton Inference Server
- ✅ **MMS** - Multi-Model Server
- ✅ **DJL Serving** - Deep Java Library
- ✅ **TensorFlow Serving** - TensorFlow models
- ✅ **TGI** - Text Generation Inference
- ✅ **TEI** - Text Embeddings Inference

## Key Differences from CustomOrchestrator

### InferenceSpec (Two-Phase):
```python
# Phase 1: Load (once)
model = inference_spec.load(model_dir)

# Phase 2: Invoke (per request)
result = inference_spec.invoke(input, model)
```

### CustomOrchestrator (Single-Phase):
```python
# Everything in one method
result = orchestrator.handle(data)
# (handles loading internally if needed)
```

## Summary

**InferenceSpec** is:
- A **standardized interface** for model loading and inference
- **Two methods**: `load()` (once) and `invoke()` (per request)
- **Compatible** with multiple model servers
- **Structured** approach separating loading from inference
- **Flexible** enough for custom preprocessing/postprocessing

It's the **standard way** to define custom inference logic when using model servers like TorchServe, Triton, etc., providing a clean separation between model loading and inference execution.
