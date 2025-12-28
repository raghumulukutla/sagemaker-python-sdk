# CustomOrchestrator vs InferenceSpec: Complete Comparison

## Quick Summary

| Aspect | CustomOrchestrator | InferenceSpec |
|--------|-------------------|---------------|
| **Methods** | `handle(data)` | `load(model_dir)`, `invoke(input, model)` |
| **Model Servers** | SMD only | TorchServe, Triton, MMS, DJL, TGI, TEI, etc. |
| **Structure** | Single-phase | Two-phase |
| **Control** | Complete control | Structured control |
| **Use Case** | Custom APIs, complex logic | Standard inference patterns |
| **When Model Loads** | In handle() (or before) | Separate load() method |

## Detailed Comparison

### 1. Method Structure

#### CustomOrchestrator
```python
class MyOrchestrator(AsyncCustomOrchestrator):
    async def handle(self, data, context=None):
        # Everything happens here:
        # - Load model (if not already loaded)
        # - Preprocess
        # - Predict
        # - Postprocess
        return result
```

**Single method** - Everything in `handle()`

#### InferenceSpec
```python
class MyInferenceSpec(InferenceSpec):
    def load(self, model_dir: str):
        # Load model ONCE
        return model
    
    def invoke(self, input_object: object, model: object):
        # Run inference per request
        return prediction
```

**Two methods** - Separation of concerns

### 2. Model Loading

#### CustomOrchestrator
```python
class MyOrchestrator(CustomOrchestrator):
    def __init__(self, model):
        super().__init__()
        self.model = model  # Model passed in constructor
    
    def handle(self, data, context=None):
        # Model already available
        return self.model.predict(data)
```

- Model can be passed in constructor
- Or loaded inside `handle()` if needed
- You control when/how model loads

#### InferenceSpec
```python
class MyInferenceSpec(InferenceSpec):
    def load(self, model_dir: str):
        # Called ONCE by model server
        return load_model(model_dir)
    
    def invoke(self, input_object: object, model: object):
        # Model passed as parameter
        return model.predict(input_object)
```

- `load()` called once at container startup
- Model kept in memory
- `invoke()` receives model as parameter
- Model server manages model lifecycle

### 3. Model Server Compatibility

#### CustomOrchestrator
```python
# ONLY works with SMD
model_builder = ModelBuilder(
    inference_spec=MyOrchestrator(model),
    # model_server=ModelServer.SMD  # Automatically set!
)
```

- **SMD only** - SageMaker Distribution container
- Automatically forced by ModelBuilder
- Cannot use with TorchServe, Triton, etc.

#### InferenceSpec
```python
# Works with multiple model servers
model_builder = ModelBuilder(
    inference_spec=MyInferenceSpec(),
    model_server=ModelServer.TORCHSERVE,  # Can choose!
    # or ModelServer.TRITON
    # or ModelServer.MMS
    # or ModelServer.DJL_SERVING
    # etc.
)
```

- **Multiple model servers** supported
- You choose which one to use
- Each server has its own optimizations

### 4. Request Flow

#### CustomOrchestrator Flow
```
Request arrives
    ↓
handler() receives request
    ↓
orchestrator.handle(data)  ← Your code
    ↓
    ├─→ Load model (if needed)
    ├─→ Preprocess
    ├─→ Predict
    └─→ Postprocess
    ↓
Return result
```

**You control the entire flow**

#### InferenceSpec Flow
```
Container Startup:
    ↓
inference_spec.load(model_dir)  ← Called ONCE
    ↓
Model loaded, kept in memory
    ↓
─────────────────────────────────
Request arrives
    ↓
Model server receives request
    ↓
inference_spec.invoke(input, model)  ← Your code
    ↓
    ├─→ Preprocess (optional)
    ├─→ Predict
    └─→ Postprocess (optional)
    ↓
Return result
```

**Model server handles routing, you handle inference**

### 5. Code Examples

#### CustomOrchestrator Example
```python
from sagemaker.serve.spec.inference_base import AsyncCustomOrchestrator

class FastAPIOrchestrator(AsyncCustomOrchestrator):
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    async def handle(self, data, context=None):
        # Parse input
        if isinstance(data, bytes):
            import json
            data = json.loads(data.decode('utf-8'))
        
        # Extract features
        features = data.get('features', [])
        
        # Predict
        result = self.model.predict(features)
        
        # Format response
        return {
            "prediction": result.tolist(),
            "status": "success"
        }

# Usage
model_builder = ModelBuilder(
    model=my_model,
    inference_spec=FastAPIOrchestrator(my_model),
    # SMD automatically used
    role_arn="...",
    instance_type="ml.m5.xlarge"
)
```

#### InferenceSpec Example
```python
from sagemaker.serve.spec.inference_spec import InferenceSpec
import pickle

class MyInferenceSpec(InferenceSpec):
    def load(self, model_dir: str):
        # Load model once
        with open(f"{model_dir}/model.pkl", 'rb') as f:
            return pickle.load(f)
    
    def invoke(self, input_object: object, model: object):
        # Preprocess
        if isinstance(input_object, dict):
            features = input_object.get('features', [])
        else:
            features = input_object
        
        # Predict
        result = model.predict(features)
        
        return result.tolist()

# Usage
model_builder = ModelBuilder(
    model=my_model,
    inference_spec=MyInferenceSpec(),
    model_server=ModelServer.TORCHSERVE,  # Choose server!
    role_arn="...",
    instance_type="ml.m5.xlarge"
)
```

### 6. When to Use Each

#### Use CustomOrchestrator When:

✅ **You need complete control**
- Custom request routing
- Complex preprocessing/postprocessing
- Multiple models or services

✅ **Building custom APIs**
- FastAPI server
- Custom REST endpoints
- GraphQL APIs

✅ **Async operations**
- Async/await support
- Concurrent request handling
- Background tasks

✅ **Custom logic**
- Business rules in inference
- Multi-step workflows
- Conditional routing

#### Use InferenceSpec When:

✅ **Using specific model servers**
- TorchServe optimizations
- Triton performance
- MMS multi-model support

✅ **Standard inference patterns**
- Simple load → predict → return
- Framework-specific optimizations
- Model server features (batching, etc.)

✅ **Separation of concerns**
- Clean load/invoke separation
- Model lifecycle management
- Standardized interface

✅ **Multiple model servers**
- Want flexibility to switch servers
- Testing different servers
- Production with specific server

### 7. Key Differences Table

| Feature | CustomOrchestrator | InferenceSpec |
|---------|-------------------|---------------|
| **Entry Point** | `handle(data)` | `load()`, `invoke()` |
| **Model Loading** | In handle() or constructor | Separate `load()` method |
| **Model Server** | SMD only | Multiple options |
| **Request Handling** | You control everything | Model server handles routing |
| **Async Support** | Yes (AsyncCustomOrchestrator) | Depends on model server |
| **Flexibility** | Very high | Medium |
| **Structure** | Single method | Two methods |
| **Use Case** | Custom APIs, complex logic | Standard inference |
| **Model Lifecycle** | You manage | Model server manages |
| **Batching** | You implement | Model server handles |
| **Multi-Model** | You implement | Model server supports |

### 8. Side-by-Side Code Comparison

#### Same Functionality, Different Approaches

**CustomOrchestrator:**
```python
class MyOrchestrator(AsyncCustomOrchestrator):
    def __init__(self, model):
        super().__init__()
        self.model = model  # Loaded here
    
    async def handle(self, data, context=None):
        # Everything in one place
        processed = self.preprocess(data)
        result = self.model.predict(processed)
        return self.postprocess(result)
```

**InferenceSpec:**
```python
class MyInferenceSpec(InferenceSpec):
    def load(self, model_dir: str):
        # Load once
        return load_model(model_dir)
    
    def invoke(self, input_object: object, model: object):
        # Invoke per request
        processed = self.preprocess(input_object)
        result = model.predict(processed)
        return self.postprocess(result)
```

### 9. Real-World Scenarios

#### Scenario 1: Simple Model Inference

**CustomOrchestrator:**
```python
class SimpleOrchestrator(CustomOrchestrator):
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    def handle(self, data, context=None):
        return self.model.predict(data)
```
✅ Works, but forces SMD

**InferenceSpec:**
```python
class SimpleInferenceSpec(InferenceSpec):
    def load(self, model_dir: str):
        return load_model(model_dir)
    
    def invoke(self, input_object: object, model: object):
        return model.predict(input_object)
```
✅ Better - can use any model server

#### Scenario 2: Custom API with FastAPI

**CustomOrchestrator:**
```python
class FastAPIOrchestrator(AsyncCustomOrchestrator):
    async def handle(self, data, context=None):
        # Custom API logic
        return custom_api_response(data)
```
✅ Perfect - complete control

**InferenceSpec:**
```python
class FastAPIInferenceSpec(InferenceSpec):
    # Would need to work within model server constraints
    # Less flexible for custom APIs
```
❌ Not ideal - constrained by model server

#### Scenario 3: Using TorchServe Optimizations

**CustomOrchestrator:**
```python
# Cannot use TorchServe - forced to SMD
```
❌ Not possible

**InferenceSpec:**
```python
model_builder = ModelBuilder(
    inference_spec=MyInferenceSpec(),
    model_server=ModelServer.TORCHSERVE  # Use TorchServe!
)
```
✅ Perfect - can use TorchServe

### 10. Migration Guide

#### From InferenceSpec to CustomOrchestrator

```python
# Before (InferenceSpec)
class MyInferenceSpec(InferenceSpec):
    def load(self, model_dir: str):
        self.model = load_model(model_dir)
        return self.model
    
    def invoke(self, input_object: object, model: object):
        return model.predict(input_object)

# After (CustomOrchestrator)
class MyOrchestrator(CustomOrchestrator):
    def __init__(self, model):
        super().__init__()
        self.model = model  # Load in constructor
    
    def handle(self, data, context=None):
        return self.model.predict(data)
```

#### From CustomOrchestrator to InferenceSpec

```python
# Before (CustomOrchestrator)
class MyOrchestrator(CustomOrchestrator):
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    def handle(self, data, context=None):
        return self.model.predict(data)

# After (InferenceSpec)
class MyInferenceSpec(InferenceSpec):
    def load(self, model_dir: str):
        return load_model(model_dir)
    
    def invoke(self, input_object: object, model: object):
        return model.predict(input_object)
```

## Decision Tree

```
Do you need complete control over request handling?
├─ YES → Use CustomOrchestrator
│   └─ Building custom API? → CustomOrchestrator
│   └─ Complex logic? → CustomOrchestrator
│   └─ Async operations? → AsyncCustomOrchestrator
│
└─ NO → Use InferenceSpec
    └─ Want specific model server? → InferenceSpec
    └─ Standard inference? → InferenceSpec
    └─ Need model server features? → InferenceSpec
```

## Summary

**CustomOrchestrator:**
- Single `handle()` method
- Complete control
- SMD only
- Best for custom APIs and complex logic

**InferenceSpec:**
- `load()` and `invoke()` methods
- Structured approach
- Multiple model servers
- Best for standard inference patterns

Choose based on your needs:
- **Control & Flexibility** → CustomOrchestrator
- **Model Server Features** → InferenceSpec
