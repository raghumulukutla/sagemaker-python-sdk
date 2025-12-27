# SMD Usage Clarification

## When is SMD Required?

Based on the codebase analysis:

### ✅ SMD is REQUIRED when:
1. **Using CustomOrchestrator or AsyncCustomOrchestrator**
   - Code at `model_builder.py:2665-2666` FORCES SMD:
   ```python
   self.image_uri = self._get_smd_image_uri(processing_unit=cpu_or_gpu_instance)
   self.model_server = ModelServer.SMD
   ```
   - This happens automatically when ModelBuilder detects a CustomOrchestrator
   - You CANNOT override this - it's hardcoded

### ❌ SMD is NOT required when:
1. **Using custom containers with other model servers**
   - You CAN use custom `image_uri` with TORCHSERVE, TRITON, DJL_SERVING, etc.
   - You just need to specify `model_server` explicitly
   - Example:
   ```python
   model_builder = ModelBuilder(
       model=my_model,
       image_uri="my-custom-torchserve-image:latest",
       model_server=ModelServer.TORCHSERVE,  # Not SMD!
       ...
   )
   ```

2. **Using InferenceSpec (not CustomOrchestrator)**
   - Regular `InferenceSpec` can work with different model servers
   - SMD is only forced for CustomOrchestrator

## For the FastAPI Example

The FastAPI example I provided uses `AsyncCustomOrchestrator`, which means:
- ✅ **SMD is REQUIRED** for that specific example
- ✅ The code is correct as written
- ✅ ModelBuilder will automatically use SMD when it detects CustomOrchestrator

## Alternative: FastAPI without CustomOrchestrator

If you want to use FastAPI with a different model server, you would need to:
1. Structure your FastAPI server to work with that model server's contract
2. NOT use CustomOrchestrator
3. Use InferenceSpec instead (or no inference_spec)
4. Specify the appropriate model_server

Example:
```python
# This would use your custom FastAPI container with TorchServe
model_builder = ModelBuilder(
    model=my_model,
    image_uri="my-fastapi-torchserve:latest",
    model_server=ModelServer.TORCHSERVE,  # Not SMD
    inference_spec=MyInferenceSpec(),  # Not CustomOrchestrator
    ...
)
```

But this would require your FastAPI server to follow TorchServe's contract, not SageMaker's standard contract.

## Summary

| Use Case | SMD Required? | Notes |
|----------|---------------|-------|
| CustomOrchestrator | ✅ YES | Automatically forced |
| AsyncCustomOrchestrator | ✅ YES | Automatically forced |
| Custom image + TORCHSERVE | ❌ NO | Use ModelServer.TORCHSERVE |
| Custom image + TRITON | ❌ NO | Use ModelServer.TRITON |
| Custom image + InferenceSpec | ❌ NO | Depends on model_server |
| FastAPI + CustomOrchestrator | ✅ YES | Our example - correct! |
| FastAPI + Other model server | ❌ NO | Different structure needed |

## Code Evidence

From `model_builder.py:2656-2666`:
```python
if isinstance(self.inference_spec, (CustomOrchestrator, AsyncCustomOrchestrator)):
    # ... validation ...
    self.image_uri = self._get_smd_image_uri(processing_unit=cpu_or_gpu_instance)
    self.model_server = ModelServer.SMD  # FORCED!
```

From `model_builder.py:871-875`:
```python
if self.image_uri and not is_1p_image_uri(self.image_uri) and self.model_server is None:
    raise ValueError(
        f"Model_server must be set when non-first-party image_uri is set. "
        f"Supported model servers: {SUPPORTED_MODEL_SERVERS}"
    )
```

This validation allows custom images with any model server, as long as you specify it.
