"""
Side-by-Side Comparison: CustomOrchestrator vs InferenceSpec

This file shows the same functionality implemented both ways.
"""

from sagemaker.serve.spec.inference_base import AsyncCustomOrchestrator
from sagemaker.serve.spec.inference_spec import InferenceSpec
from sagemaker.serve.model_builder import ModelBuilder
from sagemaker.serve.utils.types import ModelServer
import numpy as np


# ============================================================================
# SAME FUNCTIONALITY: Two Different Approaches
# ============================================================================

# Goal: Predict with a model, handling dict input and returning formatted output

# ----------------------------------------------------------------------------
# Approach 1: CustomOrchestrator
# ----------------------------------------------------------------------------

class CustomOrchestratorApproach(AsyncCustomOrchestrator):
    """
    CustomOrchestrator approach:
    - Single handle() method
    - Complete control
    - SMD only
    """
    
    def __init__(self, model):
        super().__init__()
        self.model = model  # Model loaded in constructor
    
    async def handle(self, data, context=None):
        """
        Handle inference request.
        Everything happens here.
        """
        # Parse input
        if isinstance(data, bytes):
            import json
            data = json.loads(data.decode('utf-8'))
        
        # Extract features
        if isinstance(data, dict):
            features = data.get('features', data.get('data', []))
        else:
            features = data
        
        # Convert to numpy
        features = np.array(features)
        if len(features.shape) == 1:
            features = features.reshape(1, -1)
        
        # Predict
        prediction = self.model.predict(features)
        
        # Format output
        return {
            "prediction": prediction.tolist(),
            "status": "success"
        }


# ----------------------------------------------------------------------------
# Approach 2: InferenceSpec
# ----------------------------------------------------------------------------

class InferenceSpecApproach(InferenceSpec):
    """
    InferenceSpec approach:
    - load() and invoke() methods
    - Structured separation
    - Works with multiple model servers
    """
    
    def load(self, model_dir: str):
        """
        Load model once at startup.
        Called by model server.
        """
        import pickle
        model_path = f"{model_dir}/model.pkl"
        with open(model_path, 'rb') as f:
            return pickle.load(f)
    
    def invoke(self, input_object: object, model: object):
        """
        Run inference per request.
        Called by model server for each request.
        """
        # Extract features
        if isinstance(input_object, dict):
            features = input_object.get('features', input_object.get('data', []))
        else:
            features = input_object
        
        # Convert to numpy
        features = np.array(features)
        if len(features.shape) == 1:
            features = features.reshape(1, -1)
        
        # Predict
        prediction = model.predict(features)
        
        # Format output
        return {
            "prediction": prediction.tolist(),
            "status": "success"
        }


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

def example_custom_orchestrator():
    """Example using CustomOrchestrator."""
    
    from sklearn.linear_model import LogisticRegression
    
    # Create model
    X = np.array([[1, 2], [3, 4], [5, 6]])
    y = np.array([0, 1, 0])
    model = LogisticRegression()
    model.fit(X, y)
    
    # Create orchestrator
    orchestrator = CustomOrchestratorApproach(model)
    
    # Use with ModelBuilder
    model_builder = ModelBuilder(
        model=model,
        inference_spec=orchestrator,
        # model_server=ModelServer.SMD  # Automatically set!
        role_arn="arn:aws:iam::123456789012:role/SageMakerExecutionRole",
        instance_type="ml.m5.xlarge"
    )
    
    model_resource = model_builder.build()
    endpoint = model_builder.deploy()
    
    # Test
    result = endpoint.invoke(data={"features": [1, 2]})
    print(f"CustomOrchestrator result: {result}")
    
    return endpoint


def example_inference_spec():
    """Example using InferenceSpec."""
    
    from sklearn.linear_model import LogisticRegression
    
    # Create model
    X = np.array([[1, 2], [3, 4], [5, 6]])
    y = np.array([0, 1, 0])
    model = LogisticRegression()
    model.fit(X, y)
    
    # Create InferenceSpec
    inference_spec = InferenceSpecApproach()
    
    # Use with ModelBuilder
    model_builder = ModelBuilder(
        model=model,
        inference_spec=inference_spec,
        model_server=ModelServer.TORCHSERVE,  # Can choose!
        role_arn="arn:aws:iam::123456789012:role/SageMakerExecutionRole",
        instance_type="ml.m5.xlarge"
    )
    
    model_resource = model_builder.build()
    endpoint = model_builder.deploy()
    
    # Test
    result = endpoint.invoke(data={"features": [1, 2]})
    print(f"InferenceSpec result: {result}")
    
    return endpoint


# ============================================================================
# KEY DIFFERENCES DEMONSTRATED
# ============================================================================

def demonstrate_differences():
    """
    Show the key differences between the two approaches.
    """
    
    print("=" * 60)
    print("CUSTOM ORCHESTRATOR")
    print("=" * 60)
    print("""
    ✅ Single handle() method
    ✅ Complete control over request flow
    ✅ Model loaded in constructor or handle()
    ✅ Works with SMD only
    ✅ Best for custom APIs (FastAPI, etc.)
    ✅ Async support built-in
    
    Code structure:
        class MyOrchestrator(AsyncCustomOrchestrator):
            def __init__(self, model):
                self.model = model
            
            async def handle(self, data, context=None):
                # Everything here
                return result
    """)
    
    print("\n" + "=" * 60)
    print("INFERENCE SPEC")
    print("=" * 60)
    print("""
    ✅ Two methods: load() and invoke()
    ✅ Structured separation of concerns
    ✅ Model loaded once in load()
    ✅ Works with multiple model servers
    ✅ Best for standard inference patterns
    ✅ Model server manages lifecycle
    
    Code structure:
        class MyInferenceSpec(InferenceSpec):
            def load(self, model_dir: str):
                # Load once
                return model
            
            def invoke(self, input_object, model):
                # Invoke per request
                return prediction
    """)
    
    print("\n" + "=" * 60)
    print("WHEN TO USE WHICH")
    print("=" * 60)
    print("""
    Use CustomOrchestrator when:
    - Building custom APIs (FastAPI, Flask, etc.)
    - Need complete control over request handling
    - Complex preprocessing/postprocessing logic
    - Async operations required
    - Custom routing logic needed
    
    Use InferenceSpec when:
    - Using specific model servers (TorchServe, Triton, etc.)
    - Standard inference patterns
    - Want model server optimizations
    - Need model server features (batching, multi-model)
    - Prefer structured load/invoke separation
    """)


if __name__ == "__main__":
    demonstrate_differences()
    
    # Uncomment to run examples:
    # endpoint1 = example_custom_orchestrator()
    # endpoint2 = example_inference_spec()
