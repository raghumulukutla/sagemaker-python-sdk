"""
Simple CustomOrchestrator Example

This shows the most basic CustomOrchestrator implementation.
"""

from sagemaker.serve.spec.inference_base import CustomOrchestrator, AsyncCustomOrchestrator
from sagemaker.serve.model_builder import ModelBuilder
from sagemaker.serve.utils.types import ModelServer


# ============================================================================
# Example 1: Simple Synchronous CustomOrchestrator
# ============================================================================

class SimpleOrchestrator(CustomOrchestrator):
    """
    Simplest possible CustomOrchestrator.
    
    Just takes data and returns a result.
    """
    
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    def handle(self, data, context=None):
        """
        Handle inference request.
        
        Args:
            data: Input data (can be anything: dict, list, bytes, etc.)
            context: Optional context (usually None)
            
        Returns:
            Prediction result
        """
        # Simple: just call model.predict()
        return self.model.predict(data)


# ============================================================================
# Example 2: Async CustomOrchestrator
# ============================================================================

class SimpleAsyncOrchestrator(AsyncCustomOrchestrator):
    """
    Simple async CustomOrchestrator.
    
    Same as sync version but with async/await support.
    """
    
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    async def handle(self, data, context=None):
        """Async version of handle."""
        # Can use await here for async operations
        return self.model.predict(data)


# ============================================================================
# Example 3: CustomOrchestrator with Preprocessing
# ============================================================================

class PreprocessingOrchestrator(CustomOrchestrator):
    """
    CustomOrchestrator with preprocessing logic.
    """
    
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    def handle(self, data, context=None):
        # Step 1: Parse input
        if isinstance(data, bytes):
            import json
            data = json.loads(data.decode('utf-8'))
        
        # Step 2: Extract features
        if isinstance(data, dict):
            features = data.get('features', data.get('data', list(data.values())))
        else:
            features = data
        
        # Step 3: Run inference
        prediction = self.model.predict(features)
        
        # Step 4: Format output
        return {"prediction": prediction.tolist()}


# ============================================================================
# Example 4: Complete Workflow
# ============================================================================

def example_usage():
    """Complete example of using CustomOrchestrator."""
    
    # Assume you have a trained model
    from sklearn.linear_model import LogisticRegression
    import numpy as np
    
    # Create a simple model
    X = np.array([[1, 2], [3, 4], [5, 6]])
    y = np.array([0, 1, 0])
    model = LogisticRegression()
    model.fit(X, y)
    
    # Create orchestrator
    orchestrator = SimpleOrchestrator(model)
    
    # Use with ModelBuilder
    model_builder = ModelBuilder(
        model=model,
        inference_spec=orchestrator,  # Pass your CustomOrchestrator here
        # Note: model_server=ModelServer.SMD is automatically set!
        role_arn="arn:aws:iam::123456789012:role/SageMakerExecutionRole",
        instance_type="ml.m5.xlarge"
    )
    
    # Build and deploy
    model_resource = model_builder.build()
    endpoint = model_builder.deploy(endpoint_name="my-orchestrator-endpoint")
    
    # Test inference
    result = endpoint.invoke(data={"features": [1, 2]})
    print(f"Prediction: {result}")
    
    return endpoint


# ============================================================================
# Key Points to Remember
# ============================================================================

"""
1. CustomOrchestrator requires ONE method: handle(data, context=None)

2. handle() receives raw data - you decide how to parse it

3. handle() returns the result - you decide the format

4. ModelBuilder automatically:
   - Uses SMD container
   - Packages your orchestrator to serve.pkl
   - Sets up inference.py to call handle()

5. You have complete control over:
   - Input parsing
   - Preprocessing
   - Model inference
   - Postprocessing
   - Output formatting
"""


if __name__ == "__main__":
    # Run example
    endpoint = example_usage()
