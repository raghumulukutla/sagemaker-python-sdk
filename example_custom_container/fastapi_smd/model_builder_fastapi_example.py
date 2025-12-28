"""
Example: Using ModelBuilder with FastAPI Server and SMD

This example shows how to create a FastAPI server that works with
ModelBuilder's CustomOrchestrator and SMD (SageMaker Distribution).
"""

import boto3
from sagemaker.serve.model_builder import ModelBuilder
from sagemaker.serve.spec.inference_base import AsyncCustomOrchestrator
from sagemaker.serve.utils.types import ModelServer
from sagemaker.serve.mode.function_pointers import Mode
import numpy as np


# ============================================================================
# Example 1: FastAPI with AsyncCustomOrchestrator
# ============================================================================

class FastAPIAsyncOrchestrator(AsyncCustomOrchestrator):
    """
    Custom orchestrator that works with FastAPI.
    
    The FastAPI server will call this handle method for inference.
    """
    
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    async def handle(self, data, context=None):
        """
        Handle inference request.
        
        Args:
            data: Input data (can be dict, list, numpy array, etc.)
            context: Optional context (not used in this example)
            
        Returns:
            Prediction result
        """
        # Preprocess if needed
        processed_data = self.preprocess(data)
        
        # Run inference
        if hasattr(self.model, 'predict'):
            prediction = self.model.predict(processed_data)
        elif hasattr(self.model, '__call__'):
            prediction = self.model(processed_data)
        else:
            raise ValueError("Model does not support prediction")
        
        # Postprocess if needed
        result = self.postprocess(prediction)
        
        return result
    
    def preprocess(self, data):
        """Preprocess input data."""
        # Example: Convert dict to numpy array if needed
        if isinstance(data, dict):
            if 'features' in data:
                return np.array(data['features'])
            elif 'data' in data:
                return np.array(data['data'])
        elif isinstance(data, list):
            return np.array(data)
        return data
    
    def postprocess(self, prediction):
        """Postprocess prediction output."""
        # Convert numpy arrays to lists for JSON serialization
        if isinstance(prediction, np.ndarray):
            return prediction.tolist()
        return prediction


# ============================================================================
# Example 2: Simple Model with FastAPI
# ============================================================================

def example_fastapi_with_simple_model():
    """Example using a simple scikit-learn model with FastAPI."""
    
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.datasets import make_classification
    
    # Create a simple model for demonstration
    X, y = make_classification(n_samples=100, n_features=4, random_state=42)
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)
    
    # Create orchestrator
    orchestrator = FastAPIAsyncOrchestrator(model)
    
    # Custom FastAPI container image (must be built and pushed to ECR)
    custom_image_uri = "123456789012.dkr.ecr.us-west-2.amazonaws.com/fastapi-model:latest"
    
    # Create ModelBuilder
    model_builder = ModelBuilder(
        model=model,  # Your trained model
        inference_spec=orchestrator,  # Custom orchestrator
        image_uri=custom_image_uri,
        model_server=ModelServer.SMD,  # Use SMD for custom orchestrators
        role_arn="arn:aws:iam::123456789012:role/SageMakerExecutionRole",
        instance_type="ml.m5.xlarge",
        mode=Mode.SAGEMAKER_ENDPOINT,
    )
    
    # Build model
    print("Building model...")
    model_resource = model_builder.build()
    print(f"Model created: {model_resource.model_name}")
    
    # Deploy endpoint
    print("Deploying endpoint...")
    endpoint = model_builder.deploy(
        endpoint_name="fastapi-model-endpoint",
        initial_instance_count=1,
        wait=True
    )
    print(f"Endpoint deployed: {endpoint.endpoint_name}")
    
    # Test inference
    print("Testing inference...")
    test_data = {"features": [1.0, 2.0, 3.0, 4.0]}
    result = endpoint.invoke(data=test_data)
    print(f"Prediction: {result}")
    
    return endpoint


# ============================================================================
# Example 3: FastAPI with Custom Preprocessing
# ============================================================================

class AdvancedFastAPIOrchestrator(AsyncCustomOrchestrator):
    """Advanced orchestrator with custom preprocessing and postprocessing."""
    
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    async def handle(self, data, context=None):
        """Handle inference with advanced preprocessing."""
        # Extract features from different input formats
        if isinstance(data, dict):
            # Handle different input formats
            if 'features' in data:
                features = data['features']
            elif 'data' in data:
                features = data['data']
            elif 'input' in data:
                features = data['input']
            else:
                # Use all values as features
                features = list(data.values())
        elif isinstance(data, list):
            features = data
        else:
            features = data
        
        # Convert to numpy array
        if not isinstance(features, np.ndarray):
            features = np.array(features)
        
        # Reshape if needed (for single sample)
        if len(features.shape) == 1:
            features = features.reshape(1, -1)
        
        # Run inference
        prediction = self.model.predict(features)
        probabilities = None
        
        if hasattr(self.model, 'predict_proba'):
            probabilities = self.model.predict_proba(features)
        
        # Format response
        result = {
            "prediction": prediction.tolist() if isinstance(prediction, np.ndarray) else prediction,
            "probabilities": probabilities.tolist() if probabilities is not None else None,
            "input_shape": features.shape
        }
        
        return result


# ============================================================================
# Example 4: Complete FastAPI Workflow
# ============================================================================

def complete_fastapi_workflow():
    """Complete workflow for FastAPI with ModelBuilder."""
    
    # Step 1: Train or load your model
    from sklearn.linear_model import LogisticRegression
    from sklearn.datasets import make_classification
    
    X, y = make_classification(n_samples=1000, n_features=10, random_state=42)
    model = LogisticRegression(random_state=42)
    model.fit(X, y)
    
    # Step 2: Create custom orchestrator
    orchestrator = FastAPIAsyncOrchestrator(model)
    
    # Step 3: Build and push FastAPI container (see README)
    # docker build -t fastapi-model:latest -f Dockerfile .
    # docker tag fastapi-model:latest 123456789012.dkr.ecr.us-west-2.amazonaws.com/fastapi-model:latest
    # docker push 123456789012.dkr.ecr.us-west-2.amazonaws.com/fastapi-model:latest
    
    custom_image_uri = "123456789012.dkr.ecr.us-west-2.amazonaws.com/fastapi-model:latest"
    
    # Step 4: Create session
    session = boto3.Session(region_name='us-west-2')
    
    # Step 5: Create ModelBuilder
    model_builder = ModelBuilder(
        model=model,
        inference_spec=orchestrator,
        image_uri=custom_image_uri,
        model_server=ModelServer.SMD,
        role_arn="arn:aws:iam::123456789012:role/SageMakerExecutionRole",
        instance_type="ml.m5.xlarge",
        sagemaker_session=session,
        env_vars={
            "LOG_LEVEL": "INFO",
            "SAGEMAKER_BIND_TO_PORT": "8080"
        }
    )
    
    # Step 6: Build model
    print("Building model...")
    model_resource = model_builder.build()
    print(f"✅ Model created: {model_resource.model_name}")
    
    # Step 7: Deploy endpoint
    print("Deploying endpoint...")
    endpoint = model_builder.deploy(
        endpoint_name="fastapi-smd-endpoint",
        initial_instance_count=1,
        wait=True
    )
    print(f"✅ Endpoint deployed: {endpoint.endpoint_name}")
    
    # Step 8: Test inference
    print("\nTesting inference...")
    test_cases = [
        {"features": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]},
        {"data": [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5]},
    ]
    
    for i, test_data in enumerate(test_cases):
        print(f"\nTest case {i+1}: {test_data}")
        result = endpoint.invoke(data=test_data)
        print(f"Result: {result}")
    
    # Step 9: Test FastAPI endpoints directly (if using invoke_endpoint with custom endpoint)
    # You can also call /ping, /model-info, /predict endpoints
    
    return endpoint, model_builder


# ============================================================================
# Example 5: Testing FastAPI Endpoints Locally
# ============================================================================

def test_fastapi_locally():
    """Test FastAPI server locally before deploying."""
    
    # This would be run in your FastAPI container locally
    # python code/inference.py
    
    # Or use the ModelBuilder local mode
    from sklearn.linear_model import LogisticRegression
    from sklearn.datasets import make_classification
    
    X, y = make_classification(n_samples=100, n_features=4, random_state=42)
    model = LogisticRegression(random_state=42)
    model.fit(X, y)
    
    orchestrator = FastAPIAsyncOrchestrator(model)
    
    model_builder = ModelBuilder(
        model=model,
        inference_spec=orchestrator,
        image_uri="fastapi-model:latest",  # Local image
        model_server=ModelServer.SMD,
        role_arn="arn:aws:iam::123456789012:role/SageMakerExecutionRole",
        mode=Mode.LOCAL_CONTAINER,  # Test locally
    )
    
    # Build and deploy locally
    model_resource = model_builder.build()
    local_endpoint = model_builder.deploy()
    
    # Test locally
    result = local_endpoint.invoke(data={"features": [1.0, 2.0, 3.0, 4.0]})
    print(f"Local test result: {result}")
    
    return local_endpoint


if __name__ == "__main__":
    # Run complete workflow
    endpoint, model_builder = complete_fastapi_workflow()
    
    print("\n" + "="*50)
    print("FastAPI Model Server deployed successfully!")
    print("="*50)
    print(f"\nEndpoint: {endpoint.endpoint_name}")
    print(f"Model: {model_builder.built_model.model_name}")
    print("\nYou can now:")
    print("1. Call endpoint.invoke(data={...}) for predictions")
    print("2. Access FastAPI endpoints: /ping, /model-info, /predict")
    print("3. Use standard SageMaker /invocations endpoint")
    
    # Clean up (optional)
    # endpoint.delete()
    # model_builder.built_model.delete()
