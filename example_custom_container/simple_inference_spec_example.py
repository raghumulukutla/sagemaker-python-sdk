"""
Simple InferenceSpec Example

This shows the most basic InferenceSpec implementation.
"""

from sagemaker.serve.spec.inference_spec import InferenceSpec
from sagemaker.serve.model_builder import ModelBuilder
from sagemaker.serve.utils.types import ModelServer
import pickle
import numpy as np


# ============================================================================
# Example 1: Basic InferenceSpec
# ============================================================================

class SimpleInferenceSpec(InferenceSpec):
    """
    Simplest possible InferenceSpec.
    
    Just loads model and runs predictions.
    """
    
    def load(self, model_dir: str):
        """
        Load model from directory.
        Called ONCE when container starts.
        """
        model_path = f"{model_dir}/model.pkl"
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        return model
    
    def invoke(self, input_object: object, model: object):
        """
        Run inference.
        Called for EACH request.
        """
        # Simple: just call model.predict()
        return model.predict(input_object)


# ============================================================================
# Example 2: InferenceSpec with Preprocessing
# ============================================================================

class PreprocessingInferenceSpec(InferenceSpec):
    """
    InferenceSpec with input preprocessing.
    """
    
    def load(self, model_dir: str):
        """Load model."""
        model_path = f"{model_dir}/model.pkl"
        with open(model_path, 'rb') as f:
            return pickle.load(f)
    
    def invoke(self, input_object: object, model: object):
        """Run inference with preprocessing."""
        # Preprocess input
        processed = self.preprocess(input_object)
        
        # Run prediction
        prediction = model.predict(processed)
        
        return prediction
    
    def preprocess(self, input_data: object):
        """Preprocess input data."""
        # Extract features from dict
        if isinstance(input_data, dict):
            features = input_data.get('features', input_data.get('data'))
        else:
            features = input_data
        
        # Convert to numpy array
        if not isinstance(features, np.ndarray):
            features = np.array(features)
        
        # Reshape if single sample
        if len(features.shape) == 1:
            features = features.reshape(1, -1)
        
        return features


# ============================================================================
# Example 3: Complete InferenceSpec
# ============================================================================

class CompleteInferenceSpec(InferenceSpec):
    """
    Complete InferenceSpec with preprocessing and postprocessing.
    """
    
    def load(self, model_dir: str):
        """Load model and any configs."""
        model_path = f"{model_dir}/model.pkl"
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        # Could load additional configs here
        # self.config = load_config(f"{model_dir}/config.json")
        
        return model
    
    def invoke(self, input_object: object, model: object):
        """Complete inference pipeline."""
        # Step 1: Preprocess
        processed_input = self.preprocess(input_object)
        
        # Step 2: Predict
        prediction = model.predict(processed_input)
        
        # Step 3: Postprocess
        result = self.postprocess(prediction)
        
        return result
    
    def preprocess(self, input_data: object):
        """Custom preprocessing."""
        if isinstance(input_data, dict):
            features = input_data.get('features', input_data.get('data', []))
        else:
            features = input_data
        
        # Convert to numpy
        features = np.array(features)
        
        # Reshape if needed
        if len(features.shape) == 1:
            features = features.reshape(1, -1)
        
        return features
    
    def postprocess(self, predictions: object):
        """Custom postprocessing."""
        # Format output
        if isinstance(predictions, np.ndarray):
            return {
                "prediction": predictions.tolist(),
                "status": "success"
            }
        return {"prediction": predictions}


# ============================================================================
# Example 4: Usage with ModelBuilder
# ============================================================================

def example_usage():
    """Complete example of using InferenceSpec."""
    
    # Assume you have a trained model
    from sklearn.linear_model import LogisticRegression
    
    # Create and train a model
    X = np.array([[1, 2], [3, 4], [5, 6]])
    y = np.array([0, 1, 0])
    model = LogisticRegression()
    model.fit(X, y)
    
    # Save model (ModelBuilder will handle this, but showing for clarity)
    # import pickle
    # with open('model.pkl', 'wb') as f:
    #     pickle.dump(model, f)
    
    # Create InferenceSpec
    inference_spec = SimpleInferenceSpec()
    
    # Use with ModelBuilder
    model_builder = ModelBuilder(
        model=model,  # Your trained model
        inference_spec=inference_spec,
        model_server=ModelServer.TORCHSERVE,  # Can use any model server!
        role_arn="arn:aws:iam::123456789012:role/SageMakerExecutionRole",
        instance_type="ml.m5.xlarge"
    )
    
    # Build and deploy
    model_resource = model_builder.build()
    endpoint = model_builder.deploy(endpoint_name="my-inference-spec-endpoint")
    
    # Test inference
    result = endpoint.invoke(data={"features": [1, 2]})
    print(f"Prediction: {result}")
    
    return endpoint


# ============================================================================
# Key Points to Remember
# ============================================================================

"""
1. InferenceSpec requires TWO methods:
   - load(model_dir) - Called ONCE at startup
   - invoke(input_object, model) - Called for EACH request

2. load() returns the model object, which is kept in memory

3. invoke() receives:
   - input_object: The input data
   - model: The loaded model from load()

4. ModelBuilder automatically:
   - Saves InferenceSpec to serve.pkl
   - Model server loads it and calls load() once
   - For each request, calls invoke()

5. Works with multiple model servers:
   - TorchServe
   - Triton
   - MMS
   - DJL Serving
   - TensorFlow Serving
   - etc.

6. Optional methods:
   - preprocess() - Custom preprocessing
   - postprocess() - Custom postprocessing
   - prepare() - Custom preparation
"""


if __name__ == "__main__":
    # Run example
    endpoint = example_usage()
