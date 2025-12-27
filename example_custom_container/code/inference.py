"""
Custom inference handler for SageMaker ModelBuilder with custom container.

This handler follows SageMaker's inference contract and can work with
different model servers based on how ModelBuilder packages your model.
"""

import json
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Model will be loaded once at container startup
model = None


def model_fn(model_dir):
    """
    Load the model from the model directory.
    
    This function is called once when the container starts.
    
    Args:
        model_dir (str): Path to directory containing model artifacts
        
    Returns:
        Loaded model object
    """
    global model
    
    logger.info(f"Loading model from {model_dir}")
    
    # Example: Load a simple model
    # Replace this with your actual model loading logic
    model_path = Path(model_dir) / "model.pkl"
    
    if model_path.exists():
        import pickle
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        logger.info("Model loaded successfully")
    else:
        # For ModelBuilder with InferenceSpec, model might be in code directory
        code_dir = Path(model_dir) / "code"
        serve_pkl = code_dir / "serve.pkl"
        
        if serve_pkl.exists():
            import cloudpickle
            with open(serve_pkl, 'rb') as f:
                model = cloudpickle.load(f)
            logger.info("Model loaded from serve.pkl")
        else:
            raise FileNotFoundError(f"Model file not found in {model_dir}")
    
    return model


def input_fn(request_body, request_content_type):
    """
    Deserialize and prepare the prediction input.
    
    Args:
        request_body: The request payload
        request_content_type: The content type of the request
        
    Returns:
        Deserialized input object
    """
    logger.info(f"Received content type: {request_content_type}")
    
    if request_content_type == 'application/json':
        input_data = json.loads(request_body)
        return input_data
    elif request_content_type == 'text/csv':
        # Handle CSV input
        import pandas as pd
        import io
        return pd.read_csv(io.StringIO(request_body), header=None)
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")


def predict_fn(input_data, model):
    """
    Perform prediction on the deserialized input.
    
    Args:
        input_data: Deserialized input data
        model: The loaded model object
        
    Returns:
        Prediction result
    """
    logger.info("Making prediction")
    
    # Example: If model has a predict method
    if hasattr(model, 'predict'):
        result = model.predict(input_data)
    elif hasattr(model, '__call__'):
        result = model(input_data)
    else:
        # For InferenceSpec models, the model might be a callable
        result = model(input_data)
    
    return result


def output_fn(prediction, response_content_type):
    """
    Serialize the prediction result.
    
    Args:
        prediction: The prediction result
        response_content_type: The desired response content type
        
    Returns:
        Serialized prediction response
    """
    logger.info(f"Serializing prediction with content type: {response_content_type}")
    
    if response_content_type == 'application/json':
        return json.dumps(prediction)
    elif response_content_type == 'text/csv':
        import pandas as pd
        import io
        output = io.StringIO()
        pd.DataFrame(prediction).to_csv(output, header=False, index=False)
        return output.getvalue()
    else:
        raise ValueError(f"Unsupported response content type: {response_content_type}")


# Alternative handler for async/custom orchestrator style
# This is used when ModelBuilder uses SMD with CustomOrchestrator
async def handler(request):
    """
    Custom handler for async inference (used with SMD/CustomOrchestrator).
    
    Args:
        request: Request object with body attribute
        
    Returns:
        Prediction result
    """
    global model
    
    if model is None:
        model_dir = os.getenv("SAGEMAKER_INFERENCE_CODE_DIRECTORY", "/opt/ml/model")
        model = model_fn(model_dir)
    
    input_data = input_fn(request.body, request.headers.get('Content-Type', 'application/json'))
    prediction = predict_fn(input_data, model)
    response = output_fn(prediction, request.headers.get('Accept', 'application/json'))
    
    return response
