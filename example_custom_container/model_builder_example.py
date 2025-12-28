"""
Example: Using ModelBuilder with a Custom Container

This example shows three ways to use ModelBuilder with custom containers:
1. Custom container with InferenceSpec
2. Custom container with model_server specified
3. Custom container with SMD (SageMaker Distribution)
"""

import boto3
from sagemaker.serve.model_builder import ModelBuilder
from sagemaker.serve.spec.inference_spec import InferenceSpec
from sagemaker.serve.utils.types import ModelServer
from sagemaker.serve.mode.function_pointers import Mode


# ============================================================================
# Example 1: Custom Container with InferenceSpec
# ============================================================================

class MyCustomInferenceSpec(InferenceSpec):
    """Custom inference specification for your model."""
    
    def load(self, model_dir: str):
        """Load your model from the model directory."""
        import pickle
        model_path = f"{model_dir}/model.pkl"
        with open(model_path, 'rb') as f:
            return pickle.load(f)
    
    def invoke(self, input_object: object, model: object):
        """Run inference with your model."""
        # Your custom inference logic here
        return model.predict(input_object)


def example_1_custom_container_with_inference_spec():
    """Example using custom container with InferenceSpec."""
    
    # Your custom container image URI (must be in ECR)
    custom_image_uri = "123456789012.dkr.ecr.us-west-2.amazonaws.com/my-custom-model:latest"
    
    # Create ModelBuilder with custom container
    model_builder = ModelBuilder(
        model="path/to/your/model.pkl",  # Or your model object
        inference_spec=MyCustomInferenceSpec(),
        image_uri=custom_image_uri,
        model_server=ModelServer.SMD,  # Required for custom containers
        role_arn="arn:aws:iam::123456789012:role/SageMakerExecutionRole",
        instance_type="ml.m5.xlarge",
        mode=Mode.SAGEMAKER_ENDPOINT,
    )
    
    # Build the model
    model = model_builder.build()
    
    # Deploy to endpoint
    endpoint = model_builder.deploy(endpoint_name="my-custom-endpoint")
    
    # Make predictions
    result = endpoint.invoke(data={"features": [1, 2, 3]})
    
    return endpoint


# ============================================================================
# Example 2: Custom Container with Specific Model Server
# ============================================================================

def example_2_custom_container_with_model_server():
    """Example using custom container with a specific model server."""
    
    # Your custom TorchServe container (for example)
    custom_torchserve_image = "123456789012.dkr.ecr.us-west-2.amazonaws.com/my-torchserve:latest"
    
    model_builder = ModelBuilder(
        model=my_pytorch_model,  # Your PyTorch model object
        image_uri=custom_torchserve_image,
        model_server=ModelServer.TORCHSERVE,  # Specify the model server
        role_arn="arn:aws:iam::123456789012:role/SageMakerExecutionRole",
        instance_type="ml.g5.xlarge",
    )
    
    model = model_builder.build()
    endpoint = model_builder.deploy()
    
    return endpoint


# ============================================================================
# Example 3: Custom Container with Pre-built Model Artifacts
# ============================================================================

def example_3_custom_container_passthrough():
    """Example using custom container in passthrough mode (image only)."""
    
    # Custom container that already has everything configured
    custom_image = "123456789012.dkr.ecr.us-west-2.amazonaws.com/my-complete-model:latest"
    
    # For passthrough, you need to set up the container definition manually
    # This is more advanced and requires the container to handle everything
    from sagemaker.core.resources import Model, ContainerDefinition
    
    container_def = ContainerDefinition(
        image=custom_image,
        model_data_source={
            "s3_data_source": {
                "s3_uri": "s3://my-bucket/my-model/model.tar.gz",
                "s3_data_type": "S3Prefix"
            }
        },
        environment={
            "SAGEMAKER_PROGRAM": "inference.py",
            "SAGEMAKER_REGION": "us-west-2"
        }
    )
    
    model = Model.create(
        model_name="my-passthrough-model",
        primary_container=container_def,
        execution_role_arn="arn:aws:iam::123456789012:role/SageMakerExecutionRole"
    )
    
    return model


# ============================================================================
# Example 4: Building and Pushing Custom Container
# ============================================================================

def build_and_push_custom_container():
    """
    Steps to build and push your custom container to ECR:
    
    1. Build the Docker image:
       docker build -t my-custom-model:latest .
    
    2. Create ECR repository (if not exists):
       aws ecr create-repository --repository-name my-custom-model --region us-west-2
    
    3. Get login token:
       aws ecr get-login-password --region us-west-2 | docker login --username AWS --password-stdin 123456789012.dkr.ecr.us-west-2.amazonaws.com
    
    4. Tag the image:
       docker tag my-custom-model:latest 123456789012.dkr.ecr.us-west-2.amazonaws.com/my-custom-model:latest
    
    5. Push to ECR:
       docker push 123456789012.dkr.ecr.us-west-2.amazonaws.com/my-custom-model:latest
    
    6. Use the image URI in ModelBuilder:
       image_uri = "123456789012.dkr.ecr.us-west-2.amazonaws.com/my-custom-model:latest"
    """
    pass


# ============================================================================
# Example 5: Complete Workflow with Custom Container
# ============================================================================

def complete_custom_container_example():
    """Complete example workflow."""
    
    # Step 1: Build and push your custom container (see build_and_push_custom_container)
    custom_image_uri = "123456789012.dkr.ecr.us-west-2.amazonaws.com/my-custom-model:latest"
    
    # Step 2: Create session
    session = boto3.Session(region_name='us-west-2')
    
    # Step 3: Create ModelBuilder with custom container
    model_builder = ModelBuilder(
        model="path/to/model.pkl",
        inference_spec=MyCustomInferenceSpec(),
        image_uri=custom_image_uri,
        model_server=ModelServer.SMD,  # Required when using custom image_uri
        role_arn="arn:aws:iam::123456789012:role/SageMakerExecutionRole",
        instance_type="ml.m5.xlarge",
        sagemaker_session=session,
        env_vars={
            "CUSTOM_ENV_VAR": "value",
            "LOG_LEVEL": "INFO"
        }
    )
    
    # Step 4: Build model (creates SageMaker Model resource)
    print("Building model...")
    model = model_builder.build()
    print(f"Model created: {model.model_name}")
    
    # Step 5: Deploy endpoint
    print("Deploying endpoint...")
    endpoint = model_builder.deploy(
        endpoint_name="my-custom-endpoint",
        initial_instance_count=1,
        wait=True
    )
    print(f"Endpoint deployed: {endpoint.endpoint_name}")
    
    # Step 6: Test inference
    print("Testing inference...")
    test_data = {"features": [1.0, 2.0, 3.0]}
    result = endpoint.invoke(data=test_data)
    print(f"Prediction result: {result}")
    
    return endpoint


if __name__ == "__main__":
    # Run the complete example
    endpoint = complete_custom_container_example()
    
    # Clean up (optional)
    # endpoint.delete()
    # model_builder.built_model.delete()
