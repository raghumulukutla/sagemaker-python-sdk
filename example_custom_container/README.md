# Custom Container Example for SageMaker ModelBuilder

This example demonstrates how to use ModelBuilder with a custom Docker container.

## Overview

When using ModelBuilder with a custom container, you have several options:

1. **Custom container with InferenceSpec** - ModelBuilder packages your model and uses your custom container
2. **Custom container with model_server** - Use a custom container that implements a specific model server (TorchServe, Triton, etc.)
3. **Custom container passthrough** - Container handles everything, ModelBuilder just creates the Model resource

## Requirements

- Docker installed locally
- AWS CLI configured
- SageMaker execution role with appropriate permissions
- ECR repository for your container image

## Files Structure

```
example_custom_container/
├── Dockerfile              # Custom container definition
├── requirements.txt        # Python dependencies
├── code/
│   └── inference.py       # Inference handler code
├── model_builder_example.py  # ModelBuilder usage examples
└── README.md              # This file
```

## Quick Start

### 1. Build Your Custom Container

```bash
# Build the Docker image
docker build -t my-custom-model:latest .

# Create ECR repository (if needed)
aws ecr create-repository --repository-name my-custom-model --region us-west-2

# Get ECR login token
aws ecr get-login-password --region us-west-2 | \
  docker login --username AWS --password-stdin \
  123456789012.dkr.ecr.us-west-2.amazonaws.com

# Tag the image
docker tag my-custom-model:latest \
  123456789012.dkr.ecr.us-west-2.amazonaws.com/my-custom-model:latest

# Push to ECR
docker push 123456789012.dkr.ecr.us-west-2.amazonaws.com/my-custom-model:latest
```

### 2. Use ModelBuilder with Custom Container

```python
from sagemaker.serve.model_builder import ModelBuilder
from sagemaker.serve.utils.types import ModelServer

model_builder = ModelBuilder(
    model=my_model,
    inference_spec=MyInferenceSpec(),
    image_uri="123456789012.dkr.ecr.us-west-2.amazonaws.com/my-custom-model:latest",
    model_server=ModelServer.SMD,  # Required for custom containers
    role_arn="arn:aws:iam::123456789012:role/SageMakerExecutionRole",
    instance_type="ml.m5.xlarge"
)

model = model_builder.build()
endpoint = model_builder.deploy()
```

## Important Notes

### Container Requirements

Your custom container must:

1. **Expose port 8080** - SageMaker uses this port for inference requests
2. **Handle SageMaker inference contract** - Implement the handler functions:
   - `model_fn(model_dir)` - Load model
   - `input_fn(request_body, content_type)` - Deserialize input
   - `predict_fn(input_data, model)` - Run inference
   - `output_fn(prediction, content_type)` - Serialize output

3. **Follow SageMaker directory structure**:
   ```
   /opt/ml/model/          # Model artifacts location
   /opt/ml/model/code/     # Inference code location
   ```

### ModelBuilder Behavior

- When `image_uri` is provided, ModelBuilder will:
  - Package your model artifacts
  - Upload them to S3
  - Create a Model resource pointing to your custom container
  - The container receives model artifacts at `/opt/ml/model/`

- **model_server is required** when using a custom `image_uri` that is not a first-party SageMaker image

### Supported Model Servers

You can specify these model servers with custom containers:
- `ModelServer.SMD` - SageMaker Distribution (recommended for custom orchestrators)
- `ModelServer.TORCHSERVE` - TorchServe
- `ModelServer.TRITON` - NVIDIA Triton
- `ModelServer.DJL_SERVING` - Deep Java Library
- `ModelServer.TENSORFLOW_SERVING` - TensorFlow Serving
- `ModelServer.MMS` - Multi-Model Server
- `ModelServer.TGI` - Text Generation Inference
- `ModelServer.TEI` - Text Embeddings Inference

## Example Use Cases

1. **Custom Pre/Post Processing** - Add custom data transformations
2. **Specialized Hardware** - Use containers optimized for specific hardware
3. **Legacy Models** - Deploy models that require specific runtime environments
4. **Multi-Framework** - Combine multiple frameworks in one container
5. **Custom Security** - Add custom security or compliance checks

## Troubleshooting

### Container doesn't start
- Check CloudWatch logs for the endpoint
- Verify port 8080 is exposed
- Ensure handler function exists and is callable

### Model not found
- Verify model artifacts are uploaded to S3
- Check `/opt/ml/model/` path in container
- Ensure ModelBuilder uploaded artifacts correctly

### Inference errors
- Verify input/output serialization matches your handler
- Check content types match between client and handler
- Review CloudWatch logs for detailed error messages

## References

- [SageMaker ModelBuilder Documentation](https://docs.aws.amazon.com/sagemaker/latest/dg/model-building.html)
- [SageMaker Inference Container Contract](https://docs.aws.amazon.com/sagemaker/latest/dg/your-algorithms-inference-code.html)
- [ECR Documentation](https://docs.aws.amazon.com/ecr/)
