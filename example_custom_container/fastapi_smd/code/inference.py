"""
FastAPI Server for SageMaker ModelBuilder with SMD (SageMaker Distribution).

This FastAPI server integrates with ModelBuilder's CustomOrchestrator
and works with SageMaker's inference contract.
"""

import os
import logging
import asyncio
from pathlib import Path
from typing import Optional, Dict, Any
import cloudpickle
import uvicorn
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import json

from sagemaker.serve.validations.check_integrity import perform_integrity_check

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Initialize FastAPI app
app = FastAPI(title="SageMaker Model Server", version="1.0.0")

# Global model variable
model = None
custom_orchestrator = None


class PredictionRequest(BaseModel):
    """Request model for predictions."""
    data: Any  # Can be dict, list, etc.


class PredictionResponse(BaseModel):
    """Response model for predictions."""
    prediction: Any
    status: str = "success"


def load_model_from_modelbuilder():
    """Load model from ModelBuilder's packaged artifacts."""
    global model, custom_orchestrator
    
    code_dir = os.getenv("SAGEMAKER_INFERENCE_CODE_DIRECTORY", "/opt/ml/model/code")
    serve_pkl = Path(code_dir) / "serve.pkl"
    
    if serve_pkl.exists():
        logger.info(f"Loading model from {serve_pkl}")
        
        # Integrity check
        metadata_path = Path(code_dir) / "metadata.json"
        if metadata_path.exists():
            with open(serve_pkl, "rb") as f:
                buffer = f.read()
            perform_integrity_check(buffer=buffer, metadata_path=metadata_path)
        
        # Load the model/orchestrator
        with open(serve_pkl, "rb") as f:
            loaded_obj = cloudpickle.load(f)
        
        # Check if it's a CustomOrchestrator or a model
        if hasattr(loaded_obj, 'handle'):
            custom_orchestrator = loaded_obj
            logger.info("Loaded CustomOrchestrator")
        else:
            model = loaded_obj
            logger.info("Loaded model object")
    else:
        logger.warning(f"serve.pkl not found at {serve_pkl}")
        # Fallback: try to load from model directory
        model_dir = "/opt/ml/model"
        model_path = Path(model_dir) / "model.pkl"
        if model_path.exists():
            with open(model_path, "rb") as f:
                model = cloudpickle.load(f)
            logger.info(f"Loaded model from {model_path}")


@app.on_event("startup")
async def startup_event():
    """Load model when server starts."""
    logger.info("Starting FastAPI server...")
    load_model_from_modelbuilder()
    logger.info("Server ready!")


@app.get("/ping")
async def ping():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "SageMaker FastAPI Model Server",
        "status": "running",
        "model_loaded": model is not None or custom_orchestrator is not None
    }


@app.post("/invocations", response_model=PredictionResponse)
async def invocations(request: Request):
    """
    SageMaker inference endpoint.
    
    This endpoint follows SageMaker's inference contract and can handle
    both CustomOrchestrator and direct model inference.
    """
    try:
        # Get request body
        body = await request.body()
        content_type = request.headers.get("Content-Type", "application/json")
        
        # Parse input based on content type
        if content_type == "application/json":
            try:
                input_data = json.loads(body)
            except json.JSONDecodeError:
                input_data = body.decode('utf-8')
        else:
            input_data = body
        
        # Use CustomOrchestrator if available
        if custom_orchestrator:
            logger.info("Using CustomOrchestrator for inference")
            if asyncio.iscoroutinefunction(custom_orchestrator.handle):
                result = await custom_orchestrator.handle(input_data)
            else:
                # Run sync handler in thread pool
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    None, custom_orchestrator.handle, input_data
                )
        elif model:
            logger.info("Using direct model inference")
            # Direct model inference
            if hasattr(model, 'predict'):
                result = model.predict(input_data)
            elif hasattr(model, '__call__'):
                result = model(input_data)
            else:
                raise ValueError("Model does not have predict or __call__ method")
        else:
            raise HTTPException(
                status_code=500,
                detail="Model not loaded. Please check model artifacts."
            )
        
        return PredictionResponse(prediction=result, status="success")
        
    except Exception as e:
        logger.error(f"Error during inference: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict")
async def predict(request: PredictionRequest):
    """
    Alternative prediction endpoint (non-SageMaker standard).
    
    Useful for testing and direct API calls.
    """
    try:
        input_data = request.data
        
        if custom_orchestrator:
            if asyncio.iscoroutinefunction(custom_orchestrator.handle):
                result = await custom_orchestrator.handle(input_data)
            else:
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    None, custom_orchestrator.handle, input_data
                )
        elif model:
            if hasattr(model, 'predict'):
                result = model.predict(input_data)
            elif hasattr(model, '__call__'):
                result = model(input_data)
            else:
                raise ValueError("Model does not have predict or __call__ method")
        else:
            raise HTTPException(
                status_code=500,
                detail="Model not loaded"
            )
        
        return {"prediction": result, "status": "success"}
        
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/model-info")
async def model_info():
    """Get information about the loaded model."""
    info = {
        "model_loaded": model is not None or custom_orchestrator is not None,
        "model_type": None,
        "has_custom_orchestrator": custom_orchestrator is not None,
    }
    
    if model:
        info["model_type"] = type(model).__name__
        if hasattr(model, '__dict__'):
            info["model_attributes"] = list(model.__dict__.keys())
    
    if custom_orchestrator:
        info["orchestrator_type"] = type(custom_orchestrator).__name__
    
    return info


if __name__ == "__main__":
    # Run FastAPI server with uvicorn
    port = int(os.getenv("SAGEMAKER_BIND_TO_PORT", "8080"))
    host = os.getenv("SAGEMAKER_BIND_TO_PORT", "0.0.0.0")
    
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info",
        access_log=True
    )
