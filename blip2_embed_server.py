#!/usr/bin/env python3
"""
BLIP2 Embedding Server for DFIY
OpenAI-API compatible embedding server using BLIP2 model
"""

import os
import io
import base64
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
from functools import lru_cache

import torch
import torch.nn.functional as F
from PIL import Image
import httpx
from transformers import Blip2Processor, Blip2Model
from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global FastAPI app instance
app = FastAPI(title="BLIP2 Embedding Server", version="1.0.0")

# --- Data Models ---
class EmbeddingRequest(BaseModel):
    input: List[str] = Field(..., description="List of input texts or image URLs/Base64 strings")
    model: str = Field(default="blip2-embed", description="Model name")
    encoding_format: str = Field(default="float", description="Encoding format")
    user: Optional[str] = Field(default=None, description="User identifier")

class EmbeddingResponse(BaseModel):
    object: str = Field(default="list", description="Response object type")
    data: List[Dict[str, Any]] = Field(..., description="List of embedding data")
    model: str = Field(..., description="Model name")
    usage: Dict[str, int] = Field(..., description="Usage statistics")

class EmbeddingData(BaseModel):
    object: str = Field(default="embedding", description="Embedding object type")
    embedding: List[float] = Field(..., description="Embedding vector")
    index: int = Field(..., description="Index")

class SimilarityRequest(BaseModel):
    embedding1: List[float] = Field(..., description="First embedding vector")
    embedding2: List[float] = Field(..., description="Second embedding vector")
    metric: str = Field(default="cosine", description="Similarity metric")

class SimilarityResponse(BaseModel):
    similarity: float = Field(..., description="Similarity score")
    metric: str = Field(..., description="Used metric")

# --- Model and Utility Functions ---
@lru_cache(maxsize=1)
def get_blip2_model():
    """Load the BLIP2 model using lru_cache for singleton pattern."""
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")
        
        model_name = "Salesforce/blip2-opt-2.7b"
        logger.info(f"Loading embedding model: {model_name}")
        
        processor = Blip2Processor.from_pretrained(model_name)
        model = Blip2Model.from_pretrained(model_name)
        model.to(device)
        model.eval()
        
        logger.info("Model loaded successfully.")
        return processor, model, device
    except Exception as e:
        logger.error(f"Model loading failed: {e}")
        raise RuntimeError(f"Model loading failed: {e}")

async def load_image_from_url(url: str) -> Image.Image:
    """Load image from URL asynchronously."""
    async with httpx.AsyncClient() as client:
        response = await client.get(url, timeout=10)
        response.raise_for_status()
        return Image.open(io.BytesIO(response.content))

def load_image_from_base64(base64_string: str) -> Image.Image:
    """Load image from base64 string."""
    if ',' in base64_string:
        base64_string = base64_string.split(',')[1]
    image_data = base64.b64decode(base64_string)
    return Image.open(io.BytesIO(image_data))

def is_valid_url(text: str) -> bool:
    """Checks if a string is a valid URL."""
    return text.startswith('http://') or text.startswith('https://')

def is_valid_base64(text: str) -> bool:
    """Checks if a string is a valid base64-encoded image."""
    return text.startswith('data:image/') or (len(text) > 100 and not is_valid_url(text) and ' ' not in text)

# --- API Endpoints ---
@app.on_event("startup")
async def startup_event():
    """Load the model at application startup."""
    try:
        get_blip2_model()
    except RuntimeError as e:
        logger.error(f"Startup failed: {e}")
        raise

@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "BLIP2 Embedding Server is running"}

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        processor, model, device = get_blip2_model()
        return {
            "status": "healthy",
            "model": "blip2-embed",
            "device": str(device),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Health check failed: {str(e)}")

@app.post("/v1/embeddings", response_model=EmbeddingResponse)
async def create_embeddings(
    request: EmbeddingRequest,
    model_deps: tuple = Depends(get_blip2_model)
):
    """Creates embedding vectors for texts or images."""
    processor, model, device = model_deps
    
    embeddings = []
    total_tokens = 0
    
    for i, input_item in enumerate(request.input):
        try:
            # Check for image input first
            if is_valid_url(input_item) or is_valid_base64(input_item):
                if is_valid_url(input_item):
                    image = await load_image_from_url(input_item)
                else:
                    image = load_image_from_base64(input_item)
                
                # --- 关键修复：直接调用模型的视觉部分 ---
                inputs = processor(images=image, return_tensors="pt").to(device)
                
                with torch.no_grad():
                    image_features = model.vision_model(inputs.pixel_values)[0]
                    # 获取图像特征的平均池化版本作为嵌入
                    embedding = F.normalize(image_features.mean(dim=1), p=2, dim=1).cpu().numpy().flatten().tolist()
                token_count = 1 
            else:
                # If not an image, it must be text
                inputs = processor(text=input_item, return_tensors="pt").to(device)
                
                with torch.no_grad():
                    text_features = model.language_model.get_input_embeddings()(inputs.input_ids)
                    embedding = F.normalize(text_features.mean(dim=1), p=2, dim=1).cpu().numpy().flatten().tolist()
                
                token_count = inputs.input_ids.shape[1]
            
            embedding_data = EmbeddingData(
                object="embedding",
                embedding=embedding,
                index=i
            )
            embeddings.append(embedding_data.dict())
            total_tokens += token_count
            
        except HTTPException as e:
            raise e
        except Exception as e:
            logger.error(f"Failed to process input {i}: {e}")
            raise HTTPException(status_code=400, detail=f"Failed to process input {i}: {str(e)}")
    
    response = EmbeddingResponse(
        object="list",
        data=embeddings,
        model=request.model,
        usage={
            "prompt_tokens": total_tokens,
            "total_tokens": total_tokens
        }
    )
    
    return response

@app.post("/v1/similarity", response_model=SimilarityResponse)
async def calculate_embedding_similarity(request: SimilarityRequest):
    """Calculates the similarity between two embedding vectors."""
    try:
        similarity = calculate_similarity(
            request.embedding1, 
            request.embedding2, 
            request.metric
        )
        return SimilarityResponse(similarity=similarity, metric=request.metric)
    except Exception as e:
        logger.error(f"Failed to calculate similarity: {e}")
        raise HTTPException(status_code=400, detail=f"Failed to calculate similarity: {str(e)}")

@app.get("/models")
async def list_models():
    """Lists available models."""
    return {
        "object": "list",
        "data": [{"id": "blip2-embed", "object": "model", "created": 1640995200, "owned_by": "blip2-embedding-server", "permission": [], "root": "blip2-embed", "parent": None, "embedding_dimensions": 768}]
    }

@app.get("/v1/models")
async def list_models_v1():
    """OpenAI-compatible models list API."""
    return await list_models()

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(status_code=500, content={"error": {"message": "Internal server error", "type": "internal_server_error", "code": 500}})

if __name__ == "__main__":
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8001"))
    
    logger.info(f"Starting BLIP2 embedding server at: {host}:{port}")
    uvicorn.run(app, host=host, port=port)
