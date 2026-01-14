"""
Runpod Serverless Handler for vLLM with LoRA (HTTP Proxy Mode)
===============================================================
Runs a FastAPI server with AsyncLLMEngine for continuous batching.
"""

import os
import uuid
import asyncio
import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI
from pydantic import BaseModel
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm import SamplingParams
from vllm.lora.request import LoRARequest
from transformers import AutoTokenizer
import threading

MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"
LORA_ADAPTER_ID = "moo3030/Llama-3.2-1B-QLoRA-Summarizer-adapters"

# Get ports from environment
PORT = int(os.environ.get("PORT", 8000))
PORT_HEALTH = int(os.environ.get("PORT_HEALTH", 8080))

# Global model references
engine = None
tokenizer = None
lora_request = None


class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 256
    temperature: float = 0.7
    use_lora: bool = False


class GenerateResponse(BaseModel):
    response: str
    use_lora: bool


def load_model():
    """Load the model and tokenizer."""
    global engine, tokenizer, lora_request

    print(f"Loading model: {MODEL_ID}")

    engine_args = AsyncEngineArgs(
        model=MODEL_ID,
        gpu_memory_utilization=0.9,
        max_model_len=4096,
        enable_lora=True,
        max_loras=1,
        max_lora_rank=64,
    )
    engine = AsyncLLMEngine.from_engine_args(engine_args)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    lora_request = LoRARequest(
        lora_name="summarizer",
        lora_int_id=1,
        lora_path=LORA_ADAPTER_ID,
    )
    print("Model and LoRA adapter loaded successfully")


# Health check app (runs on PORT_HEALTH)
health_app = FastAPI(title="Health Check API")


@health_app.get("/ping")
@health_app.get("/health")
def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


# Main API app (runs on PORT)
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: model already loaded before uvicorn starts
    yield
    # Shutdown: cleanup if needed


app = FastAPI(title="vLLM Inference API", lifespan=lifespan)


@app.get("/ping")
@app.get("/health")
def api_health():
    """Health check for main app."""
    return {"status": "healthy", "model_loaded": engine is not None}


@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    """
    Generate text from prompt using AsyncLLMEngine.
    Supports continuous batching - multiple concurrent requests
    are batched together automatically by vLLM.
    """
    if engine is None:
        return {"error": "Model not loaded"}

    # Apply chat template
    messages = [{"role": "user", "content": request.prompt}]
    formatted_prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    # Generate with unique request ID for batching
    request_id = str(uuid.uuid4())
    params = SamplingParams(
        max_tokens=request.max_tokens,
        temperature=request.temperature,
        seed=42,
    )

    current_lora = lora_request if request.use_lora else None

    # Async generation - allows continuous batching
    results = engine.generate(
        formatted_prompt, params, request_id, lora_request=current_lora
    )

    final_output = None
    async for result in results:
        final_output = result

    response_text = final_output.outputs[0].text
    print(f"Generated {len(response_text)} chars (LoRA: {request.use_lora})")

    return GenerateResponse(response=response_text, use_lora=request.use_lora)


def run_health_server():
    """Run the health check server on PORT_HEALTH."""
    uvicorn.run(health_app, host="0.0.0.0", port=PORT_HEALTH, log_level="warning")


if __name__ == "__main__":
    # Load model first (before starting servers)
    load_model()

    # Start health check server in background thread
    health_thread = threading.Thread(target=run_health_server, daemon=True)
    health_thread.start()
    print(f"Health server running on port {PORT_HEALTH}")

    # Start main API server
    print(f"Starting main API server on port {PORT}")
    uvicorn.run(app, host="0.0.0.0", port=PORT)
