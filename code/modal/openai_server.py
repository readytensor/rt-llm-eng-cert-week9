"""
vLLM on Modal using vllm serve (OpenAI-compatible API)
======================================================
Deploy: modal deploy app_server.py

This runs vLLM's built-in server, giving you a drop-in OpenAI API replacement.
"""

import modal
import subprocess

MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"
LORA_ADAPTER_ID = "moo3030/Llama-3.2-1B-QLoRA-Summarizer-adapters"
VLLM_PORT = 8000

app = modal.App("vllm-inference-openai")

image = modal.Image.debian_slim(python_version="3.11").pip_install("vllm==0.6.6")

volume = modal.Volume.from_name("llm-models", create_if_missing=True)


@app.function(
    gpu="L4",
    image=image,
    volumes={"/models": volume},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    container_idle_timeout=300,
    allow_concurrent_inputs=100,
    timeout=600,
)
@modal.web_server(port=VLLM_PORT, startup_timeout=300)
def serve():
    cmd = [
        "vllm",
        "serve",
        MODEL_ID,
        "--host",
        "0.0.0.0",
        "--port",
        str(VLLM_PORT),
        "--download-dir",
        "/models",
        "--gpu-memory-utilization",
        "0.9",
        "--max-model-len",
        "4096",
        "--enable-lora",
        "--lora-modules",
        f"summarizer={LORA_ADAPTER_ID}",
        "--max-lora-rank",
        "64",
    ]

    print(f"Starting vLLM server: {' '.join(cmd)}")
    subprocess.Popen(cmd)
