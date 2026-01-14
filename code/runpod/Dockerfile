# Runpod Serverless vLLM Worker
# Based on: https://docs.runpod.io/serverless/quickstart

FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

WORKDIR /

# Install dependencies
RUN pip install --no-cache-dir \
    runpod \
    vllm==0.6.6 \
    transformers \
    huggingface_hub

# Set Hugging Face cache directory
ENV HF_HOME=/runpod-volume/huggingface

# Copy handler
COPY code/runpod/handler.py /handler.py

# Start the worker
CMD ["python3", "-u", "/handler.py"]

