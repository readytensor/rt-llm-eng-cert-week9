"""
Client for vLLM server using OpenAI SDK.

Usage: python client_openai.py
"""

from openai import OpenAI

# Point to your Modal vLLM server
BASE_URL = "https://mohamedabdelhamid3030--vllm-inference-openai-serve.modal.run/v1"

client = OpenAI(
    base_url=BASE_URL,
    api_key="not-needed",  # vLLM doesn't require auth
)

# Base model
print("=== Base Model ===")
response = client.chat.completions.create(
    model="meta-llama/Llama-3.2-1B-Instruct",
    messages=[{"role": "user", "content": "What is serverless computing?"}],
    max_tokens=100,
)
print(response.choices[0].message.content)

# LoRA adapter - use the adapter name from --lora-modules
print("\n=== LoRA Adapter (summarizer) ===")
response = client.chat.completions.create(
    model="summarizer",  # Use the LoRA adapter name
    messages=[
        {
            "role": "user",
            "content": "Summarize: The quick brown fox jumps over the lazy dog.",
        }
    ],
    max_tokens=100,
)
print(response.choices[0].message.content)

# Streaming with base model
print("\n=== Streaming ===")
stream = client.chat.completions.create(
    model="meta-llama/Llama-3.2-1B-Instruct",
    messages=[{"role": "user", "content": "Write a haiku about clouds."}],
    max_tokens=50,
    stream=True,
)

for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
print()
