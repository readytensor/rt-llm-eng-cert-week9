"""
Client for Runpod Serverless vLLM endpoint using OpenAI SDK.

Usage:
    python client_openai.py                          # Chat with model
    python client_openai.py --list-models            # List available models
    python client_openai.py --stream                 # Streaming response
    python client_openai.py --prompt "Your prompt"   # Custom prompt
    python client_openai.py --apply-chat-template    # Apply chat template locally

Environment variables:
    RUNPOD_API_KEY: Your Runpod API key
    RUNPOD_ENDPOINT_ID: Your serverless endpoint ID
"""

import os
import argparse
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# Lazy-loaded tokenizer
_tokenizer = None


def get_tokenizer(model_name: str):
    """Load tokenizer from model repo (cached)."""
    global _tokenizer
    if _tokenizer is None:
        from transformers import AutoTokenizer

        print(f"Loading tokenizer from {model_name}...")
        _tokenizer = AutoTokenizer.from_pretrained(model_name)
    return _tokenizer


# Configuration - update these or set via environment variables
RUNPOD_API_KEY = os.environ.get("RUNPOD_API_KEY", None)
ENDPOINT_ID = os.environ.get("RUNPOD_ENDPOINT_ID", None)

MODEL_NAME = "moo3030/llama-3.2-1b-summarizer-merged"

# Runpod OpenAI-compatible endpoint
BASE_URL = f"https://api.runpod.ai/v2/{ENDPOINT_ID}/openai/v1"

client = OpenAI(
    api_key=RUNPOD_API_KEY,
    base_url=BASE_URL,
)


def list_models():
    """List all available models on the endpoint."""
    print("Fetching available models...")
    try:
        models = client.models.list()
        print("\nAvailable models:")
        for model in models.data:
            print(f"  - {model.id}")
    except Exception as e:
        print(f"Error listing models: {e}")


def chat(
    prompt: str, model: str, stream: bool = False, apply_chat_template: bool = False
):
    """Send a chat completion request."""
    print(f"\n{'='*50}")
    print(f"Model: {model}")
    print(f"Streaming: {stream}")
    print(f"Apply chat template: {apply_chat_template}")
    print(f"Prompt: {prompt[:100]}{'...' if len(prompt) > 100 else ''}")
    print(f"{'='*50}\n")

    messages = [{"role": "user", "content": prompt}]

    try:
        if apply_chat_template:
            # Apply chat template locally and use completions endpoint
            tokenizer = get_tokenizer(model)
            formatted_prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            print(f"Formatted prompt:\n{formatted_prompt}\n")

            if stream:
                print("Response: ", end="", flush=True)
                response = client.completions.create(
                    model=model,
                    prompt=formatted_prompt,
                    max_tokens=256,
                    temperature=0.01,
                    stream=True,
                )
                for chunk in response:
                    if chunk.choices[0].text:
                        print(chunk.choices[0].text, end="", flush=True)
                print("\n")
            else:
                response = client.completions.create(
                    model=model,
                    prompt=formatted_prompt,
                    max_tokens=256,
                    temperature=0.01,
                )
                print(f"Response: {response.choices[0].text}")
        else:
            # Use chat completions endpoint (server applies template)
            if stream:
                print("Response: ", end="", flush=True)
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_tokens=256,
                    temperature=0.01,
                    stream=True,
                )
                for chunk in response:
                    if chunk.choices[0].delta.content:
                        print(chunk.choices[0].delta.content, end="", flush=True)
                print("\n")
            else:
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_tokens=256,
                    temperature=0.01,
                )
                print(f"Response: {response.choices[0].message.content}")

    except Exception as e:
        print(f"Error: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Client for Runpod vLLM OpenAI-compatible endpoint"
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List available models",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Enable streaming response",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="What is serverless computing? Explain in 2 sentences.",
        help="Prompt to send",
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Override model name",
    )
    parser.add_argument(
        "--apply-chat-template",
        action="store_true",
        help="Apply chat template locally using tokenizer (uses completions endpoint)",
    )

    args = parser.parse_args()

    # Check configuration
    if RUNPOD_API_KEY is None:
        print("Warning: Set RUNPOD_API_KEY environment variable")
    if ENDPOINT_ID is None:
        print("Warning: Set RUNPOD_ENDPOINT_ID environment variable")
        return

    if args.list_models:
        list_models()
        return

    model = args.model if args.model else MODEL_NAME
    chat(
        args.prompt,
        model,
        stream=args.stream,
        apply_chat_template=args.apply_chat_template,
    )


if __name__ == "__main__":
    main()
