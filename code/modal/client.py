"""
Test client for Modal endpoint with streaming support.

Usage:
    python client.py                    # Non-streaming, base model
    python client.py --stream           # Streaming, base model
    python client.py --stream --lora    # Streaming, LoRA model
    python client.py --lora             # Non-streaming, LoRA model
"""

import argparse
import requests

BASE_URL = "https://mohamedabdelhamid3030--vllm-inference-model"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stream", action="store_true", help="Use streaming endpoint")
    parser.add_argument("--lora", action="store_true", help="Use LoRA adapter")
    parser.add_argument("--prompt", type=str, default="Write a long story about a cat.")
    parser.add_argument("--max-tokens", type=int, default=100)
    args = parser.parse_args()

    payload = {
        "prompt": args.prompt,
        "max_tokens": args.max_tokens,
        "temperature": 0.7,
    }

    if args.stream:
        # Streaming endpoint
        url = f"{BASE_URL}-stream.modal.run"
        payload["use_lora"] = args.lora

        print(f"Streaming from: {url}")
        print(f"LoRA: {args.lora}\n")
        print("Response: ", end="", flush=True)

        response = requests.post(url, json=payload, stream=True)

        # Manually parse SSE events
        for line in response.iter_lines(decode_unicode=True):
            if line:
                if line.startswith("data: "):
                    data = line[6:]  # Skip "data: " (6 chars), preserve token spaces
                    if data == "[DONE]":
                        break
                    print(data, end="", flush=True)
        print("\n")

    else:
        # Non-streaming endpoint
        endpoint = "lora" if args.lora else "base"
        url = f"{BASE_URL}-{endpoint}.modal.run"

        print(f"Requesting from: {url}")
        print(f"LoRA: {args.lora}\n")

        response = requests.post(url, json=payload)
        result = response.json()
        print(f"Response: {result.get('response', result)}")


if __name__ == "__main__":
    main()
