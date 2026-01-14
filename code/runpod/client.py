"""
Runpod Serverless Client
Usage:
    python client.py --prompt "Your prompt here"
    python client.py --prompt "Summarize this text" --lora
    python client.py --async  # For async requests
"""

import argparse
import requests
import os
import time

# Configuration
RUNPOD_API_KEY = os.environ.get("RUNPOD_API_KEY", "your-api-key-here")
ENDPOINT_ID = os.environ.get("RUNPOD_ENDPOINT_ID", "z4j5ve2roz82h7")

BASE_URL = f"https://api.runpod.ai/v2/{ENDPOINT_ID}"


def run_sync(prompt: str, max_tokens: int, temperature: float, use_lora: bool):
    """Send a synchronous request and wait for the response."""
    url = f"{BASE_URL}/runsync"

    headers = {
        "Authorization": f"Bearer {RUNPOD_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "input": {
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "use_lora": use_lora,
        }
    }

    print(f"Sending request to: {url}")
    print(f"LoRA: {use_lora}")
    print(f"Prompt: {prompt[:50]}...")

    response = requests.post(url, json=payload, headers=headers)

    # Debug info
    print(f"\nStatus Code: {response.status_code}")
    print(f"Response Headers: {dict(response.headers)}")
    print(f"Raw Response: {response.text[:500] if response.text else '(empty)'}")

    if not response.text:
        print("\nError: Empty response. Check your API key and endpoint ID.")
        return None

    result = response.json()

    if response.status_code == 200:
        if "output" in result:
            print(f"\nResponse: {result['output']}")
        else:
            print(f"\nResult: {result}")
    else:
        print(f"\nError: {result}")

    return result


def run_async(prompt: str, max_tokens: int, temperature: float, use_lora: bool):
    """Send an async request and poll for the result."""
    url = f"{BASE_URL}/run"

    headers = {
        "Authorization": f"Bearer {RUNPOD_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "input": {
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "use_lora": use_lora,
        }
    }

    print(f"Sending async request to: {url}")
    print(f"LoRA: {use_lora}")

    # Submit the job
    response = requests.post(url, json=payload, headers=headers)
    result = response.json()

    if "id" not in result:
        print(f"Error submitting job: {result}")
        return result

    job_id = result["id"]
    print(f"Job submitted. ID: {job_id}")

    # Poll for status
    status_url = f"{BASE_URL}/status/{job_id}"

    while True:
        status_response = requests.get(status_url, headers=headers)
        status = status_response.json()

        job_status = status.get("status")
        print(f"Status: {job_status}")

        if job_status == "COMPLETED":
            print(f"\nResponse: {status.get('output')}")
            return status
        elif job_status in ["FAILED", "CANCELLED"]:
            print(f"\nJob failed: {status}")
            return status

        time.sleep(1)


def check_health():
    """Check the endpoint health via Runpod API."""
    url = f"{BASE_URL}/health"

    headers = {
        "Authorization": f"Bearer {RUNPOD_API_KEY}",
    }

    print(f"Checking health at: {url}")
    response = requests.get(url, headers=headers)
    print(f"Status Code: {response.status_code}")
    print(f"Raw Response: {response.text[:500] if response.text else '(empty)'}")

    if response.text:
        print(f"Health check: {response.json()}")
        return response.json()
    return None


def check_ping():
    """Test the direct /ping endpoint (HTTP Proxy mode)."""
    direct_url = f"https://{ENDPOINT_ID}.api.runpod.ai/ping"

    print(f"Pinging: {direct_url}")

    try:
        response = requests.get(direct_url, timeout=30)
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.text[:500] if response.text else '(empty)'}")
        return response
    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")
        return None


def generate_http(prompt: str, max_tokens: int, temperature: float, use_lora: bool):
    """Send request via HTTP Proxy mode (direct to worker)."""
    direct_url = f"https://{ENDPOINT_ID}.api.runpod.ai/generate"

    payload = {
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "use_lora": use_lora,
    }

    print(f"Sending HTTP request to: {direct_url}")
    print(f"LoRA: {use_lora}")
    print(f"Prompt: {prompt[:50]}...")

    try:
        response = requests.post(direct_url, json=payload, timeout=120)
        print(f"\nStatus Code: {response.status_code}")

        if response.text:
            result = response.json()
            print(f"Response: {result.get('response', result)}")
            return result
        else:
            print("Empty response")
            return None
    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Client for Runpod vLLM endpoint.")
    parser.add_argument(
        "--prompt", type=str, default="Write a short story about a robot."
    )
    parser.add_argument("--max-tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--lora", action="store_true", help="Use LoRA adapter")
    parser.add_argument(
        "--async", dest="async_mode", action="store_true", help="Use async mode"
    )
    parser.add_argument(
        "--health", action="store_true", help="Check endpoint health (API mode)"
    )
    parser.add_argument(
        "--ping", action="store_true", help="Test /ping endpoint (HTTP Proxy mode)"
    )
    parser.add_argument(
        "--http", action="store_true", help="Use HTTP Proxy mode (direct to worker)"
    )
    args = parser.parse_args()

    if args.ping:
        check_ping()
        return

    if args.health:
        check_health()
        return

    if args.http:
        generate_http(args.prompt, args.max_tokens, args.temperature, args.lora)
    elif args.async_mode:
        run_async(args.prompt, args.max_tokens, args.temperature, args.lora)
    else:
        run_sync(args.prompt, args.max_tokens, args.temperature, args.lora)


if __name__ == "__main__":
    main()
