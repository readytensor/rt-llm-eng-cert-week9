"""
Runpod Serverless Handler for vLLM with LoRA
============================================
Deploy LLMs with LoRA adapters on Runpod Serverless.
"""

import runpod
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from transformers import AutoTokenizer

MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"
LORA_ADAPTER_ID = "moo3030/Llama-3.2-1B-QLoRA-Summarizer-adapters"

# Load model and tokenizer once at startup
print(f"Loading model: {MODEL_ID}")
llm = LLM(
    model=MODEL_ID,
    gpu_memory_utilization=0.9,
    max_model_len=4096,
    enable_lora=True,
    max_loras=1,
    max_lora_rank=64,
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# Create LoRA request
lora_request = LoRARequest(
    lora_name="summarizer",
    lora_int_id=1,
    lora_path=LORA_ADAPTER_ID,
)
print("Model and LoRA adapter loaded successfully")


def handler(event):
    """
    Process incoming requests.

    Args:
        event (dict):
            {
                "input": {
                    "prompt": "Your prompt here",
                    "max_tokens": 256,
                    "temperature": 0.7,
                    "use_lora": true/false
                }
            }

    Returns:
        dict: The generated response
    """
    print("Worker received request")

    input_data = event.get("input", {})

    prompt = input_data.get("prompt")
    if not prompt:
        return {"error": "prompt is required"}

    max_tokens = input_data.get("max_tokens", 256)
    temperature = input_data.get("temperature", 0.7)
    use_lora = input_data.get("use_lora", False)

    # Apply chat template
    messages = [{"role": "user", "content": prompt}]
    formatted_prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    # Generate
    params = SamplingParams(max_tokens=max_tokens, temperature=temperature)

    if use_lora:
        outputs = llm.generate([formatted_prompt], params, lora_request=lora_request)
    else:
        outputs = llm.generate([formatted_prompt], params)

    response_text = outputs[0].outputs[0].text

    print(f"Generated {len(response_text)} chars (LoRA: {use_lora})")

    return {"response": response_text, "use_lora": use_lora}


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
