import json
import openai
from sglang.utils import print_highlight

with open("config.json", "r") as f:
    config = json.load(f)

model_path = config.get("model_path", "meta-llama/Llama-3.2-1B-Instruct")
server_url = config.get("server_url", "http://127.0.0.1:30000")
base_url = f"{server_url}/v1"

client = openai.Client(base_url=base_url, api_key="None")

response = client.chat.completions.create(
    model=model_path,
    messages=[
        {"role": "user", "content": "List 3 countries and their capitals."},
    ],
    temperature=0,
    max_tokens=64,
)

print_highlight(f"Response: {response}")