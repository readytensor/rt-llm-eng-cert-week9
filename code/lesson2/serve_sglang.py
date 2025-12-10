import json
from sglang.test.doc_patch import launch_server_cmd
from sglang.utils import wait_for_server, print_highlight, terminate_process

# Load model from config
with open("config.json", "r") as f:
    config = json.load(f)
    model_path = config.get("model_path", "meta-llama/Llama-3.2-1B-Instruct")

server_process, port = launch_server_cmd(
    f"python3 -m sglang.launch_server --model-path {model_path} --host 0.0.0.0 --log-level warning"
)

server_url = f"http://localhost:{port}"
wait_for_server(server_url)
print(f"Server started on {server_url}")

# Save server URL to config
config["server_url"] = server_url
with open("config.json", "w") as f:
    json.dump(config, f, indent=2)
print("Server URL saved to config.json")