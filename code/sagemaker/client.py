import boto3
import json


class SummarizerClient:
    def __init__(self, endpoint_name, region_name="us-east-1"):
        self.sm_runtime = boto3.client("sagemaker-runtime", region_name=region_name)
        self.endpoint_name = endpoint_name

    def summarize(
        self, text, system_prompt="You are a professional summarizer.", temperature=0.1
    ):
        payload = {
            "model": "moo3030/Llama-3.2-1B-Summarizer-merged",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Summarize the following text:\n\n{text}"},
            ],
            "temperature": temperature,
            "max_tokens": 250,
            # vLLM-specific parameters are passed in the root of the JSON
            # for direct HTTP/SageMaker calls.
            "add_generation_prompt": True,
            "echo": False,
        }

        response = self.sm_runtime.invoke_endpoint(
            EndpointName=self.endpoint_name,
            ContentType="application/json",
            Body=json.dumps(payload),
        )

        response_body = json.loads(response["Body"].read().decode("utf-8"))
        return response_body["choices"][0]["message"]["content"]


# --- Usage ---
client = SummarizerClient(endpoint_name="llama-vllm-hf-endpoint")

text = """
Victoria: God I'm really broke, I spent way to much this month
Victoria: At least we get paid soon..
Magda: Yeah, don't remind me, I know the feeling
Magda: I just paid my car insurance, I feel robbed
Victoria: Thankfully mine is paid for the rest of the year
"""

result = client.summarize(text)
print(result)
