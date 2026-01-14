import uuid
import modal
from sse_starlette.sse import EventSourceResponse

MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"
LORA_ADAPTER_ID = "moo3030/Llama-3.2-1B-QLoRA-Summarizer-adapters"

app = modal.App("vllm-inference")

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "vllm==0.6.6", "transformers", "sse-starlette"
)

volume = modal.Volume.from_name("llm-models", create_if_missing=True)


@app.cls(
    gpu="L4",
    image=image,
    volumes={"/models": volume},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    container_idle_timeout=300,
    allow_concurrent_inputs=100,
)
class Model:
    @modal.enter()
    def load(self):
        from vllm.engine.arg_utils import AsyncEngineArgs
        from vllm.engine.async_llm_engine import AsyncLLMEngine
        from vllm.lora.request import LoRARequest
        from transformers import AutoTokenizer

        engine_args = AsyncEngineArgs(
            model=MODEL_ID,
            download_dir="/models",
            gpu_memory_utilization=0.9,
            enable_lora=True,
            max_loras=1,
            max_lora_rank=64,
        )
        self.engine = AsyncLLMEngine.from_engine_args(engine_args)
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        self.lora_request = LoRARequest(
            lora_name="adapter",
            lora_int_id=1,
            lora_path=LORA_ADAPTER_ID,
        )
        volume.commit()

    async def _generate(
        self,
        messages: list[dict],
        max_tokens: int = 256,
        temperature: float = 0.7,
        use_lora: bool = False,
    ) -> str:
        from vllm import SamplingParams

        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        request_id = str(uuid.uuid4())
        params = SamplingParams(max_tokens=max_tokens, temperature=temperature, seed=42)

        lora_request = self.lora_request if use_lora else None
        results = self.engine.generate(
            prompt, params, request_id, lora_request=lora_request
        )
        async for result in results:
            final = result
        return final.outputs[0].text

    async def _generate_stream(
        self,
        messages: list[dict],
        max_tokens: int = 256,
        temperature: float = 0.7,
        use_lora: bool = False,
    ):
        from vllm import SamplingParams

        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        request_id = str(uuid.uuid4())
        params = SamplingParams(max_tokens=max_tokens, temperature=temperature, seed=42)

        lora_request = self.lora_request if use_lora else None
        results = self.engine.generate(
            prompt, params, request_id, lora_request=lora_request
        )

        previous_text = ""
        async for result in results:
            current_text = result.outputs[0].text
            new_text = current_text[len(previous_text) :]
            previous_text = current_text
            if new_text:
                yield new_text

    @modal.web_endpoint(method="POST")
    async def base(self, request: dict) -> dict:
        if "prompt" not in request:
            return {"error": "prompt is required"}

        output = await self._generate(
            [{"role": "user", "content": request["prompt"]}],
            request.get("max_tokens", 256),
            request.get("temperature", 0.7),
            use_lora=False,
        )
        return {"response": output}

    @modal.web_endpoint(method="POST")
    async def lora(self, request: dict) -> dict:
        if "prompt" not in request:
            return {"error": "prompt is required"}

        output = await self._generate(
            [{"role": "user", "content": request["prompt"]}],
            request.get("max_tokens", 256),
            request.get("temperature", 0.7),
            use_lora=True,
        )
        return {"response": output}

    @modal.web_endpoint(method="POST")
    async def stream(self, request: dict):

        if "prompt" not in request:
            return {"error": "prompt is required"}

        use_lora = request.get("use_lora", False)

        async def event_generator():
            async for chunk in self._generate_stream(
                [{"role": "user", "content": request["prompt"]}],
                request.get("max_tokens", 256),
                request.get("temperature", 0.7),
                use_lora=use_lora,
            ):
                yield {"data": chunk}
            yield {"data": "[DONE]"}

        return EventSourceResponse(event_generator())
