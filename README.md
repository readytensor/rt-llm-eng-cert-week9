# LLM Engineering & Deployment - Week 9 Code Examples

**Week 9: LLM Production Deployment**  
Part of the LLM Engineering & Deployment Certification Program

This repository contains code examples for deploying LLMs to production on various cloud platforms. The module covers:

- **Modal** - Serverless GPU functions with pay-per-second billing
- **Runpod** - On-demand GPU inference with custom handlers
- **SageMaker** - Enterprise ML deployment on AWS
- **Bedrock** - Managed foundation models (zero infrastructure)

---

## Prerequisites

- Python 3.10+
- Modal account (free tier available)
- Runpod account (for Runpod deployment)
- AWS account with configured credentials (for SageMaker, Bedrock)
- Hugging Face account with accepted Llama license

---

## Setup

### 1. Environment Setup

Create a virtual environment:

```bash
python -m venv venv
```

Activate the virtual environment:

```bash
# On Windows:
venv\Scripts\activate

# On Mac/Linux:
source venv/bin/activate
```

### 2. Dependency Installation

Install all dependencies:

```bash
pip install -r requirements.txt
```

### 3. Platform Authentication

**Modal:**

```bash
modal setup
```

**AWS:**

```bash
aws configure
# Enter your Access Key ID, Secret Access Key, and region
```

**Hugging Face (for gated models):**

```bash
huggingface-cli login
```

### 4. Environment Variables

Create a `.env` file for Runpod and Bedrock clients:

```bash
RUNPOD_API_KEY=your-runpod-api-key
RUNPOD_ENDPOINT_ID=your-endpoint-id
AWS_ACCESS_KEY_ID=your-aws-access-key
AWS_SECRET_ACCESS_KEY=your-aws-secret-key
```

---

## Running the Code Examples

### Modal Deployment

The Modal server demonstrates vLLM deployment with LoRA adapter support, streaming, and continuous batching.

**Deploy the server:**

```bash
cd code/modal
modal deploy server.py
```

**Endpoints:**

| Endpoint | Description |
|----------|-------------|
| `POST /base` | Generate with base model |
| `POST /lora` | Generate with LoRA adapter |
| `POST /stream` | Streaming generation (supports both) |

**Test with client:**

```bash
# Base model
python client.py --prompt "What is serverless computing?"

# LoRA adapter
python client.py --lora --prompt "Summarize: The quick brown fox..."

# Streaming
python client.py --stream --prompt "Write a story about a robot."
```

### Modal with OpenAI-Compatible API

Alternative deployment using vLLM's built-in OpenAI-compatible server:

**Deploy:**

```bash
modal deploy openai_server.py
```

**Test with OpenAI SDK:**

```bash
python client_openai.py
```

### Runpod Serverless

Client for Runpod's serverless vLLM endpoints with OpenAI-compatible API.

**Test with client:**

```bash
cd code/runpod

# Chat with model
python client_openai.py --prompt "What is serverless computing?"

# Streaming response
python client_openai.py --stream --prompt "Write a haiku"

# List available models
python client_openai.py --list-models
```

### SageMaker Deployment

Deploy vLLM on AWS SageMaker using pre-built containers.

**Create model:**

```bash
aws sagemaker create-model \
    --model-name llama-vllm-model \
    --primary-container '{
        "Image": "763104351884.dkr.ecr.us-east-1.amazonaws.com/vllm:0.11-gpu-py312-cu129-ubuntu22.04-sagemaker-v1",
        "Environment": {
            "SM_VLLM_MODEL": "moo3030/Llama-3.2-1B-Summarizer-merged",
            "HUGGING_FACE_HUB_TOKEN": "your-hf-token-here"
        }
    }' \
    --execution-role-arn arn:aws:iam::YOUR_ACCOUNT_ID:role/SageMaker-Execution-Role
```

**Create endpoint config:**

```bash
aws sagemaker create-endpoint-config \
    --endpoint-config-name llama-vllm-config \
    --production-variants "[{
        \"VariantName\": \"AllTraffic\",
        \"ModelName\": \"llama-vllm-model\",
        \"InstanceType\": \"ml.g6.2xlarge\",
        \"InitialInstanceCount\": 1,
        \"ContainerStartupHealthCheckTimeoutInSeconds\": 900
    }]"
```

**Create endpoint:**

```bash
aws sagemaker create-endpoint \
    --endpoint-name llama-vllm-endpoint \
    --endpoint-config-name llama-vllm-config
```

**Test with client:**

```bash
python code/sagemaker/client.py
```

**Cleanup (important!):**

```bash
aws sagemaker delete-endpoint --endpoint-name llama-vllm-endpoint
aws sagemaker delete-endpoint-config --endpoint-config-name llama-vllm-config
aws sagemaker delete-model --model-name llama-vllm-model
```

### Bedrock Deployment

Use AWS Bedrock with custom imported models.

**Test with client:**

```bash
python code/bedrock/bedrock_client.py
```

---

## Lessons Overview

| Lesson | Topic | Key Concepts |
|--------|-------|--------------|
| Overview | Unit Introduction | Platform comparison, decision framework |
| 1 | Modal | Serverless GPU, pay-per-second, cold start optimization |
| 2 | Runpod | Custom handlers, worker scaling, OpenAI-compatible API |
| 3 | SageMaker | Enterprise deployment, IAM, VPC, pre-built vLLM containers |
| 4 | Bedrock | Managed models, custom model import, per-token pricing |
| 5 | On-Premise | Private cloud, Kubernetes, cost analysis, security |

---

## Project Structure

```
rt-llm-eng-cert-week9/
├── code/
│   ├── modal/
│   │   ├── server.py           # Modal vLLM server with LoRA
│   │   ├── client.py           # Test client for Modal
│   │   ├── openai_server.py    # Modal with vLLM serve
│   │   └── client_openai.py    # OpenAI SDK client
│   ├── runpod/
│   │   └── client_openai.py    # Runpod OpenAI-compatible client
│   ├── sagemaker/
│   │   └── client.py           # SageMaker endpoint client
│   └── bedrock/
│       └── bedrock_client.py   # Bedrock client
├── lessons/
│   ├── overview.md             # Unit overview and decision framework
│   ├── lesson1-modal.md        # Modal deployment guide
│   ├── lesson2-runpod.md       # Runpod deployment guide
│   ├── lesson3-sagemaker.md    # SageMaker deployment guide
│   ├── lesson4-bedrock.md      # Bedrock deployment guide
│   └── lesson5-onpremise.md    # On-premise deployment guide
├── requirements.txt            # Python dependencies
└── README.md
```

---

## Platform Comparison

| Platform | Scaling | Complexity | Cost Model | Cold Starts |
|----------|---------|------------|------------|-------------|
| **Modal** | Automatic | Low | Per-second GPU | Optimizable |
| **Runpod** | Automatic | Medium | Per-second GPU | Configurable |
| **SageMaker** | Policy-based | Medium-High | Per-hour instance | Warm pools |
| **Bedrock** | Automatic | Low | Per-token | None |

---

## Decision Framework

**Choose Modal or Runpod if:**
- You want serverless simplicity with GPU support
- Traffic is bursty or unpredictable
- Pay-per-second billing makes sense

**Choose SageMaker if:**
- You're in an AWS-native organization
- You need enterprise features (IAM, VPC, compliance)
- You're deploying multiple models

**Choose Bedrock if:**
- You want zero infrastructure management
- Available models meet your needs
- You prefer per-token pricing

---

## Hardware Considerations

- **Modal L4**: 24GB VRAM, good for 1-7B models
- **SageMaker ml.g6.2xlarge**: L4 GPU, CUDA 12.x compatible
- **SageMaker ml.g5.xlarge**: A10G GPU, CUDA 11.4 only

> ⚠️ **CUDA Compatibility**: The vLLM container (`cu129`) requires CUDA 12.x. Use `ml.g6.*` instances on SageMaker. `ml.g5.*` instances use CUDA 11.4 and will fail with "CannotStartContainerError".

---

## License

This work is licensed under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).

**You are free to:**

- Share and adapt this material for non-commercial purposes
- Must give appropriate credit and indicate changes made
- Must distribute adaptations under the same license

See [LICENSE](LICENSE) for full terms.

---

## Contact

For questions or issues related to this repository, please refer to the course materials or contact your instructor.
