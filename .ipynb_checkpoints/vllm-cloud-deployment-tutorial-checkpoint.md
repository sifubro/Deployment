# Deploying Models with vLLM on AWS and GCP: A Complete Beginner's Guide

> A comprehensive tutorial for deploying LLMs, Vision-Language Models, and embedding models like CLIP at scale using vLLM on AWS and GCP with multi-GPU support.

---

## Table of Contents

1. [Part 1: Understanding vLLM Fundamentals](#part-1-understanding-vllm-fundamentals)
2. [Part 2: Local Setup and Your First vLLM Server](#part-2-local-setup-and-your-first-vllm-server)
3. [Part 3: Multi-GPU Parallelism Deep Dive](#part-3-multi-gpu-parallelism-deep-dive)
4. [Part 4: Deploying Vision-Language Models (VLMs)](#part-4-deploying-vision-language-models-vlms)
5. [Part 5: Deploying Embedding Models (CLIP-style)](#part-5-deploying-embedding-models-clip-style)
6. [Part 6: AWS Deployment Walkthrough](#part-6-aws-deployment-walkthrough)
7. [Part 7: GCP Deployment Walkthrough](#part-7-gcp-deployment-walkthrough)
8. [Part 8: Production Considerations](#part-8-production-considerations)
9. [Part 9: Troubleshooting & Cost Optimization](#part-9-troubleshooting--cost-optimization)

---

## Part 1: Understanding vLLM Fundamentals

### What is vLLM?

vLLM is an open-source inference engine originally built at UC Berkeley's Sky Computing Lab. Think of it as a highly optimized "web server" specifically designed for serving large neural networks. Where you would normally use Flask or FastAPI to serve a Python function, vLLM is purpose-built to serve transformer models at massive scale.

vLLM supports 200+ model architectures from Hugging Face, including decoder-only LLMs (Llama, Qwen, Gemma), Mixture-of-Expert models (Mixtral, DeepSeek-V3), multimodal/vision-language models (LLaVA, Qwen-VL, Pixtral), and embedding/retrieval models (E5-Mistral, GTE).

### Why vLLM Instead of Plain Hugging Face Transformers?

If you've used `transformers.pipeline()` to run a model, you know it works but is slow when serving multiple users. The naive approach has three big problems:

**Problem 1: Memory waste from KV cache.** During text generation, transformers store "key-value" tensors for every token they have already generated. Naive implementations pre-allocate huge contiguous memory blocks based on the maximum possible output length, even if the actual response is short. Most of that memory sits empty.

**Problem 2: Sequential request processing.** If user A asks a question and is mid-generation, user B has to wait until A is completely done. GPUs sit idle between tokens.

**Problem 3: Poor batching.** Even when you batch requests, traditional batching forces all sequences in the batch to finish together. The fastest sequence waits for the slowest.

### The Three Key Innovations of vLLM

**1. PagedAttention.** Borrowing the idea of virtual memory from operating systems, vLLM splits the KV cache into small fixed-size "pages" (blocks of tokens). Memory is allocated on demand, page by page, instead of one giant chunk per sequence. This typically reclaims 60-80% of "wasted" memory and lets you fit more concurrent requests in the same VRAM.

**2. Continuous Batching.** Instead of waiting for an entire batch to finish, vLLM continuously swaps finished sequences out and new requests in at every generation step. The GPU is never idle waiting for the slowest sequence to complete. This is the single biggest source of throughput gains over naive serving.

**3. Optimized CUDA Kernels.** vLLM uses hand-tuned attention kernels (FlashAttention, FlashInfer, Triton-based kernels) and supports a wide range of quantization formats including FP8, MXFP4, NVFP4, INT8, INT4, GPTQ, AWQ, and GGUF, plus speculative decoding methods like n-gram, EAGLE, and DFlash.

### When You Should Use vLLM

vLLM is the right tool when you need to serve a model to many users concurrently, when latency and throughput both matter, when you want an OpenAI-compatible API drop-in (so existing client code works without changes), and when you have access to NVIDIA GPUs (it also supports AMD/ROCm, Intel XPU, AWS Neuron, and TPUs to varying degrees).

It's overkill if you're running a model once on your laptop or only have one user at a time — for those cases, `transformers` or `llama.cpp` are simpler.

### The Mental Model

Picture vLLM as having three layers stacked on top of each other:

The bottom layer is the **engine** — this is where the actual GPU work happens. It loads model weights, manages the KV cache, and runs the forward passes. The middle layer is the **scheduler** — at every step it decides which sequences to run, which to pause, and which to swap to CPU memory. The top layer is the **API server** — an OpenAI-compatible HTTP server that accepts requests and feeds them into the engine. When you run `vllm serve <model>`, all three layers spin up together.

---

## Part 2: Local Setup and Your First vLLM Server

Before deploying to the cloud, get vLLM running locally (or on a single cloud GPU instance) so you understand what's happening before you scale it up.

### Hardware Requirements

vLLM officially supports NVIDIA GPUs with compute capability 7.0+ (Volta, Turing, Ampere, Ada Lovelace, Hopper, Blackwell). Practically, this means V100 and newer. Some quantization formats (FP8 block-wise) require Ada Lovelace or newer (compute capability ≥ 8.9). For learning, a single T4, L4, or A10G is enough to serve a 7B model.

### Installation

The recommended installation uses `uv`, a fast Python package manager. The traditional `pip` also works:

```bash
# Option 1: Using uv (recommended — much faster)
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv vllm-env --python 3.12
source vllm-env/bin/activate
uv pip install vllm

# Option 2: Using pip
python -m venv vllm-env
source vllm-env/bin/activate
pip install vllm
```

Verify the installation by checking the CLI:

```bash
vllm --help
```

If you see a list of subcommands (`serve`, `chat`, `bench`, etc.), you're set.

### Your First Server

Pick a small model that fits on your GPU. For a 24GB card (L4, A10G, RTX 4090), `Qwen/Qwen2.5-7B-Instruct` is a good starting point:

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct \
    --host 0.0.0.0 \
    --port 8000
```

The first run will download the model from Hugging Face (~15GB), then load it into VRAM and start the server. You'll see logs like `Application startup complete` when it's ready.

### Testing the Server

vLLM exposes an OpenAI-compatible API, so any OpenAI client library works. Here's a Python example:

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed",  # vLLM doesn't require auth by default
)

response = client.chat.completions.create(
    model="Qwen/Qwen2.5-7B-Instruct",
    messages=[
        {"role": "user", "content": "Explain PagedAttention in one paragraph."}
    ],
    max_tokens=200,
)
print(response.choices[0].message.content)
```

Or with curl:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-7B-Instruct",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 50
  }'
```

### Key Configuration Flags You'll Use Constantly

A handful of flags come up in nearly every deployment:

`--gpu-memory-utilization` (default 0.9) controls what fraction of GPU memory vLLM is allowed to use. The remainder is left for CUDA overhead, the OS, and other processes. Lower it to 0.85 if you hit OOM errors; raise it to 0.95 if you have spare headroom and want a bigger KV cache.

`--max-model-len` caps the maximum context length. By default it uses the model's full advertised context, which can be huge (256K for some models) and reserves enormous KV cache space. If you only need 8K context, set `--max-model-len 8192` to free that memory for batching more requests.

`--max-num-seqs` (default varies) caps how many sequences run concurrently. Higher values mean more throughput but more memory pressure.

`--dtype` controls precision. `auto` (default) picks `bfloat16` on Ampere+ and `float16` elsewhere. You can force `float16`, `bfloat16`, or `float32`.

`--quantization` enables a quantization format. Common choices: `fp8`, `awq`, `gptq`, `bitsandbytes`. You can also load pre-quantized models without this flag if their `config.json` already specifies the format.

---

## Part 3: Multi-GPU Parallelism Deep Dive

This is the section that determines whether your deployment is fast and cheap or slow and expensive. There are three core strategies, and the choice matters a lot.

### Strategy 1: Tensor Parallelism (TP)

Tensor parallelism shards each individual layer across multiple GPUs. When the model computes a matrix multiplication for, say, the attention or MLP layer, that matrix is split column-wise across GPUs. Each GPU computes its slice, then the results are combined via an `AllReduce` collective communication.

You enable TP with a single flag:

```bash
vllm serve meta-llama/Llama-3.1-70B-Instruct --tensor-parallel-size 4
```

The above runs the 70B model split across 4 GPUs. Tensor parallelism is what you need when a model is too big to fit on one GPU (Llama-70B in BF16 needs ~140GB, so you need 2x or more H100s).

There's a critical hardware caveat: tensor parallelism produces a lot of inter-GPU communication on every single layer. On NVLink-connected datacenter GPUs (H100, A100), you can expect roughly 1.84x speedup per additional GPU — close to ideal. On consumer GPUs connected by PCIe (4090s, 5090s in a workstation), the ceiling is closer to 1.4x per card because the interconnect can't keep up. If you're deploying on cloud, NVLink-equipped instances (e.g. AWS `p4de`, `p5`, GCP `a3` family) are worth the price premium for TP-heavy workloads.

The TP size must evenly divide the number of attention heads. Most models have 32, 64, 96, or 128 heads, so common valid TP sizes are 2, 4, 8.

### Strategy 2: Pipeline Parallelism (PP)

Pipeline parallelism splits the model *across layers* instead of within layers. GPU 0 runs layers 1–20, GPU 1 runs layers 21–40, and so on. Activations are passed forward between GPUs.

```bash
vllm serve gpt2 --tensor-parallel-size 4 --pipeline-parallel-size 2
```

This uses 8 GPUs total: TP=4 within each "stage", PP=2 across stages.

Pipeline parallelism shines when you have multiple machines (a multi-node cluster) and inter-node bandwidth is the bottleneck. Communication only happens at stage boundaries (much less frequently than TP), so PP across machines + TP within a machine is the classic recipe for very large models. The drawback is that PP introduces "bubbles" — periods where some GPUs wait for others to finish their stage. vLLM mitigates this by running multiple requests through the pipeline concurrently.

### Strategy 3: Data Parallelism (DP)

Data parallelism doesn't split the model at all. Each GPU has a full copy of the weights, and incoming requests are distributed across the replicas. This is used when the model is small enough to fit on one GPU but you want more throughput.

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct --data-parallel-size 4
```

DP gives near-linear throughput scaling because there's almost no cross-GPU communication. The catch is memory: you pay 4x the model size in VRAM to run 4x the requests.

### Choosing Between TP, PP, and DP

A practical decision tree:

- **Model fits on 1 GPU, you want more throughput** → use DP.
- **Model doesn't fit on 1 GPU, but fits on 1 machine** → use TP across all the GPUs in that machine.
- **Model doesn't fit on 1 machine** → use TP within each machine + PP across machines.
- **Model is a Mixture-of-Experts (MoE) like Mixtral or DeepSeek** → look into Expert Parallelism (EP), which is `--enable-expert-parallel` combined with TP or DP. MoE models route different experts to different GPUs, which avoids duplicating expert weights.

For multimodal models, vLLM exposes `--mm-encoder-tp-mode data`, which runs the (small) vision encoder in DP mode while the (large) language model uses TP. This is almost always faster than tensor-paralleling a small ViT, because the communication overhead would dominate.

### Multi-Node Setup with Ray

For deployments that span multiple machines, vLLM uses Ray as the distributed runtime. The basic pattern:

```bash
# On the head node
ray start --head --port=6379

# On each worker node
ray start --address=<HEAD_NODE_IP>:6379

# Now launch vLLM from the head node — it will discover the cluster
vllm serve /path/to/model \
    --tensor-parallel-size 16 \
    --distributed-executor-backend ray
```

Every node must have an identical execution environment (same Python packages, same model files, same CUDA version). The simplest way to enforce this is to use the official vLLM Docker image (`vllm/vllm-openai:latest`) on every node.

---

## Part 4: Deploying Vision-Language Models (VLMs)

Vision-language models accept both text and images and generate text. Examples: LLaVA, Qwen-VL, Pixtral, Llama 4 Scout, Idefics3. vLLM has first-class support for them.

### Launching a VLM Server

Qwen2.5-VL is a strong open-source choice with broad support. To serve the 7B variant:

```bash
vllm serve Qwen/Qwen2.5-VL-7B-Instruct \
    --host 0.0.0.0 \
    --port 8000 \
    --max-model-len 32768 \
    --limit-mm-per-prompt '{"image": 4}'
```

The `--limit-mm-per-prompt` flag caps how many images can be in a single request — this is important because vLLM reserves memory based on the maximum number of multimodal inputs it might see. Setting this to a realistic value frees up memory for KV cache.

For the 72B variant on 4 GPUs:

```bash
vllm serve Qwen/Qwen2.5-VL-72B-Instruct \
    --tensor-parallel-size 4 \
    --mm-encoder-tp-mode data \
    --gpu-memory-utilization 0.95 \
    --max-model-len 32768
```

`--mm-encoder-tp-mode data` is the key flag here. The vision encoder (~675M parameters in Qwen2.5-VL-72B) is tiny relative to the language model (72B), so tensor-paralleling the encoder adds communication overhead without speedup. Running it in data-parallel mode keeps each GPU running its own copy of the small encoder while the big language model uses TP.

### Calling a VLM

The OpenAI multimodal API format works directly:

```python
import base64
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

# Method A: image as URL
response = client.chat.completions.create(
    model="Qwen/Qwen2.5-VL-7B-Instruct",
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "What's happening in this image?"},
            {"type": "image_url", "image_url": {"url": "https://example.com/cat.jpg"}},
        ],
    }],
)

# Method B: image as base64 (for local files or privacy)
with open("cat.jpg", "rb") as f:
    img_b64 = base64.b64encode(f.read()).decode()

response = client.chat.completions.create(
    model="Qwen/Qwen2.5-VL-7B-Instruct",
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "Describe this image."},
            {"type": "image_url", "image_url": {
                "url": f"data:image/jpeg;base64,{img_b64}"
            }},
        ],
    }],
)

print(response.choices[0].message.content)
```

### VLM-Specific Memory Tips

VLMs are memory-hungry because each image expands into hundreds or thousands of tokens after encoding. A few practical rules:

If you only ever serve images (no video), pass `--limit-mm-per-prompt '{"video": 0}'` to skip allocating video buffer memory. If you're serving a model that supports text-only mode (Llama-4, Qwen-3.5), passing `--language-model-only` skips the vision encoder entirely and dedicates that memory to KV cache for higher text throughput.

For the latest Qwen3-VL models (235B MoE variants), serving requires at least 8 GPUs with 80GB each (A100/H100/H200), and FP8 quantized checkpoints (`Qwen/Qwen3-VL-235B-A22B-Instruct-FP8`) cut memory roughly in half if you're on Ada/Hopper.

---

## Part 5: Deploying Embedding Models (CLIP-style)

This is where things get interesting and where vLLM's coverage is uneven.

### Important Caveat About CLIP

CLIP itself (the original OpenAI/LAION dual-encoder model) is **not directly supported** as a first-class architecture in vLLM as of late 2025/early 2026. CLIP is a very different beast from generative LLMs — it's a contrastive dual-encoder, and the inference workload (single forward pass through small ViT + text encoder, no autoregressive generation) doesn't benefit from vLLM's main optimizations (PagedAttention, continuous batching).

You have three realistic options:

**Option 1: Use CLIP through Hugging Face Transformers directly.** For pure CLIP serving, a FastAPI wrapper around `transformers.CLIPModel` running on a single GPU is usually the right tool. vLLM is overkill.

**Option 2: Use vLLM's pooling/embedding mode for supported embedding models.** vLLM supports many text embedding models (E5-Mistral, GTE, BGE, Jina) and reranking models via its "pooling" runner:

```bash
vllm serve intfloat/e5-mistral-7b-instruct \
    --runner pooling \
    --task embed
```

Then call the OpenAI embeddings endpoint:

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

embeddings = client.embeddings.create(
    model="intfloat/e5-mistral-7b-instruct",
    input=["First sentence", "Second sentence"],
)
print(len(embeddings.data[0].embedding))  # e.g., 4096
```

**Option 3: Use a vision-language model that has been adapted for embeddings.** Models like `nomic-ai/nomic-embed-vision-v1.5` or some SigLIP-based encoders are loadable through vLLM's Transformers backend. These give you joint image-text embeddings similar to CLIP but with a different architecture.

### Supported Multimodal Embedding Models

Through vLLM's pooling runner, models that produce image+text embeddings include some BGE-VL variants, certain CLIP-derivative models implemented in HF Transformers, and SigLIP-based encoders. Support is checked at startup — if vLLM doesn't recognize the architecture as a pooling model, you'll need to pass `--hf-overrides` with the correct architecture name, or fall back to a transformers-based server.

A pragmatic recommendation: **for production CLIP-style image embeddings, run a separate FastAPI service with the HF model. For text embeddings, use vLLM.** Trying to force CLIP through vLLM tends to be more pain than it's worth.

---

## Part 6: AWS Deployment Walkthrough

Now we put the pieces together. We'll deploy a multi-GPU vLLM server on AWS using EC2 directly. Other paths (SageMaker, EKS) work too but EC2 is the most transparent for learning.

### Step 1: Choose an Instance Type

AWS has several GPU instance families relevant to vLLM:

| Family | GPU | VRAM/GPU | GPUs | Use case |
|--------|-----|----------|------|----------|
| `g5` | A10G | 24 GB | 1, 4, 8 | 7B-13B models, dev/test |
| `g6` | L4 | 24 GB | 1, 4, 8 | Cost-efficient inference for 7B-13B |
| `g6e` | L40S | 48 GB | 1, 4, 8 | 30B-70B models with quantization |
| `p4d` / `p4de` | A100 | 40 / 80 GB | 8 | Production 70B+ |
| `p5` / `p5e` | H100 | 80 GB | 8 | Production large models, tensor parallelism |
| `p5en` | H200 | 141 GB | 8 | Largest open models |

For a first multi-GPU deployment, `g5.12xlarge` (4x A10G, 96 GB total VRAM) is a good balance of cost and capability. It runs Llama-3.1-8B with TP=4, or Qwen2.5-VL-32B comfortably.

### Step 2: Launch the Instance

The cleanest path is the **AWS Deep Learning AMI**, which has CUDA, NCCL, and Python pre-installed.

From the AWS console (or CLI), launch with these settings:

- **AMI**: "Deep Learning OSS NVIDIA Driver AMI GPU PyTorch" (latest version)
- **Instance type**: `g5.12xlarge` (or your choice)
- **Storage**: At least 200 GB EBS (model weights + cache add up fast). gp3 is fine for most cases.
- **Security group**: Allow SSH (port 22) from your IP, and port 8000 (or whichever you'll serve on) — for production, don't expose 8000 to the internet directly; use an ALB or VPN.
- **IAM role**: Optional but useful — attach a role with S3 read access if your model weights live in S3.

Equivalent CLI command:

```bash
aws ec2 run-instances \
    --image-id ami-0abcdef1234567890 \
    --instance-type g5.12xlarge \
    --key-name your-key-pair \
    --security-group-ids sg-xxxxxxxx \
    --subnet-id subnet-xxxxxxxx \
    --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":200,"VolumeType":"gp3"}}]' \
    --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=vllm-server}]'
```

(Replace the AMI ID with the current Deep Learning AMI for your region — find it in AWS Marketplace or via `aws ec2 describe-images`.)

### Step 3: SSH and Verify GPUs

```bash
ssh -i your-key.pem ubuntu@<instance-public-ip>
nvidia-smi
```

You should see 4 A10G GPUs listed, each with 23 GB free VRAM. The Deep Learning AMI ships with the right NVIDIA drivers; verify with `nvidia-smi`. If you see something off, the AMI version may have shifted — you can reinstall drivers via `sudo apt install nvidia-driver-535` and reboot.

### Step 4: Install vLLM

```bash
# Activate the pre-installed PyTorch environment, or use uv for a fresh one
source activate pytorch
pip install vllm

# Or, for cleanliness:
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv vllm-env --python 3.12
source vllm-env/bin/activate
uv pip install vllm
```

### Step 5: Launch the Server

```bash
vllm serve meta-llama/Llama-3.1-8B-Instruct \
    --tensor-parallel-size 4 \
    --host 0.0.0.0 \
    --port 8000 \
    --gpu-memory-utilization 0.90 \
    --max-model-len 16384
```

If you're using a gated model (Llama family, for example), set `HF_TOKEN` first:

```bash
export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxx
```

The first launch downloads the model — this takes a few minutes for an 8B model. Subsequent launches use the cache.

### Step 6: Production-ize with Docker and systemd

Running `vllm serve` in an SSH session is fine for testing but fragile. For a real deployment, use the official Docker image and a systemd service.

```bash
# /etc/systemd/system/vllm.service
[Unit]
Description=vLLM Inference Server
After=docker.service
Requires=docker.service

[Service]
Restart=always
RestartSec=10
ExecStartPre=-/usr/bin/docker stop vllm
ExecStartPre=-/usr/bin/docker rm vllm
ExecStart=/usr/bin/docker run --rm --name vllm \
    --gpus all \
    --ipc=host \
    -p 8000:8000 \
    -v /home/ubuntu/.cache/huggingface:/root/.cache/huggingface \
    -e HF_TOKEN=${HF_TOKEN} \
    vllm/vllm-openai:latest \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --tensor-parallel-size 4 \
    --gpu-memory-utilization 0.90 \
    --max-model-len 16384

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now vllm
sudo systemctl status vllm
journalctl -u vllm -f  # follow logs
```

The `--ipc=host` flag is required because vLLM's Ray-based multi-GPU coordination uses shared memory that exceeds Docker's default limit.

### Step 7: Put it Behind a Load Balancer (Optional but Recommended)

For production traffic, place an Application Load Balancer in front of one or more vLLM instances. Configure the ALB to health-check `/health`, terminate TLS at the ALB with an ACM certificate, and route to the EC2 target group on port 8000. Now your clients hit `https://api.yourcompany.com/v1/chat/completions` and never see the EC2 instance directly.

For multiple replicas (auto-scaling), put the EC2 instances in an Auto Scaling Group. Note that vLLM startup is slow (model load can take minutes), so set generous health check grace periods and keep the minimum replica count ≥ 1 unless you can tolerate cold starts.

### Step 8: Multi-Node with EKS (Brief Pointer)

For very large models that span multiple machines, the cleanest production path is **EKS with the LeaderWorkerSet (LWS) operator** plus the official vLLM Helm chart. The LWS pattern provisions one "leader" pod and N "worker" pods that join via Ray, enabling TP+PP across machines. This is a substantial topic on its own — the vLLM documentation has a dedicated guide, and the AWS samples repo has reference Terraform.

---

## Part 7: GCP Deployment Walkthrough

GCP's GPU offerings are organized similarly to AWS but with different naming and some unique features (like TPU support, which we won't cover here).

### Step 1: Choose a Machine Type

Relevant GCP GPU machine families:

| Family | GPU | VRAM/GPU | GPUs | Notes |
|--------|-----|----------|------|-------|
| `g2` | L4 | 24 GB | 1, 2, 4, 8 | Cost-efficient, broad availability |
| `a2` | A100 | 40 or 80 GB | 1, 2, 4, 8, 16 | Standard production workhorse |
| `a3` (high) | H100 | 80 GB | 8 | Production, NVLink within node |
| `a3` (mega/edge/ultra) | H100 / H200 | 80 / 141 GB | 8 | Latest and largest |

For learning, `g2-standard-48` (4x L4, 96 GB total VRAM) is a near-direct analog of AWS's `g5.12xlarge` and similarly priced.

### Step 2: Choose Your Deployment Surface

GCP gives you three reasonable paths:

**Compute Engine (GCE) VM** — the most transparent option, equivalent to AWS EC2.

**Google Kubernetes Engine (GKE)** — for production at scale, GKE with the GPU and TPU operators is well-supported and the official path Google recommends.

**Vertex AI Online Prediction** — fully managed model serving. You give it a model and a container; Vertex handles autoscaling and load balancing. Good if you don't want to manage infrastructure but more opaque than GCE.

I'll walk through the GCE path since it's the most instructive. Adapting to GKE is straightforward once you have the Docker command working.

### Step 3: Create the VM

You need an image with NVIDIA drivers and CUDA pre-installed. GCP's "Deep Learning VM" images (`deeplearning-platform-release` project) ship with PyTorch, CUDA, and drivers ready.

```bash
gcloud compute instances create vllm-server \
    --zone=us-central1-a \
    --machine-type=g2-standard-48 \
    --accelerator=type=nvidia-l4,count=4 \
    --image-family=common-cu124-ubuntu-2204-py310 \
    --image-project=deeplearning-platform-release \
    --boot-disk-size=200GB \
    --boot-disk-type=pd-ssd \
    --maintenance-policy=TERMINATE \
    --restart-on-failure \
    --metadata="install-nvidia-driver=True" \
    --tags=vllm-server
```

A few details worth understanding:

`--maintenance-policy=TERMINATE` is required for GPU instances — they can't live-migrate. `--metadata="install-nvidia-driver=True"` triggers automatic driver installation on first boot for Deep Learning VM images. The image family I chose (`common-cu124-ubuntu-2204-py310`) has CUDA 12.4 and Python 3.10 — pick one that matches your vLLM version's CUDA requirement.

### Step 4: Open Firewall Rules

GCE VMs are not exposed to the internet by default. Open SSH and your serving port:

```bash
gcloud compute firewall-rules create allow-vllm \
    --direction=INGRESS \
    --action=ALLOW \
    --rules=tcp:8000 \
    --source-ranges=YOUR.IP.ADDRESS.HERE/32 \
    --target-tags=vllm-server
```

**Don't open port 8000 to `0.0.0.0/0`** unless the server is behind authentication. vLLM has no built-in auth.

### Step 5: SSH and Set Up vLLM

```bash
gcloud compute ssh vllm-server --zone=us-central1-a

# On the VM:
nvidia-smi  # confirm 4x L4 visible
pip install --upgrade pip
pip install vllm
```

### Step 6: Launch the Server

```bash
export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxx

vllm serve Qwen/Qwen2.5-VL-7B-Instruct \
    --tensor-parallel-size 4 \
    --host 0.0.0.0 \
    --port 8000 \
    --max-model-len 16384 \
    --limit-mm-per-prompt '{"image": 4}'
```

Test from your local machine:

```bash
curl http://<VM_EXTERNAL_IP>:8000/v1/models
```

### Step 7: Containerize and Use a Managed Instance Group (Production)

For production on GCE, the pattern mirrors AWS:

1. Bake a custom image with vLLM and Docker pre-installed (use Packer or `gcloud compute instances create` followed by `gcloud compute images create`).
2. Create an instance template that uses the custom image and runs vLLM via `docker run` in a startup script.
3. Create a managed instance group (MIG) from the template.
4. Put a Google Cloud Load Balancer (HTTP(S) LB) in front of the MIG with health checks on `/health`.

Sample Docker invocation for a startup script:

```bash
#!/bin/bash
docker run -d --restart=unless-stopped \
    --gpus all \
    --ipc=host \
    -p 8000:8000 \
    -v /var/cache/huggingface:/root/.cache/huggingface \
    -e HF_TOKEN="$(gcloud secrets versions access latest --secret=hf-token)" \
    --name vllm \
    vllm/vllm-openai:latest \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --tensor-parallel-size 4 \
    --gpu-memory-utilization 0.90
```

Notice the use of `gcloud secrets` — store the Hugging Face token in Secret Manager rather than baking it into images.

### Step 8: GKE Path (Brief Outline)

For Kubernetes-native deployments on GCP:

1. Create a GKE cluster with a GPU node pool (`gcloud container node-pools create vllm-pool --accelerator type=nvidia-l4,count=4 --machine-type=g2-standard-48`).
2. Install the NVIDIA GPU operator (GKE auto-installs the daemonset for default node pools, but you may need to install for custom configurations).
3. Deploy the vLLM Helm chart, which includes a `Deployment`, `Service`, and `HorizontalPodAutoscaler`.
4. For multi-node TP+PP, install the LeaderWorkerSet operator and use the LWS-based vLLM example manifests.

GKE Autopilot also supports GPU workloads (`cloud.google.com/gke-accelerator: nvidia-l4` plus a request for `nvidia.com/gpu`) and removes most node management overhead.

---

## Part 8: Production Considerations

Once you have a server running, several things separate a demo from a production deployment.

### Observability

vLLM exposes Prometheus metrics on `/metrics` by default. The metrics you care most about:

- `vllm:num_requests_running` — currently active requests
- `vllm:num_requests_waiting` — queued requests (if this grows, you're capacity-bound)
- `vllm:gpu_cache_usage_perc` — KV cache pressure (if near 1.0, you'll see throughput drop)
- `vllm:e2e_request_latency_seconds` — request latency histogram
- `vllm:time_to_first_token_seconds` — TTFT (matters for streaming UX)

Scrape these into Prometheus + Grafana, or push them to CloudWatch/Cloud Monitoring. Set alerts on queue depth and cache usage.

### Authentication

vLLM has a simple API key option (`--api-key sk-yourkey`) that requires clients to send `Authorization: Bearer sk-yourkey`. For anything more sophisticated (per-user quotas, JWT validation, rate limiting), put a gateway in front — typical choices are Kong, Tyk, or a managed gateway like AWS API Gateway / GCP API Gateway.

### Health Checks

The endpoints `/health` and `/ping` return 200 once the engine is ready. Use `/health` for load balancer health checks. Note that this endpoint returns 200 even if the GPU is overloaded — it doesn't reflect throughput pressure. For more nuanced health checks, scrape `/metrics` and gate on `vllm:num_requests_waiting < threshold`.

### Streaming

The OpenAI API supports streaming responses (`stream=True`), and vLLM passes tokens out as they're generated. This dramatically improves perceived latency. Make sure your load balancer supports SSE (Server-Sent Events) and doesn't buffer responses — for AWS ALB this works by default; for some CDNs you have to disable response buffering.

### Cost Management

GPU instances are expensive. A few tactics:

- **Spot/Preemptible instances**: For batch workloads or stateless services with multiple replicas, AWS Spot or GCP Spot/Preemptible instances cost 60-90% less. Add tolerance for interruptions.
- **Quantization**: An FP8 model uses ~half the VRAM of BF16. This often lets you use a smaller (cheaper) instance type. Modern checkpoints like `Qwen/Qwen3.5-397B-A17B-FP8` ship pre-quantized.
- **Right-size with `--max-model-len`**: Don't reserve KV cache for 256K context if your workload uses 4K. The freed memory either lets you batch more requests or run a bigger model.
- **Auto-scale on queue depth**: Don't scale on CPU or GPU utilization — those stay near 100% under any load. Scale on `vllm:num_requests_waiting`.

### Security

Beyond authentication, consider that vLLM trusts the model weights it loads. If you allow `--trust-remote-code` (required for some HF models), you're letting that model's repository run arbitrary Python at load time. Pin model versions, audit the code for new models, and run vLLM in a container with no excess privileges.

Network-wise: vLLM should never be directly internet-accessible. Put it in a private subnet, and route traffic through a load balancer or API gateway in your DMZ.

---

## Part 9: Troubleshooting & Cost Optimization

A grab-bag of issues that come up repeatedly.

### "CUDA out of memory" at startup

The model + KV cache reservation doesn't fit. In order of what to try: lower `--gpu-memory-utilization` to 0.85; lower `--max-model-len` to what you actually need; switch to a quantized variant of the model (`-FP8`, `-AWQ`); add more GPUs and bump `--tensor-parallel-size`.

### Throughput is much lower than expected on multi-GPU

Check if your GPUs have NVLink — `nvidia-smi topo -m` shows the topology. PCIe-only links cap TP scaling. On the cloud, the NVLink-equipped families are `p4d`/`p4de`/`p5` on AWS and `a2`/`a3` on GCP. Cheaper families (`g5`, `g6`, `g2`) typically have PCIe between GPUs, so TP across all 4 GPUs may scale poorly — sometimes DP=2 with TP=2 is faster than TP=4.

### "Found tensor parallel size N, but model has M attention heads not divisible by N"

The model's number of attention heads must be divisible by your TP size. Common heads: 32, 64, 96, 128. So TP=3 or TP=5 won't work with most models. Stick to powers of 2 or use a different parallelism strategy.

### The server hangs at startup with multi-GPU

Almost always a CUDA IPC issue when running in Docker. Make sure you passed `--ipc=host` to `docker run`. If it persists, also try `--shm-size=10gb`.

### Hugging Face download is slow or fails

Set `HF_HUB_ENABLE_HF_TRANSFER=1` (and `pip install hf_transfer`) for parallel downloads. For repeated deployments, mirror the model to S3 or GCS once and pull from there — much faster within the same region.

### Cold starts are too slow

For models that take minutes to load, two options:
1. Keep a "warm pool" — minimum replica count of 1, so users always hit a warm server.
2. Pre-bake the model into your Docker image. The image is bigger, but startup is just "load weights from local disk", which is much faster than re-downloading from Hugging Face.

### How do I benchmark my deployment?

vLLM ships with a benchmarking tool:

```bash
vllm bench serve \
    --backend openai-chat \
    --endpoint /v1/chat/completions \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --dataset-name sharegpt \
    --num-prompts 1000 \
    --request-rate 10
```

This sends 1000 ShareGPT-distribution prompts at 10 RPS and reports throughput, TTFT, and end-to-end latency percentiles. Use this to size your instances before going live.

---

## Wrapping Up

You now have a complete picture: the fundamentals of how vLLM differs from naive serving, the three parallelism strategies and when to use each, how to deploy LLMs and VLMs (with the honest caveat about CLIP), and end-to-end recipes for AWS and GCP. The next steps depend on your workload — for high-traffic production, invest in EKS/GKE with auto-scaling and observability; for research or moderate traffic, a single VM with systemd is often plenty.

A few resources to bookmark:

- vLLM main docs: https://docs.vllm.ai
- vLLM model recipes (model-specific tuning): https://docs.vllm.ai/projects/recipes/
- vLLM GitHub releases (changelog, new features): https://github.com/vllm-project/vllm/releases
- AWS Deep Learning AMI release notes
- GCP Deep Learning VM image catalog

Happy serving.
