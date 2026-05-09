# Deploying Models with NVIDIA Triton Inference Server on AWS and GCP

A practical, beginner-friendly guide to serving CLIP encoders, Vision Language Models (VLMs), and other deep learning models efficiently across multiple GPUs in the cloud.

---

## Table of Contents

1. [Why Triton, and What Problem Does It Solve?](#1-why-triton)
2. [Triton Fundamentals: Core Concepts](#2-triton-fundamentals)
3. [The Model Repository — Triton's Heart](#3-the-model-repository)
4. [Local Setup: Your First Triton Server](#4-local-setup)
5. [Deploying a CLIP Encoder](#5-deploying-clip)
6. [Deploying a Vision Language Model (LLaVA)](#6-deploying-vlm)
7. [Multi-GPU Strategies](#7-multi-gpu)
8. [Cloud Deployment on AWS (EKS)](#8-aws-eks)
9. [Cloud Deployment on GCP (GKE)](#9-gcp-gke)
10. [Monitoring, Autoscaling, and Production Hardening](#10-production)
11. [Troubleshooting Cheat Sheet](#11-troubleshooting)

---

## 1. Why Triton, and What Problem Does It Solve? <a name="1-why-triton"></a>

Imagine you've trained or downloaded several models — a CLIP image encoder, a CLIP text encoder, a YOLO object detector, and a LLaVA vision-language model. You now want to serve all of them in production. The naive approach is to wrap each in its own Flask or FastAPI server. This works for a demo. It falls apart in production for several reasons:

- **GPU underutilization.** A single Flask process loading one model on an A100 GPU might use 20% of its memory and a fraction of its compute. The other 80% sits idle.
- **No batching.** Each request goes through the model one at a time. GPUs are massively parallel — they want batches.
- **Hard to scale.** Adding a new model means a new container, new ingress rules, new monitoring, new everything.
- **No standard protocol.** Every team invents their own JSON schema.

Triton Inference Server is NVIDIA's answer to this. It's an open-source server that hosts any number of models from any framework (PyTorch, TensorFlow, ONNX, TensorRT, vLLM, Python custom code), exposes them through a standard HTTP/gRPC API, and squeezes maximum performance out of your GPUs.

Triton's distinguishing feature is being **backend-agnostic**. A single Triton process can simultaneously serve an LLM via the vLLM backend, a CLIP encoder via the ONNX backend, and a Python preprocessing model — all sharing the same GPUs and the same API endpoint. This is exactly what you want for multimodal pipelines (think: image comes in → CLIP encodes it → vector DB lookup → VLM generates answer).

For pure single-model LLM serving, vLLM or TensorRT-LLM standalone are simpler. Triton wins when your fleet is diverse, which is the realistic case for most production systems.

The current release as of mid-2026 is Triton 2.68 (container 26.04). It is now branded "NVIDIA Dynamo-Triton," but everyone still calls it Triton, and the concepts and APIs in this tutorial are unchanged.

---

## 2. Triton Fundamentals: Core Concepts <a name="2-triton-fundamentals"></a>

Before any code, internalize these five concepts. Everything else is detail.

### 2.1 The Model Repository

A directory on disk (or S3, GCS, Azure Blob) that holds all your models. Triton reads it on startup and on demand. This is the single source of truth for what Triton serves. Structure matters and is strict — we cover it in section 3.

### 2.2 Backends

A backend is the engine that actually runs a model. Triton ships with several:

- **TensorRT** — fastest, requires you to compile your model into a `.plan` file targeting specific GPU architectures.
- **ONNX Runtime** — great middle ground; export PyTorch/TensorFlow to ONNX once, runs anywhere.
- **PyTorch (LibTorch)** — runs TorchScript files directly.
- **TensorFlow** — runs SavedModel directories.
- **vLLM** — for LLMs; brings PagedAttention and continuous batching.
- **TensorRT-LLM** — NVIDIA's optimized LLM backend, faster than vLLM but requires compilation.
- **Python** — write a `model.py` file with `initialize`, `execute`, `finalize` methods. The escape hatch for anything that doesn't fit a standard backend.
- **OpenVINO**, **FIL** (forest models), **DALI** (data loading), and others.

You pick a backend per model in its config file.

### 2.3 Dynamic Batching

This is where most of Triton's performance comes from. When ten clients each send a single image at the same time, Triton can wait a few milliseconds, gather them into a batch of ten, run one forward pass on the GPU, then split the results back. The clients don't know — they each just see their own response. You enable it with one block in the model config:

```
dynamic_batching {
  preferred_batch_size: [4, 8, 16]
  max_queue_delay_microseconds: 5000
}
```

`max_queue_delay_microseconds` is your latency-throughput dial. Higher values mean bigger batches and more throughput but more tail latency.

### 2.4 Concurrent Model Execution and Instance Groups

By default Triton loads one copy of your model on the assigned device. If your model is small enough that one copy uses, say, 4 GB on a 24 GB GPU, you're wasting memory. You can tell Triton to load multiple instances:

```
instance_group [
  { count: 2, kind: KIND_GPU, gpus: [0] },
  { count: 2, kind: KIND_GPU, gpus: [1] }
]
```

Now Triton runs four copies — two on GPU 0, two on GPU 1 — and routes incoming requests across them. Each instance can serve a request independently, so concurrency scales accordingly. This is called **instance groups**, and tuning it is a big part of getting the most out of your hardware.

### 2.5 Ensembles and BLS

Real pipelines have multiple steps. For a CLIP-based image search, a request might flow: raw image bytes → preprocess (resize, normalize) → CLIP encoder → return embedding. Triton supports two ways to chain models:

- **Ensembles** — declarative. You define a graph in a config file: outputs of one model become inputs to another. Fast, no extra Python overhead.
- **BLS (Business Logic Scripting)** — a Python backend that programmatically calls other models. Use this when you need conditionals, loops, or variable-length pipelines.

Use ensembles when the pipeline is fixed; use BLS when it isn't.

### 2.6 Quick Mental Model

Think of Triton as a smart router and batcher in front of a pool of model instances, all behind a single HTTP/gRPC endpoint. Your job is to lay out the model repository correctly and write good config files. Triton handles the rest.

---

## 3. The Model Repository — Triton's Heart <a name="3-the-model-repository"></a>

The repository is a directory that looks exactly like this:

```
model_repository/
├── clip_image_encoder/
│   ├── config.pbtxt
│   └── 1/
│       └── model.onnx
├── clip_text_encoder/
│   ├── config.pbtxt
│   └── 1/
│       └── model.onnx
└── llava_vlm/
    ├── config.pbtxt
    └── 1/
        └── model.py
```

Three rules:

1. Each top-level directory is a model name. Clients address it by this name.
2. Inside each model directory you have a `config.pbtxt` (the model's configuration in protobuf text format) and one or more numbered subdirectories (`1/`, `2/`, ...) each holding a version of the model file.
3. The filename inside the version directory must match the backend's expectations: `model.onnx`, `model.pt`, `model.plan`, `model.py`, etc.

### A minimal config.pbtxt

Here's the bare minimum for an ONNX CLIP image encoder:

```protobuf
name: "clip_image_encoder"
backend: "onnxruntime"
max_batch_size: 32

input [
  {
    name: "pixel_values"
    data_type: TYPE_FP32
    dims: [3, 224, 224]
  }
]

output [
  {
    name: "image_embeds"
    data_type: TYPE_FP32
    dims: [512]
  }
]

instance_group [
  { count: 1, kind: KIND_GPU }
]

dynamic_batching {
  preferred_batch_size: [8, 16]
  max_queue_delay_microseconds: 5000
}
```

Notes on this file:

- `max_batch_size: 32` enables batching with a cap of 32. Triton automatically prepends a batch dimension to your inputs and outputs, so you write `dims: [3, 224, 224]` not `[N, 3, 224, 224]`.
- Set `max_batch_size: 0` if your model already includes the batch dimension internally and you don't want Triton to manage it.
- Names in `input` and `output` must match the actual tensor names in your ONNX/PyTorch model. Use `netron` (a free model viewer) to inspect them if unsure.

---

## 4. Local Setup: Your First Triton Server <a name="4-local-setup"></a>

Before going to the cloud, get it working on your laptop or a single GPU machine. This is non-negotiable — debugging Triton inside Kubernetes is far harder than on a local Docker daemon.

### 4.1 Prerequisites

- A machine with an NVIDIA GPU (or skip GPU steps and use CPU initially).
- Docker installed.
- NVIDIA Container Toolkit installed so Docker can see the GPU. Test with `docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi`. You should see your GPU listed.

### 4.2 Pull the Triton image

NVIDIA publishes monthly releases on NGC. As of this writing, the current tag is `26.04-py3`. Use whatever the most recent stable tag is when you read this:

```bash
docker pull nvcr.io/nvidia/tritonserver:26.04-py3
```

For LLMs with vLLM specifically:

```bash
docker pull nvcr.io/nvidia/tritonserver:26.04-vllm-python-py3
```

These images are large — around 10–15 GB. Pull while you set up your model repository.

### 4.3 Run the server

Assuming your model repository is at `/home/you/model_repository`:

```bash
docker run --gpus all --rm \
  -p 8000:8000 -p 8001:8001 -p 8002:8002 \
  -v /home/you/model_repository:/models \
  nvcr.io/nvidia/tritonserver:26.04-py3 \
  tritonserver --model-repository=/models
```

The three ports are: 8000 for HTTP, 8001 for gRPC, 8002 for Prometheus metrics. Always expose all three.

When Triton finishes loading, you'll see a table of models with status `READY`. If a model fails to load, the log explains why — typically a config mismatch or a missing file. Read it carefully.

### 4.4 First inference call

Health check:

```bash
curl -v localhost:8000/v2/health/ready
```

A 200 OK means Triton is up. To send actual inference requests, use the `tritonclient` Python library:

```bash
pip install tritonclient[all]
```

```python
import numpy as np
import tritonclient.http as httpclient

client = httpclient.InferenceServerClient(url="localhost:8000")

# Dummy preprocessed image: shape [1, 3, 224, 224], normalized
image = np.random.randn(1, 3, 224, 224).astype(np.float32)

inputs = [httpclient.InferInput("pixel_values", image.shape, "FP32")]
inputs[0].set_data_from_numpy(image)

outputs = [httpclient.InferRequestedOutput("image_embeds")]

response = client.infer("clip_image_encoder", inputs, outputs=outputs)
embedding = response.as_numpy("image_embeds")
print("Got embedding shape:", embedding.shape)  # (1, 512)
```

If this works locally, the cloud part is mostly Kubernetes plumbing.

---

## 5. Deploying a CLIP Encoder <a name="5-deploying-clip"></a>

CLIP is a great first real model: small, fast, and useful for image search, zero-shot classification, and as a building block for multimodal RAG.

### 5.1 Export CLIP to ONNX

We'll use the OpenAI CLIP ViT-B/32 from Hugging Face:

```python
import torch
from transformers import CLIPModel, CLIPProcessor

model_id = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(model_id).eval()
processor = CLIPProcessor.from_pretrained(model_id)

# Export the vision tower
dummy_pixel_values = torch.randn(1, 3, 224, 224)
torch.onnx.export(
    model.vision_model,
    dummy_pixel_values,
    "clip_image_encoder.onnx",
    input_names=["pixel_values"],
    output_names=["last_hidden_state", "pooler_output"],
    dynamic_axes={
        "pixel_values": {0: "batch"},
        "last_hidden_state": {0: "batch"},
        "pooler_output": {0: "batch"},
    },
    opset_version=17,
)

# Export the text tower
dummy_input_ids = torch.ones(1, 77, dtype=torch.long)
dummy_attention_mask = torch.ones(1, 77, dtype=torch.long)
torch.onnx.export(
    model.text_model,
    (dummy_input_ids, dummy_attention_mask),
    "clip_text_encoder.onnx",
    input_names=["input_ids", "attention_mask"],
    output_names=["last_hidden_state", "pooler_output"],
    dynamic_axes={
        "input_ids": {0: "batch"},
        "attention_mask": {0: "batch"},
        "last_hidden_state": {0: "batch"},
        "pooler_output": {0: "batch"},
    },
    opset_version=17,
)
```

Two important notes. First, `dynamic_axes` is what allows variable batch sizes — without it the ONNX model is locked to batch size 1. Second, in production you typically want to project the pooled output through CLIP's projection head to get the canonical 512-dim embedding; here we simplified by exporting just the towers.

### 5.2 Lay out the repository

```
model_repository/
├── clip_image_encoder/
│   ├── config.pbtxt
│   └── 1/
│       └── model.onnx
└── clip_text_encoder/
    ├── config.pbtxt
    └── 1/
        └── model.onnx
```

`clip_image_encoder/config.pbtxt`:

```protobuf
name: "clip_image_encoder"
backend: "onnxruntime"
max_batch_size: 64

input [
  {
    name: "pixel_values"
    data_type: TYPE_FP32
    dims: [3, 224, 224]
  }
]

output [
  {
    name: "pooler_output"
    data_type: TYPE_FP32
    dims: [768]
  }
]

instance_group [
  { count: 2, kind: KIND_GPU }
]

dynamic_batching {
  preferred_batch_size: [8, 16, 32]
  max_queue_delay_microseconds: 3000
}

optimization {
  execution_accelerators {
    gpu_execution_accelerator: [
      { name: "tensorrt"
        parameters { key: "precision_mode" value: "FP16" }
        parameters { key: "max_workspace_size_bytes" value: "1073741824" }
      }
    ]
  }
}
```

That `optimization` block is gold — it tells ONNX Runtime to convert eligible ops to TensorRT FP16 on the fly the first time the model loads. You typically get a 2–5× speedup over plain ONNX with no extra effort. The first request after startup will be slow (TensorRT is compiling); subsequent requests are fast.

The text encoder config follows the same pattern with `input_ids` and `attention_mask` of dtype `TYPE_INT64` and dims `[-1]` (variable length up to model max).

### 5.3 Add a preprocessing model and an ensemble

Clients shouldn't have to know how to resize and normalize images. Add a Python backend model that handles preprocessing:

```
model_repository/
├── clip_preprocess/
│   ├── config.pbtxt
│   └── 1/
│       └── model.py
├── clip_image_encoder/
│   └── ...
└── clip_pipeline/                  # The ensemble
    ├── config.pbtxt
    └── 1/                          # Empty but must exist
```

`clip_preprocess/1/model.py`:

```python
import numpy as np
import triton_python_backend_utils as pb_utils
from PIL import Image
import io

class TritonPythonModel:
    def initialize(self, args):
        self.mean = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
        self.std = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)

    def execute(self, requests):
        responses = []
        for request in requests:
            raw_bytes = pb_utils.get_input_tensor_by_name(request, "image_bytes").as_numpy()
            batch = []
            for item in raw_bytes:
                img = Image.open(io.BytesIO(item[0])).convert("RGB").resize((224, 224))
                arr = np.asarray(img, dtype=np.float32) / 255.0
                arr = (arr - self.mean) / self.std
                arr = arr.transpose(2, 0, 1)  # HWC -> CHW
                batch.append(arr)
            out = np.stack(batch, axis=0)
            tensor = pb_utils.Tensor("pixel_values", out)
            responses.append(pb_utils.InferenceResponse(output_tensors=[tensor]))
        return responses
```

`clip_pipeline/config.pbtxt`:

```protobuf
name: "clip_pipeline"
platform: "ensemble"
max_batch_size: 32

input [
  { name: "image_bytes", data_type: TYPE_STRING, dims: [1] }
]

output [
  { name: "image_embeds", data_type: TYPE_FP32, dims: [768] }
]

ensemble_scheduling {
  step [
    {
      model_name: "clip_preprocess"
      model_version: -1
      input_map { key: "image_bytes" value: "image_bytes" }
      output_map { key: "pixel_values" value: "preprocessed" }
    },
    {
      model_name: "clip_image_encoder"
      model_version: -1
      input_map { key: "pixel_values" value: "preprocessed" }
      output_map { key: "pooler_output" value: "image_embeds" }
    }
  ]
}
```

Now clients send raw JPEG bytes to `clip_pipeline` and get back embeddings. The two underlying models stay independently versionable, batchable, and tunable.

---

## 6. Deploying a Vision Language Model (LLaVA) <a name="6-deploying-vlm"></a>

VLMs are heavier and benefit from a different backend. The two strong options are:

- **vLLM backend** — easy. Hugging Face model name, one config file, you're done. PagedAttention and continuous batching come free.
- **TensorRT-LLM backend** — faster but requires a compilation step targeting your specific GPU architecture (compile on H100, the engine runs only on H100).

Start with vLLM. Move to TensorRT-LLM when you've measured and need more.

### 6.1 vLLM backend setup

Use the vLLM-specific Triton image: `nvcr.io/nvidia/tritonserver:26.04-vllm-python-py3`.

Repository structure:

```
model_repository/
└── llava/
    ├── config.pbtxt
    └── 1/
        └── model.json
```

`llava/config.pbtxt`:

```protobuf
name: "llava"
backend: "vllm"
max_batch_size: 0

model_transaction_policy {
  decoupled: True
}

input [
  { name: "text_input", data_type: TYPE_STRING, dims: [1] },
  { name: "image", data_type: TYPE_STRING, dims: [1], optional: true },
  { name: "sampling_parameters", data_type: TYPE_STRING, dims: [1], optional: true }
]

output [
  { name: "text_output", data_type: TYPE_STRING, dims: [-1] }
]

instance_group [
  { count: 1, kind: KIND_MODEL }
]
```

Two things to call out:

- `decoupled: True` enables streaming responses. Each token comes back as a separate response on the same request. Essential for chat UX.
- `kind: KIND_MODEL` instead of `KIND_GPU` because vLLM manages GPU placement itself, including tensor parallelism across multiple GPUs.

`llava/1/model.json`:

```json
{
  "model": "llava-hf/llava-1.5-7b-hf",
  "disable_log_requests": true,
  "gpu_memory_utilization": 0.9,
  "tensor_parallel_size": 2,
  "max_model_len": 4096,
  "dtype": "bfloat16"
}
```

`tensor_parallel_size: 2` tells vLLM to shard the model across 2 GPUs. Use 4 or 8 for larger models like LLaVA-1.6-34B. Make sure the number of GPUs visible to the container is at least this number.

### 6.2 Launching

```bash
docker run --gpus all --rm --shm-size=4g \
  -p 8000:8000 -p 8001:8001 -p 8002:8002 \
  -v /home/you/model_repository:/models \
  nvcr.io/nvidia/tritonserver:26.04-vllm-python-py3 \
  tritonserver --model-repository=/models
```

The `--shm-size=4g` flag is required — vLLM uses shared memory for inter-process communication during tensor parallelism, and Docker's default 64 MB is far too small. This is one of the most common deployment-stage gotchas.

### 6.3 Client call with image and streaming

```python
import json
import base64
import tritonclient.grpc as grpcclient

with open("photo.jpg", "rb") as f:
    image_b64 = base64.b64encode(f.read()).decode()

client = grpcclient.InferenceServerClient(url="localhost:8001")

# Streaming requires a callback
results = []
def callback(result, error):
    if error: raise error
    results.append(result.as_numpy("text_output"))

client.start_stream(callback=callback)
inputs = [
    grpcclient.InferInput("text_input", [1], "BYTES"),
    grpcclient.InferInput("image", [1], "BYTES"),
    grpcclient.InferInput("sampling_parameters", [1], "BYTES"),
]
inputs[0].set_data_from_numpy(np.array([b"Describe this image."]).reshape(1))
inputs[1].set_data_from_numpy(np.array([image_b64.encode()]).reshape(1))
inputs[2].set_data_from_numpy(np.array([json.dumps({"max_tokens": 256, "temperature": 0.7}).encode()]).reshape(1))

client.async_stream_infer(model_name="llava", inputs=inputs)
client.stop_stream()
print(b"".join(r[0] for r in results).decode())
```

Tokens arrive incrementally on the stream. For an HTTP-based interface, use server-sent events from the same `decoupled: True` config.

---

## 7. Multi-GPU Strategies <a name="7-multi-gpu"></a>

There are three distinct multi-GPU patterns. Pick the right one for each model — they have different tradeoffs.

### 7.1 Data parallelism (replicas)

Best for models that fit comfortably on one GPU. Load N copies, route requests across them. Throughput scales nearly linearly with N.

```protobuf
instance_group [
  { count: 1, kind: KIND_GPU, gpus: [0] },
  { count: 1, kind: KIND_GPU, gpus: [1] },
  { count: 1, kind: KIND_GPU, gpus: [2] },
  { count: 1, kind: KIND_GPU, gpus: [3] }
]
```

### 7.2 Tensor parallelism (sharding)

Best for models too large to fit on one GPU. Split the model's weight matrices across GPUs; each forward pass uses all of them. Latency goes up modestly; the model now fits.

For vLLM, set `tensor_parallel_size` in `model.json`. For TensorRT-LLM, you specify it during the engine compilation step.

### 7.3 Pipeline parallelism

Split the model's layers across GPUs sequentially. Mostly used for very large models (70B+). vLLM and TensorRT-LLM support it via `pipeline_parallel_size`. Less efficient than tensor parallelism unless you have many concurrent requests to keep all stages busy.

### 7.4 Mixing models on the same GPUs

A common production layout for multimodal pipelines: load CLIP encoders (small) on every GPU, load the VLM (large) sharded across all GPUs. Triton handles the request routing transparently. The only constraint is total memory: sum(model sizes) + KV cache for the LLM + dynamic batching headroom must fit.

A practical starting allocation on a 4×A100-80GB node serving CLIP + LLaVA-13B:

| Model              | Strategy                | GPUs        | Memory per GPU |
|--------------------|-------------------------|-------------|----------------|
| clip_image_encoder | 4 replicas              | 0,1,2,3     | ~1 GB          |
| clip_text_encoder  | 4 replicas              | 0,1,2,3     | ~0.5 GB        |
| llava_vlm          | tensor parallel size 4  | 0,1,2,3     | ~30 GB         |

Leaving roughly 48 GB per GPU for KV cache and batching — generous and safe.

---

## 8. Cloud Deployment on AWS (EKS) <a name="8-aws-eks"></a>

We'll deploy on Elastic Kubernetes Service. The pattern is: GPU node group → Triton Deployment → LoadBalancer Service → optional autoscaler.

### 8.1 Prerequisites

- AWS account with a quota for GPU instances. P-series and G-series have separate quotas; both default low. Request increases ahead of time.
- `aws`, `kubectl`, `eksctl`, and `helm` installed.
- ECR repo or use NVIDIA's NGC images directly (works fine; no need to mirror).

### 8.2 Pick the GPU instance type

| Instance     | GPU         | GPUs | GPU Memory | When to use                         |
|--------------|-------------|------|------------|-------------------------------------|
| g5.xlarge    | A10G        | 1    | 24 GB      | CLIP, small models, dev             |
| g5.12xlarge  | A10G        | 4    | 96 GB      | CLIP fleet + 7B VLM                 |
| g6.12xlarge  | L4          | 4    | 96 GB      | Cost-optimized inference            |
| p4d.24xlarge | A100        | 8    | 320 GB     | 13B–70B VLMs, high throughput       |
| p5.48xlarge  | H100        | 8    | 640 GB     | Frontier models, max performance    |

Pricing changes constantly — check the live AWS pricing page when planning.

### 8.3 Create the cluster

A reproducible `eksctl` config file is best:

```yaml
# eks-triton-cluster.yaml
apiVersion: eksctl.io/v1alpha5
kind: ClusterConfig
metadata:
  name: triton-cluster
  region: us-west-2
  version: "1.30"

managedNodeGroups:
  - name: cpu-nodes
    instanceType: m6i.large
    desiredCapacity: 2
    minSize: 1
    maxSize: 3
  - name: gpu-nodes
    instanceType: g5.12xlarge
    desiredCapacity: 1
    minSize: 0
    maxSize: 4
    volumeSize: 200
    labels: { role: gpu }
    taints:
      - key: nvidia.com/gpu
        value: "true"
        effect: NoSchedule
    tags:
      k8s.io/cluster-autoscaler/enabled: "true"
      k8s.io/cluster-autoscaler/triton-cluster: "owned"

iam:
  withOIDC: true
```

```bash
eksctl create cluster -f eks-triton-cluster.yaml
```

This takes 15–20 minutes. The taint on the GPU nodes ensures only pods that explicitly tolerate it land there — keeping random workloads off your expensive GPUs.

### 8.4 Install the NVIDIA GPU Operator

The GPU Operator handles the device plugin, drivers, and DCGM exporter for monitoring in one shot:

```bash
helm repo add nvidia https://helm.ngc.nvidia.com/nvidia
helm repo update
helm install --wait gpu-operator nvidia/gpu-operator \
  -n gpu-operator --create-namespace
```

After a few minutes, `kubectl describe node <gpu-node>` should show `nvidia.com/gpu: 4` under Allocatable.

### 8.5 Set up the model repository in S3

Create a bucket and upload your repo:

```bash
aws s3 mb s3://my-triton-models --region us-west-2
aws s3 sync ./model_repository s3://my-triton-models/model_repository
```

Triton can read directly from S3, so you don't need to bake models into the container or use a persistent volume. Give the pods access via IRSA (IAM Roles for Service Accounts):

```bash
eksctl create iamserviceaccount \
  --cluster=triton-cluster \
  --namespace=triton \
  --name=triton-sa \
  --attach-policy-arn=arn:aws:iam::aws:policy/AmazonS3ReadOnlyAccess \
  --approve
```

### 8.6 Deploy Triton

```yaml
# triton-deployment.yaml
apiVersion: v1
kind: Namespace
metadata: { name: triton }
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: triton-server
  namespace: triton
spec:
  replicas: 1
  selector:
    matchLabels: { app: triton }
  template:
    metadata:
      labels: { app: triton }
    spec:
      serviceAccountName: triton-sa
      tolerations:
        - key: nvidia.com/gpu
          operator: Equal
          value: "true"
          effect: NoSchedule
      nodeSelector:
        role: gpu
      containers:
        - name: triton
          image: nvcr.io/nvidia/tritonserver:26.04-vllm-python-py3
          args:
            - tritonserver
            - --model-repository=s3://my-triton-models/model_repository
            - --strict-model-config=false
            - --log-verbose=1
          ports:
            - { containerPort: 8000, name: http }
            - { containerPort: 8001, name: grpc }
            - { containerPort: 8002, name: metrics }
          resources:
            limits:
              nvidia.com/gpu: 4
              memory: 64Gi
              cpu: "16"
          readinessProbe:
            httpGet: { path: /v2/health/ready, port: 8000 }
            initialDelaySeconds: 60
            periodSeconds: 10
          livenessProbe:
            httpGet: { path: /v2/health/live, port: 8000 }
            initialDelaySeconds: 120
            periodSeconds: 30
          volumeMounts:
            - { name: dshm, mountPath: /dev/shm }
      volumes:
        - name: dshm
          emptyDir: { medium: Memory, sizeLimit: 8Gi }
---
apiVersion: v1
kind: Service
metadata:
  name: triton-server
  namespace: triton
  annotations:
    service.beta.kubernetes.io/aws-load-balancer-type: nlb
spec:
  type: LoadBalancer
  selector: { app: triton }
  ports:
    - { name: http, port: 8000, targetPort: 8000 }
    - { name: grpc, port: 8001, targetPort: 8001 }
    - { name: metrics, port: 8002, targetPort: 8002 }
```

```bash
kubectl apply -f triton-deployment.yaml
```

`initialDelaySeconds: 60` on readiness gives Triton time to load models from S3. For a 7B VLM this might need to be 300+ seconds. Watch with `kubectl logs -f deployment/triton-server -n triton`.

### 8.7 Test

```bash
TRITON_URL=$(kubectl get svc triton-server -n triton -o jsonpath='{.status.loadBalancer.ingress[0].hostname}')
curl http://${TRITON_URL}:8000/v2/health/ready
curl http://${TRITON_URL}:8000/v2/models/clip_image_encoder/config | jq
```

For production, place an internal NLB behind a private API gateway rather than exposing 8000 to the public internet. The example above keeps things simple.

---

## 9. Cloud Deployment on GCP (GKE) <a name="9-gcp-gke"></a>

The structure mirrors AWS exactly: Kubernetes cluster + GPU node pool + Deployment + LoadBalancer. The plumbing is different.

### 9.1 Pick the GPU machine type

| Machine          | GPU       | GPUs | When to use                       |
|------------------|-----------|------|-----------------------------------|
| g2-standard-12   | L4        | 1    | CLIP, small models                |
| g2-standard-48   | L4        | 4    | Mixed fleet, 7B VLMs              |
| a2-highgpu-1g    | A100-40GB | 1    | Mid-sized models                  |
| a2-ultragpu-8g   | A100-80GB | 8    | 13B–70B VLMs                      |
| a3-highgpu-8g    | H100      | 8    | Frontier models                   |

### 9.2 Create the cluster and node pool

```bash
gcloud container clusters create triton-cluster \
  --region us-central1 \
  --release-channel regular \
  --num-nodes 1 \
  --machine-type e2-standard-4

gcloud container node-pools create gpu-pool \
  --cluster triton-cluster \
  --region us-central1 \
  --machine-type g2-standard-48 \
  --accelerator type=nvidia-l4,count=4,gpu-driver-version=latest \
  --num-nodes 1 \
  --min-nodes 0 \
  --max-nodes 4 \
  --enable-autoscaling \
  --node-taints nvidia.com/gpu=true:NoSchedule \
  --node-labels role=gpu
```

`gpu-driver-version=latest` is the easy mode — GKE installs and updates the NVIDIA driver for you. Without it you have to apply a daemonset manually.

### 9.3 Get cluster credentials

```bash
gcloud container clusters get-credentials triton-cluster --region us-central1
kubectl get nodes
```

### 9.4 Upload your repository to GCS

```bash
gcloud storage buckets create gs://my-triton-models --location us-central1
gcloud storage cp -r ./model_repository gs://my-triton-models/
```

### 9.5 Configure Workload Identity for the pod

This is GKE's equivalent of IRSA — lets your pod read GCS without static credentials:

```bash
# Create namespace and service accounts
kubectl create namespace triton
kubectl create serviceaccount triton-sa -n triton

gcloud iam service-accounts create triton-gcs-reader \
  --display-name "Triton GCS Reader"

gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member "serviceAccount:triton-gcs-reader@${PROJECT_ID}.iam.gserviceaccount.com" \
  --role "roles/storage.objectViewer"

gcloud iam service-accounts add-iam-policy-binding \
  triton-gcs-reader@${PROJECT_ID}.iam.gserviceaccount.com \
  --role "roles/iam.workloadIdentityUser" \
  --member "serviceAccount:${PROJECT_ID}.svc.id.goog[triton/triton-sa]"

kubectl annotate serviceaccount triton-sa -n triton \
  iam.gke.io/gcp-service-account=triton-gcs-reader@${PROJECT_ID}.iam.gserviceaccount.com
```

### 9.6 Deploy Triton

The deployment manifest is nearly identical to the AWS one — just change the model repository URI to `gs://my-triton-models/model_repository`:

```yaml
# Same as AWS deployment, except:
args:
  - tritonserver
  - --model-repository=gs://my-triton-models/model_repository
  - --strict-model-config=false
```

Apply it:

```bash
kubectl apply -f triton-deployment-gcp.yaml
```

GKE creates a regional Network Load Balancer for the LoadBalancer service automatically.

### 9.7 Test

```bash
TRITON_IP=$(kubectl get svc triton-server -n triton -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
curl http://${TRITON_IP}:8000/v2/health/ready
```

### 9.8 GCP-specific bonus: Inference Quickstart

GCP recently added a feature called Inference Quickstart that auto-generates tuned manifests for common LLMs on GKE. Worth checking when you start a new deployment — it can shortcut a lot of the config tuning. It's surfaced inside the GKE console under AI/ML → Models.

---

## 10. Monitoring, Autoscaling, and Production Hardening <a name="10-production"></a>

### 10.1 Metrics

Triton exposes Prometheus metrics on port 8002 by default. The metrics that matter most:

- `nv_inference_request_success` — request count.
- `nv_inference_request_duration_us` — end-to-end latency.
- `nv_inference_queue_duration_us` — time spent waiting in the dynamic batching queue. A growing queue is your earliest signal that you need to scale.
- `nv_inference_compute_infer_duration_us` — GPU compute time.
- `nv_gpu_utilization` and `nv_gpu_memory_used_bytes` — GPU saturation.

Install Prometheus and Grafana via kube-prometheus-stack on either cluster:

```bash
helm install kube-prom-stack prometheus-community/kube-prometheus-stack \
  -n monitoring --create-namespace
```

Then a `ServiceMonitor` to scrape Triton:

```yaml
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata: { name: triton, namespace: triton }
spec:
  selector: { matchLabels: { app: triton } }
  endpoints: [{ port: metrics, interval: 15s }]
```

NVIDIA publishes a Grafana dashboard on grafana.com (search for "Triton Inference Server"). Import it directly.

### 10.2 Horizontal Pod Autoscaling

Standard CPU-based HPA isn't useful for GPU workloads. Use a custom metric instead — most often `nv_inference_queue_duration_us`:

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata: { name: triton, namespace: triton }
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: triton-server
  minReplicas: 1
  maxReplicas: 8
  metrics:
    - type: Pods
      pods:
        metric:
          name: nv_inference_queue_duration_us
        target:
          type: AverageValue
          averageValue: 50000   # 50 ms average queue time
```

For this to work you need the Prometheus Adapter installed and configured to expose Triton metrics to the Kubernetes metrics API. The combination of a cluster autoscaler (scaling GPU nodes) and HPA (scaling Triton pods) gives you end-to-end elasticity.

### 10.3 Model Analyzer

NVIDIA ships a tool called Model Analyzer that sweeps configurations (batch size, instance count, precision) for a given model and finds the Pareto frontier of latency vs. throughput on your actual hardware. Run it once per model when you onboard it; the output is a recommended `config.pbtxt`. Worth doing — it can find 2–3× improvements on models you thought were already tuned.

### 10.4 Security checklist

- Don't expose port 8000 to the public internet directly. Put it behind an authenticated API gateway.
- Triton supports HTTPS — configure with `--http-tls-cert-file` and friends.
- Enable client authentication if exposed beyond your VPC.
- The model repository in S3/GCS should be private with IAM-only access. Never ship a public bucket.
- Pin the Triton image to a specific tag (e.g., `26.04-py3`), never `latest`.
- Set Pod Security Standards to `restricted` on the namespace.

### 10.5 Cost control

- Use spot/preemptible GPU nodes for non-critical workloads. GCP preemptible A100s are often less than half the on-demand price; AWS Spot G5s have similar discounts. Configure pod disruption budgets and tolerations carefully.
- Keep `minSize: 0` on GPU node pools. When idle they cost nothing.
- Use L4 GPUs (g6 on AWS, g2 on GCP) over A10G/A100 when models fit. They're roughly half the price for inference workloads and FP8 capable.
- Aggressively profile and quantize. INT8 or FP8 typically halve memory and double throughput vs FP16 with negligible quality loss for CLIP and most VLMs.

---

## 11. Troubleshooting Cheat Sheet <a name="11-troubleshooting"></a>

| Symptom | Likely cause | Fix |
|---|---|---|
| Model status `UNAVAILABLE` on startup | Bad config.pbtxt or filename mismatch | Check Triton logs; usually one explicit line tells you |
| Tensor parallelism errors with vLLM | shm too small | Add `--shm-size=4g` (Docker) or memory-backed `emptyDir` (K8s) |
| First request very slow, subsequent fast | TensorRT compiling | Expected; warm up at startup with a dummy request |
| Pod stays Pending | No GPU node available | Check `kubectl describe pod`; typically autoscaler hasn't fired |
| 503 errors under load | Queue backed up | Increase replicas or increase batch size |
| OOM during model load | Too many instances | Reduce `count` in `instance_group` |
| S3/GCS access denied | IRSA/Workload Identity misconfigured | Re-check the binding; service account annotation matters |
| Throughput plateaus despite low GPU util | Batching not engaging | Check `max_queue_delay_microseconds`; raise it slightly |
| Grad-different outputs vs PyTorch | ONNX export precision drift | Re-export with opset 17+, or use `dtype=float32` end-to-end |

---

## Closing Thoughts

The mental model that makes all of this click: Triton is a smart router and batcher in front of a pool of model instances. The model repository tells it what models exist; config files tell it how each one wants to be run; the cloud layer (EKS/GKE) gives it the GPUs. Everything else — ensembles, BLS, dynamic batching, multi-GPU strategies, autoscaling — is configuration applied to that core architecture.

A reasonable path through this material if you're learning by doing:

1. Get the local CLIP example from section 5 working on any machine with a GPU.
2. Add the preprocessing model and ensemble. Convince yourself with `curl` that one HTTP call now does the whole pipeline.
3. Deploy that same model repository to either EKS or GKE — pick whichever cloud you already have an account in.
4. Add the LLaVA VLM from section 6.
5. Add monitoring and HPA.
6. Run Model Analyzer on each model and apply its recommendations.

By that point you'll have a production-grade multimodal inference platform, and the next model you add will take an afternoon, not a sprint.

For the latest features and breaking changes, the canonical references are the [Triton GitHub repository](https://github.com/triton-inference-server/server) and the [official documentation](https://docs.nvidia.com/deeplearning/triton-inference-server/). NVIDIA publishes a new container release monthly — there's almost always something useful in the changelog.
