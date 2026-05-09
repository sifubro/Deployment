# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository purpose

Reference implementations for deploying image/text embedding models to production. Two independent deployment tracks live side-by-side:

- `retrieval/resnet50/sagemaker/` — ResNet50 image embedder packaged as a FastAPI container for **AWS SageMaker**.
- `retrieval/clip/` — CLIP (image + text) deployed on **NVIDIA Triton Inference Server** as a multi-model pipeline (Python preprocess + TorchScript image/text encoders).

`cls/resnet50/` is currently a placeholder (empty). The two retrieval tracks do **not** share code — treat them as separate projects.

## Track 1: ResNet50 → SageMaker (`retrieval/resnet50/sagemaker/`)

The container exposes the two endpoints SageMaker mandates: `GET /ping` (health) and `POST /invocations` (inference). Body is `{"image_b64": "<base64 jpeg>"}`, response is `{"embedding": [...], "dim": 2048}`. Embeddings are L2-normalised inside `ResNetEmbedder` so dot products are cosine similarity.

Key files:
- `embedder.py` — `ResNetEmbedder` (ResNet50 minus FC layer) + ImageNet `preprocess` transform. Imported by both the server and tests.
- `app.py` — FastAPI app; instantiates the model **once at import time** (`model = ResNetEmbedder()` at module scope). Don't move it inside the request handler.
- `Dockerfile` — Pre-downloads weights at build time via `RUN python -c "from embedder import ResNetEmbedder; ResNetEmbedder()"` so cold starts don't fetch from torchvision. Installs PyTorch CUDA 11.8 wheels from the PyTorch index, then `requirements.txt` (FastAPI/uvicorn/pillow/pydantic).
- `deploy.py` — `sagemaker.Model.deploy(...)` wrapper. The `finally: predictor.delete_endpoint()` at the bottom tears the endpoint down at the end of the script — fine for smoke tests, **remove before any real deployment**. Region is hard-coded `eu-west-1`, account `586917955410`, role `SageMakerExecutionRole`. Default instance is `ml.t2.medium` (CPU); switch to `ml.g4dn.xlarge` for GPU.
- `infer.py` — boto3 `sagemaker-runtime` client for hitting the deployed endpoint.
- `test_embedder.py` — local sanity check using cat/car pairs in `images/`; verifies same-class pairs score higher than cross-class.

Common commands (run from `retrieval/resnet50/sagemaker/`):

```bash
docker build -t depop-embedder:latest .
docker run -p 8080:8080 depop-embedder:latest
curl http://localhost:8080/ping

# Smoke-test inference (Git Bash / Linux)
IMG_B64=$(base64 -w 0 test.jpg)
echo "{\"image_b64\": \"$IMG_B64\"}" > payload.json
curl -X POST http://localhost:8080/invocations -H "Content-Type: application/json" -d @payload.json

# Deploy to SageMaker (run from base conda env, not the torch env — see README note)
python deploy.py
python infer.py   # hits the live endpoint

# Local correctness check (no Docker)
python test_embedder.py
```

The README pins one environment-specific install command for the user's machine: `sagemaker` is installed into `py3.10_torch_2.2.2_cu118` via `--target` rather than `pip install sagemaker` in the active env.

## Track 2: CLIP → Triton Inference Server (`retrieval/clip/`)

A 3-model Triton ensemble-style layout (no actual ensemble config wired up yet — clients call `clip_image` directly after preprocessing client-side, see `test_triton.py`):

- `clip_preprocess` — Python backend. Decodes raw image bytes via PIL, runs `CLIPProcessor` to produce `[3,224,224]` FP32 pixel values. CPU, 2 instances. `max_batch_size: 0` (no batching).
- `clip_image` — `pytorch_libtorch` backend. Input `pixel_values [3,224,224]` FP32, output `embedding [512]` FP32. GPU, dynamic batching (preferred sizes 8/16/32, max queue delay 5ms).
- `clip_text` — `pytorch_libtorch` backend. Inputs `input_ids` + `attention_mask` (INT64, length 77), output `embedding [512]`. GPU, dynamic batching (preferred 16/32/64).

The TorchScript artifacts (`model_repository/clip_*/1/model.pt`) are **not committed** — `1/` directories are empty. Generate them with `export_to_torchscript.py`, which wraps HuggingFace `CLIPModel` in two `nn.Module`s (`ImageEncoder`, `TextEncoder`) that include the projection head and L2-normalise outputs, then `torch.jit.trace`s them to the right paths.

Important architectural constraint: the TorchScript wrappers in `export_to_torchscript.py` and the eager `CLIPEmbedder` in `clip_embedder.py` must stay numerically equivalent — both apply `visual_projection`/`text_projection` then `F.normalize(dim=-1)`. If you change normalisation or projection in one place, mirror it in the other or downstream similarity scores diverge.

Common commands (run from `retrieval/clip/`):

```bash
# Generate TorchScript model files (writes into model_repository/clip_*/1/model.pt)
python export_to_torchscript.py

# Start Triton (image/tag is whatever you have locally)
docker run --gpus=all --rm -p 8000:8000 -p 8001:8001 -p 8002:8002 \
  -v "$PWD/model_repository:/models" \
  nvcr.io/nvidia/tritonserver:<tag>-py3 tritonserver --model-repository=/models

# Eager-mode local sanity check
python clip_embedder.py     # expects ./jacket.jpg

# HTTP smoke test against running Triton
python test_triton.py       # expects ./jacket.jpg, hits localhost:8000
```

`test_triton.py` does CLIP preprocessing **on the client** with HuggingFace's `CLIPProcessor` and posts numpy arrays directly to `clip_image` — it bypasses the `clip_preprocess` Triton model. To exercise the Python backend instead, post raw image bytes to `clip_preprocess` and feed its output into `clip_image` (or wire up an actual Triton ensemble).
