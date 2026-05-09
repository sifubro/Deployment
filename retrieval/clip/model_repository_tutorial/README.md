# Triton Inference Server — CLIP deployment tutorial

A line-by-line walkthrough of every important `config.pbtxt` setting,
demonstrated on a production-shaped CLIP deployment. Companion piece to
[`../../VLM/model_repository_tutorial/`](../../VLM/model_repository_tutorial/) —
both stand alone, but the VLM tutorial covers `decoupled` streaming and
`sequence_batching` in more depth, while this one drills into

- **two parallel modality paths** that share an output contract,
- **`tensorrt_plan`** with optimisation profiles and per-arch engines,
- **BLS** (Business Logic Scripting) for branching on optional inputs,
- aggressive **`response_cache`** use — embeddings are deterministic and
  real-world query distributions repeat heavily.

## Pipeline overview

```
                                                       ┌────────────────────┐
                                            ┌─────────▶│  clip_router (BLS) │──────────────┐
                                            │          └─────┬─────┬────────┘              │
                                            │                │     │                       │
                                            │    image path  │     │   text path           ▼
       (one of)                             │                ▼     ▼                  ┌────────────┐
   image_bytes ──────────┐                  │     ┌────────────┐  ┌────────────┐       │ embedding  │
                         │                  │     │ clip_image │  │ clip_text  │       │  [512]     │
   text          ────────┴──────────────────┘     │ _ensemble  │  │ _ensemble  │       │  modality  │
                                                  └─────┬──────┘  └─────┬──────┘       └────────────┘
                                                        │               │
                                            ┌───────────▼───────────┐   │
                                            │ clip_image_preprocess │   │       (ensembles can be
                                            │     Python · CPU      │   │        called directly,
                                            └───────────┬───────────┘   │        bypassing the router)
                                                        │ pixel_values  │
                                            ┌───────────▼───────────┐   │
                                            │  clip_image_encoder   │   │
                                            │   TorchScript · GPU   │   │
                                            │   (or _trt variant)   │   │
                                            └───────────┬───────────┘   │
                                                        │               │
                                                        ▼               ▼
                                                  embedding[512]   ┌──────────────────────┐
                                                                   │ clip_text_preprocess │
                                                                   │    Python · CPU      │
                                                                   └──────────┬───────────┘
                                                                              │ input_ids,
                                                                              │ attn_mask
                                                                   ┌──────────▼───────────┐
                                                                   │   clip_text_encoder  │
                                                                   │    ONNX + TRT · GPU  │
                                                                   └──────────┬───────────┘
                                                                              │
                                                                              ▼
                                                                       embedding[512]
```

Both encoders emit an L2-normalised 512-D embedding so dot products are
cosine similarity. Vector-DB code stays modality-agnostic.

## Layout

```
retrieval/clip/model_repository_tutorial/
├── README.md                                ← this file
├── clip_image_preprocess/   config.pbtxt    ← Python · CPU
├── clip_text_preprocess/    config.pbtxt    ← Python · CPU · response cache for repeated queries
├── clip_image_encoder/      config.pbtxt    ← TorchScript · multi-GPU instance_group · CUDA graphs
├── clip_text_encoder/       config.pbtxt    ← ONNX + TensorRT EP · engine cache
├── clip_image_encoder_trt/  config.pbtxt    ← tensorrt_plan · optimization profiles · cc_model_filenames
├── clip_image_ensemble/     config.pbtxt    ← bytes → embedding
├── clip_text_ensemble/      config.pbtxt    ← text → embedding
└── clip_router/             config.pbtxt    ← BLS: branch on optional inputs
```

In a real deployment each model directory also contains numbered
version subdirs (`1/`, `2/`, ...) holding the actual weights
(`model.py`, `model.pt`, `model.onnx`, `model.plan`, etc.). Those are
out of scope for this tutorial.

## `platform` vs `backend`

Every model declares **exactly one**:

| use `platform` | use `backend` |
| --- | --- |
| `pytorch_libtorch` (TorchScript `.pt`) | `python` (your `model.py`, including BLS) |
| `onnxruntime_onnx` (`.onnx`) | `tensorrtllm` |
| `tensorrt_plan` (`.plan`) | `vllm` |
| `tensorflow_savedmodel` / `tensorflow_graphdef` | `dali`, `fil`, `openvino`, custom |
| `ensemble` (no weights) | |

If you have a serialised framework artefact, prefer `platform`. If you
need to run arbitrary code (preprocess, postprocess, BLS), use
`backend: "python"`.

## Section-by-section reference

### `name`
Optional but always set it explicitly — catches typos in CI before they
land in production.

### `max_batch_size`
- `0` — Triton does NOT add a batch dim. Inputs arrive in the exact
  shape declared under `dims`. Required for backends that batch
  internally (TRT-LLM, vLLM) or for stages that genuinely process one
  request at a time (router, image preprocess).
- `>0` — Triton prepends a batch dim and may group requests via
  `dynamic_batching`. The value is the absolute maximum, picked to fit
  in GPU memory at peak shape.

### `input` / `output`

| field | meaning |
| --- | --- |
| `name` | Tensor name; must match the framework / Python code. |
| `data_type` | `TYPE_FP32` `TYPE_FP16` `TYPE_BF16` `TYPE_INT8/16/32/64` `TYPE_UINT8` `TYPE_BOOL` `TYPE_STRING`. |
| `dims` | Shape EXCLUDING batch dim when `max_batch_size > 0`. `-1` = variable. |
| `format` | `FORMAT_NCHW` / `FORMAT_NHWC` / `FORMAT_NONE`. Metadata for tooling and clients. |
| `optional` | Clients may omit this input. Critical for the BLS router's branching. |
| `allow_ragged_batch` | Stack different-length requests without padding. Backend must support it. NOT applicable to CLIP since seq is always 77. |
| `is_shape_tensor` | TensorRT shape tensors only. |
| `reshape { shape: [...] }` | Reshape between client view and model view. |

### `instance_group`
One entry creates `count` parallel copies of the model.
- `kind`: `KIND_CPU` / `KIND_GPU` / `KIND_AUTO` / `KIND_MODEL`.
- `gpus: [0,1]` — pin to specific devices. Use multiple entries (one
  per GPU) instead of `gpus: [0,1]` when you want explicit control.
- `name` — label for logs and metric tags. Useful when entries differ
  by purpose (e.g. `fast_lane` vs `throughput_lane`).
- `profile: ["fast"]` — TensorRT optimisation profile names. Lets you
  pin instances to specific shape ranges baked into the plan.
- `rate_limiter { resources [...] priority }` — cooperative scheduling
  across models that share a finite resource.

Adding a second instance on the same GPU only helps when one instance
leaves it under-utilised. Each copy duplicates weights —
`nvidia-smi dmon -s mu` is your friend.

### `dynamic_batching` — the biggest throughput knob

Triton holds incoming requests up to `max_queue_delay_microseconds`,
trying to assemble a batch close to one of `preferred_batch_size`,
then dispatches as one forward pass.

- `preserve_ordering: true` — return responses in arrival order. Costs
  ~1–5% throughput. Embedding clients almost never need this.
- `priority_levels` + `default_priority_level` — multiple queues.
  Lower number = higher priority. Pattern: priority 1 = interactive
  search, priority 2 = bulk re-indexing.
- `default_queue_policy` (or per-level `priority_queue_policy`):
  - `timeout_action: REJECT | DELAY` — fail-fast vs. flag-late.
  - `default_timeout_microseconds` — per-request deadline.
  - `allow_timeout_override: true` — clients can shorten via header.
  - `max_queue_size` — back-pressure threshold; (max+1)th queued
    request fails immediately.

### `optimization`

| key | what it does |
| --- | --- |
| `graph { level: N }` | Backend graph optimiser level (ORT: -1/0/1/2/99). |
| `priority` | `PRIORITY_DEFAULT` / `PRIORITY_MAX` / `PRIORITY_MIN`. |
| `cuda { graphs: true graph_spec [...] busy_wait_events output_copy_stream }` | Capture CUDA graphs for listed batch sizes. Required for low-latency CLIP serving — kernel-launch overhead dominates without it. |
| `input_pinned_memory { enable: true }` / `output_pinned_memory` | Async H2D/D2H DMA. Default ON. |
| `gather_kernel_buffer_threshold` | When to use a gather kernel vs per-request memcpy. 0 disables. |
| `execution_accelerators { gpu_execution_accelerator | cpu_execution_accelerator }` | Plug TensorRT into ORT, OpenVINO into CPU sessions. See `clip_text_encoder/config.pbtxt`. |

### `model_warmup`
Synthetic requests run at LOAD time so the first real request doesn't
pay the cost of CUDA-graph capture / engine deserialisation / JIT
compile. Configure one entry per (batch_size, shape) you actually
serve. **Mandatory** if `cuda.graphs: true` — otherwise the first
request after every deploy spikes p99 by hundreds of milliseconds.

### `version_policy`
- `latest { num_versions: 1 }` — only the highest-numbered (default).
- `latest { num_versions: 2 }` — current + previous, for blue/green.
- `all {}` — every version on disk loaded simultaneously. For
  per-request version pinning. Costs one model's RAM per extra version.
- `specific { versions: [1, 3] }` — pin exactly these.

### `parameters`
Free-form key/value strings handed to the backend.
- Python: read in `initialize()` from `model_config["parameters"]`.
- ORT: `intra_op_thread_count`, `execution_mode`, `enable_mem_pattern`.
- TensorRT EP: `precision_mode`, `trt_engine_cache_enable`,
  `trt_int8_calibration_table_name`, ...

Always prefer `parameters` over hard-coded constants in `model.py` —
the same artefact then loads in dev/staging/prod.

### `response_cache`

This is the standout knob for CLIP. Embeddings are **deterministic**
and real-world traffic distributions are **heavy-tailed** — the same
texts and images are queried over and over.

| stage | hit rate (typical) | enable? |
| --- | --- | --- |
| `clip_image_preprocess` | < 1% (image-bytes hash collisions are rare) | **no** |
| `clip_text_preprocess`  | 30–60% on real e-commerce queries | **yes** |
| `clip_image_encoder`    | 5–20% on product catalogues | **yes** |
| `clip_text_encoder`     | 30–60% | **yes** |
| `clip_image_ensemble`   | inherits image_encoder hit rate | **yes** |
| `clip_text_ensemble`    | high — entire pipeline skipped | **yes** |
| `clip_router`           | n/a — would double-cache | **no** |

You **must** also start tritonserver with a cache backend:

```powershell
--cache-config=local,size=1073741824
# or, for shared cache across replicas:
--cache-config=redis,host=cache.internal,port=6379,size=4294967296
```

**Never** enable `response_cache` for stochastic outputs (sampling
LLMs, dropout-at-inference). For embedding models it's almost
always a win.

### `default_model_filename` / `cc_model_filenames`

Override the file Triton looks for in `1/`. `cc_model_filenames`
selects a different file per GPU compute capability — **critical for
TensorRT plans**, which are tied to the exact GPU arch they were built
on:

```protobuf
default_model_filename: "model_a10.plan"
cc_model_filenames [
  { key: "7.5" value: "model_t4.plan"   },   # T4
  { key: "8.0" value: "model_a100.plan" },   # A100
  { key: "8.6" value: "model_a10.plan"  },   # A10
  { key: "8.9" value: "model_l4.plan"   },   # L4
  { key: "9.0" value: "model_h100.plan" }    # H100
]
```

For TorchScript and ONNX you can usually ship one file across archs.

### `model_repository_agents`
Hooks that fire on model load/unload. Built-in agents include
`checksum` (verify weights weren't corrupted in transit). You can
write your own — decrypt-from-KMS, download-from-S3,
run-static-analysis. Wire with:

```protobuf
model_repository_agents {
  agents [
    { name: "checksum"  parameters [{ key: "MD5"  value: "<hex>" }] }
  ]
}
```

### `metric_tags`
Extra labels appended to every Prometheus metric this model emits.
Used heavily in this tutorial via `parameters` (Triton exposes those
under `nv_inference_*` labels on recent versions). Slice your
dashboards by `modality`, `stage`, `engine`.

### `runtime`
Override the backend shared-library file. Rarely needed.

## TensorRT plan: optimisation profiles in depth

A single `.plan` can hold **multiple optimisation profiles**, each
defining (min, opt, max) shapes for every dynamic input. Build with
`trtexec`:

```bash
trtexec --onnx=clip_image.onnx \
  --minShapes=pixel_values:1x3x224x224 \
  --optShapes=pixel_values:8x3x224x224 \
  --maxShapes=pixel_values:8x3x224x224 \
  --profile=fast \
  --minShapes=pixel_values:1x3x224x224 \
  --optShapes=pixel_values:32x3x224x224 \
  --maxShapes=pixel_values:32x3x224x224 \
  --profile=throughput \
  --fp16 --workspace=2048 \
  --saveEngine=model_a10.plan
```

Then in `config.pbtxt` you pin instances to profiles by name:

```protobuf
instance_group [
  { count: 1  kind: KIND_GPU  gpus: [0]  name: "fast_lane"        profile: ["fast"] },
  { count: 1  kind: KIND_GPU  gpus: [0]  name: "throughput_lane"  profile: ["throughput"] }
]
```

This gives you a **latency-optimised lane** (small batches, captured
at opt=8) and a **throughput-optimised lane** (big batches, opt=32)
sharing the same GPU.

Set `dynamic_batching.preferred_batch_size` to the union of the opt
shapes (`[8, 32]`) so the scheduler aims for sizes the engine was
actually optimised for.

Plans are **arch-bound** — see `cc_model_filenames` above.

## Ensemble vs. BLS

| | Ensemble | BLS (Python) |
| --- | --- | --- |
| Defined by | Static DAG in `config.pbtxt` | Imperative Python in `1/model.py` |
| Branching | No | Yes (`if`, loops, retries) |
| Variable steps | No | Yes |
| Latency overhead | Lowest (zero-copy GPU↔GPU) | Slightly higher (Python dispatch) |
| Best for | Fixed pipeline | Conditional logic, fan-out, data-dependent retries |
| Decoupled support | Yes — but the ensemble itself must also be decoupled | Yes — flexible per-call |

Pick ensemble first. Reach for BLS only when control flow is genuinely
data-dependent. The `clip_router` config in this tutorial is the
textbook BLS use case: branch on which optional input was supplied.

A decoupled ensemble step requires the ensemble itself to be decoupled:

```protobuf
model_transaction_policy { decoupled: true }
```

CLIP is fully synchronous so we don't need this here; see the VLM
tutorial for streaming examples.

## Deployment

```powershell
# 1. Layout (you've already done this)
#    retrieval/clip/model_repository_tutorial/
#      <model_name>/config.pbtxt
#      <model_name>/1/<weights or model.py>

# 2. Validate configs in CI
docker run --rm `
  -v "${PWD}/retrieval/clip/model_repository_tutorial:/models" `
  nvcr.io/nvidia/tritonserver:24.10-py3 `
  tritonserver --model-repository=/models `
               --strict-model-config=true `
               --exit-on-error=true

# 3. Serve (with response cache enabled)
docker run --gpus=all --rm `
  -p 8000:8000 -p 8001:8001 -p 8002:8002 `
  -v "${PWD}/retrieval/clip/model_repository_tutorial:/models" `
  nvcr.io/nvidia/tritonserver:24.10-py3 `
  tritonserver --model-repository=/models `
               --log-verbose=1 `
               --cache-config=local,size=1073741824

# 4. Smoke-test
curl -s localhost:8000/v2/health/ready
curl -s localhost:8000/v2/models/clip_image_ensemble | jq .
curl -s localhost:8000/v2/models/clip_text_ensemble  | jq .
```

Image embedding request:

```python
import base64, requests
with open("jacket.jpg", "rb") as f:
    img_b64 = base64.b64encode(f.read()).decode()
resp = requests.post(
    "http://localhost:8000/v2/models/clip_image_ensemble/infer",
    json={
        "inputs": [
            { "name": "image_bytes", "shape": [1], "datatype": "BYTES",
              "data": [img_b64] }
        ]
    }
)
embedding = resp.json()["outputs"][0]["data"]   # 512 floats, L2-normalised
```

Text embedding request:

```python
resp = requests.post(
    "http://localhost:8000/v2/models/clip_text_ensemble/infer",
    json={
        "inputs": [
            { "name": "text", "shape": [1], "datatype": "BYTES",
              "data": ["a red leather jacket"] }
        ]
    }
)
```

Modality-agnostic via the router:

```python
# Image
resp = requests.post(
    "http://localhost:8000/v2/models/clip_router/infer",
    json={
        "inputs": [
            { "name": "image_bytes", "shape": [1], "datatype": "BYTES",
              "data": [img_b64] }
        ]
    }
)
# Text — same endpoint, different input
resp = requests.post(
    "http://localhost:8000/v2/models/clip_router/infer",
    json={
        "inputs": [
            { "name": "text", "shape": [1], "datatype": "BYTES",
              "data": ["a red leather jacket"] }
        ]
    }
)
```

## Tuning checklist

1. Measure baseline with `perf_analyzer -m clip_image_ensemble --concurrency-range 1:64:4`
   and `perf_analyzer -m clip_text_ensemble --concurrency-range 1:128:8`.
2. Set `dynamic_batching.preferred_batch_size` to the sizes where GPU
   utilisation plateaus on your benchmark — usually 16–64 for image,
   64–256 for text.
3. Add `model_warmup` for every batch size you serve.
4. Capture CUDA graphs once shapes are stable. Required for tight p99.
5. Turn `response_cache` ON at the encoder and ensemble level. Watch
   `nv_cache_num_hits` / `nv_cache_num_misses` to confirm it's working.
6. Add a TensorRT plan variant (`clip_image_encoder_trt`) only AFTER
   the ONNX+TRT-EP version is correct — captures another ~2× throughput
   at the cost of arch-binding.
7. Use `priority_levels` to keep an interactive search lane open under
   bulk indexing load.
8. Run `model-analyzer profile --model-repository=...` for an
   end-to-end sweep across batching / instance / precision settings
   before declaring it production-ready.
9. Watch the Prometheus metrics: `nv_inference_queue_duration_us`
   and `nv_inference_compute_infer_duration_us` together tell you
   whether you're bottlenecked on scheduling or compute.

## Common pitfalls

- **Different embedding norms across modalities** — the L2 normalisation
  MUST be inside the traced/exported graph for both encoders. Triton
  has no "normalise this output" hook. If one encoder returns raw
  features and the other normalises, cosine sim is silently wrong.
- **Forgetting `--cache-config`** — `response_cache { enable: true }`
  silently does nothing without a server-level cache backend.
- **TensorRT plan / GPU arch mismatch** — a plan built on sm_86 (A10)
  won't run on sm_80 (A100). Use `cc_model_filenames` and ship one
  plan per arch, or fall back to ONNX + TRT EP (which builds the
  engine on first run for the host arch).
- **`dynamic_batching` on a `tensorrt_plan` model with profiles that
  don't cover the requested shape** — TRT raises an error at execution
  time. Always make sure `preferred_batch_size` ⊆ shapes covered by
  some profile.
- **Caching the router** — `response_cache` on `clip_router` would
  double-store everything that's already cached at the inner ensemble
  level. Cache once, at the deepest stage that's still deterministic.
- **Loading too many versions with `version_policy: all {}`** — every
  loaded version takes full model memory. Use `latest { num_versions: 2 }`
  for blue/green; reserve `all {}` for the rare case you genuinely
  need per-request version pinning.
- **Skipping `model_warmup` with `cuda.graphs: true`** — first real
  request after deploy pays graph-capture cost. Looks like a network
  spike on dashboards but isn't.
