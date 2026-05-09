# Triton Inference Server — VLM deployment tutorial

A line-by-line walkthrough of every important `config.pbtxt` setting,
demonstrated on a realistic 5-stage Vision Language Model pipeline. Each
stage uses a different backend / feature so you see Triton's full surface
area in one repository.

By the time you finish reading you should be able to:
- write production configs from scratch,
- pick `platform` vs `backend` correctly,
- tune the throughput/latency trade-off knobs (dynamic batching, CUDA
  graphs, instance groups) with intent rather than guessing,
- choose between ensemble and BLS pipelines,
- decide when `decoupled`, `sequence_batching`, or in-flight batching is
  the right tool.

## Pipeline overview

The repository wires five models behind one ensemble endpoint:

```
                  ┌────────────────┐
   image_bytes ──▶│ vlm_preprocess │──▶ pixel_values ───────────┐
   prompt      ──▶│  Python · CPU  │──▶ input_ids ───────────┐  │
                  └────────────────┘──▶ attention_mask ──┐   │  │
                                                         │   │  │
                                            ┌────────────▼───▼──▼──┐
                                            │   vlm_text_encoder    │
                                            │   ONNX + TRT · GPU    │
                                            └───────────┬───────────┘
                                                        │ text_embeds
                                            ┌───────────▼───────────┐
   pixel_values ────────────────────────────│  vlm_vision_encoder   │
                                            │   TorchScript · GPU   │
                                            └───────────┬───────────┘
                                                        │ image_features
                                            ┌───────────▼───────────┐
                                            │   vlm_llm_decoder     │
                                            │  Python · DECOUPLED   │
                                            └───────────┬───────────┘
                                                        │ token_ids, is_final
                                            ┌───────────▼───────────┐
                                            │   vlm_postprocess     │
                                            │   Python · CPU        │
                                            └───────────┬───────────┘
                                                        ▼
                                                       text
```

Everything runs in-process inside `tritonserver`. The ensemble pipes
tensors GPU-to-GPU between steps without serialising over HTTP.

## Layout

```
retrieval/VLM/model_repository_tutorial/
├── README.md                              ← this file
├── vlm_preprocess/        config.pbtxt    ← Python backend, CPU, parameters
├── vlm_vision_encoder/    config.pbtxt    ← TorchScript, dynamic batching, CUDA graphs, warmup, priorities
├── vlm_text_encoder/      config.pbtxt    ← ONNX + TensorRT execution accelerator, ragged batch
├── vlm_llm_decoder/       config.pbtxt    ← decoupled (token streaming), sequence_batching alt
├── vlm_postprocess/       config.pbtxt    ← response_cache, version_policy: all
└── vlm_ensemble/          config.pbtxt    ← ensemble_scheduling, decoupled ensemble
```

In a real deployment each model directory also contains numbered version
subdirs (`1/`, `2/`, ...) holding the actual artefacts (`model.py`,
`model.pt`, `model.onnx`, etc.). Those are out of scope for this tutorial.

## `platform` vs `backend`

Every model declares **exactly one** of these. Pick by what's loading the
weights:

| use `platform` | use `backend` |
| --- | --- |
| `pytorch_libtorch` (TorchScript `.pt`) | `python` (your `model.py`) |
| `onnxruntime_onnx` (`.onnx`) | `tensorrtllm` |
| `tensorrt_plan` (`.plan`) | `vllm` |
| `tensorflow_savedmodel` / `tensorflow_graphdef` | `dali`, `fil`, `openvino`, custom |
| `ensemble` (no weights) | |

If you have a serialised framework artefact, prefer `platform`. If you
need to run arbitrary code, use `backend: "python"`.

## Per-section reference

### `name`
Optional but always set it — catches typos in CI before they land in prod.

### `max_batch_size`
- `0` — no batch dim is added. Required for stateful models, ragged inputs,
  and backends (TRT-LLM, vLLM) that batch internally.
- `>0` — Triton prepends a batch dim, dynamic_batching may group requests.
  Set to the largest batch that fits in GPU memory at peak shape.

### `input` / `output`

| field | meaning |
| --- | --- |
| `name` | Tensor name. Must match what the framework / Python code expects. |
| `data_type` | `TYPE_FP32` `TYPE_FP16` `TYPE_BF16` `TYPE_INT8/16/32/64` `TYPE_UINT8` `TYPE_BOOL` `TYPE_STRING`. |
| `dims` | Shape EXCLUDING the batch dim when `max_batch_size > 0`. `-1` = variable. |
| `format` | `FORMAT_NCHW` / `FORMAT_NHWC` / `FORMAT_NONE`. Metadata for tooling and clients. |
| `optional` | Clients may omit this input (default `false`). |
| `allow_ragged_batch` | Stack different-length requests without padding. Backend support required. |
| `is_shape_tensor` | TensorRT shape tensors only. |
| `reshape { shape: [...] }` | Reshape between client view and model view. |

### `instance_group`
One entry creates `count` parallel copies of the model.
- `kind`: `KIND_CPU` / `KIND_GPU` / `KIND_AUTO` / `KIND_MODEL`.
- `gpus: [0,1]` — pin to specific devices.
- `profile: ["fast"]` — TensorRT optimisation profile names.
- `rate_limiter { resources [...] priority }` — cooperative scheduling
  across models that share a finite resource (GPU memory, custom
  semaphore).

Adding a second instance only helps if `nvidia-smi dmon -s mu` shows the
GPU under-utilised at peak load. Each copy duplicates weights.

### `dynamic_batching` — the biggest throughput knob

Triton waits up to `max_queue_delay_microseconds` to fill a batch close
to one of `preferred_batch_size`, then dispatches.

- `preserve_ordering: true` — return responses in arrival order. Costs
  ~1–5% throughput.
- `priority_levels` + `default_priority_level` — multiple queues.
  Lower number = higher priority.
- `default_queue_policy` (or per-level `priority_queue_policy`):
  - `timeout_action: REJECT | DELAY` — fail-fast vs. flag-late.
  - `default_timeout_microseconds` — per-request deadline.
  - `allow_timeout_override: true` — clients can shorten via header.
  - `max_queue_size` — back-pressure. (max+1)th queued request fails immediately.

### `sequence_batching` — stateful models

Use **instead of** `dynamic_batching` for models that maintain per-client
state (chat KV cache, online learning). Triton routes all requests
sharing a `correlation_id` to the same instance.

```protobuf
sequence_batching {
  max_sequence_idle_microseconds: 5000000          # close idle sessions after 5 s

  oldest {                                          # or `direct {}` for strict per-slot
    max_candidate_sequences: 8
    preferred_batch_size: [ 4, 8 ]
    max_queue_delay_microseconds: 1000
  }

  control_input [
    { name: "START"   control [{ kind: CONTROL_SEQUENCE_START   fp32_false_true: [0,1] }] },
    { name: "READY"   control [{ kind: CONTROL_SEQUENCE_READY   fp32_false_true: [0,1] }] },
    { name: "END"     control [{ kind: CONTROL_SEQUENCE_END     fp32_false_true: [0,1] }] },
    { name: "CORRID"  control [{ kind: CONTROL_SEQUENCE_CORRID  data_type: TYPE_UINT64 }] }
  ]

  state [                                           # Triton-managed implicit state
    { input_name: "kv_in"  output_name: "kv_out"
      data_type: TYPE_FP16  dims: [ -1, 4096 ]
      initial_state [{ data_type: TYPE_FP16  dims: [0,4096]  zero_data: true  name: "init" }]
    }
  ]
}
```

- `direct {}` — strict per-slot routing, lowest latency.
- `oldest {}` — fill batches from the oldest pending sequence in any slot,
  highest utilisation.

Real LLM serving (TRT-LLM, vLLM) handles KV cache *inside* the backend
and uses neither dynamic nor sequence batching here.

### `model_transaction_policy`
- `decoupled: true` — one request can produce 0..N responses. Required
  for token-streaming LLMs and pub-sub patterns. Your model emits
  responses via `response_sender.send()` instead of returning from
  `execute()`. **An ensemble that contains a decoupled step must itself
  be decoupled.**

### `optimization`
| key | what it does |
| --- | --- |
| `graph { level: N }` | Backend graph optimiser level (ORT: -1/0/1/2/99). |
| `priority` | `PRIORITY_DEFAULT` / `PRIORITY_MAX` / `PRIORITY_MIN`. |
| `cuda { graphs: true graph_spec [...] busy_wait_events output_copy_stream }` | Capture CUDA graphs for listed batch sizes. Eliminates kernel-launch overhead. |
| `input_pinned_memory { enable: true }` / `output_pinned_memory` | Async H2D/D2H DMA. Default ON. |
| `gather_kernel_buffer_threshold` | When to use a gather kernel vs per-request memcpy. 0 disables. |
| `execution_accelerators { gpu_execution_accelerator | cpu_execution_accelerator }` | Plug TRT into ORT, OpenVINO into CPU sessions. |

### `model_warmup`
Synthetic requests run at LOAD time so the first real request doesn't
pay the cost of CUDA graph capture / JIT compile / engine deserialise.
Configure one entry per (batch_size, shape) combination you actually
serve. Required if you turned on `cuda.graphs`.

### `version_policy`
- `latest { num_versions: 1 }` — only the highest-numbered (default).
- `all {}` — every version on disk loaded simultaneously. For blue/green.
- `specific { versions: [1, 3] }` — pin exactly these.

### `parameters`
Free-form key/value strings handed to the backend.
- Python: read in `initialize()` from `model_config["parameters"]`.
- ORT: `intra_op_thread_count`, `execution_mode`, `enable_mem_pattern`.
- TensorRT-LLM: engine path, KV cache fraction, scheduler policy.
- vLLM: model path, gpu_memory_utilization, max_model_len.

Always prefer `parameters` over hard-coded constants in `model.py` —
the same artefact then works across dev/staging/prod.

### `response_cache`
Per-model cache. Enable here AND start the server with
`--cache-config=local,size=1073741824` (or `redis,host=...`). Key is a
hash of (model name + version + inputs). **Only safe for deterministic
models** — never enable for sampling LLMs.

### `default_model_filename` / `cc_model_filenames`
Override the file Triton looks for in `1/`. `cc_model_filenames` selects
a different file per GPU compute capability — ship one TRT plan per
arch (sm_80, sm_86, sm_90):

```protobuf
default_model_filename: "model_a10.plan"
cc_model_filenames [
  { key: "8.0" value: "model_a100.plan" },
  { key: "8.6" value: "model_a10.plan"  },
  { key: "9.0" value: "model_h100.plan" }
]
```

### `model_repository_agents`
Hooks that fire on model load/unload. Built-in agents include `checksum`
(verify weights weren't corrupted) and you can write your own
(decrypt-from-KMS, download-from-S3, run-static-analysis). Wire with:

```protobuf
model_repository_agents {
  agents [
    { name: "checksum"  parameters [{ key: "MD5"  value: "<hex>" }] }
  ]
}
```

### `metric_tags`
Extra labels on every Prometheus metric this model emits. Slice
dashboards by tenant / family / stage:

```protobuf
metric_tags [
  { key: "tenant"  value: "depop" },
  { key: "stage"   value: "vision_encoder" }
]
```

### `runtime`
Override the backend shared library file. Rarely needed.

## Ensemble vs. BLS (Business Logic Scripting)

|                  | Ensemble                  | BLS                                |
| ---              | ---                       | ---                                |
| Defined by       | Static DAG in config.pbtxt | Imperative Python that calls other models |
| Branching        | No                        | Yes (`if`, loops, retries)         |
| Variable steps   | No                        | Yes                                |
| Latency overhead | Lowest                    | Slightly higher                    |
| Best for         | Fixed pipeline            | Conditional logic, fan-out, retries |

Pick ensemble first; reach for BLS only when control flow is genuinely
data-dependent (e.g. *"skip the vision encoder if no image was supplied"*
or *"if confidence < 0.5, re-run with a bigger model"*).

## Deployment

```powershell
# 1. Layout (you've already done this)
#    retrieval/VLM/model_repository_tutorial/
#      vlm_<stage>/config.pbtxt
#      vlm_<stage>/1/model.{py,pt,onnx,plan}

# 2. Validate configs (optional, recommended in CI)
docker run --rm `
  -v "${PWD}/retrieval/VLM/model_repository_tutorial:/models" `
  nvcr.io/nvidia/tritonserver:24.10-py3 `
  tritonserver --model-repository=/models `
               --strict-model-config=true `
               --exit-on-error=true

# 3. Serve
docker run --gpus=all --rm `
  -p 8000:8000 -p 8001:8001 -p 8002:8002 `
  -v "${PWD}/retrieval/VLM/model_repository_tutorial:/models" `
  nvcr.io/nvidia/tritonserver:24.10-py3 `
  tritonserver --model-repository=/models `
               --log-verbose=1 `
               --cache-config=local,size=1073741824

# 4. Smoke-test
#    HTTP    : 8000          gRPC : 8001          metrics : 8002/metrics
curl -s localhost:8000/v2/health/ready
curl -s localhost:8000/v2/models/vlm_ensemble | jq .
```

Send a request:

```python
import base64, requests, json

with open("jacket.jpg", "rb") as f:
    img_b64 = base64.b64encode(f.read()).decode()

resp = requests.post(
    "http://localhost:8000/v2/models/vlm_ensemble/infer",
    json={
        "inputs": [
            { "name": "image_bytes", "shape": [1], "datatype": "BYTES",
              "data": [img_b64] },
            { "name": "prompt",      "shape": [1], "datatype": "BYTES",
              "data": ["Describe this item for an e-commerce listing."] }
        ]
    }
)
```

For streaming responses (decoupled mode), use the gRPC client's
`stream_infer()` — HTTP/SSE is not yet GA.

## Tuning checklist

1. Start with `max_batch_size` = whatever fits in memory and
   `dynamic_batching { preferred_batch_size: [...] }` matching your
   benchmark sweet-spot.
2. Add `model_warmup` for every batch size you serve (mandatory if
   `cuda.graphs: true`).
3. Capture CUDA graphs once you trust shapes are stable.
4. Add a second `instance_group` only if the GPU is under-utilised at
   peak load.
5. Plug TensorRT in via the ONNX-Runtime EP **before** building a
   hand-crafted `.plan` — captures ~80% of the speedup with ~10% of the
   brittleness.
6. Turn on `response_cache` only for deterministic stages (postprocess,
   embeddings of stable inputs). Never for sampling LLMs.
7. Use `priority_levels` to keep an interactive lane open under bulk load.
8. Run `perf_analyzer -m vlm_ensemble --concurrency-range 1:64:4` and
   `model-analyzer profile` before declaring it production-ready.
9. Watch the Prometheus metrics: `nv_inference_queue_duration_us` and
   `nv_inference_compute_infer_duration_us` together tell you whether
   you're bottlenecked on scheduling or compute.

## Common pitfalls

- **Decoupled step in a non-decoupled ensemble** — fails to load with a
  cryptic error. Always set `model_transaction_policy.decoupled: true`
  on the ensemble too.
- **`max_batch_size` mismatch in an ensemble** — the ensemble's
  `max_batch_size` must be ≤ every step's `max_batch_size` (steps with 0
  are unconstrained). Mismatches surface only at runtime.
- **Forgetting `--cache-config`** — `response_cache { enable: true }`
  silently does nothing if the server wasn't started with a cache backend.
- **Hard-coding paths in `model.py`** — use `parameters` so the same
  artefact loads in every environment.
- **Bumping `instance_group.count` blindly** — multiple instances
  multiply weight memory. Profile first; you might just need a bigger
  `preferred_batch_size`.
- **Skipping `model_warmup` with `cuda.graphs: true`** — the first real
  request pays the graph-capture cost. Looks like a network spike on
  dashboards but isn't.
