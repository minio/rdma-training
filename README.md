# S3-over-RDMA for PyTorch: benchmarks and reference pipelines

A reproducible benchmark suite comparing **MinIO AIStor GPU-Direct S3-over-RDMA**
against a tuned **S3-over-HTTP** baseline for the two storage operations that
matter in deep-learning training:

1. **Feeding the GPUs** — reading training data, measured as end-to-end ResNet-50
   step time and as raw storage throughput.
2. **Checkpointing** — saving and restoring model + optimizer state, which stalls
   training while it runs.

Everything here ran on real hardware against a real cluster. Every number is
backed by raw JSON in [`results/`](results/), and every RDMA measurement carries
server-side proof that RDMA was actually used — because the client library falls
back to HTTP silently, and we caught it doing exactly that (three separate ways)
during bring-up.

---

## Read this first: does S3-over-RDMA apply to your workload?

This is the most useful thing in the repo, and it is a measured constraint rather
than an opinion:

> **GPU-Direct RDMA engages only when the destination is CUDA device memory.**
> Host memory — even CUDA-pinned host memory — silently falls back to TCP.
>
> **GPU JPEG decoders require the encoded bitstream in host memory.**

So the question is not how fast your network is. It is:

### Are the bytes in your object store already in the layout your GPU consumes?

| Your data | RDMA on the data path | Why |
| --- | --- | --- |
| **Checkpoints** (model + optimizer state) | ✅ **Yes** | tensors, GPU-native by construction |
| Pre-decoded image shards (FFCV/DALI-style raw uint8) | ✅ Yes | tensors |
| Embeddings, activations, feature stores | ✅ Yes | tensors |
| Model weights (safetensors, sharded) | ✅ Yes | tensors — **measured: 16 GB in 0.77 s** |
| Tokenized text / `.npy` int32 | ✅ Yes | tensors |
| KV-cache offload | ✅ Yes | tensors |
| **JPEG / PNG / WebP image datasets** | ❌ **No** | decode needs the bytes on the host |
| Video (h264/h265) | ⚠️ Untested | only if NVDEC can decode from VRAM |

If your input pipeline is JPEG, the data path cannot use RDMA as normally built —
convert the dataset to a GPU-native layout first, and only then does the transport
matter. We measured that conversion at **1.76× on ResNet-50**, which is larger than
anything the transport did. **Checkpointing benefits regardless of what your input
pipeline looks like.**

Where RDMA helps and where it cannot:
[`docs/05-interpreting-results.md`](docs/05-interpreting-results.md).

---

## Headline results

Hardware: 2 × storage node (24 × 7 TB NVMe each, 48 drives, EC:4), 1 × client with
8 × H200 and dual 400 GbE RoCE. **All client measurements use a single 400 GbE
rail**, whose practical ceiling (`iperf3`) is 41.5 GB/s.

### Storage throughput into GPU memory

| Path | Peak | Client CPU | Per core |
| --- | --- | --- | --- |
| **S3-over-RDMA** (1 Python process, 64 streams) | **42.4 GB/s** | **0.95 cores** | **44.6 GB/s** |
| S3-over-HTTP (`warp`, native Go, 64 streams) | 39.4 GB/s | 68.2 cores | 0.58 GB/s |
| S3-over-HTTP (1 Python process, best) | 3.0 GB/s | 2.8 cores | 1.06 GB/s |

**Both transports saturate the NIC.** The difference is not bandwidth — it is that
HTTP spends ~68 CPU cores to get there and RDMA spends one. On a training node
those are the cores you wanted for augmentation. RDMA also reaches line rate from a
*single Python process*, which HTTP cannot do at all.

→ raw runs: `results/b1-fetch-*.json`

### ResNet-50 training — 200 batches on 8 × H200 (DDP, batch 256/GPU)

| Data layout | Transport | Seconds for 200 batches | images/s | Host CPU | Storage wait |
| --- | --- | --- | --- | --- | --- |
| **raw** (GPU-native) | **RDMA** | **14.31 s** (13.26–15.17) | 27,339 | **8.2 cores** | 0.00 s |
| **raw** (GPU-native) | HTTP | 16.68 s (16.45–16.98) | 24,553 | 11.3 cores | 0.00 s |
| **jpeg** (tar shards) | HTTP | 28.25 s | 14,450 | 9.3 cores | 0.00 s |

Four repeats each at matched fetch concurrency. The honest reading:

1. **The data layout is worth 1.76×** — JPEG → GPU-native took 28.25 s to 16.08 s on
   the *same* transport. Still the largest single effect.
2. **The transport is worth 14%**, with non-overlapping ranges. But **neither arm
   waits on storage** — `fetch_wait` is 0.00 s on both — so this is *not* a
   storage-throughput win.
3. **The mechanism is CPU contention.** Compute alone differs by 2.24 s (14.11 s vs
   16.35 s) on identical work, because HTTP's 11.3 cores of storage handling compete
   with the training loop's own CPU. A storage path that costs 11 cores is not free
   even when you are never blocked on it.

→ raw runs: `results/b3-resnet50-*.json`

### Checkpointing

Per process, checkpoint written from / restored into GPU memory:

| Checkpoint | Operation | RDMA | Tuned HTTP | `torch.save`/`load` + HTTP |
| --- | --- | --- | --- | --- |
| 16 GiB | save | **0.65 s** | 12.16 s | 62.55 s |
| 16 GiB | restore | **0.50 s** | 5.46 s | 34.52 s |
| 32 GiB | save | **1.42 s** | 24.52 s | — |
| 32 GiB | restore | **0.94 s** | 10.97 s | — |
| ResNet-50 (195 MiB) | save | **0.12 s** | 0.55 s | 0.67 s |

CPU during a 32 GiB save: **0.3 cores** (RDMA) vs 5.8 (HTTP). Every restore was
verified tensor-by-tensor against the original.

Those are *per-process* figures. In the configuration that favours HTTP most — 8 DDP
ranks each writing its own 4 GiB shard, 32 GiB total — RDMA is still **3.4× faster
with 4.3× less CPU**:

| | Per-rank save | Aggregate | Whole-host CPU |
| --- | --- | --- | --- |
| **RDMA** | **1.28 s** | **28.34 GB/s** | **4.8 cores** |
| Tuned HTTP | 4.15 s | 8.23 GB/s | 20.6 cores |

At LLM scale — a 112 GiB checkpoint (8B model + Adam), 8 DDP ranks × 14 GiB each:

| | Per-rank save | Aggregate | Whole-host CPU |
| --- | --- | --- | --- |
| **RDMA** | **4.33 s** | **28.03 GB/s** | **3.7 cores** |
| Tuned HTTP | 14.42 s | 8.43 GB/s | 30.4 cores |

Since ranks write concurrently, per-rank time *is* the training stall. MoE is the
sharpest case: checkpoint size scales with *total* params while speed scales with
*active* params, so a DeepSeek-V3-class model (671B/37B) writes **18× more
checkpoint bytes per training FLOP** than a dense model of equal throughput.

→ raw runs: `results/b4b5-checkpoint-*.json`

### Inference cold start — model weights into VRAM

Real Hugging Face checkpoints, unmodified layout, tensors verified element-wise
against `safetensors` itself:

| Model | **RDMA** | HTTP | local NVMe (warm cache) | download then load |
| --- | --- | --- | --- | --- |
| Llama-3.1-8B-Instruct (16.06 GB) | **0.77 s** | 2.90 s | 2.29 s | 30.96 s |
| Qwen3-32B (65.52 GB) | **3.10 s** | 13.77 s | 8.37 s | 125.06 s |

~21 GB/s sustained on **0.12–0.15 CPU cores**, holding across a 4× change in model
size. Two ways to read it: **40× faster than a cold node's real path**, and — more
interestingly — **3× faster than reading the same weights off the machine's own
NVMe**. We used the warm-page-cache local number as the baseline because it flatters
the alternative; against a genuinely cold local disk (6.28 s) it is 8.2×.

This is also the use case that **requires
[PR #258](https://github.com/minio/minio-cpp/pull/258)**: HF's default
`max_shard_size="5GB"` is decimal, so standard shards (5.00 GB here) sit *above*
cuObject's 4 GiB registration limit. A whole-object RDMA GET on one silently falls
back to HTTP; only ranged GETs can carry it, and then the server accounts 100% of
the payload as RDMA.

→ raw runs: `results/b8-weights-*.json`

### The LLM *data* path is not a storage problem

Worth knowing before pitching it. Tokenized int32 is GPU-native so RDMA applies —
it is just irrelevant. One loader process, measured:

| | tokens/s | Storage | CPU | H200s of Llama-3 8B it could feed |
| --- | --- | --- | --- | --- |
| **RDMA** | 6.75 B | 27.0 GB/s | 0.51 cores | 727,856 |
| HTTP | 785 M | 3.14 GB/s | 2.69 cores | **84,739** |

RDMA is 8.6× faster, on a quantity with four orders of magnitude of headroom: an
8B job needs ~0.3 MB/s per 8 GPUs, and a *single HTTP process* already over-serves
~85,000 GPUs. Tokens are read in long contiguous runs, so low byte volume comes
with low IOPS too (~7 MB/s and a few hundred IOPS for a 1024-GPU MoE job).

→ raw runs: `results/b7-tokens-*.json`

---

## What's in here

```
├── docs/                     how to run it on your own hardware
│   ├── [01-setup-aistor-rdma.md](docs/01-setup-aistor-rdma.md)      cluster + client bring-up
│   ├── [02-prepare-dataset.md](docs/02-prepare-dataset.md)        building shards
│   ├── [03-training-benchmark.md](docs/03-training-benchmark.md)
│   ├── [04-checkpoint-benchmark.md](docs/04-checkpoint-benchmark.md)
│   ├── [05-interpreting-results.md](docs/05-interpreting-results.md)   incl. when RDMA will NOT help you
│   └── [troubleshooting.md](docs/troubleshooting.md)           every silent-fallback trap we hit
├── plans/
│   ├── s3-over-rdma-training-benchmark-plan.md
│   ├── demo-manual.md        runbook for demonstrating this live
│   ├── s3-over-rdma-deck.html  presentation slides (MinIO design system)
│   └── reports/              the findings, with raw numbers
├── src/s3rdma_train/
│   ├── rdma_client.py        ctypes binding to libminiocpp's C ABI
│   ├── s3.py                 ObjectStore: http | rdma | local, one interface
│   ├── shards.py             JPEG tar shard format + byte index
│   ├── dataset.py            GPU-native and JPEG loaders
│   ├── checkpoint.py         flat contiguous checkpoints, chunked + parallel
│   ├── weights.py            safetensors -> VRAM as zero-copy views
│   ├── train_resnet50.py     B3: step-time breakdown by backend
│   ├── bench_fetch.py        B1: raw throughput
│   ├── bench_ckpt.py         B4/B5: checkpoint save/restore
│   ├── bench_tokens.py       B7: tokenized int32 loader throughput
│   ├── bench_weights.py      B8: model weights into VRAM (cold start)
│   └── metrics.py            RDMAWitness + CPU/GPU/NIC/fabric counters
├── scripts/
│   ├── setup/                cluster provisioning, client env, verification
│   ├── ingest/               dataset builders (jpeg / raw uint8 / int32 tokens)
│   └── run/                  benchmark runners
└── results/                  raw JSON for every run (this is the evidence)
```

---

## Quickstart

### Prerequisites

**Storage nodes** — Linux, NVMe drives, RoCE v2 or InfiniBand NIC with an
`ibv_devinfo` port in `PORT_ACTIVE`, and an AIStor licence.

**Client** — NVIDIA GPU, driver with `nvidia_peermem`, RDMA NIC on the same fabric,
PCIe ACS Redirect cleared on the GPU/NIC bridges, and `uv`.

You will also need a `libminiocpp.so` built from minio-cpp **with RDMA** that sends
the *bare 81-character* cuObject descriptor. Older builds append `:addr:size` and
will either silently fall back to HTTP or segfault —
[`docs/troubleshooting.md`](docs/troubleshooting.md) explains how to tell.

### 1. Bring up the cluster

```bash
# on each storage node
scripts/setup/01-stop-memkv.sh          # stop whatever is using the drives
scripts/setup/02-format-drives.sh       # dry run: lists what it would wipe
scripts/setup/02-format-drives.sh --apply
AISTOR1_RAIL0=<ip> AISTOR2_RAIL0=<ip> \
    scripts/setup/03-configure-hosts.sh # also run on the client
scripts/setup/04-roce-lossless.sh --install-service   # also on the client
scripts/setup/05-install-aistor.sh
MINIO_LICENSE="$(cat minio.license)" scripts/setup/06-start-aistor.sh
```

`02-format-drives.sh` **destroys data** on every matching drive. It refuses to
touch the root device and identifies data drives by model string; run it without
`--apply` first and read the list.

### 2. Set up the client and *prove* RDMA works

```bash
scripts/setup/10-client-env.sh --libs-from /path/to/rdma-libs
scripts/setup/12-cufile-config.sh          # auto-detects your RoCE NIC addresses
source env.sh
python scripts/setup/11-verify-rdma.py
```

The last step is not optional. It PUTs and GETs from VRAM and then checks the
**server's** `minio_api_rdma_*` counters. Anything less can't distinguish RDMA from
a silent TCP fallback:

```
[4] PUT 64 MiB straight from VRAM
    server RDMA write delta: 67,108,864 B in 1 ops
    PROOF: server accounted these bytes as RDMA
RESULT: PASS - GPU-Direct RDMA is live
```

### 3. Run the benchmarks

```bash
# raw storage throughput, both transports
scripts/run/run-b1-fetch.sh

# checkpointing: a real ResNet-50 checkpoint plus a size sweep
python -m s3rdma_train.bench_ckpt --backends rdma,http --resnet50 \
    --sizes-gib 1,4,16,32 --chunk-mib 512 --concurrency 32

# build the GPU-native dataset, then train
python scripts/ingest/build_raw_shards.py --src /path/to/imagenet/train \
    --bucket imagenet-raw --prefix train --size 256 --workers 8
python -m s3rdma_train.train_resnet50 --backend rdma --layout raw --steps 200
```

---

## Things that will bite you

Each of these produced a *working* transfer with no error while quietly using TCP,
or crashed the process. All are documented in
[`docs/troubleshooting.md`](docs/troubleshooting.md).

| Symptom | Cause |
| --- | --- |
| Transfers work, RDMA counters stay 0 | `/etc/cufile.json` ships `rdma_dev_addr_list: []`; without your NIC addresses, buffer registration fails and the SDK falls back |
| `libcuobjclient.so.1: undefined symbol: _Z20cuFileRDMADescStrGet...` | PyTorch bundles cuFile 1.15; cuObject needs ≥ 1.18, and same SONAME means first-loaded wins. `LD_PRELOAD` the newer one |
| Server logs `unexpected suffix on client RDMA token: got length 115, expected exactly 81` | client library predates the 2026-08-07 AIStor release; it appends `:addr:size`, which current servers reject |
| Segfault in `miniocpp_put_object` at ~4 GiB | cuObject's registration limit. **Fixed upstream in minio-cpp #241**; older builds crash |
| Segfault as soon as you use a thread pool | cuFile needs a bound CUDA context in *each* thread; a fresh Python thread has none |
| Segfault under concurrency with shared buffers | cuObject registers by address; two concurrent transfers into one buffer crash — and PyTorch's allocator recycles addresses, so a fresh buffer per transfer hits this too |
| Segfault with 8 DDP ranks × >1 RDMA fetch thread | **Fixed upstream**; but throughput still anti-scales across processes ([minio-cpp #259](https://github.com/minio/minio-cpp/issues/259)) — 8 processes move less than 1 |

---

## Upstream contributions from this work

- **[minio-cpp #258](https://github.com/minio/minio-cpp/pull/258)** — expose ranged
  GET through the C ABI. `GetObjectArgs::offset` already drove a ranged RDMA GET;
  the C ABI had no way to set it, so bindings could only fetch whole objects.
- **[minio-cpp #259](https://github.com/minio/minio-cpp/issues/259)** — GPU-Direct
  throughput anti-scales across processes (27.84 GB/s at 1 process → 1.91 GB/s at
  8), with a silent HTTP fallback at large buffer sizes. Filed rather than patched:
  it does not appear to live in minio-cpp.

Two blockers reported in earlier revisions of this repo (the 4 GiB segfault and the
multi-process crash) turned out to be **already fixed upstream**; the library we
first benchmarked predated the fixes.

## Honest limits of these numbers

- **Client-NIC-bound.** Both transports hit ~40 GB/s against a 41.5 GB/s
  single-rail ceiling, so this does not measure either path's true maximum. A
  dual-rail client run is not yet done.
- **Two storage nodes, 48 drives.** Nothing here says how it scales out.
- **HTTP figures from Python are per-process-limited** (~2 GB/s read, ~1.4 GB/s
  write). That is a real constraint on real PyTorch pipelines, and it is reported
  separately from the native-client ceiling (`warp`: 39.4 GB/s) so the two are not
  conflated.
- **The fabric's switch-side PFC was not configured by us.** RDMA hardware error
  counters were zero throughout, which is good evidence the traffic was lossless,
  but it is not the same as a verified lossless switch config.
- **ResNet-50 is compute-bound at 8 GPUs** — storage wait is 0.00 s on both arms, so
  none of the training result is a storage-throughput claim. A workload with a higher
  bytes-per-FLOP ratio is where raw throughput would show up, and that is untested.
- **RDMA anti-scales past ~4 fetch streams per rank** at 8 ranks
  ([#259](https://github.com/minio/minio-cpp/issues/259)). This workload does not
  need the headroom; a storage-bound one would.

## Licence

Apache-2.0 — see [LICENSE](LICENSE).

No third-party code or binaries are redistributed here, and the components this
project depends on keep their own terms: `libminiocpp`/minio-cpp is Apache-2.0 and
you build it yourself, `libcuobjclient` and `libcufile` are NVIDIA components under
the NVIDIA Software License Agreement, and MinIO AIStor requires a commercial
licence. ImageNet is subject to its own terms of access and is not included.
Details in [NOTICE](NOTICE).
