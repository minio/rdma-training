# 3. The training benchmark (B3): ResNet-50 step time

Answers the question "how long do N batches take, per storage backend?" — and
answers it as a **breakdown**, not a single number, because a single number hides
the thing that decides whether storage matters at all.

```
step time = fetch_wait + decode/augment + forward/backward/optimizer
```

If `fetch_wait` is near zero the job is compute-bound, and no transport can make it
faster however much faster the transport is in isolation. Reporting only "seconds
for N batches" would let a compute-bound tie be read as "RDMA doesn't work". The
breakdown says *why*.

---

## Which combinations are valid

| `--layout` | `--backend rdma` | `--backend http` | `--backend local` |
| --- | --- | --- | --- |
| `raw` (GPU-native uint8) | ✅ | ✅ | ✅ |
| `jpeg` (tar shards) | ❌ refused | ✅ | ✅ |

`--layout jpeg --backend rdma` exits with an explanation rather than running.
RDMA needs a CUDA destination; nvJPEG needs a host source. Silently measuring an
HTTP fallback and labelling it RDMA would be worse than refusing.

Use **`--layout raw`** for the storage comparison. Use `--layout jpeg` to see where
a conventional pipeline's time actually goes.

---

## Running it

```bash
source env.sh

# single GPU
# scripts/run/run-b2b3-training.sh runs every valid (layout, backend) pair
# in one go; the invocations below are what it wraps.
python -m s3rdma_train.train_resnet50 --backend rdma --layout raw \
    --steps 200 --warmup 20 --batch-size 256

# 8 GPUs, DDP — this is what makes aggregate storage demand large enough to matter
torchrun --standalone --nproc-per-node=8 \
    -m s3rdma_train.train_resnet50 --backend rdma --layout raw \
    --steps 200 --warmup 20 --batch-size 256
```

Then the same command with `--backend http`, and optionally `--backend local
--local-root /path` as a "no network at all" control.

### Options worth knowing

| Flag | Default | Why |
| --- | --- | --- |
| `--warmup` | 20 | discarded; the first steps pay cudnn autotuning, RDMA buffer registration, connection setup |
| `--steps` | 200 | measured steps — the N in "seconds for N batches" |
| `--batch-size` | 256 | per GPU |
| `--amp` | `bf16` | what anyone would actually use on Hopper |
| `--prefetch` | 2 | shards fetched ahead on a background thread |
| `--epochs-over-shards` | 4 | repeats the shard list so long runs don't run out of data |
| `--max-shards` | 0 (all) | cap the working set |

TF32 and `channels_last` are on by default. Leaving them off would inflate compute
time and flatter the storage layer.

---

## Reading the output

```
==============================================================================
B3 ResNet-50 | backend=rdma layout=raw gpus=8 batch=256/gpu
==============================================================================
  seconds for 200 batches         : <wall>
  images/s                        : <throughput>
  median step time                : <ms>
  storage read                    : <GiB> @ <GB/s>
  time blocked on storage         : <s> (<pct>% of wall)
  host CPU                        : <cores>
  server RDMA bytes               : <n>
  fabric errors                   : 0
  rank0 breakdown                 : fetch_wait .. | decode .. | augment .. | compute ..
```

**`time blocked on storage`** is the line that matters.

- **High** (say >20% of wall) → the job is storage-bound, and the transport
  difference shows up directly in step time.
- **Near zero** → compute-bound. Both backends will report nearly the same step
  time, and that is the correct answer, not a failed measurement. What RDMA buys
  you here is the **host CPU** line: cores handed back to the rest of the pipeline.

**`server RDMA bytes`** is the proof line. If it is far below `storage read`, the
run silently used HTTP and prints a warning. Do not report such a run as RDMA.

---

## What to expect, honestly

**`--layout raw`** is the configuration where storage is the limit: no decode, no
per-sample host work, so bytes/second is what feeds the GPUs. This is where a
transport difference translates into step time.

**`--layout jpeg`** is decode-bound. ResNet-50 on 8 × H200 wants roughly 25–30k
images/s; at ~110 KB per JPEG that is only ~3 GB/s of storage — far below what
either transport delivers (report 01: 42.4 GB/s RDMA, 39.4 GB/s native HTTP). The
nvJPEG decode and per-image resize become the bottleneck long before the network
does. That is not a flaw in the benchmark, it is the finding: **for a JPEG pipeline
the storage transport is not your problem**, and converting the dataset to a
GPU-native layout is the larger speedup.

---

## Comparing fairly

Everything downstream of the fetch is byte-identical between backends — same model,
same augmentation code, same AMP settings, same seeds — so the delta is
attributable to the storage path.

Things that will make a comparison meaningless if you get them wrong:

- **Don't compare across layouts.** `raw` and `jpeg` do different amounts of work
  per sample.
- **Keep `--warmup` non-zero.** RDMA registers its destination buffers on first
  touch; charging that to step 1 penalises RDMA.
- **Give each rank disjoint shards** (the loader does this automatically) — two
  ranks reading one shard inflates cache hits and duplicates samples.
- **Check `fabric errors` is 0.** Non-zero means the fabric is dropping and the
  RDMA number is fabric-limited, not transport-limited.

---

## Isolating the storage layer instead

If you want the storage ceiling without a model in the way, that is B1:

```bash
scripts/run/run-b1-fetch.sh                 # both backends, size x concurrency sweep
```

B3 tells you whether storage limits *your* training. B1 tells you what the storage
path can do at all. They answer different questions and the reports keep them
separate.

Next: [4. The checkpoint benchmark](04-checkpoint-benchmark.md).
