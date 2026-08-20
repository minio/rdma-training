# 2. Preparing the dataset

Three layouts, because they answer different questions. Which one you need depends
entirely on whether RDMA can touch your data path at all:

- **GPU-Direct RDMA engages only for CUDA device memory.** Pinned host memory
  silently falls back to TCP.
- **GPU JPEG decoders need the encoded bitstream in host memory.**

Those two facts are incompatible, so **a JPEG pipeline cannot use RDMA on its data
path.** Decoding once at ingest removes the conflict.

| Layout | Bucket | Object | RDMA usable | What it is for |
| --- | --- | --- | --- | --- |
| **raw** | `imagenet-raw` | `train-000000.raw` + `.json` | ✅ yes | the storage comparison |
| **jpeg** | `imagenet-shards` | `train-000000.tar` + `.index.json` | ❌ no | the conventional baseline |
| **tokens** | `llm-tokens` | `train-000000.bin` + `.json` | ✅ yes | documenting that the LLM data path is *not* storage-bound |

The two image layouts are built from the same file list with the same shuffle seed,
so they contain the same samples in the same shard order and are directly
comparable. The token layout is independent and synthetic.

---

## Why shards at all

ImageNet averages ~110 KB per JPEG. At that size a GET is dominated by
per-operation cost, not data movement: our sweep measured **0.6 GB/s at 1 MiB
objects versus 42.4 GB/s at 256 MiB** — a 70× spread on identical hardware. Small
objects also get *worse* with concurrency (1 MiB: 0.58 → 0.38 GB/s going from 8 to
64 streams) while CPU climbs.

No transport fixes that; the data layout does. This is why real large-scale training
uses sharded formats, and why `bench_fetch --sizes 1` is still published — for
anyone whose data really is millions of small objects, the honest answer is that
the fix is the layout, not the network.

---

## Layout A — GPU-native raw shards (use this for RDMA)

Images are decoded once at ingest and stored as fixed-size `256×256×3` uint8. A
shard is then a flat array the GPU consumes directly.

```bash
python scripts/ingest/build_raw_shards.py \
    --src /path/to/imagenet/ILSVRC/Data/CLS-LOC/train \
    --bucket imagenet-raw --prefix train \
    --size 256 --samples-per-shard 4096 --workers 8
```

- `--size 256` with a 224 training crop: **random crop and horizontal flip still
  happen per epoch, on the GPU**, so augmentation is not sacrificed. Only the JPEG
  decode moves off the hot path. This is the trade FFCV and DALI's raw format make.
- `--samples-per-shard 4096` → 768 MiB per shard, comfortably inside the 256 MiB+
  range where throughput is good and well under cuObject's 4 GiB registration
  limit.
- `--workers 8` runs one process per GPU, each batch-decoding with nvJPEG.

Objects written:

```
train/train-000000.raw     4096 × 256 × 256 × 3 uint8, contiguous, no header
train/train-000000.json    {"samples", "height", "width", "channels",
                            "sample_bytes", "labels": [...], "keys": [...]}
```

Fixed-size samples mean no offset table: sample *i* begins at `i * 256*256*3`.

**Capacity cost:** 197 KB per image versus ~110 KB as JPEG, so ImageNet train grows
from 140 GB to ~252 GB. That is the price of making the data GPU-native, and on a
251 TiB cluster it is noise.

**A training step then is:** RDMA the shard into VRAM → index a batch out of it →
crop/flip/normalise on the GPU. No host memory, no decode, no CPU in the data path.

---

## Layout B — JPEG tar shards (the conventional baseline)

WebDataset-compatible tars, plus a sidecar byte index.

```bash
python scripts/ingest/build_shards.py \
    --src /path/to/imagenet/ILSVRC/Data/CLS-LOC/train \
    --bucket imagenet-shards --prefix train \
    --samples-per-shard 8192 --workers 24
```

```
train/train-000000.tar         <key>.jpg and <key>.cls per sample
train/train-000000.index.json  {"shard", "size",
                                "samples":[{"key","offset","length","label"}]}
```

The tar is a normal tar — `tar tf` works and any WebDataset reader can consume it.
The index records each JPEG's byte range so a loader can slice batches without
parsing 512-byte headers on the hot path.

Offsets are exact, not approximate: the writer uses `USTAR` format (not the default
PAX, which would insert extra header blocks), asserts member names fit the ustar
name field, checks the payload landed where predicted, and spot-checks that indexed
ranges start with a JPEG `SOI` marker. The unit check in the repo compares every
index offset against what `tarfile` itself reports.

> Layout B is deliberately **HTTP-only**. `train_resnet50.py --layout jpeg
> --backend rdma` refuses to run and explains why, rather than quietly measuring
> an HTTP fallback and labelling it RDMA.

---

## Layout C — tokenized int32 shards (the LLM data path)

Built to *document a negative result*, not to accelerate anything. Tokenized text
is GPU-native (int32 ids) so RDMA genuinely applies — it is simply irrelevant,
because the volume is four orders of magnitude below what an image pipeline needs.

```bash
python scripts/ingest/build_token_shards.py --bucket llm-tokens --shards 16
python -m s3rdma_train.bench_tokens --backend rdma --bucket llm-tokens
python -m s3rdma_train.bench_tokens --backend http --bucket llm-tokens
```

```
train/train-000000.bin     sequences x seq_len int32 token ids, contiguous
train/train-000000.json    {"sequences","seq_len","dtype","vocab_size",...}
```

Fixed-length sequences mean no index: sequence *i* starts at `i * seq_len * 4`.
Token values are synthetic — byte volume and access pattern are what is being
measured, and neither depends on what the tokens mean. `int32` (not `uint16`) is
the pessimistic choice for a bandwidth argument and is what a >65535 vocab forces.

`bench_tokens` reports tokens/s and converts it into *how many H200s of a given
model that loader could feed*, which is the number that settles the question:

| | tokens/s | Storage | CPU | H200s of Llama-3 8B feedable |
| --- | --- | --- | --- | --- |
| RDMA | 6.75 B | 27.0 GB/s | 0.51 cores | 727,856 |
| HTTP | 785 M | 3.14 GB/s | 2.69 cores | 84,739 |

A single HTTP process over-serves an 8B pretraining job by ~85,000 GPUs
(`results/b7-tokens-*.json`).

## Verifying what you built

```bash
mc ls --recursive myalias/imagenet-raw/train/ | head
mc du myalias/imagenet-raw

python - <<'PY'
import json, sys
sys.path.insert(0, "src")
from s3rdma_train.s3 import StoreConfig, make_store
st = make_store("http", StoreConfig(bucket="imagenet-raw"))
keys = sorted(k for k, _ in st.list("train") if k.endswith(".raw"))
print(f"{len(keys)} shards")
m = json.loads(st.get_bytes(keys[0].replace(".raw", ".json")).decode())
print({k: v for k, v in m.items() if k not in ("labels", "keys")})
print("size on server:", st.stat(keys[0]),
      "expected:", m["samples"] * m["sample_bytes"])
print("labels:", len(m["labels"]), "distinct:", len(set(m["labels"])))
PY
```

All three builders skip shards that already exist, so an interrupted run resumes by
re-running the same command (`--no-skip-existing` forces a rebuild).

---

## Layout D -- model weights (for the cold-start benchmark)

Benchmark B8 loads a real model's weights straight into VRAM, so the
`model-weights` bucket has to hold **standard, unmodified** Hugging Face
safetensors shards -- same shard names, same `model.safetensors.index.json`. That
fidelity is the whole point: it is what an inference server would actually read,
including the fact that HF's default `max_shard_size="5GB"` emits shards *above*
cuObject's 4 GiB registration limit.

From a model you already have in the HF cache:

```bash
python scripts/ingest/upload_model_weights.py \
    --snapshot ~/.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/<rev> \
    --prefix llama-3.1-8b-instruct
```

Or let it fetch the repo first (needs `huggingface_hub`, and a `HF_TOKEN` for
gated models):

```bash
python scripts/ingest/upload_model_weights.py \
    --repo-id meta-llama/Llama-3.1-8B-Instruct \
    --prefix llama-3.1-8b-instruct
```

Cache snapshots store files as symlinks into `blobs/`; those are dereferenced on
upload, so the bucket gets real objects. Re-runs skip shards already present
unless you pass `--no-skip-existing`.

Any safetensors model works -- the benchmark reads the index and shards generically.
We used an 8B and a 32B to show how the gap scales with model size.

---

## Rough cost

On our rig (8 × H200, 256 cores, 140 GB of source JPEGs on local XFS):

| | |
| --- | --- |
| `build_shards.py` (tar, no decode) | I/O bound on reading 1.28M small files |
| `build_raw_shards.py` (decode + resize) | GPU-decode bound; ~252 GB written |
| `build_token_shards.py` (synthetic tokens) | RNG + upload bound; seconds per GiB |

Ingest rate is printed continuously (`GB/s` and `img/s`) so you can tell which end
is limiting. Ingest is setup, not a benchmark — it uses plain multipart HTTP so the
tools work on machines with no functioning RDMA client.

Next: [3. The training benchmark](03-training-benchmark.md).
