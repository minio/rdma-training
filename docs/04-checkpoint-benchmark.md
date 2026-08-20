# 4. The checkpoint benchmark (B4/B5)

Checkpointing is where S3-over-RDMA applies without qualification. A checkpoint is
tensors, so it is GPU-native by construction — unlike an image data path, which
cannot use RDMA at all (see [interpreting results](05-interpreting-results.md)).
It is also the operation that **stalls training**, so seconds here are idle GPUs.

---

## Running it

```bash
source env.sh
ulimit -n 65536

# a real ResNet-50 checkpoint plus a size sweep, both transports
python -m s3rdma_train.bench_ckpt \
    --backends rdma,http --resnet50 \
    --sizes-gib 1,4,16,32 --repeats 3 \
    --chunk-mib 512 --concurrency 32
```

Three methods are measured, and the third exists so the comparison is not against a
strawman:

| Method | Backend | What it is |
| --- | --- | --- |
| `flat` | rdma | pack into one contiguous VRAM buffer, PUT chunks straight from the device pointer |
| `flat` | http | identical packing, one D2H copy into pinned host memory, then parallel multipart PUT |
| `torch-save` | http | `torch.save(state, BytesIO)` then `put_object` — what people actually write |

By default `--backends rdma,http` runs `flat` for RDMA and both `flat` and
`torch-save` for HTTP. Override with `--methods`.

Every restore is verified tensor-by-tensor (shape, dtype, full value equality)
against the original, and a cell that fails verification is reported as failed
rather than as a fast result.

---

## Chunk size is not optional

```bash
--chunk-mib 512      # keep this well under 4096
```

**A single GPU-Direct transfer cannot exceed cuObject's 4 GiB registration limit.**
minio-cpp's source says RDMA is simply not attempted above the limit and the
transfer falls back to a single HTTP PUT. Measured behaviour is worse: 3.5 GiB
transfers fine and **4 GiB segfaults**. `rdma_client.py` raises rather than letting
the process die, but you still have to chunk.

Chunking is also where the throughput comes from. A single stream reaches ~3 GB/s;
32–64 concurrent streams reach 42 GB/s. Measured effect of adding chunked
parallelism to the RDMA save path: **3.19 GB/s → 24.2 GB/s**. Chunks are views into
the one flat buffer, so parallelising costs no extra copy.

`--concurrency 32` with `--chunk-mib 512` means up to 32 in-flight transfers of
512 MiB each.

---

## Reading the output

```
            workload  backend      method     GiB   save s  save GB/s   load s  load GB/s  cores   ok
     synthetic-16gib     rdma        flat   16.00     0.65      26.50     0.50      34.32    0.3  yes
     synthetic-16gib     http        flat   16.00    12.16       1.41     5.46       3.15    5.9  yes
     synthetic-16gib     http  torch-save   16.00    62.55       0.27    34.52       0.50    1.4  yes
```

- `ok` = the restore verified against the original. Treat `NO` as a failed run.
- `cores` is whole-host CPU during the operation. For checkpointing this is nearly
  as interesting as the time: 0.3 cores versus 5.9 for the same work.
- RDMA cells that show no server-side RDMA byte movement are reported as
  `FAILED: RDMA not used` rather than as a result.

---

## Caveats you must carry with the numbers

**These are per-process figures, and the Python HTTP client is per-process limited**
to ~1.4 GB/s write and ~3 GB/s read: a raw socket with `recv_into` was no faster
than the SDK on a fabric doing 41.5 GB/s under `iperf3` (`results/b1-fetch-*.json`).

A real 8-GPU job does not checkpoint from one process: **each rank writes its own
shard**, which gives HTTP roughly 8× more headroom. Measure that configuration
rather than extrapolating:

```bash
RANKS=8 TOTAL_GIB=32 scripts/run/run-b4-ddp-sharded.sh
```

What sharding does *not* change is CPU. 5.8 cores per rank for HTTP versus 0.3 for
RDMA is ~46 cores versus ~2.4 across 8 ranks, and RDMA is already at the NIC ceiling
where HTTP is still climbing.

**If you quote the 17–19× ratio, quote it as per-process, and quote the CPU figure
next to it.**

---

## Synthetic sizes and VRAM

`--sizes-gib N` builds an *N* GiB state dict and then flattens it, which needs
**~2N GiB of VRAM**. On a 143 GB H200 that caps the sweep around 48 GiB; 64 GiB
fails to allocate. If you need to characterise larger checkpoints, build the flat
buffer in place instead of flattening from a separate state dict.

`--resnet50` uses a genuine ResNet-50 + SGD-momentum checkpoint (195 MiB flat,
481 tensors). It takes one optimizer step first so momentum buffers exist — an
empty optimizer state would understate a real checkpoint by about half.

---

## Using the checkpoint format in your own code

`src/s3rdma_train/checkpoint.py` is usable directly:

```python
from s3rdma_train.checkpoint import save_flat, load_flat
from s3rdma_train.s3 import StoreConfig, make_store

store = make_store("rdma", StoreConfig(bucket="checkpoints"))

state = {"model": model.state_dict(), "optimizer": opt.state_dict(), "step": step}
save_flat(store, f"run7/step{step}", state, chunk_bytes=1 << 30, concurrency=16)

state, buffer, timing = load_flat(store, "run7/step1000", device="cuda")
model.load_state_dict(state["model"], assign=True)   # views, no per-tensor copy
```

Two things to know:

- **Keep `buffer` alive.** Restored tensors are *views* into it; dropping it frees
  the memory they point at. `assign=True` is what makes the module adopt those views
  instead of copying into its existing parameters.
- Nested structures round-trip — optimizer `state`/`param_groups`, integer keys,
  scalars — because the manifest carries a JSON skeleton of the original object with
  tensors replaced by references.

This is a benchmark artefact, not a replacement for
`torch.distributed.checkpoint`. It exists to isolate the transport: one contiguous
buffer, one manifest, nothing clever.

Next: [5. Interpreting results](05-interpreting-results.md).
