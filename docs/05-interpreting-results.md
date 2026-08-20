# 5. Interpreting results — including when RDMA will not help you

The point of this page is to stop you drawing a conclusion the measurements do not
support. Our own first numbers were wrong twice, in both directions.

---

## First: is the result even real?

**Every RDMA number needs server-side proof.** `libminiocpp`'s buffer path is
documented to "attempt RDMA with HTTP fallback", so success and correct data prove
nothing about the transport. Every benchmark here brackets its work in
`metrics.RDMAWitness`, which reads `minio_api_rdma_{read,write}_bytes_total` from
every node and fails the run if the counters did not move.

During bring-up, RDMA was silently disabled **three separate ways** while every
transfer succeeded (empty `rdma_dev_addr_list`, a `libcufile` SONAME collision with
PyTorch, and a stale client token format). Any of them, undetected, would have
produced a benchmark comparing HTTP to HTTP.

In B1's output the witness ratio was exactly `1.00` across all 30 cells. That is
what a trustworthy RDMA result looks like:

```
   size  conc    GB/s  cores       expect_B    rdma_read_B  ratio   ops fabric_errs
   256M    64   42.39   0.95  103,079,215,104 103,079,215,104   1.00   384 0
```

**Also check `fabric_errors` is 0.** Non-zero `out_of_buffer` / `packet_seq_err` /
`local_ack_timeout_err` means the fabric is dropping, and the RDMA number is
fabric-limited rather than transport-limited.

---

## The three questions, and which benchmark answers each

| Question | Benchmark | Do not use |
| --- | --- | --- |
| What can the storage path do at all? | **B1** (`bench_fetch`) | B2 — it measures the loader, and once storage is fast enough it measures augmentation |
| How many GPUs could one loader feed? | **B2** (`--loader-only`) | B1 — no augmentation, no batching |
| Does storage limit *my* training? | **B3** (`train_resnet50`) | B1/B2 — they have no model |

Confusing these is the most common way to get an answer that is true but irrelevant.

---

## Reading B3: the line that matters is `time blocked on storage`

```
  seconds for 200 batches         : 16.07 s
  time blocked on storage         : 0.00 s (0.0% of wall)
  host CPU                        : 11.3 cores
  rank0 breakdown : fetch_wait 0.00s | decode 0.00s | augment 0.30s | compute 15.84s
```

- **Near zero** → compute-bound. Both transports will report nearly the same step
  time, and that is the *correct answer*, not a failed measurement. A faster
  transport cannot speed up a job that is not waiting on storage. What RDMA buys you
  here is the `host CPU` line.
- **High (>20%)** → storage-bound, and the transport difference shows up directly.

Our measured case: ResNet-50 on 8 × H200 needs **4.8 GB/s** against a path that
delivers **40+ GB/s**, so it is compute-bound and step time barely moves. We report
that rather than engineering around it.

---

## When S3-over-RDMA will *not* help you

Stated plainly, because knowing this saves more time than any tuning:

**1. Your data needs CPU-side parsing.**
JPEG, PNG, WebP — GPU decoders take the encoded bitstream from *host* memory, while
RDMA only engages for *device* memory. The data path cannot use RDMA. Convert to a
GPU-native layout first; we measured that conversion at **1.76× on ResNet-50**,
larger than anything the transport did.

**2. Your objects are small.**
Below ~4 MiB, per-operation cost dominates. RDMA measured **0.6 GB/s at 1 MiB**
versus **42.4 GB/s at 256 MiB** — a 70× spread on identical hardware. Small objects
also get *worse* with concurrency. No transport fixes this; the data layout does.

**3. Your job is compute-bound.**
See above. Check `time blocked on storage` before optimising storage.

**4. You are already at your NIC's line rate.**
Both transports reached ~40 GB/s against a 41.5 GB/s single-rail `iperf3` ceiling.
If you are NIC-limited, RDMA will not give you more bandwidth — it will give you the
same bandwidth for ~1 CPU core instead of ~68.

**5. You are fanning out across many client processes.**
RDMA throughput anti-scales with process count in the client library: 27.84 GB/s
from one process against 1.91 GB/s across eight. Four fetch threads inside one
process per rank is the knee. If your loader needs many independent processes each
pulling hard, measure before assuming RDMA wins.

---

## When it will help, in order of confidence

1. **Checkpointing.** Unconditional, large, and repeatable: 16 GiB saved in 0.65 s
   versus 12.16 s (tuned HTTP) or 62.55 s (`torch.save`), at 0.3 CPU cores versus
   5.9. A checkpoint is tensors, so this applies no matter what your input pipeline
   looks like.
2. **CPU-starved data pipelines.** Getting bytes in for ~1 core instead of ~68 frees
   the cores that decide how fast the model can actually be fed.
3. **Reading GPU-native data at line rate from a single Python process.** HTTP cannot
   do this at all: one CPython process caps near 2 GB/s no matter how many threads,
   measured identically for minio-py, bare `urllib3`, and a raw socket.
4. **Genuinely storage-bound training** — high resolution, cheap model, or many GPUs
   per loader. Plausible from B1 but **not measured here**, so treat it as a
   hypothesis.

---

## Comparing fairly: the traps we fell into

Each of these initially produced a wrong answer. Two flattered RDMA and two
flattered HTTP.

**Unequal concurrency.** HTTP splits each object into `--http-parts` ranged GETs; the
RDMA C ABI has no offset parameter, so one fetch is one stream. Measured naively,
HTTP looked **3.6× faster** than RDMA — the reverse of B1. Always report the
concurrency of both arms.

**A hidden copy in the baseline.** Our HTTP upload wrapped the payload in
`io.BytesIO(memoryview)`, which copies — an extra 16 GiB pass through host memory
charged to HTTP for no reason. Fixing it halved the reported checkpoint gap.

**A shared buffer.** One pinned staging buffer per size class meant concurrent HTTP
streams read different objects into the same memory: garbage data (the harness
checked only length) and a hard ~3 GB/s cap from cache-line ping-pong.

**Per-process limits mistaken for transport limits.** Python's ~2 GB/s per-process
HTTP ceiling is real and matters for PyTorch, but it is *not* HTTP's ceiling. Native
`warp` reached 39.4 GB/s. Report both, and never quote the single-process HTTP number
as "what HTTP can do".

**Measurement windows that exclude background work.** Prefetch threads fetch shards
before the measured window opens; a 768 MiB shard feeds many steps. Snapshot loader
counters at the warmup boundary or the proof check will mis-fire.

---

## A checklist before you believe a number

- [ ] RDMA runs show a server-side byte-counter delta matching expected bytes
- [ ] `fabric_errors` is 0
- [ ] Both arms had comparable fetch concurrency, and it is stated
- [ ] Warmup is non-zero (RDMA registers buffers on first touch)
- [ ] ≥3 repeats, and the ranges are reported — not just a best run
- [ ] The HTTP arm has no gratuitous copies and uses multiple processes if that is
      what a real pipeline would do
- [ ] `time blocked on storage` is reported alongside step time
- [ ] CPU cores are reported; for RDMA this is often the actual result
- [ ] Object size and layout are stated (a 1 MiB result and a 256 MiB result differ
      by 70×)
- [ ] You have said what the measurement does **not** establish
