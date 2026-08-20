# Troubleshooting GPU-Direct S3-over-RDMA

Every problem in this document produced either a **working transfer that silently
used TCP** or a **segfault**. None of them produced a useful error message. They
are in the order you are likely to hit them.

The single most important habit: **never trust a successful transfer as evidence
that RDMA was used.** `libminiocpp`'s buffer path is documented to "attempt RDMA
with HTTP fallback", so a correct byte count and correct data prove nothing about
the transport. Always confirm from the server side:

```bash
curl -s http://aistor1:9000/minio/metrics/v3/api/rdma | grep -E '^minio'
```

```
minio_api_rdma_read_bytes_total{server="aistor1:9000"} 0     # <- nothing used RDMA
minio_api_rdma_write_bytes_total{server="aistor1:9000"} 0
```

`scripts/setup/11-verify-rdma.py` does this for you and fails loudly. Run it first,
and run it again after any change to the client stack.

---

## 1. Everything works, RDMA counters stay at zero

**Cause:** cuFile has no RDMA devices configured. minio-cpp decides per request:

```cpp
bool use_rdma = (rdma_client.cuMemObjGetDescriptor(args.buf, size) == 0);
```

`cuMemObjGetDescriptor` pins the buffer via `cuFileBufRegister`. The stock
`/etc/cufile.json` ships `"rdma_dev_addr_list": []`, so registration fails,
`use_rdma` stays false, and the transfer goes over ordinary HTTP with no error.

**Diagnosis** — look in `cufile.log` (written to the process's working directory):

```
ERROR cufio_rdma:1430 failed to register with RDMA, no devices found in configuration
ERROR cufio_core:1781 cuFileBufRegister error registering buffer object for RDMA
ERROR cufio_core:1839 cuFileBufRegister error cufile buf registration error
```

Also check:

```bash
/usr/local/cuda/gds/tools/gdscheck -p | grep -A2 'Userspace RDMA'
#   --rdma devices : Not configured        <- the problem
```

**Fix:**

```bash
scripts/setup/12-cufile-config.sh          # auto-detects PORT_ACTIVE RoCE NIC IPs
source env.sh                              # exports CUFILE_ENV_PATH_JSON
```

Confirm registration now succeeds (set `CUFILE_LOG_LEVEL=TRACE` and re-run):

```
DEBUG cufio_rdma:1482 nvidia_peermem is enabled
DEBUG cufio_rdma:1562 register with RDMA success mr_size: 16777216
DEBUG cufio-multipath:288 Registered path on device mlx5_0 healthy= 1
```

The settings that matter, beyond the address list, are `rdma_peer_type: "dmabuf"`
and `use_pci_p2pdma: true` — without them the NIC cannot map GPU memory for
peer-to-peer DMA.

---

## 2. `undefined symbol: _Z20cuFileRDMADescStrGetPKvmm12CUfileOpcodePPc`

**Symptom:** `libminiocpp.so` loads fine on its own, but fails after `import torch`.

**Cause:** a SONAME collision. `libcuobjclient` needs `cuFileRDMADescStrGet`, which
exists only in **cuFile ≥ 1.18**. PyTorch's CUDA wheels bundle **cuFile 1.15.x** at
`site-packages/nvidia/*/lib/libcufile.so.0`. Both files carry SONAME
`libcufile.so.0`, so whichever is `dlopen`'d first claims the name — and `dlopen`
by absolute path will **not** override an already-loaded SONAME. Import torch
first and torch's older copy wins.

**Diagnosis:**

```bash
# which libcufile is actually mapped into the process
grep libcufile /proc/<pid>/maps

# do they have the symbol?
nm -D <path>/libcufile.so.1.18.0 | grep -c cuFileRDMADescStrGet   # 1
nm -D .venv/lib/python3.12/site-packages/nvidia/cu13/lib/libcufile.so.0 \
  | grep -c cuFileRDMADescStrGet                                   # 0
```

**Fix** — preload the newer one so it claims the SONAME first. `env.sh` does this:

```bash
export LD_PRELOAD="$REPO/vendor/rdma-libs/libcufile.so.1.18.0${LD_PRELOAD:+:$LD_PRELOAD}"
```

PyTorch works fine against 1.18. `rdma_client.py` recognises this failure and
prints the fix instead of the mangled symbol name.

---

## 3. Server returns 500: `unexpected suffix on client RDMA token`

**Symptom:** the client sends `x-amz-rdma-token`, the server replies
**500 InternalError**, and the server log says:

```
Error: unexpected suffix on client RDMA token: got length 115, expected exactly 81
```

**Cause:** a client/server contract change. The cuObject RDMA descriptor is 81
characters:

```
<addr>:<size>:<rkey>:<lid>:<qp>:<gid_present>:<gid>
 16     8      8      4     6    1             32      + 6 separators = 81
```

Older minio-cpp appended `:<addr>:<size>` (34 more chars → 115) because an older
AIStor required it. AIStor releases from **2026-08-07** onward changed the
contract: the descriptor already carries address and size in its first two fields,
so the server derives them from there and **a client must send exactly the
descriptor**. An appended suffix is rejected.

**Fix:** use a `libminiocpp` that sends the bare 81-char descriptor.

```bash
# how to tell, without reading the source: watch the server log during a transfer
sudo journalctl -u minio -f | grep -i 'rdma token'
```

Beware: the prebuilt `libminiocpp.so.0.4.0` bundled inside the AIStor source tree
still appends the suffix. It is harmless to AIStor (which links those symbols but
never calls them) and is the obvious thing for a client integrator to pick up. Against a current server it either falls back to
HTTP or segfaults.

---

## 4. Segfault in `miniocpp_put_object` at exactly 4 GiB

**Symptom:** 3.5 GiB transfers fine; 4 GiB kills the process.

**Cause:** cuObject can pin at most 4 GiB in one registration
(`kCuObjMaxMemoryRegSize`). minio-cpp's source says RDMA is simply not attempted
above the limit and the transfer falls back to a single HTTP PUT. Measured
behaviour is worse: it segfaults.

**Fix:** chunk below the limit. `checkpoint.py` defaults to 1 GiB chunks, and
`rdma_client.py` now raises rather than letting the process die:

```
RDMAError: put: 4,294,967,296 bytes is at or above the cuObject registration limit
(4,294,967,296); libminiocpp segfaults rather than falling back. Split the transfer
into chunks of <= 2,147,483,648 bytes.
```

Chunking is worth doing anyway: a single stream reaches ~3 GB/s, while 32–64
concurrent streams reach 42 GB/s.

---

## 5. Segfault the moment you use a thread pool

**Symptom:** single-threaded RDMA works. Any `ThreadPoolExecutor` crashes — even
with one client per thread, even with every call serialised under a lock.

**Cause:** cuFile/cuObject reach for the **current CUDA context** via the driver
API. CUDA's primary context is bound *per thread*, and a freshly spawned Python
thread has none until it touches CUDA. Registering or transferring a device pointer
with no current context does not return an error; it segfaults inside
`libcuobjclient`.

**Fix:** bind the context in every thread before its first RDMA call.

```python
torch.cuda.set_device(device_index)
torch.cuda.current_stream()          # force torch's lazy per-thread init
```

`RdmaStore.prepare_thread()` does this, and should be used as the
`ThreadPoolExecutor(initializer=...)`.

---

## 6. Segfault under concurrency with a shared destination buffer

**Symptom:** crashes only when two or more streams run, and only sometimes.

**Cause:** cuObject registers buffers **by address**. Two concurrent transfers
targeting the same buffer mean concurrent register/deregister of one address, which
crashes. (Over HTTP the same bug is silent data corruption instead.)

**Fix:** one destination buffer per concurrent stream, and never submit more
in-flight tasks than you have buffers. In the harness this meant submitting exactly
one task per stream, each looping over its own buffer, rather than
`streams × iterations` tasks into a `streams`-sized pool.

---

## 7. `multiprocessing.Pool` hangs forever

**Symptom:** a multi-process run prints its header and then hangs; GPUs show
allocated memory at 0% utilisation.

**Cause:** a worker died (typically a segfault from one of the causes above) and
`Pool.map` blocks indefinitely rather than reporting it.

**Fix:** use `concurrent.futures.ProcessPoolExecutor`, which raises
`BrokenProcessPool`. Diagnose a live hang with:

```bash
sudo env "PATH=$PATH" py-spy dump --pid <pid>
```

---

## 7b. Segfault at process exit, *after* the results printed

**Symptom:** the benchmark prints its table and `saved: results/….json`, then torchrun
reports `SIGSEGV` on every rank and exits non-zero. Intermittent — roughly half of
8-GPU RDMA runs.

**Your measurement is fine.** Check the ordering in the output: if `saved:` appears
before the crash, the JSON on disk is complete and valid.

**Cause:** the loader's fetch threads are daemon threads holding per-thread RDMA
clients. If a client is freed while a thread is still inside `miniocpp_get_object` —
or if it is left to garbage collection and freed during interpreter shutdown, after
the CUDA context has gone — the free faults.

**Fix (already in this repo):** three parts, and the third is the one that mattered.

1. `RdmaStore.close()` frees the clients it handed out, instead of leaving them to GC.
2. `RDMAClient.__del__` returns early when `sys.is_finalizing()`, leaking the handle
   rather than crashing a process that is exiting anyway.
3. **The prefetch threads are joined** (`_Prefetcher.join()`) before anything frees a
   client. Drain the ready queue first, or a worker blocked on a full queue never
   observes the stop flag.

If you see this on your own code built against these modules, check that you join
your fetch threads before closing the store.

## 8. `OSError: [Errno 24] Too many open files`

High stream counts multiplied by per-thread connection pools exhaust the default
1024 descriptors.

```bash
ulimit -n 65536
```

---

## 9. Throughput far below line rate over HTTP

If a single Python process caps around 2 GB/s, that is expected and is **not** a
configuration problem. We measured the same ~1.5–1.9 GB/s for minio-py, bare
`urllib3`, and a hand-rolled HTTP request over a raw socket with `recv_into`, on a
fabric that does 41.5 GB/s under `iperf3` and 39.4 GB/s under `warp`. The limit is
the CPython process.

Use worker **processes** (8 processes ≈ 13 GB/s). RDMA does not share this limit —
it reaches 42 GB/s from one process on ~1 core, because the transfer happens in the
NIC and the GIL is never in the data path.

---

## 10. RDMA works but throughput collapses under load

Check the fabric. Rising drop/retry counters mean it is not actually lossless:

```bash
grep -r . /sys/class/infiniband/mlx5_0/ports/1/hw_counters/ \
  | grep -iE 'out_of_buffer|packet_seq_err|local_ack_timeout_err'

ethtool -S <netdev> | grep -iE 'prio3_pause|discard|drop|out_of_buffer'
```

On a healthy fabric these stay flat and prio-3 pause counters rise under incast.
Host-side config (`scripts/setup/04-roce-lossless.sh`) is necessary but **not
sufficient** — PFC must be configured identically on every switch in the path, or
the switch will drop regardless of what the hosts agreed.

---

## 11. GPUDirect fails with `IBV_WC_REM_OP_ERR` (status 11)

PCIe ACS Redirect is forcing GPU↔NIC peer-to-peer traffic upstream to the root
complex. Counterintuitively this hits the GPU *closest* to the NIC (same PCIe
switch).

```bash
nvidia-smi topo -m                                    # PIX pairs are the risk
sudo lspci -vvv -s <bridge-bdf> | grep ACSCtl          # want ReqRedir- CmpltRedir-
```

Clear ACS Redirect on the bridges in the data path; AIStor's
`docs/distributed/RDMA.md` ships a systemd one-shot that walks every GPU/NIC path.
Verify it actually ran (`journalctl -u disable-pcie-acs -b`) — a unit that both
declares `Before=basic.target` and is `WantedBy=multi-user.target` forms an
ordering cycle, and systemd silently drops the job while still reporting `enabled`.

---

## Quick reference: is my RDMA path healthy?

```bash
ibv_devinfo | grep -E 'hca_id|state'                      # PORT_ACTIVE
ulimit -l                                                  # unlimited
/usr/local/cuda/gds/tools/gdscheck -p | grep -i peerdirect # Enabled
python -c "from s3rdma_train import rdma_client; print(rdma_client.is_available())"
python scripts/setup/11-verify-rdma.py                     # the only real test
```
