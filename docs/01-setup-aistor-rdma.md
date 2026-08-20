# 1. Bringing up AIStor with GPU-Direct RDMA, and a client that can use it

Two halves, and the second is the one people underestimate. The server side is
ordinary AIStor deployment plus one extra package. The client side has four
independent ways to silently end up on TCP, so it ends with a verification step
that is not optional.

---

## Server side

### Prerequisites

| | |
| --- | --- |
| OS | 64-bit Linux, kernel 5.4+ with the RDMA subsystem |
| NIC | RoCE v2 or InfiniBand, `ibv_devinfo` shows a port in `PORT_ACTIVE` |
| Packages | `libibverbs1`, `librdmacm1`, `libnuma1` |
| Limits | `ulimit -l` = `unlimited` (RDMA pins buffers) |
| Licence | an AIStor licence file |

```bash
ibv_devinfo | grep -E 'hca_id|state:|link_layer'
#   hca_id: mlx5_0
#           state:      PORT_ACTIVE (4)
#           link_layer: Ethernet        <- RoCE
ulimit -l   # unlimited
```

### 1. Prepare drives

`scripts/setup/02-format-drives.sh` puts XFS on every data NVMe and mounts it at
`/mnt/data<N>`, persisting by label in `/etc/fstab`.

```bash
scripts/setup/02-format-drives.sh            # dry run — prints what it WOULD wipe
scripts/setup/02-format-drives.sh --apply    # destroys data on those drives
```

It identifies data drives by **model string** (`DRIVE_MODEL`, default matches the
NVMes in our rig) and refuses to touch whatever backs `/`. Read the dry-run list
before applying; if your drives differ, set `DRIVE_MODEL` rather than editing the
loop. AIStor is pointed at `/mnt/data<N>/minio`, a subdirectory, so an unmounted
drive can never be silently filled up on the boot disk.

### 2. Make one erasure pool span your nodes

MinIO expands **one** server-pool argument with an ellipsis. Two separate arguments
create two *pools*, and a single-node pool cannot stripe its erasure sets across
hosts — you lose roughly half the aggregate throughput.

Real host names are rarely contiguous. If yours are `node02` and `node04`, then
`node{02...04}` expands to include a `node03` you may not even own. So
`scripts/setup/03-configure-hosts.sh` maps your two nodes onto contiguous aliases:

```bash
AISTOR1_RAIL0=10.0.0.11 AISTOR2_RAIL0=10.0.0.12 \
AISTOR1_RAIL1=10.0.1.11 AISTOR2_RAIL1=10.0.1.12 \
    scripts/setup/03-configure-hosts.sh
```

The `RAIL1` pair is optional -- set it only if your NICs are dual-railed. That
writes into `/etc/hosts`:

```
10.0.0.11   aistor1 aistor1.rail0
10.0.0.12   aistor2 aistor2.rail0
10.0.1.11   aistor1.rail1
10.0.1.12   aistor2.rail1
```

giving a single argument covering all 48 drives:

```
http://aistor{1...2}:9000/mnt/data{1...24}/minio
```

Run it on the storage nodes **and the client** — the client resolves the same names.

> **Shell trap.** Never inline that pattern as a `${VAR:-default}` default.
> Parameter expansion ends at the first unescaped `}`, so
> `${MINIO_VOLUMES:-http://aistor{1...2}:9000/...}` silently becomes
> `http://aistor{1...2:9000/...}` and MinIO rejects it with a misleading
> "allowed minimum range of 4" (there is no such minimum; a 2-host ellipsis is
> fine). Assign the default on its own single-quoted line.

### 3. Lossless RoCE (host side)

```bash
scripts/setup/04-roce-lossless.sh --install-service   # on every node AND the client
```

PFC on priority 3, DSCP 26 → priority 3, ECN on priority 3 — the NVIDIA convention.
The `--install-service` flag persists it across reboots, since `mlnx_qos` settings
are not sticky.

**This is necessary but not sufficient.** Losslessness must be configured
identically on every switch in the path; host-only settings cannot stop a switch
from dropping. Verify empirically under load instead of assuming:

```bash
ethtool -S <netdev> | grep -iE 'prio3_pause'     # should rise under incast
grep -r . /sys/class/infiniband/mlx5_0/ports/1/hw_counters/ \
  | grep -iE 'out_of_buffer|packet_seq_err|local_ack_timeout_err'   # should stay flat
```

### 4. Install the RDMA build

The plain `minio` binary **cannot** do GPU-Direct RDMA — the feature needs a CGO
build (tags `rdma,kqueue`) linked against libibverbs/librdmacm and NVIDIA's
cuObject libraries. That ships as a separate `minio.rdma` artifact.

```bash
scripts/setup/05-install-aistor.sh
minio.rdma --version | grep Features
#   Features: GPU-Direct RDMA        <- if this is missing, nothing else will work
```

The package installs the transport libraries to `/usr/lib/minio` and the binary as
`/usr/local/bin/minio.rdma`, deliberately distinct from a stock `minio`.

### 5. Start the cluster

```bash
MINIO_LICENSE="$(cat minio.license)" scripts/setup/06-start-aistor.sh
```

> **The licence must be passed as `--license <path>`.** That CLI flag has no
> EnvVar binding, unlike most MinIO flags, so `MINIO_LICENSE` /
> `MINIO_LICENSE_FILE` in the environment are silently ignored and the server comes
> up in offline mode: `All S3 operations are denied`.

The generated `/etc/minio/config.env` sets `MINIO_RDMA_INTERNODE=on` plus 400 GbE
tunables (`MINIO_RDMA_CHANNELS=256`, `MINIO_RDMA_NUM_DCIS=256`,
`MINIO_RDMA_CQ_DEPTH=8192`). Erasure is one pool, stripe 16 → EC:4 (12 data +
4 parity).

Confirm:

```bash
mc admin info myalias                 # all drives online, correct stripe/sets
sudo journalctl -u minio | grep -i 'RDMA Server listening'
sudo journalctl -u minio | grep -i 'internode initialized'
curl -s http://aistor1:9000/minio/metrics/v3/api/rdma | grep '^minio'
```

---

## Client side

### What the client needs

Three shared objects, which the server-side `minio.rdma` package does **not** ship
(the server links `libminiocpp` at build time but never calls it, so the release
omits it):

| Library | Purpose |
| --- | --- |
| `libminiocpp.so` | minio-cpp built `-DMINIO_CPP_ENABLE_RDMA=ON`; exposes the C ABI |
| `libcuobjclient.so.1` | NVIDIA cuObject client |
| `libcufile.so.0` (**≥ 1.18**) | NVIDIA GPUDirect Storage |

### Which minio-cpp version

**Build minio-cpp from tag [`v0.6.0`](https://github.com/minio/minio-cpp/releases/tag/v0.6.0)
or later.** That tag is the first release that carries every fix this project
depends on, and it is the whole client-side dependency -- **no AIStor server change
is needed**. Do *not* use the prebuilt copy bundled inside the AIStor source tree,
and do not track `main` unless you mean to: `main` has since swapped curlpp for
cpp-httplib (#257), which is unrelated churn for our purposes.

| Needed for | Fix | First in |
| --- | --- | --- |
| Any transfer at all | Send the **bare 81-char** cuObject descriptor as `x-amz-rdma-token`; older builds append `:<addr>:<size>` (115 chars) and the server rejects it | `v0.5.0` |
| Transfers ≥ 4 GiB | Guard cuObject's 4 GiB single-registration limit instead of faulting; fixes >2 GiB upload truncation | `v0.6.0` |
| HTTP fallback correctness | Stage device memory when the RDMA path is unavailable | `v0.6.0` |
| **Model weights only** (the cold-start benchmark) | `miniocpp_get_object_range()` in the C ABI | [PR #258](https://github.com/minio/minio-cpp/pull/258) -- **open**, rebases cleanly onto `v0.6.0` |

Training and checkpointing need only `v0.6.0`; both chunk below 4 GiB on their own
(raw shards are 768 MiB, checkpoint chunks 1 GiB). PR #258 matters only if you read
**standard Hugging Face safetensors shards**, whose default `max_shard_size="5GB"`
is decimal -- ~4.66 GiB, just above the registration limit -- so a whole-object GET
of one silently falls back to HTTP. `scripts/run/show-4gib-fallback.py` demonstrates
both halves of that.

See [troubleshooting §3](troubleshooting.md#3-server-returns-500-unexpected-suffix-on-client-rdma-token)
for what the token-format failure looks like in the wild.

Also required on the client:

- `nvidia_peermem` loaded (`gdscheck -p` shows `Mellanox PeerDirect : Enabled`)
- PCIe ACS Redirect cleared on the bridges above the GPUs and NIC
  (`lspci -vvv -s <bridge> | grep ACSCtl` → `ReqRedir- CmpltRedir-`)
- `ulimit -l unlimited`

### 1. Environment

```bash
scripts/setup/10-client-env.sh --libs-from /path/to/rdma-libs
```

Creates `.venv` with PyTorch + torchvision, stages the RDMA libraries into
`vendor/rdma-libs/`, asserts the C ABI is present, and writes `env.sh`.

`env.sh` sets `LD_PRELOAD` to the ≥1.18 `libcufile`, and that is **load-bearing**:
PyTorch bundles cuFile 1.15, both carry SONAME `libcufile.so.0`, and first-loaded
wins. Without the preload, `import torch` before the RDMA client yields
`undefined symbol: _Z20cuFileRDMADescStrGet...`.

### 2. cuFile configuration

```bash
scripts/setup/12-cufile-config.sh     # auto-detects PORT_ACTIVE RoCE NIC addresses
```

**Do not skip this.** The stock `/etc/cufile.json` has an empty
`rdma_dev_addr_list`, buffer registration fails, and every transfer silently
degrades to HTTP with no error anywhere except `cufile.log`.

### 3. Verify — the step that actually matters

```bash
source env.sh
python scripts/setup/11-verify-rdma.py
```

This PUTs 64 MiB from VRAM, GETs it back into VRAM, compares byte-for-byte, and
reads the **server's** RDMA counters around each operation:

```
[4] PUT 64 MiB straight from VRAM
    server RDMA write delta: 67,108,864 B in 1 ops
    PROOF: server accounted these bytes as RDMA
[5] GET 64 MiB straight into VRAM
    server RDMA read delta:  67,108,864 B in 1 ops
    PROOF: server accounted these bytes as RDMA
[6] integrity
    OK: 67,108,864 bytes identical (compared on device)
RESULT: PASS - GPU-Direct RDMA is live
```

A green round-trip without the counter check means nothing: the client falls back
to HTTP silently by design.

---

## Baseline the fabric before benchmarking anything

Know your ceilings, or you will attribute a client bottleneck to a transport.

```bash
# TCP ceiling, one rail
iperf3 -s -p 5301 --one-off          # on a storage node
iperf3 -c <node> -p 5301 -P 16 -t 8  # on the client

# native S3-over-HTTP ceiling, and what it costs in CPU
warp get --host=aistor1:9000,aistor2:9000 --access-key=... --secret-key=... \
  --bucket=warp-ceiling --obj.size=256MiB --objects=64 --concurrent=64 --duration=25s
```

On our rig: TCP 41.5 GB/s, `warp` 39.4 GB/s at 68 CPU cores, RDMA 42.4 GB/s at
~1 core. Both transports were NIC-limited, which is exactly the kind of thing you
want to know *before* reading a benchmark table.

Next: [2. Preparing the dataset](02-prepare-dataset.md).
