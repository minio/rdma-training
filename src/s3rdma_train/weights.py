# Copyright 2026 MinIO, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Load safetensors model weights from object storage straight into VRAM.

Safetensors is already laid out the way GPU-Direct RDMA wants: an 8-byte
little-endian header length, a JSON header mapping every tensor to
``{dtype, shape, data_offsets}``, and then one contiguous blob of tensor bytes.
That is the same shape as this project's checkpoint format, arrived at
independently -- one buffer plus a manifest.

So a model's weights can go from object storage to GPU memory with **no host
memory and no deserialisation**: read the blob into a device buffer, then build
each tensor as a *view* into it at the offset the header gives.

What this replaces
------------------
The usual inference cold start is: download shards to local disk, then
``safetensors.torch.load_file`` (which mmaps) and copy each tensor to the GPU.
That is two passes through host memory and a full copy of the model onto local
disk, and it is why cold start is slow enough to shape autoscaling policy.

The 4 GiB problem
-----------------
Hugging Face's default sharding is ``max_shard_size="5GB"`` -- decimal, so ~4.66
GiB, **above cuObject's 4 GiB registration limit**. Llama-3.1-8B-Instruct's shards
are 4.98/5.00/4.92/1.17 GB. A whole-object RDMA GET therefore cannot be used on a
standard HF shard at all; it silently falls back to HTTP.

Ranged GET is what fixes this: read the shard in sub-4-GiB windows into one device
buffer. That needs ``miniocpp_get_object_range``, added in minio-cpp PR #258. On a
library without it, ``load_shard`` says so rather than quietly degrading.
"""

from __future__ import annotations

import json
import struct
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import torch

# Stay under cuObject's 4 GiB registration limit with margin; also gives the
# transfer some concurrency to work with.
DEFAULT_WINDOW_BYTES = 1024 * 1024 * 1024

_SAFETENSORS_DTYPES = {
    "F64": torch.float64, "F32": torch.float32, "F16": torch.float16,
    "BF16": torch.bfloat16, "I64": torch.int64, "I32": torch.int32,
    "I16": torch.int16, "I8": torch.int8, "U8": torch.uint8, "BOOL": torch.bool,
    "F8_E4M3": getattr(torch, "float8_e4m3fn", None),
    "F8_E5M2": getattr(torch, "float8_e5m2", None),
}


@dataclass
class SafetensorsHeader:
    """Parsed safetensors header: where each tensor lives in the blob."""

    header_bytes: int  # 8 + len(json)
    total_bytes: int  # whole file
    tensors: dict[str, dict]  # name -> {dtype, shape, data_offsets}
    metadata: dict = field(default_factory=dict)

    @property
    def data_bytes(self) -> int:
        return self.total_bytes - self.header_bytes

    @property
    def num_tensors(self) -> int:
        return len(self.tensors)


def parse_header(raw: bytes, total_bytes: int) -> SafetensorsHeader:
    """Parse a safetensors header from the first bytes of the file."""
    if len(raw) < 8:
        raise ValueError("safetensors file too short to contain a header length")
    (n,) = struct.unpack("<Q", raw[:8])
    if len(raw) < 8 + n:
        raise ValueError(
            f"need {8 + n} header bytes but only {len(raw)} were read; "
            "re-read with a larger prefix"
        )
    doc = json.loads(raw[8 : 8 + n].decode("utf-8"))
    meta = doc.pop("__metadata__", {}) or {}
    return SafetensorsHeader(
        header_bytes=8 + n, total_bytes=total_bytes, tensors=doc, metadata=meta
    )


def read_header(store, key: str, probe_bytes: int = 1 << 20) -> SafetensorsHeader:
    """Fetch just enough of ``key`` to parse its header.

    Deliberately over HTTP (``get_bytes``): the header is kilobytes to a few MB,
    and routing it through the RDMA payload path would put non-payload bytes into
    the RDMA counters this project uses as proof.
    """
    total = store.stat(key)
    prefix = store.get_bytes_range(key, 0, min(probe_bytes, total))
    try:
        return parse_header(prefix, total)
    except ValueError:
        # Header longer than the probe: read exactly what it declares. Still a
        # range read -- never pull the whole ~5 GB shard just for its header.
        (n,) = struct.unpack("<Q", prefix[:8])
        return parse_header(store.get_bytes_range(key, 0, 8 + n), total)


def build_views(buf: torch.Tensor, header: SafetensorsHeader) -> dict[str, torch.Tensor]:
    """Build tensors as views into ``buf``, which holds the file's data region.

    ``buf`` must contain the bytes *after* the header, i.e. offsets in the header
    are relative to buf[0]. No data is copied; keep ``buf`` alive for as long as
    the tensors are used.
    """
    out: dict[str, torch.Tensor] = {}
    for name, spec in header.tensors.items():
        dt = _SAFETENSORS_DTYPES.get(spec["dtype"])
        if dt is None:
            raise ValueError(f"{name}: unsupported safetensors dtype {spec['dtype']!r}")
        start, end = spec["data_offsets"]
        nbytes = end - start
        chunk = buf[start:end]
        if chunk.numel() != nbytes:
            raise ValueError(f"{name}: buffer holds {chunk.numel()} of {nbytes} bytes")
        shape = tuple(spec["shape"])
        # A 0-element tensor has no bytes to view; make it directly.
        if nbytes == 0:
            out[name] = torch.empty(shape, dtype=dt, device=buf.device)
            continue
        out[name] = chunk.view(dt).reshape(shape)
    return out


def load_shard(store, key: str, device: str = "cuda",
               window_bytes: int = DEFAULT_WINDOW_BYTES,
               concurrency: int = 8) -> tuple[dict[str, torch.Tensor], torch.Tensor, dict]:
    """Load one safetensors shard into VRAM. Returns (tensors, buffer, timing).

    The data region is transferred in ``window_bytes`` windows, in parallel. Two
    reasons, both measured elsewhere in this repo: a single RDMA stream reaches a
    fraction of the throughput of 8-32 streams, and a standard HF shard is above
    the 4 GiB limit a single registration can cover.
    """
    from concurrent.futures import ThreadPoolExecutor

    t0 = time.perf_counter()
    header = read_header(store, key)
    t_header = time.perf_counter() - t0

    if header.data_bytes > window_bytes and not store.supports_range:
        raise RuntimeError(
            f"{key}: data region is {header.data_bytes:,} bytes, which needs ranged "
            f"reads, but the {store.name} backend cannot do them. For RDMA this "
            "means a libminiocpp without miniocpp_get_object_range (minio-cpp #258)."
        )

    buf = torch.empty(header.data_bytes, dtype=torch.uint8, device=device)
    windows = [
        (off, min(window_bytes, header.data_bytes - off))
        for off in range(0, header.data_bytes, window_bytes)
    ]

    def fetch(i: int) -> int:
        off, ln = windows[i]
        # Offsets in the object include the header; offsets in buf do not.
        return store.get_range_into(key, buf[off : off + ln], ln,
                                    header.header_bytes + off)

    t1 = time.perf_counter()
    if len(windows) == 1 and not store.supports_range:
        got = [store.get_into(key, buf, header.data_bytes)]
    else:
        nthreads = max(1, min(concurrency, len(windows)))
        if nthreads == 1:
            store.prepare_thread()
            got = [fetch(i) for i in range(len(windows))]
        else:
            with ThreadPoolExecutor(max_workers=nthreads,
                                    initializer=store.prepare_thread) as pool:
                got = list(pool.map(fetch, range(len(windows))))
    t_transfer = time.perf_counter() - t1

    if sum(got) != header.data_bytes:
        raise RuntimeError(
            f"{key}: transferred {sum(got):,} of {header.data_bytes:,} bytes"
        )

    t2 = time.perf_counter()
    tensors = build_views(buf, header)
    t_views = time.perf_counter() - t2

    return tensors, buf, {
        "key": key,
        "total_bytes": header.total_bytes,
        "data_bytes": header.data_bytes,
        "num_tensors": header.num_tensors,
        "windows": len(windows),
        "header_s": t_header,
        "transfer_s": t_transfer,
        "views_s": t_views,
        "total_s": t_header + t_transfer + t_views,
        "transfer_gbps": header.data_bytes / t_transfer / 1e9 if t_transfer > 0 else None,
    }


def load_model(store, prefix: str, device: str = "cuda",
               window_bytes: int = DEFAULT_WINDOW_BYTES,
               concurrency: int = 8) -> tuple[dict[str, torch.Tensor], list, dict]:
    """Load every safetensors shard under ``prefix`` into VRAM.

    Returns (tensors, buffers, timing). ``buffers`` must be kept alive: the
    tensors are views into them.
    """
    keys = sorted(k for k, _ in store.list(prefix) if k.endswith(".safetensors"))
    if not keys:
        raise RuntimeError(f"no .safetensors objects under {prefix}")

    tensors: dict[str, torch.Tensor] = {}
    buffers: list[torch.Tensor] = []
    shards = []
    t0 = time.perf_counter()
    total = 0
    for k in keys:
        t, buf, info = load_shard(store, k, device=device, window_bytes=window_bytes,
                                  concurrency=concurrency)
        # Duplicate names across shards would mean a malformed index.
        dup = set(t) & set(tensors)
        if dup:
            raise RuntimeError(f"{k}: tensor names repeat across shards: {sorted(dup)[:3]}")
        tensors.update(t)
        buffers.append(buf)
        shards.append(info)
        total += info["data_bytes"]
    wall = time.perf_counter() - t0

    return tensors, buffers, {
        "shards": len(keys),
        "num_tensors": len(tensors),
        "bytes": total,
        "wall_s": wall,
        "gbps": total / wall / 1e9 if wall > 0 else None,
        "per_shard": shards,
    }
