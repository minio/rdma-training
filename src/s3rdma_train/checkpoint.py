# Copyright 2026 MinIO, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Checkpointing that can write straight out of GPU memory.

The conventional path is ``torch.save(state_dict, buf); put_object(buf)``. That
pickles tensor-by-tensor, so every tensor is copied GPU -> host, then again into
the pickle buffer, then again by the HTTP client — three passes over host memory
and a training loop stalled for all of them. For a ResNet-50 that is ~300 MB and
nobody notices. For an LLM it is hundreds of gigabytes and it is the reason
checkpoint intervals get stretched until a crash costs hours.

This module instead lays the whole checkpoint out as **one contiguous buffer**
plus a small JSON manifest describing where each tensor lives inside it:

    buffer:    [ tensor A ][pad][ tensor B ][pad][ tensor C ] ...
    manifest:  {"skeleton": {...}, "tensors": [{"path", "dtype", "shape",
                                                "offset", "nbytes"}, ...]}

With the buffer allocated in VRAM, saving is a single RDMA PUT of a device
pointer: no host memory touched at all. Restoring is a single RDMA GET into VRAM,
after which every tensor is a *view* into that buffer — no per-tensor copy, and
the model can be loaded with ``assign=True``.

The manifest also carries a JSON "skeleton" of the original object with tensors
replaced by references, so nested structures (optimizer ``state``/``param_groups``,
step counters, scalars) round-trip rather than only flat model state dicts.

Format is versioned and deliberately boring: it is a benchmark artefact, not a
proposal to replace ``torch.distributed.checkpoint``.
"""

from __future__ import annotations

import io
import json
from dataclasses import dataclass
from typing import Any, Optional

import torch

FORMAT_VERSION = 1

# Every tensor starts on a 512-byte boundary. Two reasons: reinterpreting a byte
# slice as a typed tensor requires the storage offset to divide the element size,
# and DMA engines prefer aligned, page-friendly boundaries. The wasted bytes are
# under 0.01% for realistic checkpoints.
ALIGNMENT = 512

_TENSOR_REF = "__s3rdma_tensor__"


def _align(n: int) -> int:
    return (n + ALIGNMENT - 1) // ALIGNMENT * ALIGNMENT


@dataclass
class TensorEntry:
    path: int  # index into the tensors list, referenced from the skeleton
    dtype: str
    shape: list[int]
    offset: int
    nbytes: int


def _dtype_name(dt: torch.dtype) -> str:
    return str(dt).removeprefix("torch.")


def _dtype_from_name(name: str) -> torch.dtype:
    dt = getattr(torch, name, None)
    if not isinstance(dt, torch.dtype):
        raise ValueError(f"unknown torch dtype {name!r} in checkpoint manifest")
    return dt


def _walk(obj: Any, tensors: list[torch.Tensor]) -> Any:
    """Replace tensors with references, returning a JSON-able skeleton."""
    if isinstance(obj, torch.Tensor):
        tensors.append(obj)
        return {_TENSOR_REF: len(tensors) - 1}
    if isinstance(obj, dict):
        # Keys are stringified; a checkpoint with non-string keys (optimizer state
        # is keyed by int) round-trips through "__keys__" below.
        out: dict[str, Any] = {}
        keytypes: dict[str, str] = {}
        for k, v in obj.items():
            ks = str(k)
            if not isinstance(k, str):
                keytypes[ks] = type(k).__name__
            out[ks] = _walk(v, tensors)
        if keytypes:
            return {"__dict__": out, "__keytypes__": keytypes}
        return out
    if isinstance(obj, (list, tuple)):
        return {
            "__seq__": [_walk(v, tensors) for v in obj],
            "__tuple__": isinstance(obj, tuple),
        }
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    raise TypeError(
        f"cannot serialise {type(obj).__name__} in a checkpoint; "
        "supported: Tensor, dict, list, tuple, str, int, float, bool, None"
    )


def _unwalk(node: Any, views: list[torch.Tensor]) -> Any:
    if isinstance(node, dict):
        if _TENSOR_REF in node:
            return views[node[_TENSOR_REF]]
        if "__seq__" in node:
            seq = [_unwalk(v, views) for v in node["__seq__"]]
            return tuple(seq) if node.get("__tuple__") else seq
        if "__dict__" in node:
            kt = node.get("__keytypes__", {})
            out = {}
            for k, v in node["__dict__"].items():
                key: Any = k
                t = kt.get(k)
                if t == "int":
                    key = int(k)
                elif t == "float":
                    key = float(k)
                elif t == "bool":
                    key = k == "True"
                out[key] = _unwalk(v, views)
            return out
        return {k: _unwalk(v, views) for k, v in node.items()}
    return node


@dataclass
class FlatCheckpoint:
    buffer: torch.Tensor  # 1-D uint8, contiguous; CUDA or CPU (optionally pinned)
    manifest: dict

    @property
    def nbytes(self) -> int:
        return self.buffer.numel()

    def manifest_bytes(self) -> bytes:
        return json.dumps(self.manifest, separators=(",", ":")).encode()


def flatten(state: Any, device: Optional[torch.device] = None, pin: bool = False) -> FlatCheckpoint:
    """Pack ``state`` into one contiguous buffer plus a manifest.

    ``device`` defaults to the device of the first tensor found, which for a
    training checkpoint means the buffer is built in VRAM and can be uploaded by
    RDMA with no host involvement.
    """
    tensors: list[torch.Tensor] = []
    skeleton = _walk(state, tensors)

    entries: list[TensorEntry] = []
    offset = 0
    for i, t in enumerate(tensors):
        nb = t.numel() * t.element_size()
        entries.append(
            TensorEntry(
                path=i, dtype=_dtype_name(t.dtype), shape=list(t.shape), offset=offset, nbytes=nb
            )
        )
        offset = _align(offset + nb)
    total = offset

    if device is None:
        device = tensors[0].device if tensors else torch.device("cpu")
    if pin and device.type == "cpu":
        buffer = torch.empty(total, dtype=torch.uint8, pin_memory=True)
    else:
        buffer = torch.empty(total, dtype=torch.uint8, device=device)

    for t, e in zip(tensors, entries):
        src = t.detach()
        if not src.is_contiguous():
            src = src.contiguous()
        flat = src.reshape(-1).view(torch.uint8)
        buffer[e.offset : e.offset + e.nbytes].copy_(flat)

    if buffer.is_cuda:
        torch.cuda.current_stream().synchronize()

    manifest = {
        "format_version": FORMAT_VERSION,
        "total_bytes": total,
        "alignment": ALIGNMENT,
        "skeleton": skeleton,
        "tensors": [e.__dict__ for e in entries],
    }
    return FlatCheckpoint(buffer=buffer, manifest=manifest)


def unflatten(buffer: torch.Tensor, manifest: dict) -> Any:
    """Rebuild the original structure as **views** into ``buffer``.

    No per-tensor copy: each returned tensor aliases the buffer. Keep ``buffer``
    alive for as long as the returned structure is used, and prefer
    ``load_state_dict(..., assign=True)`` so the module adopts these views instead
    of copying into its existing parameters.
    """
    if manifest.get("format_version") != FORMAT_VERSION:
        raise ValueError(
            f"unsupported checkpoint format_version {manifest.get('format_version')!r}"
        )
    views: list[torch.Tensor] = []
    for e in manifest["tensors"]:
        off, nb = int(e["offset"]), int(e["nbytes"])
        dt = _dtype_from_name(e["dtype"])
        chunk = buffer[off : off + nb]
        views.append(chunk.view(dt).reshape(tuple(e["shape"])))
    return _unwalk(manifest["skeleton"], views)


# --------------------------------------------------------------- store paths ---

MANIFEST_SUFFIX = ".manifest.json"

# Chunk size for the payload. Two independent reasons, both binding:
#
#  1. Correctness. A single GPU-Direct RDMA transfer cannot exceed cuObject's 4 GiB
#     registration limit, and at exactly 4 GiB libminiocpp segfaults rather than
#     falling back (measured: 3.5 GiB fine, 4 GiB crash). Any checkpoint bigger than
#     that MUST be split.
#  2. Throughput. A single stream reaches only ~3 GB/s; the same path reaches
#     42 GB/s across 32-64 concurrent streams (report 01). Chunking is what lets a
#     checkpoint be written in parallel at all.
DEFAULT_CHUNK_BYTES = 1024 * 1024 * 1024
DEFAULT_CONCURRENCY = 16


def _chunk_key(key: str, i: int) -> str:
    return f"{key}.part{i:05d}"


def _chunk_ranges(total: int, chunk: int) -> list[tuple[int, int]]:
    return [(off, min(chunk, total - off)) for off in range(0, total, chunk)]


def save_flat(store, key: str, state: Any, *, device: Optional[torch.device] = None,
              pin: bool = False, chunk_bytes: int = DEFAULT_CHUNK_BYTES,
              concurrency: int = DEFAULT_CONCURRENCY) -> dict:
    """Flatten and upload as parallel chunks plus a manifest.

    Chunks are views into the one flat buffer, so no extra copy is made: each
    upload reads a distinct byte range of the same allocation. For RDMA that also
    means distinct addresses, which matters because cuObject registers buffers by
    address and concurrent registration of the *same* address crashes.
    """
    import time
    from concurrent.futures import ThreadPoolExecutor

    t0 = time.perf_counter()
    flat = flatten(state, device=device, pin=pin)
    t_flatten = time.perf_counter() - t0

    ranges = _chunk_ranges(flat.nbytes, chunk_bytes)
    nthreads = max(1, min(concurrency, len(ranges)))

    def upload(i: int) -> int:
        off, ln = ranges[i]
        return store.put_from(_chunk_key(key, i), flat.buffer[off : off + ln], ln)

    t1 = time.perf_counter()
    if nthreads == 1:
        store.prepare_thread()
        written = [upload(i) for i in range(len(ranges))]
    else:
        with ThreadPoolExecutor(max_workers=nthreads,
                                initializer=store.prepare_thread) as pool:
            written = list(pool.map(upload, range(len(ranges))))
    t_upload = time.perf_counter() - t1

    manifest = dict(flat.manifest)
    manifest["chunk_bytes"] = chunk_bytes
    manifest["num_chunks"] = len(ranges)
    raw = json.dumps(manifest, separators=(",", ":")).encode()
    # Metadata goes over HTTP explicitly (see ObjectStore.put_bytes): it is tiny,
    # and routing it through the payload path would let a host-buffer fallback
    # muddy the RDMA byte counters we use as proof.
    store.put_bytes(key + MANIFEST_SUFFIX, raw)

    return {
        "bytes": sum(written),
        "flatten_s": t_flatten,
        "upload_s": t_upload,
        "total_s": t_flatten + t_upload,
        "manifest_bytes": len(raw),
        "buffer_device": str(flat.buffer.device),
        "num_chunks": len(ranges),
        "chunk_bytes": chunk_bytes,
        "upload_threads": nthreads,
    }


def load_flat(store, key: str, device: str = "cuda",
              concurrency: int = DEFAULT_CONCURRENCY) -> tuple[Any, torch.Tensor, dict]:
    """Download chunks in parallel into one buffer and rebuild views.

    Returns (state, buffer, timing). The buffer is returned because the state
    aliases it; dropping it would free the memory the tensors point at.
    """
    import time
    from concurrent.futures import ThreadPoolExecutor

    manifest = json.loads(store.get_bytes(key + MANIFEST_SUFFIX).decode())
    total = int(manifest["total_bytes"])
    chunk = int(manifest.get("chunk_bytes", total)) or total
    ranges = _chunk_ranges(total, chunk)
    nthreads = max(1, min(concurrency, len(ranges)))

    t0 = time.perf_counter()
    buffer = torch.empty(total, dtype=torch.uint8, device=device)

    def download(i: int) -> int:
        off, ln = ranges[i]
        return store.get_into(_chunk_key(key, i), buffer[off : off + ln], ln)

    if nthreads == 1:
        store.prepare_thread()
        got = [download(i) for i in range(len(ranges))]
    else:
        with ThreadPoolExecutor(max_workers=nthreads,
                                initializer=store.prepare_thread) as pool:
            got = list(pool.map(download, range(len(ranges))))
    t_download = time.perf_counter() - t0

    t1 = time.perf_counter()
    state = unflatten(buffer, manifest)
    t_rebuild = time.perf_counter() - t1

    return state, buffer, {
        "bytes": sum(got),
        "download_s": t_download,
        "rebuild_s": t_rebuild,
        "total_s": t_download + t_rebuild,
        "num_chunks": len(ranges),
        "download_threads": nthreads,
    }


def save_torch_save(store, key: str, state: Any) -> dict:
    """The baseline everyone actually writes: ``torch.save`` into a buffer, then PUT.

    Included so the comparison is against real practice rather than only against a
    hand-optimised path. It is genuinely slower for a reason worth naming: pickling
    copies every tensor to host individually, so the cost is several passes over
    host memory before a single byte reaches the network.
    """
    import time

    t0 = time.perf_counter()
    bio = io.BytesIO()
    torch.save(state, bio)
    payload = bio.getbuffer()
    t_serialize = time.perf_counter() - t0

    t1 = time.perf_counter()
    host = torch.frombuffer(payload, dtype=torch.uint8)
    n = store.put_from(key, host, len(payload))
    t_upload = time.perf_counter() - t1

    return {
        "bytes": n,
        "serialize_s": t_serialize,
        "upload_s": t_upload,
        "total_s": t_serialize + t_upload,
    }
