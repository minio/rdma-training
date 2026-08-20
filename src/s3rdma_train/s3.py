# Copyright 2026 MinIO, Inc.
# SPDX-License-Identifier: Apache-2.0

"""One object-store interface, two transports, so benchmarks can swap them.

    store = make_store("rdma", cfg)   # or "http"
    store.get_into(key, dst_tensor, nbytes)

Both backends land bytes in a CUDA tensor. They differ only in how:

``rdma``
    ``libminiocpp`` hands the NIC the GPU pointer and the payload DMAs straight
    into VRAM. One call, no host memory involved.

``http``
    Ranged GETs over TCP into **pinned** host memory across a thread pool, then a
    single async H2D copy. This is what a well-tuned S3 loader does today, and it
    is deliberately tuned rather than naive: a single-threaded
    ``resp.read()``-into-``bytes`` baseline would lose for reasons that have
    nothing to do with the transport, and beating a strawman proves nothing.

The HTTP path's extra host round-trip is not an artefact of the comparison — it
is precisely the cost GPU-Direct RDMA removes, so it belongs in the measurement.
"""

from __future__ import annotations

import io
import os
import threading
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Iterator, Optional

import torch


@dataclass
class StoreConfig:
    endpoint: str = os.environ.get("S3_ENDPOINT", "aistor1:9000")
    access_key: str = os.environ.get("S3_ACCESS_KEY", "minioadmin")
    secret_key: str = os.environ.get("S3_SECRET_KEY", "minioadmin")
    bucket: str = "imagenet-shards"
    secure: bool = False
    region: str = ""
    # HTTP tuning. parts>1 splits an object into concurrent ranged GETs, which is
    # how you fill a 400 GbE NIC from one object with a Python client.
    http_parts: int = 8
    http_threads: int = 8
    http_chunk: int = 4 << 20
    # Additional endpoints for the HTTP path to spread connections over. A single
    # node's 400 GbE caps a client at ~50 GB/s, so aggregate reads should address
    # every node.
    endpoints: list[str] = field(default_factory=list)

    def all_endpoints(self) -> list[str]:
        return self.endpoints or [self.endpoint]


class _ZeroCopyReader(io.RawIOBase):
    """A read-only file object over a memoryview, without copying it.

    ``io.BytesIO(memoryview)`` copies the whole buffer, which for a 16 GiB
    checkpoint means an extra 16 GiB pass through host memory before the first byte
    reaches the network -- charged to the HTTP baseline for no reason other than
    convenience. minio-py only needs ``read``/``readinto``, so a thin wrapper keeps
    the comparison about the transport.
    """

    def __init__(self, view: memoryview) -> None:
        self._v = view.cast("B") if view.format != "B" else view
        self._pos = 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self._pos

    def seek(self, offset: int, whence: int = os.SEEK_SET) -> int:
        base = {os.SEEK_SET: 0, os.SEEK_CUR: self._pos, os.SEEK_END: len(self._v)}[whence]
        self._pos = max(0, min(len(self._v), base + offset))
        return self._pos

    def readinto(self, b) -> int:  # type: ignore[override]
        n = min(len(b), len(self._v) - self._pos)
        if n <= 0:
            return 0
        b[:n] = self._v[self._pos : self._pos + n]
        self._pos += n
        return n

    def read(self, size: int = -1) -> bytes:
        if size is None or size < 0:
            size = len(self._v) - self._pos
        n = min(size, len(self._v) - self._pos)
        out = bytes(self._v[self._pos : self._pos + n])
        self._pos += n
        return out


class ObjectStore(ABC):
    """Minimal surface the benchmarks need."""

    @abstractmethod
    def get_into(self, key: str, dst: torch.Tensor, nbytes: Optional[int] = None) -> int:
        """Fill ``dst`` (a contiguous uint8 CUDA tensor) with the object's bytes."""

    @abstractmethod
    def put_from(self, key: str, src: torch.Tensor, nbytes: Optional[int] = None) -> int:
        """Upload ``nbytes`` from ``src`` (a contiguous uint8 tensor, CUDA or CPU)."""

    @abstractmethod
    def stat(self, key: str) -> int:
        """Object size in bytes."""

    @abstractmethod
    def list(self, prefix: str = "") -> Iterator[tuple[str, int]]:
        """(key, size) pairs under ``prefix``."""

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def get_bytes(self, key: str) -> bytes:
        """Fetch a small object as bytes.

        Always over HTTP, in every backend. RDMA only engages for CUDA device
        memory, so a manifest read into a host buffer would silently fall back --
        and a silent fallback inside a benchmark's metadata path is exactly the
        kind of thing that makes an RDMA result unfalsifiable. Keeping metadata
        explicitly on HTTP means the RDMA byte counters only ever move for payload.
        """

    @abstractmethod
    def put_bytes(self, key: str, data: bytes) -> int:
        """Upload a small object from bytes. Always over HTTP; see get_bytes."""

    @abstractmethod
    def get_bytes_range(self, key: str, offset: int, length: int) -> bytes:
        """Fetch a small byte range as bytes, over HTTP.

        For reading a header out of a large object without transferring the whole
        thing -- a safetensors header is kilobytes inside a ~5 GB shard.
        """

    def get_range_into(self, key: str, dst: torch.Tensor, nbytes: int,
                       offset: int) -> int:
        """Read ``nbytes`` of ``key`` starting at ``offset`` into ``dst``.

        Default implementation is whole-object only and raises; backends that can
        do ranged reads override it. ``supports_range`` says which you have.
        """
        raise NotImplementedError(f"{self.name} backend cannot do ranged reads")

    @property
    def supports_range(self) -> bool:
        return False

    def prepare_thread(self) -> None:
        """Per-thread setup before the first transfer. No-op unless overridden."""

    def close(self) -> None:  # pragma: no cover - trivial
        pass

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.close()


def _check_dst(dst: torch.Tensor, nbytes: Optional[int]) -> int:
    if dst.dtype != torch.uint8:
        raise TypeError(f"destination must be uint8, got {dst.dtype}")
    if not dst.is_contiguous():
        raise ValueError("destination must be contiguous")
    n = nbytes if nbytes is not None else dst.numel()
    if n > dst.numel():
        raise ValueError(f"destination holds {dst.numel()} bytes, need {n}")
    return n


# --------------------------------------------------------------------- HTTP ---


class HttpStore(ObjectStore):
    """Tuned S3-over-HTTP: concurrent ranged GETs -> pinned host -> async H2D."""

    def __init__(self, cfg: StoreConfig) -> None:
        from minio import Minio

        self.cfg = cfg
        self._eps = cfg.all_endpoints()
        # One client per (thread, endpoint) pair: minio-py holds a urllib3 pool
        # per client and sharing one across threads serialises on it.
        self._local = threading.local()
        self._Minio = Minio
        self._rot = _EndpointRotator()

    def _client(self, idx: int = 0):
        cache = getattr(self._local, "clients", None)
        if cache is None:
            cache = self._local.clients = {}
        ep = self._eps[idx % len(self._eps)]
        c = cache.get(ep)
        if c is None:
            c = cache[ep] = self._Minio(
                ep,
                access_key=self.cfg.access_key,
                secret_key=self.cfg.secret_key,
                secure=self.cfg.secure,
                region=self.cfg.region or None,
            )
        return c

    @property
    def name(self) -> str:
        return "http"

    def _stage(self, nbytes: int) -> torch.Tensor:
        """Per-THREAD pinned staging buffer, cached by size.

        Must not be shared across threads. A single buffer per size class would
        have every concurrent stream reading a different object into the same
        memory: the bytes are then garbage (the benchmark only checks the length,
        so it would not even notice), and the cache line ping-pong across two NUMA
        nodes throttles the whole HTTP path. Both were real: sharing capped this
        client at ~3 GB/s regardless of concurrency.
        """
        cache = getattr(self._local, "staging", None)
        if cache is None:
            cache = self._local.staging = {}
        buf = cache.get(nbytes)
        if buf is None:
            buf = cache[nbytes] = torch.empty(nbytes, dtype=torch.uint8, pin_memory=True)
        return buf

    def _part_pool(self) -> Optional[ThreadPoolExecutor]:
        """Per-thread pool for split-range fetches.

        Per-thread rather than one shared pool: the callers are themselves pool
        threads, so a single shared inner pool makes every stream queue behind
        every other stream's parts, turning added concurrency into added latency.
        """
        if self.cfg.http_parts <= 1:
            return None
        pool = getattr(self._local, "parts", None)
        if pool is None:
            pool = self._local.parts = ThreadPoolExecutor(
                max_workers=self.cfg.http_parts, thread_name_prefix="httppart"
            )
        return pool

    def _fetch_range(self, key: str, view: memoryview, offset: int, length: int, idx: int) -> int:
        resp = None
        try:
            resp = self._client(idx).get_object(
                self.cfg.bucket, key, offset=offset, length=length
            )
            got = 0
            chunk = self.cfg.http_chunk
            while got < length:
                # readinto avoids materialising a bytes object per chunk; the copy
                # into the pinned buffer is then a single C memcpy.
                n = resp.readinto(view[got : min(got + chunk, length)])
                if not n:
                    break
                got += n
            return got
        finally:
            if resp is not None:
                resp.close()
                resp.release_conn()

    def get_into(self, key: str, dst: torch.Tensor, nbytes: Optional[int] = None) -> int:
        n = _check_dst(dst, nbytes)
        stage = self._stage(n)
        view = memoryview(stage.numpy())  # zero-copy view of the pinned buffer

        parts = max(1, min(self.cfg.http_parts, n // (1 << 20) or 1))
        base_ep = self._rot.index()
        pool = self._part_pool() if parts > 1 else None
        if pool is None:
            got = self._fetch_range(key, view, 0, n, base_ep)
        else:
            step = (n + parts - 1) // parts
            futs = []
            for i in range(parts):
                off = i * step
                if off >= n:
                    break
                ln = min(step, n - off)
                futs.append(
                    pool.submit(
                        self._fetch_range, key, view[off : off + ln], off, ln, base_ep + i
                    )
                )
            got = sum(f.result() for f in futs)

        if dst.is_cuda:
            dst[:n].copy_(stage[:n], non_blocking=True)
            torch.cuda.current_stream().synchronize()
        else:
            dst[:n].copy_(stage[:n])
        return got

    def put_from(self, key: str, src: torch.Tensor, nbytes: Optional[int] = None) -> int:
        n = nbytes if nbytes is not None else src.numel()
        if src.is_cuda:
            # One D2H copy into pinned memory is unavoidable for HTTP -- it is the
            # cost RDMA removes -- but it should be the ONLY host-side pass.
            stage = self._stage(n)
            stage[:n].copy_(src[:n], non_blocking=True)
            torch.cuda.current_stream().synchronize()
            host = stage
        else:
            host = src.contiguous()
        reader = _ZeroCopyReader(memoryview(host.numpy())[:n])
        self._client(self._rot.index()).put_object(
            self.cfg.bucket, key, reader, n, part_size=max(5 << 20, self.cfg.http_chunk)
        )
        return n

    def stat(self, key: str) -> int:
        return self._client(0).stat_object(self.cfg.bucket, key).size

    @property
    def supports_range(self) -> bool:
        return True

    def get_range_into(self, key: str, dst: torch.Tensor, nbytes: int,
                       offset: int) -> int:
        n = _check_dst(dst, nbytes)
        stage = self._stage(n)
        view = memoryview(stage.numpy())
        got = self._fetch_range(key, view, offset, n, self._rot.index())
        if dst.is_cuda:
            dst[:n].copy_(stage[:n], non_blocking=True)
            torch.cuda.current_stream().synchronize()
        else:
            dst[:n].copy_(stage[:n])
        return got

    def get_bytes(self, key: str) -> bytes:
        resp = self._client(0).get_object(self.cfg.bucket, key)
        try:
            return resp.read()
        finally:
            resp.close()
            resp.release_conn()

    def put_bytes(self, key: str, data: bytes) -> int:
        import io as _io

        self._client(0).put_object(self.cfg.bucket, key, _io.BytesIO(data), len(data))
        return len(data)

    def get_bytes_range(self, key: str, offset: int, length: int) -> bytes:
        resp = self._client(0).get_object(self.cfg.bucket, key, offset=offset,
                                         length=length)
        try:
            return resp.read()
        finally:
            resp.close()
            resp.release_conn()

    def list(self, prefix: str = "") -> Iterator[tuple[str, int]]:
        for o in self._client(0).list_objects(self.cfg.bucket, prefix=prefix, recursive=True):
            yield o.object_name, o.size

    def prepare_thread(self) -> None:
        """Build this thread's client, staging cache and part pool up front."""
        self._client(self._rot.index())

    def close(self) -> None:
        pass


# --------------------------------------------------------------------- RDMA ---


class _EndpointRotator:
    """Hands each thread a stable endpoint index, round-robin on first use.

    Without this every stream targets endpoint 0, so one node serves the whole
    load and the measurement reports that node's ceiling rather than the
    cluster's.
    """

    def __init__(self) -> None:
        self._n = 0
        self._lock = threading.Lock()
        self._local = threading.local()

    def index(self) -> int:
        i = getattr(self._local, "idx", None)
        if i is None:
            with self._lock:
                i = self._n
                self._n += 1
            self._local.idx = i
        return i


class RdmaStore(ObjectStore):
    """GPU-Direct S3-over-RDMA: the NIC DMAs to/from VRAM, no host bounce."""

    def __init__(self, cfg: StoreConfig) -> None:
        from . import rdma_client

        self.cfg = cfg
        self._mod = rdma_client
        if not rdma_client.is_available():
            raise RuntimeError(
                "cuObjClient is not connected to a cuObjServer. Run "
                "scripts/setup/11-verify-rdma.py for a diagnosis."
            )
        self._eps = cfg.all_endpoints()
        # The device whose context worker threads must bind. Captured at
        # construction from the caller's current device.
        self._device_index = torch.cuda.current_device() if torch.cuda.is_available() else 0
        # RDMAClient is not thread-safe; give every thread its own.
        self._local = threading.local()
        # ...but keep a registry of them so close() can free them deterministically.
        # Relying on __del__ means miniocpp_client_free() runs at whatever point GC
        # decides -- in practice during interpreter shutdown, after the CUDA context
        # has gone, which segfaults intermittently at process exit.
        self._clients: list = []
        self._clients_lock = threading.Lock()
        self._rot = _EndpointRotator()
        # minio-py for metadata only: the C ABI has no HEAD/LIST.
        from minio import Minio

        self._meta = Minio(
            cfg.endpoint,
            access_key=cfg.access_key,
            secret_key=cfg.secret_key,
            secure=cfg.secure,
            region=cfg.region or None,
        )

    def _client(self, idx: int = 0):
        cache = getattr(self._local, "clients", None)
        if cache is None:
            # MUST come before any RDMA call on a device pointer from this thread.
            #
            # cuFile/cuObject reach for the *current* CUDA context via the driver
            # API. A freshly spawned Python thread has none -- CUDA's primary
            # context is bound per thread, and torch only binds it when that
            # thread first touches CUDA. Registering or transferring a device
            # pointer with no current context does not return an error; it
            # segfaults inside libcuobjclient.
            #
            # This is why a single-threaded RDMA fetch works and any thread pool
            # crashes. set_device binds the primary context; touching the stream
            # forces torch's lazy per-thread init to actually happen.
            self._bind_cuda_context()
            cache = self._local.clients = {}
        ep = self._eps[idx % len(self._eps)]
        c = cache.get(ep)
        if c is None:
            c = cache[ep] = self._mod.RDMAClient(
                ep, self.cfg.access_key, self.cfg.secret_key,
                region=self.cfg.region, secure=self.cfg.secure,
            )
            with self._clients_lock:
                self._clients.append(c)
        return c

    def _bind_cuda_context(self) -> None:
        if torch.cuda.is_available():
            torch.cuda.set_device(self._device_index)
            torch.cuda.current_stream()

    @property
    def name(self) -> str:
        return "rdma"

    def prepare_thread(self) -> None:
        """Bind the CUDA context and build this thread's client up front.

        Call from a thread-pool initializer so the first measured transfer does
        not also pay context binding and connection setup.
        """
        self._client(self._rot.index())

    def get_into(self, key: str, dst: torch.Tensor, nbytes: Optional[int] = None) -> int:
        n = _check_dst(dst, nbytes)
        # dst must stay alive for the duration; it is the caller's tensor, and the
        # raw pointer we pass carries no reference.
        return self._client(self._rot.index()).get(self.cfg.bucket, key, dst.data_ptr(), n)

    def put_from(self, key: str, src: torch.Tensor, nbytes: Optional[int] = None) -> int:
        n = nbytes if nbytes is not None else src.numel()
        if not src.is_contiguous():
            raise ValueError("source must be contiguous")
        written, _etag, _ck = self._client(self._rot.index()).put(self.cfg.bucket, key, src.data_ptr(), n)
        return written

    def stat(self, key: str) -> int:
        return self._meta.stat_object(self.cfg.bucket, key).size

    @property
    def supports_range(self) -> bool:
        return self._mod.has_range()

    def get_range_into(self, key: str, dst: torch.Tensor, nbytes: int,
                       offset: int) -> int:
        n = _check_dst(dst, nbytes)
        return self._client(self._rot.index()).get_range(
            self.cfg.bucket, key, dst.data_ptr(), n, offset
        )

    def close(self) -> None:
        """Free every RDMA client this store handed out.

        Deterministic teardown matters here: left to garbage collection these are
        freed during interpreter shutdown, once CUDA has already torn down, and
        miniocpp_client_free() then faults. The benchmark has finished and written
        its results by that point, so the crash never corrupted a measurement --
        it just made a clean run exit non-zero about half the time.
        """
        with self._clients_lock:
            clients, self._clients = self._clients, []
        for c in clients:
            try:
                c.close()
            except Exception:
                pass

    def get_bytes(self, key: str) -> bytes:
        resp = self._meta.get_object(self.cfg.bucket, key)
        try:
            return resp.read()
        finally:
            resp.close()
            resp.release_conn()

    def put_bytes(self, key: str, data: bytes) -> int:
        import io as _io

        self._meta.put_object(self.cfg.bucket, key, _io.BytesIO(data), len(data))
        return len(data)

    def get_bytes_range(self, key: str, offset: int, length: int) -> bytes:
        resp = self._meta.get_object(self.cfg.bucket, key, offset=offset, length=length)
        try:
            return resp.read()
        finally:
            resp.close()
            resp.release_conn()

    def list(self, prefix: str = "") -> Iterator[tuple[str, int]]:
        for o in self._meta.list_objects(self.cfg.bucket, prefix=prefix, recursive=True):
            yield o.object_name, o.size


# -------------------------------------------------------------------- local ---


class LocalStore(ObjectStore):
    """Read shards from a local filesystem: the "no network at all" control.

    Useful to show where the bottleneck actually is. If the local-disk number is
    close to the network numbers, storage was never the constraint.
    """

    def __init__(self, cfg: StoreConfig, root: str) -> None:
        self.cfg = cfg
        self.root = root
        # Per-thread, for the same reason as HttpStore._stage: one buffer shared
        # across concurrent readers corrupts data and thrashes cache.
        self._local = threading.local()

    @property
    def name(self) -> str:
        return "local"

    def _path(self, key: str) -> str:
        return os.path.join(self.root, key)

    def get_into(self, key: str, dst: torch.Tensor, nbytes: Optional[int] = None) -> int:
        n = _check_dst(dst, nbytes)
        cache = getattr(self._local, "staging", None)
        if cache is None:
            cache = self._local.staging = {}
        stage = cache.get(n)
        if stage is None:
            stage = cache[n] = torch.empty(n, dtype=torch.uint8, pin_memory=True)
        view = memoryview(stage.numpy())
        got = 0
        with open(self._path(key), "rb", buffering=0) as fh:
            while got < n:
                r = fh.readinto(view[got:n])
                if not r:
                    break
                got += r
        if dst.is_cuda:
            dst[:n].copy_(stage[:n], non_blocking=True)
            torch.cuda.current_stream().synchronize()
        else:
            dst[:n].copy_(stage[:n])
        return got

    def put_from(self, key: str, src: torch.Tensor, nbytes: Optional[int] = None) -> int:
        n = nbytes if nbytes is not None else src.numel()
        host = src[:n].cpu() if src.is_cuda else src[:n]
        path = self._path(key)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as fh:
            fh.write(memoryview(host.numpy()))
        return n

    def stat(self, key: str) -> int:
        return os.path.getsize(self._path(key))

    @property
    def supports_range(self) -> bool:
        return True

    def get_range_into(self, key: str, dst: torch.Tensor, nbytes: int,
                       offset: int) -> int:
        n = _check_dst(dst, nbytes)
        cache = getattr(self._local, "staging", None)
        if cache is None:
            cache = self._local.staging = {}
        stage = cache.get(n)
        if stage is None:
            stage = cache[n] = torch.empty(n, dtype=torch.uint8, pin_memory=True)
        view = memoryview(stage.numpy())
        got = 0
        with open(self._path(key), "rb", buffering=0) as fh:
            fh.seek(offset)
            while got < n:
                r = fh.readinto(view[got:n])
                if not r:
                    break
                got += r
        if dst.is_cuda:
            dst[:n].copy_(stage[:n], non_blocking=True)
            torch.cuda.current_stream().synchronize()
        else:
            dst[:n].copy_(stage[:n])
        return got

    def get_bytes(self, key: str) -> bytes:
        with open(self._path(key), "rb") as fh:
            return fh.read()

    def put_bytes(self, key: str, data: bytes) -> int:
        path = self._path(key)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "wb") as fh:
            fh.write(data)
        return len(data)

    def get_bytes_range(self, key: str, offset: int, length: int) -> bytes:
        with open(self._path(key), "rb") as fh:
            fh.seek(offset)
            return fh.read(length)

    def list(self, prefix: str = "") -> Iterator[tuple[str, int]]:
        base = os.path.join(self.root, prefix)
        for dirpath, _dirs, files in os.walk(os.path.dirname(base) or self.root):
            for f in files:
                full = os.path.join(dirpath, f)
                rel = os.path.relpath(full, self.root)
                if rel.startswith(prefix):
                    yield rel, os.path.getsize(full)


def make_store(backend: str, cfg: StoreConfig, local_root: str = "") -> ObjectStore:
    b = backend.lower()
    if b == "http":
        return HttpStore(cfg)
    if b == "rdma":
        return RdmaStore(cfg)
    if b == "local":
        if not local_root:
            raise ValueError("backend 'local' requires local_root")
        return LocalStore(cfg, local_root)
    raise ValueError(f"unknown backend {backend!r}; expected http, rdma, or local")
