# Copyright 2026 MinIO, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Data pipelines for the two shard layouts, sharing one iterator contract.

Both yield ``(images, labels)`` with images as normalised NCHW float in
``channels_last``, so the training loop is identical and the only difference is
how bytes reach the GPU.

``RawShardLoader`` — GPU-native (``imagenet-raw``)
    Fetches a shard of fixed-size uint8 images directly into VRAM (RDMA: the NIC
    DMAs into device memory; HTTP: pinned host then one H2D), then does random
    crop, horizontal flip and normalisation on the GPU. **No decode, no host-side
    per-sample work.** This is the layout in which the transport is what limits
    training throughput.

``JpegShardLoader`` — JPEG tars (``imagenet-shards``)
    Fetches a tar into pinned host memory and batch-decodes with nvJPEG. RDMA
    cannot be used here at all: it requires a device destination, while the JPEG
    decoder requires a host source (see plans/reports/02-where-rdma-applies.md).
    Present to establish where a conventional pipeline's time actually goes.

Prefetching in both cases is a background thread fetching shard N+1 while the GPU
consumes shard N, which is what keeps the fetch off the critical path; the loop
reports how long it actually waited so a storage-bound run is distinguishable from
a compute-bound one.
"""

from __future__ import annotations

import json
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Iterator, Optional

import torch

# ImageNet channel statistics, in 0-255 units so normalisation folds into one
# fused op on the uint8 -> float conversion.
IMAGENET_MEAN = (0.485 * 255, 0.456 * 255, 0.406 * 255)
IMAGENET_STD = (0.229 * 255, 0.224 * 255, 0.225 * 255)


@dataclass
class LoaderStats:
    """Where the loader's wall clock went. The point of the whole exercise."""

    fetch_wait_s: float = 0.0  # blocked waiting for a shard to arrive
    fetch_s: float = 0.0  # time the fetch thread spent transferring
    decode_s: float = 0.0
    augment_s: float = 0.0
    bytes_fetched: int = 0
    shards: int = 0
    batches: int = 0
    samples: int = 0
    per_shard_fetch_s: list[float] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "fetch_wait_s": round(self.fetch_wait_s, 4),
            "fetch_s": round(self.fetch_s, 4),
            "decode_s": round(self.decode_s, 4),
            "augment_s": round(self.augment_s, 4),
            "bytes_fetched": self.bytes_fetched,
            "shards": self.shards,
            "batches": self.batches,
            "samples": self.samples,
            "fetch_gbps": (
                self.bytes_fetched / self.fetch_s / 1e9 if self.fetch_s > 0 else None
            ),
        }


def _normalise(batch_hwc_u8: torch.Tensor) -> torch.Tensor:
    """uint8 [B,H,W,C] -> normalised float [B,C,H,W] in channels_last."""
    x = batch_hwc_u8.permute(0, 3, 1, 2).contiguous(memory_format=torch.channels_last)
    x = x.to(torch.float32)
    mean = torch.tensor(IMAGENET_MEAN, device=x.device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=x.device).view(1, 3, 1, 1)
    return (x - mean) / std


def random_crop_flip(imgs: torch.Tensor, out_size: int, generator=None) -> torch.Tensor:
    """Per-image random crop + horizontal flip, as a single batched gather.

    ``imgs`` is uint8 [B,H,W,C]; returns uint8 [B,out,out,C].

    Doing this with one advanced-indexing gather rather than a Python loop over the
    batch matters: at a few thousand images per second per GPU, a loop issuing two
    or three small kernels per image becomes the bottleneck the whole exercise is
    trying to remove. Folding the flip into the x index means crop and flip cost one
    kernel together.
    """
    b, h, w, _ = imgs.shape
    dev = imgs.device
    max_y, max_x = h - out_size, w - out_size
    if max_y < 0 or max_x < 0:
        raise ValueError(f"cannot crop {out_size} from {h}x{w}")

    ys = torch.randint(0, max_y + 1, (b, 1, 1), device=dev, generator=generator)
    xs = torch.randint(0, max_x + 1, (b, 1, 1), device=dev, generator=generator)
    flip = torch.rand((b, 1, 1), device=dev, generator=generator) < 0.5

    ar = torch.arange(out_size, device=dev)
    iy = ys + ar.view(1, out_size, 1)
    # Reverse the x offsets for flipped images instead of calling .flip() after.
    ix_fwd = xs + ar.view(1, 1, out_size)
    ix_rev = xs + (out_size - 1 - ar).view(1, 1, out_size)
    ix = torch.where(flip, ix_rev, ix_fwd)

    ib = torch.arange(b, device=dev).view(b, 1, 1)
    return imgs[ib, iy, ix]


class _BufferPool:
    """Device buffers with stable addresses, each owned by exactly one fetch thread.

    Two properties are required, and both were learned by crashing:

    1. **Addresses must be stable.** Allocating a fresh destination buffer per shard
       looks harmless and is not: PyTorch's caching allocator recycles addresses, so
       a new fetch often lands on an address a previous fetch used. cuObject
       registers buffers *by address*, so with several fetches in flight you get
       concurrent register/deregister of one address, which segfaults inside
       libminiocpp.

    2. **A given address must belong to one thread.** Sharing one free-list across
       fetch threads still crashes, because a buffer released by thread A is then
       registered by thread B's client -- the same address seen by two clients. So
       the pool is partitioned: worker *i* only ever touches its own buffers.

    VRAM use is ``size * per_worker * workers``.
    """

    def __init__(self, size: int, workers: int, device: torch.device,
                 per_worker: int = 2) -> None:
        self.size = size
        self.workers = workers
        self._free: list[queue.Queue] = []
        self.buffers: list[torch.Tensor] = []
        for _ in range(workers):
            q: queue.Queue = queue.Queue()
            for _ in range(per_worker):
                b = torch.empty(size, dtype=torch.uint8, device=device)
                self.buffers.append(b)
                q.put(b)
            self._free.append(q)
        torch.cuda.synchronize()

    def acquire(self, worker: int) -> torch.Tensor:
        return self._free[worker % self.workers].get()

    def release(self, worker: int, buf: torch.Tensor) -> None:
        self._free[worker % self.workers].put(buf)


class _Prefetcher:
    """Fetch shards on background threads, several in flight at once.

    ``workers`` matters a great deal and is not just a tuning knob:

    The HTTP path splits every object into ``http_parts`` concurrent ranged GETs, so
    one shard fetch is already 8-way parallel. The RDMA C ABI has no offset
    parameter -- ``miniocpp_get_object`` transfers a whole object -- so an RDMA shard
    fetch is a *single* stream. Running one fetch at a time therefore hands HTTP 8x
    the concurrency and makes the comparison meaningless: measured that way HTTP
    looked 3.6x faster than RDMA, the reverse of the isolated storage benchmark.

    Fetching several shards concurrently gives the RDMA path the parallelism it
    needs (throughput at one stream is a fraction of throughput at 32), and it is
    what a real loader does anyway. Report ``workers`` alongside any result.

    Each in-flight shard needs its own destination buffer, so VRAM use is roughly
    ``2 * workers * shard_bytes``; the queue is bounded to keep that predictable.
    """

    def __init__(self, fetch_fn, keys: list[str], depth: int = 2, workers: int = 1,
                 on_thread_start=None) -> None:
        self._q: queue.Queue = queue.Queue(maxsize=max(1, depth))
        self._keys = list(keys)
        self._fetch = fetch_fn
        self._stop = threading.Event()
        self._workers = max(1, workers)
        self._on_thread_start = on_thread_start
        self._next = 0
        self._lock = threading.Lock()
        self._live = self._workers
        self._threads = [
            threading.Thread(target=self._run, args=(i,), daemon=True,
                             name=f"shard-prefetch-{i}")
            for i in range(self._workers)
        ]
        self.fetch_s = 0.0  # summed across workers; wall-clock is lower when workers>1
        self.per_shard: list[float] = []
        self.error: Optional[BaseException] = None

    def start(self) -> "_Prefetcher":
        for t in self._threads:
            t.start()
        return self

    def _claim(self) -> Optional[str]:
        with self._lock:
            if self._next >= len(self._keys):
                return None
            k = self._keys[self._next]
            self._next += 1
            return k

    def _run(self, worker: int = 0) -> None:
        try:
            # RDMA needs a bound CUDA context in each thread before its first
            # transfer, or it segfaults; see RdmaStore.prepare_thread.
            if self._on_thread_start is not None:
                self._on_thread_start()
            while not self._stop.is_set():
                k = self._claim()
                if k is None:
                    break
                t0 = time.perf_counter()
                item = self._fetch(k, worker)
                dt = time.perf_counter() - t0
                with self._lock:
                    self.fetch_s += dt
                    self.per_shard.append(dt)
                self._q.put(item)
        except BaseException as exc:  # surfaced to the consumer
            self.error = exc
        finally:
            with self._lock:
                self._live -= 1
                last = self._live == 0
            if last:
                self._q.put(None)

    def __iter__(self):
        while True:
            item = self._q.get()
            if item is None:
                if self.error is not None:
                    raise self.error
                return
            yield item

    def stop(self) -> None:
        self._stop.set()

    def join(self, timeout: float = 30.0) -> bool:
        """Wait for the fetch threads to actually leave the transport.

        Required before the store frees its clients. The workers are daemon
        threads, so without this they can still be inside get_into() when
        RdmaStore.close() frees the client underneath them -- a use-after-free
        that surfaces as an intermittent segfault at process exit.

        Call stop() and drain the ready queue first, or a worker blocked on a full
        queue will never observe the stop flag. Returns True if all threads exited.
        """
        deadline = time.perf_counter() + timeout
        for t in self._threads:
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                break
            t.join(timeout=remaining)
        return not any(t.is_alive() for t in self._threads)


class RawShardLoader:
    """Iterate batches from GPU-native raw shards.

    With ``backend="rdma"`` the shard lands in VRAM by DMA and nothing else in this
    class touches host memory.
    """

    def __init__(
        self,
        store,
        keys: list[str],
        batch_size: int,
        device: str = "cuda:0",
        crop: int = 224,
        prefetch: int = 2,
        drop_last: bool = True,
        seed: int = 0,
        fetch_workers: int = 4,
    ) -> None:
        self.store = store
        self.keys = keys
        self.batch_size = batch_size
        self.device = torch.device(device)
        self.crop = crop
        self.prefetch = prefetch
        self.fetch_workers = fetch_workers
        self.drop_last = drop_last
        self.stats = LoaderStats()
        self._gen = torch.Generator(device=self.device).manual_seed(seed)
        # Metadata is small and read over HTTP by contract; do it once up front so
        # it never appears in the measured loop.
        self._meta: dict[str, dict] = {}
        self._pool: Optional[_BufferPool] = None

    def _meta_for(self, key: str) -> dict:
        m = self._meta.get(key)
        if m is None:
            m = self._meta[key] = json.loads(
                self.store.get_bytes(key.rsplit(".", 1)[0] + ".json").decode()
            )
        return m

    def preload_metadata(self) -> int:
        total = 0
        for k in self.keys:
            m = self._meta_for(k)
            total += m["samples"]
        return total

    def _ensure_pool(self) -> "_BufferPool":
        if self._pool is None:
            # Every shard in this dataset is the same size, so one size class covers
            # the pool. Two buffers per worker lets a worker start its next fetch
            # while the consumer still holds the previous shard.
            m = self._meta_for(self.keys[0])
            size = m["samples"] * m["height"] * m["width"] * m["channels"]
            self._pool = _BufferPool(size, self.fetch_workers, self.device,
                                     per_worker=2)
        return self._pool

    def _fetch(self, key: str, worker: int = 0):
        m = self._meta_for(key)
        n, h, w, c = m["samples"], m["height"], m["width"], m["channels"]
        nbytes = n * h * w * c
        pool = self._pool
        buf = pool.acquire(worker)
        try:
            got = self.store.get_into(key, buf, nbytes)
        except BaseException:
            pool.release(worker, buf)
            raise
        if got != nbytes:
            pool.release(worker, buf)
            raise RuntimeError(f"{key}: short read {got} != {nbytes}")
        labels = torch.tensor(m["labels"][:n], dtype=torch.int64, device=self.device)
        return worker, buf, buf[:nbytes].view(n, h, w, c), labels, nbytes

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        pool = self._ensure_pool()
        pf = _Prefetcher(
            self._fetch, self.keys, depth=max(self.prefetch, self.fetch_workers),
            workers=self.fetch_workers, on_thread_start=self.store.prepare_thread,
        ).start()
        stream = pf.__iter__()
        try:
            while True:
                t0 = time.perf_counter()
                try:
                    item = next(stream)
                except StopIteration:
                    break
                self.stats.fetch_wait_s += time.perf_counter() - t0
                worker, buf, imgs, labels, nbytes = item
                self.stats.shards += 1
                self.stats.bytes_fetched += nbytes

                try:
                    n = imgs.shape[0]
                    for s in range(0, n, self.batch_size):
                        e = min(s + self.batch_size, n)
                        if self.drop_last and e - s < self.batch_size:
                            break
                        t1 = time.perf_counter()
                        cropped = random_crop_flip(imgs[s:e], self.crop, generator=self._gen)
                        x = _normalise(cropped)
                        self.stats.augment_s += time.perf_counter() - t1
                        self.stats.batches += 1
                        self.stats.samples += e - s
                        yield x, labels[s:e]
                finally:
                    # The batch tensors produced above are fresh allocations, so the
                    # shard buffer can go back to the pool as soon as this shard is
                    # consumed. Returning it in `finally` matters: a consumer that
                    # stops early (step budget reached) would otherwise starve the
                    # fetch threads on the next iteration.
                    del imgs, labels
                    pool.release(worker, buf)
        finally:
            pf.stop()
            # Drain anything already fetched, so a worker blocked on a full ready
            # queue can observe the stop flag and exit.
            try:
                while True:
                    leftover = pf._q.get_nowait()
                    if leftover is not None:
                        pool.release(leftover[0], leftover[1])
            except queue.Empty:
                pass
            # Then wait for them to leave the transport before anyone frees a client.
            pf.join()
            self.stats.fetch_s = pf.fetch_s
            self.stats.per_shard_fetch_s = pf.per_shard

    def __len__(self) -> int:
        total = sum(self._meta_for(k)["samples"] for k in self.keys)
        return total // self.batch_size


class JpegShardLoader:
    """Iterate batches from JPEG tar shards, decoding with nvJPEG.

    The tar must land in **host** memory: nvJPEG parses JPEG headers on the CPU, so
    the encoded bytes cannot come from VRAM. That is why this loader has no RDMA
    variant -- RDMA requires a device destination.
    """

    def __init__(
        self,
        store,
        keys: list[str],
        batch_size: int,
        device: str = "cuda:0",
        crop: int = 224,
        prefetch: int = 2,
        drop_last: bool = True,
        seed: int = 0,
        fetch_workers: int = 4,
    ) -> None:
        from .shards import ShardIndex

        self.store = store
        self.keys = keys
        self.batch_size = batch_size
        self.device = torch.device(device)
        self.crop = crop
        self.prefetch = prefetch
        self.fetch_workers = fetch_workers
        self.drop_last = drop_last
        self.stats = LoaderStats()
        self._ShardIndex = ShardIndex
        self._gen = torch.Generator(device=self.device).manual_seed(seed)
        self._index: dict[str, object] = {}
        # Per-thread: parallel fetchers sharing one staging buffer per size class
        # would read different shards into the same memory.
        self._staging_local = threading.local()

    def _index_for(self, key: str):
        ix = self._index.get(key)
        if ix is None:
            raw = self.store.get_bytes(self._ShardIndex.key_for(key))
            ix = self._index[key] = self._ShardIndex.from_json(raw)
        return ix

    def preload_metadata(self) -> int:
        return sum(self._index_for(k).num_samples for k in self.keys)

    def _fetch(self, key: str, worker: int = 0):
        ix = self._index_for(key)
        cache = getattr(self._staging_local, "bufs", None)
        if cache is None:
            cache = self._staging_local.bufs = {}
        buf = cache.get(ix.size)
        if buf is None:
            buf = cache[ix.size] = torch.empty(ix.size, dtype=torch.uint8, pin_memory=True)
        got = self.store.get_into(key, buf, ix.size)
        if got != ix.size:
            raise RuntimeError(f"{key}: short read {got} != {ix.size}")
        return buf, ix, ix.size

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        from torchvision.io import ImageReadMode, decode_jpeg
        import torch.nn.functional as F

        pf = _Prefetcher(
            self._fetch, self.keys, depth=max(self.prefetch, self.fetch_workers),
            workers=self.fetch_workers, on_thread_start=self.store.prepare_thread,
        ).start()
        stream = pf.__iter__()
        try:
            while True:
                t0 = time.perf_counter()
                try:
                    item = next(stream)
                except StopIteration:
                    break
                self.stats.fetch_wait_s += time.perf_counter() - t0
                buf, ix, nbytes = item
                self.stats.shards += 1
                self.stats.bytes_fetched += nbytes

                samples = ix.samples
                for s in range(0, len(samples), self.batch_size):
                    grp = samples[s : s + self.batch_size]
                    if self.drop_last and len(grp) < self.batch_size:
                        break
                    t1 = time.perf_counter()
                    views = [buf[g.offset : g.offset + g.length] for g in grp]
                    try:
                        imgs = decode_jpeg(views, mode=ImageReadMode.RGB, device=self.device)
                    except Exception:
                        imgs = []
                        for v in views:
                            try:
                                imgs.append(decode_jpeg(v, mode=ImageReadMode.RGB,
                                                        device=self.device))
                            except Exception:
                                pass
                        if not imgs:
                            continue
                    self.stats.decode_s += time.perf_counter() - t1

                    t2 = time.perf_counter()
                    # Decoded sizes vary, so resize per image before stacking. This
                    # per-image work is exactly what the raw layout removes.
                    out = torch.empty(
                        (len(imgs), self.crop, self.crop, 3), dtype=torch.uint8,
                        device=self.device,
                    )
                    for j, im in enumerate(imgs):
                        r = F.interpolate(
                            im.unsqueeze(0).float(), size=(self.crop, self.crop),
                            mode="bilinear", align_corners=False, antialias=True,
                        )
                        out[j] = r.squeeze(0).clamp_(0, 255).to(torch.uint8).permute(1, 2, 0)
                    flipped = random_crop_flip(out, self.crop, generator=self._gen) \
                        if out.shape[1] > self.crop else out
                    x = _normalise(flipped)
                    labels = torch.tensor([g.label for g in grp[: len(imgs)]],
                                          dtype=torch.int64, device=self.device)
                    self.stats.augment_s += time.perf_counter() - t2
                    self.stats.batches += 1
                    self.stats.samples += len(imgs)
                    yield x, labels
        finally:
            pf.stop()
            try:
                while True:
                    if pf._q.get_nowait() is None:
                        break
            except queue.Empty:
                pass
            pf.join()
            self.stats.fetch_s = pf.fetch_s
            self.stats.per_shard_fetch_s = pf.per_shard
