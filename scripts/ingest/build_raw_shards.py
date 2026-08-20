#!/usr/bin/env python3
# Copyright 2026 MinIO, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Build GPU-native raw shards: decode once at ingest, store fixed-size uint8.

Why this dataset exists
-----------------------
GPU-Direct S3-over-RDMA only engages when the destination is CUDA device memory,
and GPU JPEG decoders only accept the compressed bitstream from host memory (both
measured -- see plans/reports/02-where-rdma-applies.md). A JPEG pipeline therefore
cannot use RDMA on its data path at all.

Decoding once, at ingest, removes the conflict: a shard becomes a flat array of
fixed-size uint8 images, which is exactly what the GPU consumes. Then a training
step is: RDMA the shard into VRAM, index a batch out of it, crop/flip/normalise on
the GPU. No host memory, no decode, no CPU in the data path.

This is the same trade FFCV and DALI's raw format make. It costs capacity --
256x256x3 is ~197 KB per image versus ~110 KB as JPEG, so ImageNet train grows
from 140 GB to ~252 GB -- and it fixes the stored resolution. Storing 256x256 and
random-cropping to 224 on the GPU keeps per-epoch augmentation intact; only the
JPEG decode is moved off the hot path.

Layout, per shard:

    <prefix>-000000.raw    N x H x W x 3 uint8, contiguous, no header
    <prefix>-000000.json   {"samples": N, "height": H, "width": W, "channels": 3,
                            "labels": [...], "keys": [...]}

Fixed-size samples mean no per-sample offset table: sample i starts at
i * H * W * 3.

    python scripts/ingest/build_raw_shards.py \
        --src /home/minio/benchmark/imagenet/ILSVRC/Data/CLS-LOC/train \
        --bucket imagenet-raw --prefix train --size 256 --samples-per-shard 4096
"""

from __future__ import annotations

import argparse
import io
import json
import os
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

_CFG: dict = {}
_CLIENT = None
_TORCH = None


def _init_worker(cfg: dict) -> None:
    global _CFG, _CLIENT, _TORCH
    import torch

    _CFG = cfg
    _TORCH = torch
    # One GPU per worker, assigned by worker index derived from the OS pid slot.
    # ProcessPoolExecutor does not expose a worker index, so derive it from the
    # order in which workers initialise.
    idx = int(os.environ.get("S3RDMA_WORKER_SLOT", "0"))
    dev = idx % torch.cuda.device_count()
    torch.cuda.set_device(dev)
    _CFG["device"] = f"cuda:{dev}"

    from minio import Minio

    _CLIENT = Minio(cfg["endpoint"], cfg["access_key"], cfg["secret_key"], secure=cfg["secure"])


def _worker_slot_init(cfg: dict, counter) -> None:
    with counter.get_lock():
        slot = counter.value
        counter.value += 1
    os.environ["S3RDMA_WORKER_SLOT"] = str(slot)
    _init_worker(cfg)


def _decode_resize_batch(paths: list[str], size: int, device: str):
    """Read, GPU-decode and resize a batch. Returns (uint8 [B,size,size,3], kept_idx)."""
    torch = _TORCH
    from torchvision.io import ImageReadMode, decode_jpeg
    import torch.nn.functional as F

    raws, kept = [], []
    for i, p in enumerate(paths):
        try:
            with open(p, "rb") as fh:
                blob = fh.read()
        except OSError:
            continue
        if len(blob) < 4 or blob[:2] != b"\xff\xd8":
            continue
        # decode_jpeg wants the encoded bytes on the host even for a CUDA decode;
        # that host residency is exactly why this conversion is done once here
        # rather than every epoch during training.
        raws.append(torch.frombuffer(bytearray(blob), dtype=torch.uint8))
        kept.append(i)

    if not raws:
        return None, []

    try:
        imgs = decode_jpeg(raws, mode=ImageReadMode.RGB, device=device)
    except Exception:
        # One corrupt file must not lose the batch; fall back to per-image decode.
        imgs, kept2 = [], []
        for j, r in enumerate(raws):
            try:
                imgs.append(decode_jpeg(r, mode=ImageReadMode.RGB, device=device))
                kept2.append(kept[j])
            except Exception:
                pass
        kept = kept2
        if not imgs:
            return None, []

    out = torch.empty((len(imgs), size, size, 3), dtype=torch.uint8, device=device)
    for j, im in enumerate(imgs):
        # antialias matters when downscaling by >2x, which is most of ImageNet.
        r = F.interpolate(
            im.unsqueeze(0).float(), size=(size, size), mode="bilinear",
            align_corners=False, antialias=True,
        )
        out[j] = r.squeeze(0).clamp_(0, 255).to(torch.uint8).permute(1, 2, 0)
    return out, kept


def _build_shard(task: tuple[int, list[tuple[str, int]]]) -> dict:
    torch = _TORCH
    shard_idx, entries = task
    prefix = _CFG["prefix"]
    size = _CFG["size"]
    device = _CFG["device"]
    shard_key = f"{prefix}/{prefix}-{shard_idx:06d}.raw"
    meta_key = f"{prefix}/{prefix}-{shard_idx:06d}.json"

    if _CFG["skip_existing"]:
        try:
            _CLIENT.stat_object(_CFG["bucket"], shard_key)
            _CLIENT.stat_object(_CFG["bucket"], meta_key)
            return {"shard": shard_key, "skipped": True, "samples": 0, "bytes": 0}
        except Exception:
            pass

    t0 = time.perf_counter()
    sample_bytes = size * size * 3
    buf = torch.empty(len(entries) * sample_bytes, dtype=torch.uint8, device=device)
    view = buf.view(len(entries), size, size, 3)

    labels: list[int] = []
    keys: list[str] = []
    n = 0
    bs = _CFG["decode_batch"]
    for s in range(0, len(entries), bs):
        chunk = entries[s : s + bs]
        imgs, kept = _decode_resize_batch([p for p, _ in chunk], size, device)
        if imgs is None:
            continue
        view[n : n + imgs.shape[0]] = imgs
        for j in kept:
            labels.append(chunk[j][1])
            keys.append(os.path.splitext(os.path.basename(chunk[j][0]))[0])
        n += imgs.shape[0]
    torch.cuda.synchronize()
    build_s = time.perf_counter() - t0

    payload_bytes = n * sample_bytes
    t1 = time.perf_counter()
    # Upload from host: this is ingest, not the benchmark, and a plain multipart PUT
    # keeps the tool usable on machines without a working RDMA client.
    host = buf[:payload_bytes].cpu()
    _CLIENT.put_object(
        _CFG["bucket"], shard_key, io.BytesIO(memoryview(host.numpy())), payload_bytes,
        part_size=_CFG["part_size"], content_type="application/octet-stream",
    )
    meta = {
        "samples": n, "height": size, "width": size, "channels": 3,
        "dtype": "uint8", "sample_bytes": sample_bytes,
        "labels": labels, "keys": keys,
    }
    raw = json.dumps(meta, separators=(",", ":")).encode()
    _CLIENT.put_object(_CFG["bucket"], meta_key, io.BytesIO(raw), len(raw),
                       content_type="application/json")
    upload_s = time.perf_counter() - t1

    del buf, view, host
    torch.cuda.empty_cache()
    return {
        "shard": shard_key, "skipped": False, "samples": n, "bytes": payload_bytes,
        "dropped": len(entries) - n, "build_s": build_s, "upload_s": upload_s,
        "device": device,
    }


def discover(src: str) -> tuple[list[tuple[str, int]], list[str]]:
    classes = sorted(d for d in os.listdir(src) if os.path.isdir(os.path.join(src, d)))
    files: list[tuple[str, int]] = []
    for label, wnid in enumerate(classes):
        with os.scandir(os.path.join(src, wnid)) as it:
            for e in it:
                if e.is_file():
                    files.append((e.path, label))
    return files, classes


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--bucket", default="imagenet-raw")
    ap.add_argument("--prefix", default="train")
    ap.add_argument("--endpoint", default=os.environ.get("S3_ENDPOINT", "aistor1:9000"))
    ap.add_argument("--access-key", default=os.environ.get("S3_ACCESS_KEY", "minioadmin"))
    ap.add_argument("--secret-key", default=os.environ.get("S3_SECRET_KEY", "minioadmin"))
    ap.add_argument("--size", type=int, default=256,
                    help="stored square edge; training random-crops to 224 from this")
    ap.add_argument("--samples-per-shard", type=int, default=4096)
    ap.add_argument("--decode-batch", type=int, default=256)
    ap.add_argument("--max-shards", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8, help="one per GPU is a good default")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--part-size", type=int, default=64 << 20)
    ap.add_argument("--no-skip-existing", action="store_true")
    args = ap.parse_args()

    from minio import Minio

    admin = Minio(args.endpoint, args.access_key, args.secret_key, secure=False)
    if not admin.bucket_exists(args.bucket):
        admin.make_bucket(args.bucket)
        print(f"created bucket {args.bucket}")

    print(f"scanning {args.src} ...", flush=True)
    files, classes = discover(args.src)
    print(f"  {len(files):,} files in {len(classes)} classes")

    # Same seed and same shuffle as the JPEG shard builder, so the two datasets
    # contain the same samples in the same shard order and results are comparable.
    random.Random(args.seed).shuffle(files)

    sps = args.samples_per_shard
    tasks = [(i, files[i * sps : (i + 1) * sps]) for i in range((len(files) + sps - 1) // sps)]
    tasks = [t for t in tasks if t[1]]
    if args.max_shards:
        tasks = tasks[: args.max_shards]

    sample_bytes = args.size * args.size * 3
    print(f"  {len(tasks)} shards x {sps} samples x {sample_bytes / 1024:.0f} KiB "
          f"= {sps * sample_bytes / 2**30:.2f} GiB per shard")
    print(f"  total ~{len(tasks) * sps * sample_bytes / 2**40:.2f} TiB, {args.workers} workers")

    cfg = {
        "bucket": args.bucket, "prefix": args.prefix, "endpoint": args.endpoint,
        "access_key": args.access_key, "secret_key": args.secret_key, "secure": False,
        "size": args.size, "decode_batch": args.decode_batch, "part_size": args.part_size,
        "skip_existing": not args.no_skip_existing,
    }

    import multiprocessing as mp

    ctx = mp.get_context("spawn")
    counter = ctx.Value("i", 0)

    done = skipped = 0
    total_bytes = total_samples = total_dropped = 0
    t_start = time.perf_counter()

    with ProcessPoolExecutor(
        max_workers=args.workers, mp_context=ctx,
        initializer=_worker_slot_init, initargs=(cfg, counter),
    ) as pool:
        futs = {pool.submit(_build_shard, t): t[0] for t in tasks}
        for f in as_completed(futs):
            try:
                r = f.result()
            except Exception as exc:
                print(f"  shard {futs[f]:06d} FAILED: {type(exc).__name__}: {exc}", flush=True)
                continue
            done += 1
            if r["skipped"]:
                skipped += 1
            else:
                total_bytes += r["bytes"]
                total_samples += r["samples"]
                total_dropped += r.get("dropped", 0)
            el = time.perf_counter() - t_start
            print(
                f"  [{done:>4}/{len(tasks)}] {r['shard']} "
                + ("skipped" if r["skipped"] else
                   f"{r['samples']:>5} samples {r['bytes'] / 2**30:.2f} GiB "
                   f"decode {r['build_s']:.1f}s upload {r['upload_s']:.1f}s [{r['device']}]")
                + f"  | {total_bytes / 2**30:.1f} GiB, {total_bytes / el / 1e9:.2f} GB/s, "
                  f"{total_samples / el:.0f} img/s",
                flush=True,
            )

    el = time.perf_counter() - t_start
    print(f"\n== done in {el / 60:.1f} min ==")
    print(f"   shards   : {done - skipped} written, {skipped} skipped")
    print(f"   samples  : {total_samples:,} ({total_dropped} dropped as unreadable/non-JPEG)")
    print(f"   bytes    : {total_bytes / 2**30:.1f} GiB")
    print(f"   rate     : {total_samples / el:.0f} img/s, {total_bytes / el / 1e9:.2f} GB/s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
