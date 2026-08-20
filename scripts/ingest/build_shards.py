#!/usr/bin/env python3
# Copyright 2026 MinIO, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Build WebDataset-style tar shards from a local ImageNet tree and upload them.

Why shards rather than one object per image
-------------------------------------------
ImageNet averages ~110 KB per JPEG. At that size an S3 GET is dominated by
per-request overhead, not by data movement, so neither HTTP nor RDMA gets near
line rate and the comparison measures request handling instead of the transport.
It is also not how anyone trains at scale. Sharding into ~1 GiB sequential
objects is both the realistic layout and the one where the transport is the thing
under test. (`bench_fetch --sizes 1` keeps the small-object case measurable, so
the limit is reported rather than hidden.)

Each shard gets a sidecar ``.index.json`` giving every sample's byte offset,
length and label — see ``s3rdma_train.shards`` for why that is what makes
decoding directly out of VRAM possible.

    python scripts/ingest/build_shards.py \
        --src /home/minio/benchmark/imagenet/ILSVRC/Data/CLS-LOC/train \
        --bucket imagenet-shards --prefix train \
        --samples-per-shard 8192 --workers 24
"""

from __future__ import annotations

import argparse
import io
import os
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from s3rdma_train.shards import ShardIndex, ShardWriter, verify_index  # noqa: E402

# Module-level config for worker processes: set once in the initializer so each
# child does not re-derive it, and so the (unpicklable) client is per-process.
_CFG: dict = {}
_CLIENT = None


def _init_worker(cfg: dict) -> None:
    global _CFG, _CLIENT
    _CFG = cfg
    from minio import Minio

    _CLIENT = Minio(
        cfg["endpoint"], cfg["access_key"], cfg["secret_key"], secure=cfg["secure"]
    )


def _build_and_upload(task: tuple[int, list[tuple[str, int]]]) -> dict:
    shard_idx, entries = task
    prefix = _CFG["prefix"]
    shard_key = f"{prefix}/{prefix}-{shard_idx:06d}.tar"
    index_key = ShardIndex.key_for(shard_key)

    if _CFG["skip_existing"]:
        try:
            _CLIENT.stat_object(_CFG["bucket"], shard_key)
            _CLIENT.stat_object(_CFG["bucket"], index_key)
            return {"shard": shard_key, "skipped": True, "samples": 0, "bytes": 0}
        except Exception:
            pass

    t0 = time.perf_counter()
    w = ShardWriter(shard_key)
    unreadable = 0
    for path, label in entries:
        try:
            with open(path, "rb") as fh:
                blob = fh.read()
        except OSError:
            unreadable += 1
            continue
        # Skip anything that is not actually a JPEG: nvJPEG would fail on it far
        # downstream, where the cause would be much harder to see.
        if len(blob) < 4 or blob[:2] != b"\xff\xd8":
            unreadable += 1
            continue
        w.add(os.path.splitext(os.path.basename(path))[0], blob, label)

    data, index = w.finish()
    build_s = time.perf_counter() - t0

    # Spot-check the offset arithmetic before uploading; a bad index would only
    # surface much later as an opaque nvJPEG decode failure.
    verify_index(data, index, checks=8)

    t1 = time.perf_counter()
    _CLIENT.put_object(
        _CFG["bucket"], shard_key, io.BytesIO(data), len(data),
        part_size=_CFG["part_size"], content_type="application/x-tar",
    )
    raw = index.to_json()
    _CLIENT.put_object(
        _CFG["bucket"], index_key, io.BytesIO(raw), len(raw),
        content_type="application/json",
    )
    upload_s = time.perf_counter() - t1

    return {
        "shard": shard_key,
        "skipped": False,
        "samples": index.num_samples,
        "bytes": index.size,
        "unreadable": unreadable,
        "build_s": build_s,
        "upload_s": upload_s,
    }


def discover(src: str) -> tuple[list[tuple[str, int]], list[str]]:
    """Return [(path, label)] and the sorted class list.

    Labels follow the conventional ImageNet ordering: WNIDs sorted
    lexicographically, so label ids match torchvision's ImageFolder on the same
    tree and results stay comparable with published numbers.
    """
    classes = sorted(d for d in os.listdir(src) if os.path.isdir(os.path.join(src, d)))
    files: list[tuple[str, int]] = []
    for label, wnid in enumerate(classes):
        d = os.path.join(src, wnid)
        with os.scandir(d) as it:
            for e in it:
                if e.is_file():
                    files.append((e.path, label))
    return files, classes


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="ImageNet split root (dir per class)")
    ap.add_argument("--bucket", default="imagenet-shards")
    ap.add_argument("--prefix", default="train")
    ap.add_argument("--endpoint", default=os.environ.get("S3_ENDPOINT", "aistor1:9000"))
    ap.add_argument("--access-key", default=os.environ.get("S3_ACCESS_KEY", "minioadmin"))
    ap.add_argument("--secret-key", default=os.environ.get("S3_SECRET_KEY", "minioadmin"))
    ap.add_argument("--samples-per-shard", type=int, default=8192)
    ap.add_argument("--max-shards", type=int, default=0, help="0 = all")
    ap.add_argument("--workers", type=int, default=16)
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
    t0 = time.perf_counter()
    files, classes = discover(args.src)
    print(f"  {len(files):,} files in {len(classes)} classes ({time.perf_counter() - t0:.1f}s)")

    # Shuffle once with a fixed seed so each shard is class-mixed. Training can then
    # read shards in any order without a global shuffle buffer, and the shuffle is
    # reproducible across rebuilds.
    random.Random(args.seed).shuffle(files)

    tasks = [
        (i, files[i * args.samples_per_shard : (i + 1) * args.samples_per_shard])
        for i in range((len(files) + args.samples_per_shard - 1) // args.samples_per_shard)
    ]
    tasks = [t for t in tasks if t[1]]
    if args.max_shards:
        tasks = tasks[: args.max_shards]

    print(f"  {len(tasks)} shards x ~{args.samples_per_shard} samples, {args.workers} workers")

    cfg = {
        "bucket": args.bucket,
        "prefix": args.prefix,
        "endpoint": args.endpoint,
        "access_key": args.access_key,
        "secret_key": args.secret_key,
        "secure": False,
        "part_size": args.part_size,
        "skip_existing": not args.no_skip_existing,
    }

    done = 0
    total_bytes = 0
    total_samples = 0
    total_unreadable = 0
    skipped = 0
    t_start = time.perf_counter()

    with ProcessPoolExecutor(
        max_workers=args.workers, initializer=_init_worker, initargs=(cfg,)
    ) as pool:
        futs = {pool.submit(_build_and_upload, t): t[0] for t in tasks}
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
                total_unreadable += r.get("unreadable", 0)
            el = time.perf_counter() - t_start
            rate = total_bytes / el / 1e9 if el > 0 else 0
            print(
                f"  [{done:>4}/{len(tasks)}] {r['shard']} "
                + ("skipped" if r["skipped"] else
                   f"{r['samples']:>6} samples {r['bytes'] / 2**30:.2f} GiB "
                   f"build {r['build_s']:.1f}s upload {r['upload_s']:.1f}s")
                + f"  | cumulative {total_bytes / 2**30:.1f} GiB, {rate:.2f} GB/s",
                flush=True,
            )

    el = time.perf_counter() - t_start
    print(f"\n== done in {el / 60:.1f} min ==")
    print(f"   shards written : {done - skipped} (skipped {skipped})")
    print(f"   samples        : {total_samples:,}")
    print(f"   bytes          : {total_bytes / 2**30:.1f} GiB")
    if total_unreadable:
        print(f"   unreadable/non-JPEG skipped: {total_unreadable}")
    if el > 0 and total_bytes:
        print(f"   ingest rate    : {total_bytes / el / 1e9:.2f} GB/s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
