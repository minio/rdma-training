#!/usr/bin/env python3
# Copyright 2026 MinIO, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Build tokenized int32 shards — the LLM pretraining data layout.

Purpose: to *document* that the LLM pretraining data path is not a storage
problem, rather than to accelerate it. The arithmetic says so plainly, and this
dataset lets us show it with a measurement instead of an assertion:

    LLM pretraining reads ~2-4 bytes per token and spends ~6N FLOPs on it.
    A dense 8B model on 8x H200 at ~45% MFU consumes ~75,000 tokens/s
      -> ~0.3 MB/s of storage.
    A DeepSeek-V3-class MoE (37B active) is even lower per GPU-second.
    For contrast, ResNet-50 on the same 8 GPUs needs ~4,800 MB/s.

So an LLM asks for roughly four orders of magnitude less storage bandwidth than
an image model on identical hardware, and reads it in long contiguous runs rather
than as many small requests — low bytes *and* low IOPS. Anyone claiming a
transport win on the LLM data path should be asked which number they are
improving.

Token values here are synthetic (uniform over the vocab). That is deliberate and
sufficient: this measures byte volume, request pattern and layout, none of which
depend on what the tokens mean. Nothing about model quality is claimed.

Layout, matching the Megatron / litgpt `.bin` convention closely enough to be
representative:

    <prefix>-000000.bin    seqs x seq_len int32 token ids, contiguous, no header
    <prefix>-000000.json   {"sequences", "seq_len", "dtype", "vocab_size", ...}

Fixed-size sequences mean no index: sequence i starts at i * seq_len * 4.

    python scripts/ingest/build_token_shards.py --bucket llm-tokens --shards 16
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))


def build_one(cfg: dict, idx: int) -> dict:
    from minio import Minio

    client = Minio(cfg["endpoint"], cfg["access_key"], cfg["secret_key"], secure=False)
    prefix = cfg["prefix"]
    key = f"{prefix}/{prefix}-{idx:06d}.bin"
    meta_key = f"{prefix}/{prefix}-{idx:06d}.json"

    if cfg["skip_existing"]:
        try:
            client.stat_object(cfg["bucket"], key)
            client.stat_object(cfg["bucket"], meta_key)
            return {"key": key, "skipped": True, "bytes": 0, "sequences": 0}
        except Exception:
            pass

    seqs, seq_len = cfg["sequences_per_shard"], cfg["seq_len"]
    t0 = time.perf_counter()
    rng = np.random.default_rng(cfg["seed"] + idx)
    # int32 is what a >65535 vocab forces (Llama-3 is 128,256), and it is the
    # pessimistic choice for a bandwidth argument: uint16 would halve the volume.
    tokens = rng.integers(0, cfg["vocab_size"], size=seqs * seq_len, dtype=np.int32)
    payload = tokens.tobytes()
    build_s = time.perf_counter() - t0

    t1 = time.perf_counter()
    client.put_object(cfg["bucket"], key, io.BytesIO(payload), len(payload),
                      part_size=cfg["part_size"],
                      content_type="application/octet-stream")
    meta = {
        "sequences": seqs, "seq_len": seq_len, "dtype": "int32",
        "bytes_per_token": 4, "vocab_size": cfg["vocab_size"],
        "sequence_bytes": seq_len * 4, "synthetic": True,
    }
    raw = json.dumps(meta, separators=(",", ":")).encode()
    client.put_object(cfg["bucket"], meta_key, io.BytesIO(raw), len(raw),
                      content_type="application/json")
    upload_s = time.perf_counter() - t1

    return {"key": key, "skipped": False, "bytes": len(payload), "sequences": seqs,
            "build_s": build_s, "upload_s": upload_s}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bucket", default="llm-tokens")
    ap.add_argument("--prefix", default="train")
    ap.add_argument("--endpoint", default=os.environ.get("S3_ENDPOINT", "aistor1:9000"))
    ap.add_argument("--access-key", default=os.environ.get("S3_ACCESS_KEY", "minioadmin"))
    ap.add_argument("--secret-key", default=os.environ.get("S3_SECRET_KEY", "minioadmin"))
    ap.add_argument("--seq-len", type=int, default=4096)
    ap.add_argument("--sequences-per-shard", type=int, default=65536,
                    help="65536 x 4096 x 4B = 1 GiB per shard")
    ap.add_argument("--shards", type=int, default=16)
    ap.add_argument("--vocab-size", type=int, default=128256, help="Llama-3 vocab")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--part-size", type=int, default=64 << 20)
    ap.add_argument("--no-skip-existing", action="store_true")
    args = ap.parse_args()

    from minio import Minio

    admin = Minio(args.endpoint, args.access_key, args.secret_key, secure=False)
    if not admin.bucket_exists(args.bucket):
        admin.make_bucket(args.bucket)
        print(f"created bucket {args.bucket}")

    shard_bytes = args.sequences_per_shard * args.seq_len * 4
    print(f"{args.shards} shards x {args.sequences_per_shard} seqs x {args.seq_len} tok "
          f"x 4B = {shard_bytes / 2**30:.2f} GiB per shard, "
          f"{args.shards * shard_bytes / 2**30:.1f} GiB total")
    print(f"tokens: {args.shards * args.sequences_per_shard * args.seq_len / 1e9:.2f}B "
          f"(synthetic, uniform over vocab {args.vocab_size})")

    cfg = {
        "bucket": args.bucket, "prefix": args.prefix, "endpoint": args.endpoint,
        "access_key": args.access_key, "secret_key": args.secret_key,
        "seq_len": args.seq_len, "sequences_per_shard": args.sequences_per_shard,
        "vocab_size": args.vocab_size, "seed": args.seed,
        "part_size": args.part_size, "skip_existing": not args.no_skip_existing,
    }

    total_bytes = done = skipped = 0
    t_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(build_one, cfg, i): i for i in range(args.shards)}
        for f in as_completed(futs):
            r = f.result()
            done += 1
            if r["skipped"]:
                skipped += 1
            else:
                total_bytes += r["bytes"]
            el = time.perf_counter() - t_start
            print(f"  [{done:>3}/{args.shards}] {r['key']} "
                  + ("skipped" if r["skipped"] else
                     f"{r['bytes'] / 2**30:.2f} GiB gen {r['build_s']:.1f}s "
                     f"upload {r['upload_s']:.1f}s")
                  + f"  | {total_bytes / max(el, 1e-9) / 1e9:.2f} GB/s", flush=True)

    el = time.perf_counter() - t_start
    print(f"\n== done in {el:.1f}s: {done - skipped} written, {skipped} skipped, "
          f"{total_bytes / 2**30:.1f} GiB ==")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
