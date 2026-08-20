#!/usr/bin/env python3
# Copyright 2026 MinIO, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Upload a Hugging Face model's safetensors shards to object storage.

Reads from a local snapshot directory (including a HF cache snapshot, whose files
are symlinks into `blobs/` — those are dereferenced), or downloads the repo first
if you pass --repo-id and have `huggingface_hub` plus credentials.

Uploaded as-is: shard names and the index file are preserved, so what lands in the
bucket is exactly the standard HF layout an inference server expects. That matters
for the benchmark to mean anything — including the fact that HF's default
`max_shard_size="5GB"` produces shards *above* cuObject's 4 GiB registration limit.

    python scripts/ingest/upload_model_weights.py \
        --snapshot ~/.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/<rev> \
        --bucket model-weights --prefix llama-3.1-8b-instruct
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# Files an inference server needs alongside the weights.
SIDECAR_NAMES = {
    "config.json", "generation_config.json", "tokenizer.json",
    "tokenizer_config.json", "special_tokens_map.json", "tokenizer.model",
    "model.safetensors.index.json", "chat_template.jinja",
}


def resolve_snapshot(repo_id: str | None, snapshot: str | None, token: str | None) -> str:
    if snapshot:
        return os.path.realpath(os.path.expanduser(snapshot))
    if not repo_id:
        raise SystemExit("pass --snapshot or --repo-id")
    from huggingface_hub import snapshot_download

    print(f"downloading {repo_id} from Hugging Face ...", flush=True)
    return snapshot_download(
        repo_id=repo_id, token=token,
        allow_patterns=["*.safetensors", "*.json", "tokenizer.model", "*.jinja"],
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", help="local snapshot dir (HF cache snapshot is fine)")
    ap.add_argument("--repo-id", help="download this repo first (needs huggingface_hub)")
    ap.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"))
    ap.add_argument("--bucket", default="model-weights")
    ap.add_argument("--prefix", required=True, help="e.g. llama-3.1-8b-instruct")
    ap.add_argument("--endpoint", default=os.environ.get("S3_ENDPOINT", "aistor1:9000"))
    ap.add_argument("--access-key", default=os.environ.get("S3_ACCESS_KEY", "minioadmin"))
    ap.add_argument("--secret-key", default=os.environ.get("S3_SECRET_KEY", "minioadmin"))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--part-size", type=int, default=64 << 20)
    ap.add_argument("--no-skip-existing", action="store_true")
    args = ap.parse_args()

    snap = resolve_snapshot(args.repo_id, args.snapshot, args.hf_token)
    if not os.path.isdir(snap):
        raise SystemExit(f"not a directory: {snap}")

    from minio import Minio

    admin = Minio(args.endpoint, args.access_key, args.secret_key, secure=False)
    if not admin.bucket_exists(args.bucket):
        admin.make_bucket(args.bucket)
        print(f"created bucket {args.bucket}")

    files = []
    for name in sorted(os.listdir(snap)):
        path = os.path.join(snap, name)
        if not os.path.isfile(path) and not os.path.islink(path):
            continue
        if name.endswith(".safetensors") or name in SIDECAR_NAMES:
            # HF cache entries are symlinks into blobs/; stat the target.
            files.append((name, os.path.realpath(path), os.stat(path).st_size))

    weights = [f for f in files if f[0].endswith(".safetensors")]
    if not weights:
        raise SystemExit(f"no .safetensors found in {snap}")
    total = sum(f[2] for f in weights)
    biggest = max(f[2] for f in weights)

    print(f"source : {snap}")
    print(f"target : {args.bucket}/{args.prefix}/")
    print(f"weights: {len(weights)} shards, {total / 1e9:.2f} GB "
          f"(largest {biggest / 1e9:.2f} GB, "
          f"{'ABOVE' if biggest > 4 * 2**30 else 'below'} the 4 GiB RDMA registration limit)")
    print(f"sidecar: {len(files) - len(weights)} files")

    def upload(item) -> tuple[str, int, float, bool]:
        name, path, size = item
        key = f"{args.prefix}/{name}"
        client = Minio(args.endpoint, args.access_key, args.secret_key, secure=False)
        if not args.no_skip_existing:
            try:
                if client.stat_object(args.bucket, key).size == size:
                    return name, size, 0.0, True
            except Exception:
                pass
        t0 = time.perf_counter()
        with open(path, "rb") as fh:
            client.put_object(args.bucket, key, fh, size, part_size=args.part_size)
        return name, size, time.perf_counter() - t0, False

    done = uploaded_bytes = skipped = 0
    t_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(upload, f): f[0] for f in files}
        for fut in as_completed(futs):
            name, size, dt, was_skipped = fut.result()
            done += 1
            if was_skipped:
                skipped += 1
            else:
                uploaded_bytes += size
            el = time.perf_counter() - t_start
            print(f"  [{done:>3}/{len(files)}] {name:<44} "
                  + ("skipped" if was_skipped else
                     f"{size / 1e9:6.2f} GB in {dt:5.1f}s")
                  + f"  | {uploaded_bytes / max(el, 1e-9) / 1e9:.2f} GB/s", flush=True)

    el = time.perf_counter() - t_start
    print(f"\n== done in {el:.1f}s: {done - skipped} uploaded "
          f"({uploaded_bytes / 1e9:.2f} GB), {skipped} skipped ==")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
