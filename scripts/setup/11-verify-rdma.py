#!/usr/bin/env python3
# Copyright 2026 MinIO, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Verify GPU-Direct S3-over-RDMA end to end, with server-side proof.

This is the gate for everything else in this project. It answers three questions
that must all be "yes" before any benchmark number means anything:

  1. Does the client library load and see a cuObjServer?
  2. Can we PUT from CUDA device memory and GET back into CUDA device memory,
     byte-exact?
  3. Did the *server* actually account those bytes as RDMA -- or did the client
     silently fall back to HTTP and hand us a plausible-looking success?

Question 3 is the one that matters. libminiocpp's buffer path is documented to
"attempt RDMA with HTTP fallback", so a green round-trip proves nothing on its
own. We read minio_api_rdma_{read,write}_bytes_total from every node and require
the counters to move.

    python scripts/setup/11-verify-rdma.py [--size-mb 64]
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import torch  # noqa: E402

from s3rdma_train import rdma_client  # noqa: E402
from s3rdma_train.metrics import RDMAWitness  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", default=os.environ.get("S3_ENDPOINT", "aistor1:9000"))
    ap.add_argument("--access-key", default=os.environ.get("S3_ACCESS_KEY", "minioadmin"))
    ap.add_argument("--secret-key", default=os.environ.get("S3_SECRET_KEY", "minioadmin"))
    ap.add_argument("--bucket", default="rdma-verify")
    ap.add_argument("--size-mb", type=int, default=64)
    ap.add_argument(
        "--servers",
        default=os.environ.get("S3_METRICS_SERVERS", "aistor1:9000,aistor2:9000"),
        help="comma-separated nodes to scrape RDMA counters from",
    )
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    servers = [s.strip() for s in args.servers.split(",") if s.strip()]
    nbytes = args.size_mb * 1024 * 1024
    key = f"verify-{args.size_mb}mib.bin"
    ok = True

    print("=" * 72)
    print("GPU-Direct S3-over-RDMA verification")
    print("=" * 72)

    # ---------------------------------------------------------------- step 1 ---
    print("\n[1] client library")
    # Load explicitly rather than via is_available(), which swallows
    # RDMANotAvailable and returns False -- that would hide the diagnostic that
    # actually tells you what to fix.
    try:
        lib = rdma_client._load()
        avail = lib.miniocpp_rdma_available() != 0
    except rdma_client.RDMANotAvailable as exc:
        print(f"    FAIL: {exc}")
        return 1
    print(f"    libminiocpp:      {rdma_client.library_path()}")
    print(f"    rdma_available(): {avail}")
    if not avail:
        print("    FAIL: cuObjClient is not connected to a cuObjServer.")
        print("          The server must be the minio.rdma build and reachable over RoCE.")
        return 1

    # ---------------------------------------------------------------- step 2 ---
    print(f"\n[2] CUDA device memory ({args.device})")
    if not torch.cuda.is_available():
        print("    FAIL: torch reports no CUDA devices")
        return 1
    dev = torch.device(args.device)
    torch.cuda.set_device(dev)
    print(f"    device: {torch.cuda.get_device_name(dev)}")

    # A deterministic, non-trivial pattern: all-zeros would pass even if the
    # transfer silently did nothing, and a constant would hide byte offsets.
    gen = torch.Generator(device=dev).manual_seed(1234)
    src = torch.randint(0, 256, (nbytes,), dtype=torch.uint8, device=dev, generator=gen)
    dst = torch.zeros(nbytes, dtype=torch.uint8, device=dev)
    torch.cuda.synchronize()
    print(f"    src: {nbytes / 2**20:.0f} MiB @ 0x{src.data_ptr():x}")
    print(f"    dst: {nbytes / 2**20:.0f} MiB @ 0x{dst.data_ptr():x}")

    # ---------------------------------------------------------------- step 3 ---
    print(f"\n[3] bucket {args.bucket}")
    try:
        from minio import Minio

        admin = Minio(args.endpoint, args.access_key, args.secret_key, secure=False)
        if not admin.bucket_exists(args.bucket):
            admin.make_bucket(args.bucket)
            print("    created")
        else:
            print("    exists")
    except Exception as exc:
        print(f"    FAIL: could not ensure bucket: {exc}")
        return 1

    client = rdma_client.RDMAClient(args.endpoint, args.access_key, args.secret_key)

    # ---------------------------------------------------------------- step 4 ---
    print(f"\n[4] PUT {nbytes / 2**20:.0f} MiB straight from VRAM")
    with RDMAWitness(servers) as w_put:
        n, etag, checksum = client.put(args.bucket, key, src.data_ptr(), nbytes)
    print(f"    bytes={n}  etag={etag}  crc64nvme={checksum}")
    print(f"    server RDMA write delta: {w_put.delta.write_bytes:,.0f} B "
          f"in {w_put.delta.write_ops:,.0f} ops")
    if n != nbytes:
        print(f"    FAIL: wrote {n} of {nbytes}")
        ok = False
    try:
        w_put.require(write_bytes=nbytes)
        print("    PROOF: server accounted these bytes as RDMA")
    except RuntimeError as exc:
        print(f"    FAIL: {exc}")
        ok = False

    # ---------------------------------------------------------------- step 5 ---
    print(f"\n[5] GET {nbytes / 2**20:.0f} MiB straight into VRAM")
    with RDMAWitness(servers) as w_get:
        n = client.get(args.bucket, key, dst.data_ptr(), nbytes)
    print(f"    bytes={n}")
    print(f"    server RDMA read delta:  {w_get.delta.read_bytes:,.0f} B "
          f"in {w_get.delta.read_ops:,.0f} ops")
    if n != nbytes:
        print(f"    FAIL: read {n} of {nbytes}")
        ok = False
    try:
        w_get.require(read_bytes=nbytes)
        print("    PROOF: server accounted these bytes as RDMA")
    except RuntimeError as exc:
        print(f"    FAIL: {exc}")
        ok = False

    # ---------------------------------------------------------------- step 6 ---
    print("\n[6] integrity")
    torch.cuda.synchronize()
    if torch.equal(src, dst):
        print(f"    OK: {nbytes:,} bytes identical (compared on device)")
    else:
        ndiff = int((src != dst).sum().item())
        print(f"    FAIL: {ndiff:,} of {nbytes:,} bytes differ")
        ok = False

    client.close()

    print("\n" + "=" * 72)
    print("RESULT:", "PASS - GPU-Direct RDMA is live" if ok else "FAIL")
    print("=" * 72)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
