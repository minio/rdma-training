#!/usr/bin/env python3
# Copyright 2026 MinIO, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Show that a standard HF safetensors shard cannot use RDMA as a whole-object GET.

Hugging Face's default sharding is ``max_shard_size="5GB"`` -- decimal, so ~4.66 GiB,
above cuObject's 4 GiB registration limit. A whole-object GET on such a shard
therefore falls back to HTTP, and it does so *silently*: the call returns a
successful byte count either way. Only the server's RDMA counters can tell you.

This is the demonstration behind minio-cpp PR #258 (ranged GET through the C ABI),
which is what lets the same shard be read as sub-4-GiB windows and actually use RDMA.

    python scripts/run/show-4gib-fallback.py                    # whole-object
    python scripts/run/show-4gib-fallback.py --ranged           # windows, for contrast

Point MINIOCPP_LIB at a pre-PR library to show that the whole-object result is not
something the PR changed (correctly -- the registration limit is real).
"""

from __future__ import annotations

import argparse
import ctypes
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import torch  # noqa: E402

from s3rdma_train.metrics import RDMAWitness  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bucket", default="model-weights")
    ap.add_argument("--key",
                    default="llama-3.1-8b-instruct/model-00002-of-00004.safetensors")
    ap.add_argument("--endpoint", default="aistor1:9000")
    ap.add_argument("--servers", default="aistor1:9000,aistor2:9000")
    ap.add_argument("--ranged", action="store_true",
                    help="read as sub-4-GiB windows instead of one whole-object GET")
    ap.add_argument("--window-mib", type=int, default=1024)
    args = ap.parse_args()

    lib_path = os.environ.get("MINIOCPP_LIB")
    if not lib_path:
        print("set MINIOCPP_LIB (source env.sh)", file=sys.stderr)
        return 2
    lib = ctypes.CDLL(lib_path)
    lib.miniocpp_client_new.argtypes = [ctypes.c_char_p] * 5 + [ctypes.c_int]
    lib.miniocpp_client_new.restype = ctypes.c_void_p
    lib.miniocpp_get_object.argtypes = [
        ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_void_p,
        ctypes.c_size_t, ctypes.c_void_p, ctypes.c_void_p,
    ]
    lib.miniocpp_get_object.restype = ctypes.c_ssize_t
    has_range = hasattr(lib, "miniocpp_get_object_range")
    if has_range:
        lib.miniocpp_get_object_range.argtypes = [
            ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_void_p,
            ctypes.c_size_t, ctypes.c_uint64,
        ]
        lib.miniocpp_get_object_range.restype = ctypes.c_ssize_t

    from minio import Minio

    size = Minio(args.endpoint, "minioadmin", "minioadmin", secure=False) \
        .stat_object(args.bucket, args.key).size
    limit = 4 * 2**30

    print(f"library      : {os.path.basename(os.path.realpath(lib_path))}")
    print(f"ranged GET   : {'available' if has_range else 'NOT available (pre-PR #258)'}")
    print(f"shard        : {args.key.split('/')[-1]}")
    print(f"size         : {size / 1e9:.2f} GB  "
          f"({'ABOVE' if size > limit else 'below'} the {limit / 2**30:.0f} GiB "
          f"registration limit)")
    print(f"mode         : {'ranged windows' if args.ranged else 'one whole-object GET'}")

    if args.ranged and not has_range:
        print("\ncannot demonstrate the ranged path: this library predates PR #258")
        return 1

    torch.cuda.set_device(0)
    buf = torch.empty(size, dtype=torch.uint8, device="cuda")
    torch.cuda.synchronize()
    h = lib.miniocpp_client_new(args.endpoint.encode(), b"", b"minioadmin",
                                b"minioadmin", b"", 0)

    servers = [s.strip() for s in args.servers.split(",") if s.strip()]
    win = args.window_mib << 20
    with RDMAWitness(servers) as w:
        t0 = time.perf_counter()
        if args.ranged:
            total = 0
            for off in range(0, size, win):
                ln = min(win, size - off)
                rc = lib.miniocpp_get_object_range(
                    ctypes.c_void_p(h), args.bucket.encode(), args.key.encode(),
                    ctypes.c_void_p(buf.data_ptr() + off), ln, ctypes.c_uint64(off))
                if rc < 0:
                    print(f"window at {off} failed: rc={rc}")
                    return 1
                total += rc
            rc = total
        else:
            rc = lib.miniocpp_get_object(
                ctypes.c_void_p(h), args.bucket.encode(), args.key.encode(),
                ctypes.c_void_p(buf.data_ptr()), size, None, None)
        dt = time.perf_counter() - t0

    got = w.delta.read_bytes or 0
    share = 100 * got / size if size else 0
    transport = "RDMA" if got >= size * 0.5 else "HTTP (silent fallback)"
    print(f"\nresult       : rc={rc:,} ({'ok' if rc == size else 'MISMATCH'})  "
          f"{dt:.2f}s  {size / dt / 1e9:.2f} GB/s")
    print(f"server counted: {int(got):,} RDMA bytes ({share:.0f}% of payload)")
    print(f"transport    : {transport}")
    if not args.ranged and got == 0:
        print("\n-> A whole-object GET of a standard HF shard cannot use RDMA, and says\n"
              "   nothing about it. Re-run with --ranged to see the same shard at 100%.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
