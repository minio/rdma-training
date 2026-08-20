#!/usr/bin/env python3
# Copyright 2026 MinIO, Inc.
# SPDX-License-Identifier: Apache-2.0

"""B8 — inference cold start: model weights from object storage into VRAM.

This is the strongest non-checkpoint case for GPU-Direct RDMA, because safetensors
is already the right shape: header + one contiguous tensor blob. The weights go
from the object store into VRAM with no host memory and no deserialisation.

Four paths are measured, so the comparison is against what people actually run and
not only against a hand-optimised one:

``rdma``
    Ranged RDMA GETs of the tensor blob straight into a device buffer, then tensors
    as views into it. No host memory touched.

``http``
    Same structure over TCP: ranged GETs into pinned host memory, one H2D copy.

``download-then-load``
    What an inference server does today on a cold node: pull the shards to local
    disk, then ``safetensors.torch.load_file`` (mmap) and copy each tensor to the
    GPU. Includes the download, because on a cold node you pay it.

``local-then-load``
    The same minus the download, i.e. weights already cached on local NVMe. This is
    the *best case available today* and the most honest bar to clear.

    python -m s3rdma_train.bench_weights --model llama-3.1-8b-instruct \
        --backends rdma,http,download-then-load,local-then-load

Correctness is checked, not assumed: with ``--verify`` a sample of tensors is
compared element-wise against ``safetensors.torch.load_file`` on the original file.
"""

from __future__ import annotations

import argparse
import os
import shutil
import statistics
import sys
import tempfile
import time

import torch

from . import weights as W
from .metrics import CPUSampler, RDMAWitness, RunRecord, gpu_info
from .s3 import StoreConfig, make_store


def free_vram(device: str) -> None:
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)


def load_via_store(store, prefix: str, device: str, window_mib: int,
                   concurrency: int, servers: list[str]) -> dict:
    """rdma / http: object store -> VRAM, tensors as views."""
    free_vram(device)
    with RDMAWitness(servers) as w, CPUSampler() as cpu:
        t0 = time.perf_counter()
        tensors, buffers, info = W.load_model(
            store, prefix, device=device,
            window_bytes=window_mib << 20, concurrency=concurrency,
        )
        torch.cuda.synchronize()
        wall = time.perf_counter() - t0
    out = {
        "wall_s": wall,
        "bytes": info["bytes"],
        "gbps": info["bytes"] / wall / 1e9,
        "num_tensors": info["num_tensors"],
        "shards": info["shards"],
        "windows_total": sum(s["windows"] for s in info["per_shard"]),
        **cpu.as_dict(),
        "rdma_witness": w.as_dict(),
        "peak_vram_bytes": torch.cuda.max_memory_allocated(device),
    }
    return out, tensors, buffers


def load_download_then_load(store, prefix: str, device: str,
                            keep_local: str | None = None) -> dict:
    """The cold-node path: S3 -> local disk -> mmap -> per-tensor H2D copy."""
    from safetensors.torch import load_file

    free_vram(device)
    tmp = keep_local or tempfile.mkdtemp(prefix="coldstart-")
    made_tmp = keep_local is None
    try:
        keys = sorted(k for k, _ in store.list(prefix) if k.endswith(".safetensors"))
        with CPUSampler() as cpu:
            t0 = time.perf_counter()
            # 1. download
            t_dl0 = time.perf_counter()
            local = []
            for k in keys:
                dest = os.path.join(tmp, os.path.basename(k))
                if not (os.path.exists(dest) and os.path.getsize(dest) == store.stat(k)):
                    data = store.get_bytes(k)
                    with open(dest, "wb") as fh:
                        fh.write(data)
                    del data
                local.append(dest)
            t_download = time.perf_counter() - t_dl0

            # 2. mmap + copy to GPU, which is what load_file(device=...) does
            t_ld0 = time.perf_counter()
            tensors: dict[str, torch.Tensor] = {}
            nbytes = 0
            for path in local:
                d = load_file(path, device=device)
                for name, t in d.items():
                    nbytes += t.numel() * t.element_size()
                tensors.update(d)
            torch.cuda.synchronize()
            t_load = time.perf_counter() - t_ld0
            wall = time.perf_counter() - t0
        return {
            "wall_s": wall,
            "download_s": t_download,
            "load_s": t_load,
            "bytes": nbytes,
            "gbps": nbytes / wall / 1e9,
            "num_tensors": len(tensors),
            "shards": len(keys),
            **cpu.as_dict(),
            "peak_vram_bytes": torch.cuda.max_memory_allocated(device),
        }, tensors, None
    finally:
        if made_tmp:
            shutil.rmtree(tmp, ignore_errors=True)


def load_local_then_load(local_dir: str, device: str) -> dict:
    """Best case today: weights already on local NVMe, mmap + H2D."""
    from safetensors.torch import load_file

    free_vram(device)
    paths = sorted(
        os.path.join(local_dir, f) for f in os.listdir(local_dir)
        if f.endswith(".safetensors")
    )
    if not paths:
        raise RuntimeError(f"no .safetensors in {local_dir}")
    with CPUSampler() as cpu:
        t0 = time.perf_counter()
        tensors: dict[str, torch.Tensor] = {}
        nbytes = 0
        for p in paths:
            d = load_file(p, device=device)
            for _n, t in d.items():
                nbytes += t.numel() * t.element_size()
            tensors.update(d)
        torch.cuda.synchronize()
        wall = time.perf_counter() - t0
    return {
        "wall_s": wall,
        "bytes": nbytes,
        "gbps": nbytes / wall / 1e9,
        "num_tensors": len(tensors),
        "shards": len(paths),
        **cpu.as_dict(),
        "peak_vram_bytes": torch.cuda.max_memory_allocated(device),
    }, tensors, None


def verify(tensors: dict, local_dir: str, sample: int = 12) -> dict:
    """Compare a sample of loaded tensors against safetensors' own reader."""
    from safetensors.torch import load_file

    paths = sorted(
        os.path.join(local_dir, f) for f in os.listdir(local_dir)
        if f.endswith(".safetensors")
    )
    checked, mismatched = 0, []
    for p in paths:
        ref = load_file(p, device="cpu")
        names = sorted(ref)
        # Spread the sample across the shard rather than taking a prefix.
        step = max(1, len(names) // max(1, sample // max(1, len(paths))))
        for name in names[::step]:
            if name not in tensors:
                mismatched.append(f"{name}: missing")
                continue
            got, want = tensors[name], ref[name]
            if got.shape != want.shape or got.dtype != want.dtype:
                mismatched.append(
                    f"{name}: {tuple(got.shape)}/{got.dtype} vs "
                    f"{tuple(want.shape)}/{want.dtype}")
                continue
            # float8 has no CPU comparison kernels; compare the raw bytes.
            a, b = got.cpu(), want
            if a.dtype in (getattr(torch, "float8_e4m3fn", None),
                           getattr(torch, "float8_e5m2", None)):
                a, b = a.view(torch.uint8), b.view(torch.uint8)
            if not torch.equal(a, b):
                mismatched.append(f"{name}: values differ")
            checked += 1
        del ref
    return {"ok": not mismatched, "tensors_checked": checked,
            "mismatched": mismatched[:5]}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="B8: inference cold start")
    ap.add_argument("--model", required=True, help="prefix in the bucket")
    ap.add_argument("--bucket", default="model-weights")
    ap.add_argument("--backends", default="rdma,http,download-then-load,local-then-load")
    ap.add_argument("--endpoints", default="aistor1:9000,aistor2:9000")
    ap.add_argument("--servers", default="aistor1:9000,aistor2:9000")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--window-mib", type=int, default=1024,
                    help="ranged-read window; must stay under the 4 GiB RDMA limit")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--local-dir", default="",
                    help="snapshot dir for local-then-load and --verify")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--tag", default="")
    args = ap.parse_args(argv)

    backends = [b.strip() for b in args.backends.split(",") if b.strip()]
    servers = [s.strip() for s in args.servers.split(",") if s.strip()]
    cfg = StoreConfig(bucket=args.bucket)
    cfg.endpoints = [e.strip() for e in args.endpoints.split(",") if e.strip()]
    cfg.endpoint = cfg.endpoints[0]
    torch.cuda.set_device(torch.device(args.device))

    print(f"== B8 inference cold start | model={args.model} ==")
    print(f"   device={args.device} window={args.window_mib} MiB "
          f"concurrency={args.concurrency} repeats={args.repeats}")

    # Report the shard geometry, since the 4 GiB limit is the whole reason ranged
    # reads are needed here.
    probe = make_store("http", cfg)
    shard_sizes = sorted((sz for k, sz in probe.list(args.model)
                          if k.endswith(".safetensors")), reverse=True)
    probe.close()
    print(f"   {len(shard_sizes)} shards, {sum(shard_sizes) / 1e9:.2f} GB, "
          f"largest {shard_sizes[0] / 1e9:.2f} GB "
          f"({'ABOVE' if shard_sizes[0] > 4 * 2**30 else 'below'} the 4 GiB "
          f"RDMA registration limit)")

    hdr = (f"{'backend':>20} {'wall s':>9} {'GB/s':>8} {'cores':>7} "
           f"{'tensors':>8} {'verified':>9}")
    print("\n" + hdr)
    print("-" * len(hdr))

    rows = []
    for backend in backends:
        runs, verified = [], None
        err = None
        for r in range(args.repeats):
            try:
                if backend in ("rdma", "http"):
                    store = make_store(backend, cfg)
                    try:
                        if backend == "rdma" and not store.supports_range \
                                and shard_sizes[0] > args.window_mib << 20:
                            raise RuntimeError(
                                "shards exceed the window and this libminiocpp has no "
                                "miniocpp_get_object_range (minio-cpp #258)")
                        res, tensors, buffers = load_via_store(
                            store, args.model, args.device, args.window_mib,
                            args.concurrency, servers)
                    finally:
                        store.close()
                elif backend == "download-then-load":
                    store = make_store("http", cfg)
                    try:
                        res, tensors, buffers = load_download_then_load(
                            store, args.model, args.device)
                    finally:
                        store.close()
                elif backend == "local-then-load":
                    if not args.local_dir:
                        raise RuntimeError("--local-dir is required for local-then-load")
                    res, tensors, buffers = load_local_then_load(
                        args.local_dir, args.device)
                else:
                    raise ValueError(f"unknown backend {backend!r}")
            except Exception as exc:
                err = f"{type(exc).__name__}: {exc}"
                break

            if r == 0 and args.verify and args.local_dir:
                verified = verify(tensors, args.local_dir)
            runs.append(res)
            del tensors, buffers
            free_vram(args.device)

        if err or not runs:
            print(f"{backend:>20}   FAILED: {(err or 'no runs')[:70]}")
            rows.append({"backend": backend, "error": err})
            continue

        walls = [x["wall_s"] for x in runs]
        med = statistics.median(walls)
        best = min(runs, key=lambda x: x["wall_s"])
        row = {"backend": backend, "runs": runs, "verified": verified,
               "wall_s_median": med, "wall_s_min": min(walls), "wall_s_max": max(walls),
               "gbps_median": statistics.median(x["gbps"] for x in runs),
               "cores_median": statistics.median(x["cpu_cores_used"] for x in runs)}

        # Proof-of-RDMA, as everywhere in this repo.
        if backend == "rdma":
            got = runs[0]["rdma_witness"]["rdma_read_bytes"] or 0
            if got < runs[0]["bytes"] * 0.5:
                row["error"] = "rdma-not-used"
                print(f"{backend:>20}   FAILED: RDMA not used "
                      f"(server counted {got:,.0f} of {runs[0]['bytes']:,})")
                rows.append(row)
                continue

        vtxt = ("yes" if verified and verified["ok"]
                else ("NO" if verified else "-"))
        print(f"{backend:>20} {med:>9.2f} {row['gbps_median']:>8.2f} "
              f"{row['cores_median']:>7.2f} {best['num_tensors']:>8} {vtxt:>9}",
              flush=True)
        rows.append(row)

    ok = [r for r in rows if "error" not in r]
    base = next((r for r in ok if r["backend"] == "local-then-load"), None)
    if base and len(ok) > 1:
        print("\n   speedup vs local-then-load (weights already on local NVMe):")
        for r in ok:
            if r["backend"] == "local-then-load":
                continue
            print(f"     {r['backend']:<22} {base['wall_s_median'] / r['wall_s_median']:>6.2f}x")

    rec = RunRecord(
        benchmark="b8-weights" + (f"-{args.tag}" if args.tag else ""),
        backend=",".join(backends),
        params={"model": args.model, "bucket": args.bucket,
                "window_mib": args.window_mib, "concurrency": args.concurrency,
                "repeats": args.repeats, "device": args.device,
                "endpoints": cfg.endpoints,
                "shard_bytes": shard_sizes},
        results={"rows": rows},
        environment={"gpus": gpu_info(), "torch": torch.__version__},
    )
    print(f"\nsaved: {rec.save(args.results_dir)}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
