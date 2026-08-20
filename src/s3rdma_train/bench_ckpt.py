#!/usr/bin/env python3
# Copyright 2026 MinIO, Inc.
# SPDX-License-Identifier: Apache-2.0

"""B4/B5 — checkpoint save and restore, HTTP vs GPU-Direct RDMA.

Checkpointing is where S3-over-RDMA applies unconditionally: a checkpoint is
tensors, so it is GPU-native by construction — unlike a JPEG data path, which
cannot use RDMA at all (see report 02). It is also the operation that stalls
training, so seconds here are seconds of idle GPUs.

Three methods are measured, and the third exists so we are not beating a strawman:

``torch-save`` (HTTP)
    What people actually write: ``torch.save(state, BytesIO)`` then ``put_object``.
    Pickles tensor-by-tensor, so every tensor is copied D2H individually and the
    payload is walked several times in host memory before any byte leaves.

``flat`` (HTTP)
    Tuned HTTP: pack the whole checkpoint into one contiguous buffer, copy it to
    pinned host memory once, then multipart PUT. This is roughly the best a
    careful engineer does without RDMA, and it is what the RDMA number is compared
    against in the headline.

``flat`` (RDMA)
    Identical packing, except the buffer stays in VRAM and is PUT straight from
    the device pointer. No host memory is touched at all.

Sizes cover both the model in question and the sizes customers actually care
about: a ResNet-50 checkpoint is ~300 MB with optimizer state, whereas the reason
anyone asks about checkpoint bandwidth is a model two or three orders of magnitude
larger.

    python -m s3rdma_train.bench_ckpt --backends rdma,http --sizes-gib 1,4,16,64
    python -m s3rdma_train.bench_ckpt --resnet50
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time

import torch

from . import checkpoint as ckpt
from .metrics import CPUSampler, RDMAWitness, RunRecord, gpu_info, rdma_hw_counters
from .s3 import StoreConfig, make_store


def synthetic_state(total_bytes: int, device: str, n_tensors: int = 64,
                    dtype: torch.dtype = torch.bfloat16) -> dict:
    """A checkpoint-shaped state dict of approximately ``total_bytes``.

    Many tensors of mixed sizes rather than one huge one: a real checkpoint is
    hundreds or thousands of separate parameters, and per-tensor overhead is part
    of what distinguishes the packing strategies. bf16 because that is what large
    models are actually stored in.
    """
    esz = torch.tensor([], dtype=dtype).element_size()

    # Solve for the per-tensor element count so the total actually lands on
    # total_bytes. Each entry contributes its weight AND an equal-sized optimizer
    # moment, and every 4th weight is half size, so a naive
    # total_bytes/n_tensors/esz overshoots by ~1.75x.
    shape_factor = sum(1.0 if i % 4 else 0.5 for i in range(n_tensors))
    per = max(1, int(total_bytes / (2 * shape_factor * esz)))

    state: dict = {"model": {}, "optimizer": {"state": {}, "param_groups": [{"lr": 0.1}]},
                   "step": 12345, "epoch": 7}
    for i in range(n_tensors):
        # Vary shapes so the layout is not artificially uniform.
        n = per if i % 4 else max(1, per // 2)
        state["model"][f"layer{i}.weight"] = torch.empty(n, dtype=dtype, device=device).normal_()
        # Optimizer moments: the reason a checkpoint is 2-3x the model size.
        state["optimizer"]["state"][i] = {
            "exp_avg": torch.zeros(n, dtype=dtype, device=device),
            "step": float(i),
        }
    torch.cuda.synchronize()
    return state


def resnet50_state(device: str) -> tuple[dict, dict]:
    """A genuine ResNet-50 + SGD-momentum checkpoint."""
    import torchvision

    model = torchvision.models.resnet50().to(device).to(memory_format=torch.channels_last)
    opt = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
    # One step so momentum buffers exist; an empty optimizer state would understate
    # a real checkpoint by ~half.
    x = torch.randn(8, 3, 224, 224, device=device).to(memory_format=torch.channels_last)
    loss = model(x).square().mean()
    loss.backward()
    opt.step()
    opt.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    state = {"model": model.state_dict(), "optimizer": opt.state_dict(), "step": 1, "epoch": 0}
    nparams = sum(p.numel() for p in model.parameters())
    return state, {"parameters": nparams}


def _fabric(devs=("mlx5_0", "mlx5_1")) -> dict:
    return {d: rdma_hw_counters(d) for d in devs}


def _fabric_delta(a: dict, b: dict) -> dict:
    keys = ("out_of_buffer", "packet_seq_err", "local_ack_timeout_err",
            "rnr_nak_retry_err", "req_cqe_error", "resp_cqe_error")
    return {d: {k: b[d].get(k, 0) - a[d].get(k, 0) for k in keys if k in b.get(d, {})} for d in b}


def bench_save(store, key: str, state, method: str, servers: list[str], repeats: int,
               chunk_bytes: int, concurrency: int) -> dict:
    """Time repeated saves. Returns aggregate + per-iteration detail."""
    runs = []
    for r in range(repeats):
        f0 = _fabric()
        with RDMAWitness(servers) as w, CPUSampler() as cpu:
            t0 = time.perf_counter()
            if method == "flat":
                detail = ckpt.save_flat(store, f"{key}-{r}", state,
                                        chunk_bytes=chunk_bytes, concurrency=concurrency)
            elif method == "torch-save":
                detail = ckpt.save_torch_save(store, f"{key}-{r}", state)
            else:
                raise ValueError(f"unknown method {method!r}")
            wall = time.perf_counter() - t0
        runs.append({
            "wall_s": wall,
            "throughput_gbps": detail["bytes"] / wall / 1e9,
            **detail,
            **cpu.as_dict(),
            "rdma_witness": w.as_dict(),
            "fabric_health": _fabric_delta(f0, _fabric()),
        })
    return _aggregate(runs, "save")


def bench_load(store, key: str, method: str, device: str, servers: list[str],
               repeats: int, concurrency: int, reference=None) -> dict:
    runs = []
    for r in range(repeats):
        k = f"{key}-{min(r, repeats - 1)}"
        f0 = _fabric()
        with RDMAWitness(servers) as w, CPUSampler() as cpu:
            t0 = time.perf_counter()
            if method == "flat":
                state, buffer, detail = ckpt.load_flat(store, k, device=device,
                                                       concurrency=concurrency)
            else:
                import io

                raw = store.get_bytes(k)
                state = torch.load(io.BytesIO(raw), map_location=device, weights_only=False)
                buffer = None
                detail = {"bytes": len(raw), "download_s": None, "rebuild_s": None}
            wall = time.perf_counter() - t0
        row = {
            "wall_s": wall,
            "throughput_gbps": detail["bytes"] / wall / 1e9,
            **{k2: v for k2, v in detail.items()},
            **cpu.as_dict(),
            "rdma_witness": w.as_dict(),
            "fabric_health": _fabric_delta(f0, _fabric()),
        }
        # Verify the restore on the first iteration: a fast restore that returns
        # wrong tensors is not a result.
        if r == 0 and reference is not None:
            row["verified"] = _verify(reference, state)
        runs.append(row)
        del state, buffer
        torch.cuda.empty_cache()
    return _aggregate(runs, "load")


def _verify(ref, got) -> dict:
    """Compare two checkpoint structures tensor-by-tensor."""
    n_checked = 0
    mismatched = []

    def walk(a, b, path="") -> None:
        nonlocal n_checked
        if isinstance(a, torch.Tensor):
            n_checked += 1
            if not isinstance(b, torch.Tensor):
                mismatched.append(f"{path}: not a tensor")
            elif a.shape != b.shape or a.dtype != b.dtype:
                mismatched.append(f"{path}: {tuple(a.shape)}/{a.dtype} vs {tuple(b.shape)}/{b.dtype}")
            elif not torch.equal(a.to(b.device), b):
                mismatched.append(f"{path}: values differ")
            return
        if isinstance(a, dict):
            for k in a:
                walk(a[k], b[k], f"{path}.{k}")
            return
        if isinstance(a, (list, tuple)):
            for i, v in enumerate(a):
                walk(v, b[i], f"{path}[{i}]")

    try:
        walk(ref, got)
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}", "tensors_checked": n_checked}
    return {
        "ok": not mismatched,
        "tensors_checked": n_checked,
        "mismatched": mismatched[:5],
    }


def _aggregate(runs: list[dict], phase: str) -> dict:
    walls = [r["wall_s"] for r in runs]
    thr = [r["throughput_gbps"] for r in runs]
    return {
        "phase": phase,
        "repeats": len(runs),
        "bytes": runs[0]["bytes"],
        "wall_s_median": statistics.median(walls),
        "wall_s_min": min(walls),
        "wall_s_max": max(walls),
        "throughput_gbps_median": statistics.median(thr),
        "throughput_gbps_best": max(thr),
        "cpu_cores_median": statistics.median(r["cpu_cores_used"] for r in runs),
        "runs": runs,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="B4/B5: checkpoint save/restore")
    ap.add_argument("--backends", default="rdma,http")
    ap.add_argument("--methods", default="", help="default: flat for rdma; flat,torch-save for http")
    ap.add_argument("--sizes-gib", default="1,4,16",
                    help="synthetic checkpoint sizes; 0 to skip synthetic")
    ap.add_argument("--resnet50", action="store_true", help="also benchmark a real ResNet-50 ckpt")
    ap.add_argument("--bucket", default="checkpoints")
    ap.add_argument("--endpoint", default=None)
    ap.add_argument("--endpoints", default="aistor1:9000,aistor2:9000")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--servers", default="aistor1:9000,aistor2:9000")
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--tag", default="")
    ap.add_argument("--skip-load", action="store_true")
    ap.add_argument("--chunk-mib", type=int, default=1024,
                    help="payload chunk size; must stay under cuObject's 4 GiB "
                         "registration limit, and smaller chunks buy parallelism")
    ap.add_argument("--concurrency", type=int, default=16,
                    help="parallel chunk transfers")
    args = ap.parse_args(argv)

    backends = [b.strip() for b in args.backends.split(",") if b.strip()]
    servers = [s.strip() for s in args.servers.split(",") if s.strip()]
    sizes = [float(x) for x in args.sizes_gib.split(",") if x.strip() and float(x) > 0]

    cfg = StoreConfig(bucket=args.bucket)
    if args.endpoint:
        cfg.endpoint = args.endpoint
    if args.endpoints:
        cfg.endpoints = [e.strip() for e in args.endpoints.split(",") if e.strip()]
        cfg.endpoint = cfg.endpoints[0]

    torch.cuda.set_device(torch.device(args.device))

    from minio import Minio

    admin = Minio(cfg.endpoint, cfg.access_key, cfg.secret_key, secure=cfg.secure)
    if not admin.bucket_exists(args.bucket):
        admin.make_bucket(args.bucket)

    workloads: list[tuple[str, dict, dict]] = []
    if args.resnet50:
        st, meta = resnet50_state(args.device)
        nb = ckpt.flatten(st).nbytes
        workloads.append((f"resnet50", st, {**meta, "bytes": nb}))
    for g in sizes:
        nb_target = int(g * 2**30)
        workloads.append((f"synthetic-{g:g}gib", synthetic_state(nb_target, args.device),
                          {"target_bytes": nb_target}))

    print("== B4/B5 checkpoint save/restore ==")
    print(f"   endpoints: {cfg.endpoints or [cfg.endpoint]}   device: {args.device}   repeats: {args.repeats}")

    hdr = (f"{'workload':>20} {'backend':>8} {'method':>11} {'GiB':>7} "
           f"{'save s':>8} {'save GB/s':>10} {'load s':>8} {'load GB/s':>10} {'cores':>6} {'ok':>4}")
    print("\n" + hdr)
    print("-" * len(hdr))

    results = []
    for wname, state, wmeta in workloads:
        for backend in backends:
            methods = (
                [m.strip() for m in args.methods.split(",") if m.strip()]
                if args.methods
                else (["flat"] if backend == "rdma" else ["flat", "torch-save"])
            )
            for method in methods:
                store = make_store(backend, cfg)
                key = f"{wname}/{backend}-{method}"
                row: dict = {"workload": wname, "backend": backend, "method": method,
                             "meta": wmeta}
                try:
                    row["save"] = bench_save(store, key, state, method, servers,
                                             args.repeats, args.chunk_mib << 20,
                                             args.concurrency)
                    if not args.skip_load:
                        row["load"] = bench_load(store, key, method, args.device, servers,
                                                 args.repeats, args.concurrency,
                                                 reference=state)
                except Exception as exc:
                    row["error"] = f"{type(exc).__name__}: {exc}"
                finally:
                    store.close()

                if "error" in row:
                    print(f"{wname:>20} {backend:>8} {method:>11}   FAILED: {row['error'][:60]}")
                    results.append(row)
                    continue

                sv = row["save"]
                ld = row.get("load")
                gib = sv["bytes"] / 2**30
                # Proof-of-RDMA: without a counter delta this is an HTTP result
                # wearing an RDMA label.
                if backend == "rdma":
                    wrote = sv["runs"][0]["rdma_witness"]["rdma_write_bytes"] or 0
                    if wrote < sv["bytes"] * 0.5:
                        row["error"] = "rdma-not-used-on-save"
                        print(f"{wname:>20} {backend:>8} {method:>11}   FAILED: RDMA not used "
                              f"(server counted {wrote:,.0f} of {sv['bytes']:,})")
                        results.append(row)
                        continue
                ok = (ld or {}).get("runs", [{}])[0].get("verified", {}).get("ok")
                print(f"{wname:>20} {backend:>8} {method:>11} {gib:>7.2f} "
                      f"{sv['wall_s_median']:>8.2f} {sv['throughput_gbps_median']:>10.2f} "
                      + (f"{ld['wall_s_median']:>8.2f} {ld['throughput_gbps_median']:>10.2f} "
                         if ld else f"{'-':>8} {'-':>10} ")
                      + f"{sv['cpu_cores_median']:>6.1f} "
                      + (f"{'yes' if ok else ('NO' if ok is False else '-'):>4}"), flush=True)
                results.append(row)

    rec = RunRecord(
        benchmark="b4b5-checkpoint" + (f"-{args.tag}" if args.tag else ""),
        backend=",".join(backends),
        params={
            "sizes_gib": sizes, "resnet50": args.resnet50, "repeats": args.repeats,
            "bucket": args.bucket, "endpoints": cfg.endpoints, "device": args.device,
            "chunk_mib": args.chunk_mib, "concurrency": args.concurrency,
        },
        results={"rows": results},
        environment={"gpus": gpu_info(), "torch": torch.__version__},
    )
    print(f"\nsaved: {rec.save(args.results_dir)}")
    return 0 if any("error" not in r for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
